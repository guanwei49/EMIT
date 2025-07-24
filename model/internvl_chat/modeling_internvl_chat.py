# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
import shutil
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from swift.utils import get_env_args, is_deepspeed_enabled

from .conversation import get_conv_template
from model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM, GenerationMixin)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn

import kornia as Kor

import torch

logger = logging.get_logger(__name__)



def zero_out_far_values(flat_matrices, coords, rows, neighbor_dis):
    repeat_times = flat_matrices.shape[1]//(rows**2)
    # Create row and column indices for the grid
    row_indices = torch.arange(rows).unsqueeze(1).expand(rows, rows)
    col_indices = torch.arange(rows).unsqueeze(0).expand(rows, rows)

    # Prepare tensors for broadcasting
    x_coords = coords[:, 0].view(-1, 1, 1)
    y_coords = coords[:, 1].view(-1, 1, 1)

    # Calculate the distance masks
    row_distance_mask = (torch.abs(row_indices - x_coords) > neighbor_dis)
    col_distance_mask = (torch.abs(col_indices - y_coords) > neighbor_dis)

    # Calculate overall masks
    total_mask = row_distance_mask | col_distance_mask
    total_mask = total_mask.flatten(1, 2).repeat(1,repeat_times)
    # Apply masks to the flat matrices
    flat_matrices[total_mask] = 0

    # Return the modified matrices
    return flat_matrices

def chunked_cosine_similarity(query_patch_tokens_reshaped, normal_tokens_reshaped, chunk_size):
    # 获取维度大小
    batch_size, query_tokens, _, token_dim = query_patch_tokens_reshaped.shape
    _, _, normal_tokens, _ = normal_tokens_reshaped.shape

    # 初始化结果张量
    result = torch.empty(batch_size, query_tokens, normal_tokens, dtype=query_patch_tokens_reshaped.dtype,
                         device=query_patch_tokens_reshaped.device)

    # 按正常tokens进行分块计算
    for start in range(0, normal_tokens, chunk_size):
        end = min(start + chunk_size, normal_tokens)

        # 提取块
        normal_chunk = normal_tokens_reshaped[:, :, start:end, :]

        # 计算相似性
        chunk_similarity = F.cosine_similarity(query_patch_tokens_reshaped, normal_chunk, dim=-1)

        # 存储结果
        result[:, :, start:end] = chunk_similarity

    return result

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_dir = os.path.dirname(current_file_path)

class CustomizedInternVLChatModel(PreTrainedModel,GenerationMixin):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        self.image_size = image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        self.layers = [4, 8, 16, 20]
        self.rotates = [0,1,2,3]
        # self.rotates = [0,1,2]

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.comp_context_token_id = None

        # self.projector = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=llm_hidden_size // 2, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels=llm_hidden_size // 2, out_channels=llm_hidden_size, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(),
        #     # nn.AdaptiveAvgPool2d((3, 3))
        # )

        # self.projector_out_num_tokens = ((self.image_size // 14) // 4) * ((self.image_size // 14) // 4)

        self.projector = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=llm_hidden_size, kernel_size=4, stride=2, padding=1),
        )
        self.projector_out_num_tokens = ((self.image_size // 14) // 2) * ((self.image_size // 14) // 2)

        self.neighbor_dis = get_env_args('neighbor_dis', int, 6)


        self.num_soft_prompt_tokens = 9
        # self.soft_prompt = nn.Parameter(torch.randn(self.num_soft_prompt_tokens, llm_hidden_size), requires_grad=True)
        self.soft_prompt = nn.Embedding(self.num_soft_prompt_tokens, llm_hidden_size)

        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)



    def _init_weights(self, module):
        import torch.nn.init as init
        for module in self.projector:
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

        # Initialize the soft prompt embeddings
        self.soft_prompt.weight = nn.Parameter(torch.load(os.path.join(current_dir,'soft_prompt_ini.pt')), requires_grad=True)


    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def encode_image_for_one_shot(self, pixel_values):
        # Generate outputs
        with torch.no_grad():
            outputs = self.vision_model(pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            patch_features = [hidden_states[layer][:,1:,] for layer in self.layers]

        return patch_features   #list [(batch_size,1024,1024)]


    def encode_image_for_one_shot_with_aug(self, pixel_values):
        batch_size = pixel_values.shape[0]
        rotated_images = []

        # ---- 1. 多角度旋转图像 ----
        for j, degree in enumerate(self.rotates):
            rotated_img = torch.rot90(pixel_values, k=degree, dims=(2, 3))
            rotated_images.append(rotated_img)
        # 合并所有旋转后图片，batch变大了
        pixel_values = torch.cat(rotated_images, dim=0)  # [batch_size * len(rotates), C, H, W]

        # ---- 2. 特征提取 ----
        with torch.no_grad():
            outputs = self.vision_model(pixel_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple, 每层shape都是 [big_batch, n_token, dims]

            patch_features = [hidden_states[layer][:, 1:, ] for layer in self.layers]

            dims = hidden_states[0].shape[-1]
            num_patch = hidden_states[0].shape[1] - 1  # 通常ViT第0位是cls token


            for i in range(len(patch_features)):
                patch_features[i] = torch.stack(torch.split(patch_features[i], batch_size, dim=0), dim =0).transpose(0,1).reshape(
                    batch_size, len(self.rotates) * num_patch, dims)

        return patch_features

    def small_ref(self, normal_pixel_values, image_pixel_values):
        query_patch_tokens = self.encode_image_for_one_shot(image_pixel_values)
        # normal_patch_tokens = self.encode_image_for_one_shot(normal_pixel_values)
        normal_patch_tokens = self.encode_image_for_one_shot_with_aug(normal_pixel_values)

        sims = []

        batch_size, num_patch, dims = query_patch_tokens[-1].shape

        for i in range(len(query_patch_tokens)):
            query_patch_tokens_reshaped = query_patch_tokens[i].view(batch_size, num_patch, 1, dims)
            normal_tokens_reshaped = normal_patch_tokens[i].reshape(batch_size, 1, -1, dims)
            cosine_similarity_matrix = chunked_cosine_similarity(query_patch_tokens_reshaped, normal_tokens_reshaped, chunk_size=512)
            # cosine_similarity_matrix = F.cosine_similarity(query_patch_tokens_reshaped, normal_tokens_reshaped, dim=-1)

            width = int(np.sqrt(num_patch))
            # Convert coords to tensors
            coords = torch.tensor([(i, j) for i in range(width) for j in range(width)])
            coords = coords.repeat(batch_size, 1)

            cosine_similarity_matrix = zero_out_far_values(cosine_similarity_matrix.flatten(0,1), coords, rows=width, neighbor_dis= self.neighbor_dis).reshape((batch_size,num_patch,-1))

            sim_max, _ = torch.max(cosine_similarity_matrix, dim=-1)
            sims.append(sim_max)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(1 - sims[3][0].reshape(32, 32).detach().cpu().float().numpy(),
        #            cmap='viridis')  # 选择 colormap 颜色映射，比如 'viridis'
        # plt.colorbar()  # 添加颜色条以表示数值大小
        # plt.title('Matrix Visualization with Matplotlib')  # 添加标题
        # plt.savefig('/mnt/vlr/laishi/code/myswift/4.jpg')

        sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape((batch_size, 1, width, width))
        # sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
        anomaly_map_all = 1 - sim  # (batch_size, 1, 32, 32)

        del query_patch_tokens
        del normal_patch_tokens
        del sim
        del sims

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(anomaly_map_all[0,0].detach().cpu().float().numpy(), cmap='viridis')  # 选择 colormap 颜色映射，比如 'viridis'
        # plt.colorbar()  # 添加颜色条以表示数值大小
        # plt.title('Matrix Visualization with Matplotlib')  # 添加标题
        # plt.savefig('/mnt/vlr/laishi/code/myswift/0.jpg')

        return self.projector(anomaly_map_all).permute(0,2,3,1).flatten(1,2)    # (batch_size, 9,llm_dim)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            comp_pixel_values: Optional[List] = None,
            inputs_embeds: torch.FloatTensor = None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(f'pixel_values.shape: {pixel_values.shape}')
        # print(f'input_ids: {input_ids[:20]}')
        # print(f'comp_pixel_values: {comp_pixel_values.shape}')

        soft_prompt = self.soft_prompt(torch.arange(self.num_soft_prompt_tokens).to(input_ids.device))
        if comp_pixel_values is None:
            if is_deepspeed_enabled():
                comp_pixel_values_temp = torch.rand((1, 3, self.image_size, self.image_size)).to(torch.bfloat16).to(
                    input_ids.device)
                # comp_pixel_values_temp = torch.rand((1, 3, self.image_size, self.image_size)).to(pixel_values.dtype).to(
                #     input_ids.device)
                ref_tokens = torch.cat([soft_prompt.unsqueeze(0).repeat(1, 1, 1),
                                         self.small_ref(comp_pixel_values_temp, comp_pixel_values_temp)], dim=1)
        else:
            ref_tokens = torch.cat([soft_prompt.unsqueeze(0).repeat(comp_pixel_values.size()[0], 1, 1),
                        self.small_ref(comp_pixel_values[:, 0], comp_pixel_values[:, 1])], dim=1)

        print(ref_tokens.shape)


        # print(f'self.small_ref(comp_pixel_values[:,0], comp_pixel_values[:,1] {self.small_ref(comp_pixel_values[:,0], comp_pixel_values[:,1])}')
        # print(f'ref_tokens in forward {ref_tokens}')
        # print(f'self.soft_prompt.weight {self.soft_prompt.weight}')
        # print(f'self.soft_prompt.unsqueeze(0).repeat(batch_size, 1, 1){self.soft_prompt.unsqueeze(0).repeat(batch_size, 1, 1)}')
        # print(f'self.soft_prompt{self.soft_prompt}')

        # distances = torch.cdist(ref_tokens.flatten(0, 1).float(), self.language_model.model.embed_tokens.weight.float(), p=2)
        #
        # # 对每一个 ref_token，找出距离最近的 embed_token 的 ID
        # nearest_embed_token_ids = torch.argmin(distances, dim=1)
        # print(f'distances{distances}')
        # nearest_embed_token_ids = nearest_embed_token_ids.reshape((-1, 19))
        # print(f'distances{nearest_embed_token_ids}')

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        if pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values)
            # vit_embeds = vit_embeds[image_flags == 1]
            vit_batch_size = pixel_values.shape[0]

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
                if statistics is not None:
                    num_samples, num_padding_tokens, num_padding_images = statistics.tolist()
                    self.num_samples += num_samples
                    print(f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}')

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                      f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
                ignore_flag = True

            if comp_pixel_values is not None:
                selected = (input_ids == self.comp_context_token_id)
                # print(f'selected.sum() FORWARD {selected.sum()}')
                # print(f'ref_tokens.shape() FORWARD {ref_tokens.shape}')
                input_embeds[selected] = input_embeds[selected] * 0.0 + ref_tokens.reshape(-1, C)
            # else:
            #     print('ref_tokens NoNE IN FORWARD')


            input_embeds = input_embeds.reshape(B, N, C)
        else:
            ignore_flag = False

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        # print(labels[..., 1:][labels[..., 1:]!=-100],torch.argmax(logits[..., :-1, :][labels[..., 1:]!=-100], dim=-1))

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # torch.set_printoptions(sci_mode=False, precision=4)
            # a = torch.softmax(shift_logits, dim=1)
            # ax = a[shift_labels!=-100]
            # lab = shift_labels[shift_labels!=-100]
            # print(ax[list(range(len(lab))),lab.tolist()])

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def replace_i_image_with_text(self, input_string, replace, i):
        # Split the string into a list by "image"
        parts = input_string.split("<image>")

        # Check if there are enough "image" parts to replace the i-th one
        if i <= 0 or i >= len(parts):
            return "Invalid index: i must be between 1 and the number of 'image' occurrences."

        # Reconstruct the string, replacing the i-th "image" with "text"
        output_string = "<image>".join(parts[:i]) + replace + "<image>".join(parts[i:])

        return output_string

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, comp_pixel_values, questions, use_comps, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',   COMP_CONTEXT_TOKEN='<COMP_CONTEXT>', verbose=False, image_counts=None):

        device = pixel_values.device
        
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.comp_context_token_id = tokenizer.convert_tokens_to_ids(COMP_CONTEXT_TOKEN)

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, (num_patches, use_comp) in enumerate(zip(num_patches_list, use_comps)):
            question = questions[idx]

            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            for ith, num_patch in enumerate(num_patches):
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patch + IMG_END_TOKEN
                query = self.replace_i_image_with_text(query, image_tokens, 1)

            if use_comp:
                parts = query.split("<|im_end|>")

                query = "<|im_end|>".join(parts[:-1]) + COMP_CONTEXT_TOKEN *(self.num_soft_prompt_tokens + self.projector_out_num_tokens)  + "<|im_end|>" + parts[-1]

            queries.append(query)

        self.tokenizer = tokenizer

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            comp_pixel_values=comp_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses


    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            comp_pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        ref_tokens = None

        if comp_pixel_values is not None:
            soft_prompt = self.soft_prompt(torch.arange(self.num_soft_prompt_tokens).to(input_ids.device))

            ref_tokens = torch.cat([soft_prompt.unsqueeze(0).repeat(comp_pixel_values.size()[0], 1, 1),
                                    self.small_ref(comp_pixel_values[:, 0], comp_pixel_values[:, 1])], dim=1)




            # print(f'ref_tokens in generate {ref_tokens}')

            # distances = torch.cdist(ref_tokens.flatten(0, 1).float(),
            #                         self.language_model.model.embed_tokens.weight.float(), p=2)
            #
            # # 对每一个 ref_token，找出距离最近的 embed_token 的 ID
            # nearest_embed_token_ids = torch.argmin(distances, dim=1)
            # print(f'distances{distances}')
            # nearest_embed_token_ids = nearest_embed_token_ids.reshape((-1, self.num_soft_prompt_tokens + self.projector_out_num_tokens))
            # print(f'distances{nearest_embed_token_ids}')
            # print(f'decode{self.tokenizer.batch_decode(nearest_embed_token_ids)}')


        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            if ref_tokens is not None:
                selected = (input_ids == self.comp_context_token_id)
                print(f'selected.sum() {selected.sum()}')
                print(f'ref_tokens.shape() {ref_tokens.shape}')
                input_embeds[selected] = ref_tokens.reshape(-1, C)
            else:
                selected = (input_ids == self.comp_context_token_id)
                # print(f'selected.sum() {selected.sum()}')
                print('ref_tokens NoNE')

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
