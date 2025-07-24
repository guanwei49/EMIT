from swift.llm import TrainArguments, ModelArch, TemplateType, Model, ModelGroup, ModelMeta, register_model, ModelInfo, \
    register_template, Template
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.llm.template.vision_utils import transform_image
from swift.llm.train import SwiftSft

import torch
from torch import nn
from typing import List, Union, Literal
from typing import Any, Dict
from PIL import Image
from swift.utils import get_model_parameter_info, get_env_args
import numpy as np

from utils import if_use_comp

try:
    import orjson as json
except:
    import json

from model.internvl_chat import (InternVLChatConfig,CustomizedInternVLChatModel)

from transformers import AutoTokenizer
from model.dataset import build_transform, dynamic_preprocess
from model.constants import (QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN, COMP_CONTEXT_TOKEN)


def load_image(image, input_size=448, use_thumbnail=True, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    image = image.convert('RGB')
    transform = build_transform(is_train=True, input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class CustomizedInternvlTemplate(Template):
    skip_prompt = False
    num_image_token = None
    num_soft_prompt_token = None
    placeholder_tokens = ['<IMG_CONTEXT>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if self.mode == 'vllm':
            image_context = ['<image>\n']
        else:
            image_context = ['<img>', [-100], '</img>\n']
        return image_context


    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)

        image_size = get_env_args('image_size', int, 448)
        max_num = get_env_args('max_num', int, 12)
        if self.num_image_token is None:
            self.num_image_token = int((image_size // 14) ** 2 * (0.5 ** 2))
            # self.num_soft_prompt_token = 9 + ((image_size // 14) // 4) * ((image_size // 14) // 4)
            self.num_soft_prompt_token = 9 + ((image_size // 14) // 2) * ((image_size // 14) // 2)

        input_ids = encoded['input_ids']
        pixel_values = None
        images = inputs.images
        comp_pixel_values = None

        input_content = inputs.messages[0]['content']
        use_comp = (images is not None) and (len(images) == 2 and if_use_comp(input_content)) ## Set if the soft prompt is adopted.

        if images:
            num_patches_list = []
            pixel_values_images = []
            labels = encoded.get('labels')
            if use_comp:
                comp_pixel_values = []

            for image in images:
                temp = load_image(image, input_size=image_size, max_num=max_num, use_thumbnail=True)
                if use_comp:   #the first image is noraml for reference
                    comp_pixel_values.append(temp[-1].unsqueeze(0))
                pixel_values_images.append(temp)
                num_patches_list.append(temp.size(0))

            pixel_values = torch.cat(pixel_values_images, dim=0).to(self.model_info.torch_dtype)
            if use_comp:
                comp_pixel_values = torch.cat(comp_pixel_values, dim=0).to(self.model_info.torch_dtype)

            for num_patch in num_patches_list:
                index = findall(input_ids, -100)[0]
                img_tokens = self.processor.encode(IMG_CONTEXT_TOKEN, add_special_tokens=False) * self.num_image_token * num_patch
                input_ids = input_ids[:index] + img_tokens + input_ids[index + 1:]
                if labels is not None:
                    labels = labels[:index] + [-100] * len(img_tokens) + labels[index + 1:]

            if use_comp:
                if labels is not None:
                    index = (np.array(labels) < 0).sum()-5
                else:
                    index = len(input_ids)-5

                input_ids = input_ids[:index] + self.processor.encode(COMP_CONTEXT_TOKEN, add_special_tokens=False) * self.num_soft_prompt_token + input_ids[index:]

                if labels is not None:
                    labels = labels[:index] + [-100] * self.num_soft_prompt_token + labels[index:]

            # print(num_patches_list,self.processor.decode(input_ids))

            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
        encoded['pixel_values'] = pixel_values
        encoded['comp_pixel_values'] = comp_pixel_values
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # print(f'inputs.keys()  in customized_post_encode:{inputs.keys()}')
        # print(f'inputs in customized_post_encode:{inputs}')
        return inputs

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # multimodal
        res = {}
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        comp_pixel_values = [b['comp_pixel_values'] for b in batch if b.get('comp_pixel_values') is not None]

        if len(comp_pixel_values) > 0:
            res['comp_pixel_values'] = torch.stack(comp_pixel_values, 0)
        else:
            res['comp_pixel_values'] = None

        if len(pixel_values) > 0:
            res['pixel_values'] = torch.concat(pixel_values)

            image_sizes = [b['image_sizes'] for b in batch if b.get('image_sizes') is not None]
            if len(image_sizes) > 0:
                res['image_sizes'] = torch.concat(image_sizes)
        else:
            res['pixel_values'] = None

        pixel_values_videos = [b['pixel_values_videos'] for b in batch if b.get('pixel_values_videos') is not None]
        if len(pixel_values_videos) > 0:
            res['pixel_values_videos'] = torch.concat(pixel_values_videos)
        return res


def get_model_tokenizer_customizedinternvl(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True)
    tokenizer.tokenizer_path = tokenizer_path
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN, COMP_CONTEXT_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    comp_context_token_id = tokenizer.convert_tokens_to_ids(COMP_CONTEXT_TOKEN)

    config = InternVLChatConfig.from_pretrained(model_dir)
    if config.llm_config.model_type == 'internlm2':
        config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM

    else:
        config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA

    # model = CustomizedInternVLChatModel(config=config)
    # model = model.from_pretrained(
    #      model_dir, torch_dtype= model_info.torch_dtype, config=config,**model_kwargs)
    model = CustomizedInternVLChatModel.from_pretrained(
        model_dir, torch_dtype= model_info.torch_dtype, config=config,**model_kwargs)

    model.img_context_token_id = img_context_token_id
    model.comp_context_token_id = comp_context_token_id

    patch_size = model.config.vision_config.patch_size

    model.num_image_token = int(
        (model.config.force_image_size // patch_size) ** 2 * (model.config.downsample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True

    model_info.config = config if model is None else model.config

    return model, tokenizer
