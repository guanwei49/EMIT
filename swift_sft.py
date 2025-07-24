import os

# 设置环境变量 'CUDA_VISIBLE_DEVICES'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 仅使用第一个GPU

from swift import get_logger
from swift.llm import TrainArguments,  Model, ModelGroup, ModelMeta, register_model, \
    register_template, register_model_arch, MultiModelKeys
from swift.llm.model.model.internlm import get_model_tokenizer_internvl
from swift.llm.template.template.internvl import InternvlTemplate
from swift.llm.template.template.utils import ChatmlTemplateMeta
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

from sft_Trainer import CustomizedSeq2SeqTrainer
from reg_class import CustomizedInternvlTemplate, get_model_tokenizer_customizedinternvl

try:
    import orjson as json
except:
    import json


logger = get_logger()


register_template(
    ChatmlTemplateMeta(
        'custom',
        # default_system='You are an AI assistant whose name is InternLM (书生·浦语).',
        default_system='None',
        # template_cls=InternvlTemplate,
        template_cls=CustomizedInternvlTemplate,
        auto_add_bos=True)
)

register_model_arch(
    MultiModelKeys(
        'custom_arc',
        language_model='language_model',
        aligner=[
            'projector',
            'soft_prompt'
        ],
        vision_tower=['vision_model','mlp1'],
    ))


register_model(
    ModelMeta(
        model_type='custom',
        model_groups=[
            ModelGroup([
                Model('CustomizedInternVL3-8B', 'CustomizedInternVL3-8B'),
            ]),
        ],
        template ='custom',
        get_function = get_model_tokenizer_customizedinternvl,

        # template=TemplateType.internvl2_5,
        # get_function = get_model_tokenizer_internvl,
        architectures=['CustomedInternVLChatModel'],
        model_arch = 'custom_arc',
        requires=['transformers>=4.37.2', 'timm'],
        is_multimodal=True,
        tags=['vision', 'video']
    )
)


class CustomizedSwiftSft(SwiftSft):
    args_class = TrainArguments
    args: args_class

    def __init__(self, args: Union[List[str], TrainArguments, None] = None) -> None:
        super().__init__(args)

    def run(self):
        args = self.args

        train_dataset, val_dataset = self._get_dataset()
        self._save_val_dataset(args.output_dir, val_dataset)
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        args.save_args()

        data_collator = self._get_data_collator()
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer = CustomizedSeq2SeqTrainer(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        trainer.set_ori_modelPath(self.args.model)
        return self.train(trainer)


def sft_main(args: Union[List[str], TrainArguments, None] = None):
    return CustomizedSwiftSft(args).main()

if __name__ == '__main__':
    sft_main()

