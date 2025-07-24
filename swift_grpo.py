from swift import get_logger
from swift.llm import TrainArguments, ModelArch, TemplateType, Model, ModelGroup, ModelMeta, register_model, ModelInfo, \
    register_template, Template, register_model_arch, MultiModelKeys, RLHFArguments, HfConfigFactory
from swift.llm.template.template.utils import ChatmlTemplateMeta
from swift.llm.train import SwiftSft
from swift.llm.train.kto import prepare_kto_dataset
from typing import List, Union

from swift.utils import get_model_parameter_info

from grpo_Trainer import GRPOTrainer
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

class SwiftRLHF(SwiftSft):
    args_class = RLHFArguments
    args: args_class

    def _prepare_model_tokenizer(self):
        from swift.llm.infer.utils import prepare_adapter
        args = self.args
        for key in ['ref', 'reward', 'value']:
            origin_key = key
            setattr(self, f'{key}_model', None)
            if key == 'value':
                if args.rlhf_type == 'ppo':
                    key = 'reward'
                else:
                    continue
            model_id_or_path = getattr(args, f'{key}_model')
            if model_id_or_path is None:
                continue
            model_type = getattr(args, f'{key}_model_type')
            model_revision = getattr(args, f'{key}_model_revision')
            adapters = args.adapters if key == 'ref' else args.reward_adapters
            if origin_key == 'ref':
                task_type = args.task_type
                num_labels = None
            else:
                task_type = 'seq_cls'
                num_labels = 1
            # Be aware of the unexpected behavior caused by double monkey patching.
            model, processor = args.get_model_processor(
                model=model_id_or_path,
                model_type=model_type,
                model_revision=model_revision,
                task_type=task_type,
                num_labels=num_labels)

            model = prepare_adapter(args, model, adapters)
            if origin_key in {'ref', 'reward'}:
                model.requires_grad_(False).eval()
            else:
                model = self.prepare_model(args, model, task_type=task_type)
                logger.info(f'value_model: {model}')
                model_parameter_info = get_model_parameter_info(model)
                self.train_msg['value_model_parameter_info'] = model_parameter_info
                logger.info(f'value_model_parameter_info: {model_parameter_info}')
            setattr(self, f'{origin_key}_model', model)
            HfConfigFactory.set_model_config_attr(model, 'use_cache', False)
            if origin_key == 'reward' and args.rlhf_type == 'grpo':
                reward_template = self.args.get_template(processor, processor.model_meta.template)
                if reward_template.use_model:
                    reward_template.model = model
                self.reward_template = reward_template

        super()._prepare_model_tokenizer()

    def _prepare_template(self) -> None:
        args = self.args
        super()._prepare_template()
        model_mapping = {'kto': 'kto', 'ppo': 'pt', 'grpo': 'pt'}
        self.template.set_mode(model_mapping.get(args.rlhf_type, 'rlhf'))

        if args.rlhf_type == 'ppo':
            args.training_args.stop_token_id = self.template.template_meta.stop_token_id

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _get_trainer_kwargs(self):
        trainer_kwargs = {}
        for key in ['ref', 'reward', 'value']:
            key = f'{key}_model'
            model = getattr(self, key, None)
            if model or self.args.rlhf_type == 'ppo':
                trainer_kwargs[key] = model
        if hasattr(self, 'reward_template'):
            trainer_kwargs['reward_template'] = self.reward_template
        if self.args.rlhf_type == 'grpo':
            trainer_kwargs['reward_funcs'] = self.args.reward_funcs
            trainer_kwargs['vllm_client'] = self.args.vllm_client
        return trainer_kwargs

    def run(self):
        args = self.args

        train_dataset, val_dataset = self._get_dataset()
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)

        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        args.save_args()

        data_collator = self._get_data_collator()
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer = GRPOTrainer(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)

def rlhf_main(args: Union[List[str], RLHFArguments, None] = None):
    return SwiftRLHF(args).main()


# from swift.llm import rlhf_main

if __name__ == '__main__':
    rlhf_main()
