import os
from typing import Optional

from swift import Seq2SeqTrainingArguments, get_logger, SwiftModel
from swift.trainers import Seq2SeqTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_peft_available
from peft import PeftModel
import torch
import shutil

logger = get_logger()

class CustomizedSeq2SeqTrainer(Seq2SeqTrainer):
    args: Seq2SeqTrainingArguments

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def set_ori_modelPath(self,path):
        self.ori_modelPath = path


    def _save_checkpoint(self, *args, **kwargs):
        result = super()._save_checkpoint(*args, **kwargs)

        if args[0].device.index == 0:
            ckpt = set([f for f in os.listdir(self.state.last_model_checkpoint) if
                   os.path.isfile(os.path.join(self.state.last_model_checkpoint, f))])
            ori = set([f for f in os.listdir(self.ori_modelPath) if
                 os.path.isfile(os.path.join(self.ori_modelPath, f))])
            ori = {filename for filename in ori if not filename.endswith('.safetensors')}

            copy_file_names = ori - ckpt
            copy_file_names.discard('README.md')

            def copy_files_to_folder(file_paths, destination_folder):
                try:
                    for file_path in file_paths:
                        # 拷贝每个文件到目标文件夹
                        shutil.copy(file_path, destination_folder)
                        # logger.info(f"File {file_path} is copied to {destination_folder}")

                except Exception as e:
                    logger.info(f"Error when copying file: {e}")

            copy_files = [os.path.join(self.ori_modelPath,file_name) for file_name in copy_file_names]
            copy_files_to_folder(copy_files,self.state.last_model_checkpoint)

        return result