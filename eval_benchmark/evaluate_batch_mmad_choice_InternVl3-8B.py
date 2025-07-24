import argparse
import json
import os
import random
import re
import time
import torch
from torch.utils.data import DataLoader
from difflib import get_close_matches
from model.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm
import json
import functools
import itertools
import multiprocessing as mp
from multiprocessing import Pool
from model.internvl_chat import (InternVLChatConfig, CustomizedInternVLChatModel)
from transformers import AutoTokenizer

from summary import caculate_accuracy_mmad
from summary_finer import caculate_accuracy_mmad_finer 



def load_image(image_file, input_size=448, use_thumbnail=True, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class MMADDataset(torch.utils.data.Dataset):
    def __init__(self, root, input_size=448, dynamic_image_size=True, max_num=6,  is_one_shot=True, similar_template=True, think =True):
        self.root = root
        with open(os.path.join(root, 'mmad.json'), "r") as file:
            self.data = json.load(file)
        with open(os.path.join(root, 'domain_knowledge.json'), "r") as file:
            self.domain_knowledge = json.load(file)
        self.indexer = []
        for img_path, v in self.data.items():
            total_question_num = len(v['conversation'])
            for i in range(total_question_num):
                self.indexer.append((img_path, i))

        # self.indexer = self.indexer[:20]

        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.max_num = max_num
        self.is_one_shot = is_one_shot
        self.similar_template = similar_template
        self.transform = build_transform(is_train=False, input_size=input_size)

        if think:
            self.instruct = "Begin by documenting your analysis process within <think> </think> tags. Conclude with your final answer enclosed in <answer> </answer> tags. The final answer should be either 'A' or 'B' or 'C' or 'D'. The output format should be as follows: <think>...</think><answer>...</answer>. Please adhere strictly to the prescribed format."
        else:
            self.instruct = "Do not include any text in your response."

    def __len__(self):
        return len(self.indexer)

    def parse_conversation(self, text_gt):
        Question = []
        Answer = []
        # 想要匹配的关键字
        keyword = "conversation"

        # 遍历字典中的所有键
        for key in text_gt.keys():
            # 如果键以关键字开头
            if key.startswith(keyword):
                # 获取对应的值
                conversation = text_gt[key]
                for i, QA in enumerate(conversation):
                    options_items = list(QA['Options'].items())
                    options_text = ""
                    for j, (key, value) in enumerate(options_items):
                        options_text += f"{key}. {value}\n"
                    questions_text = QA['Question']
                    Question.append(
                        {
                            "type": "text",
                            "text": f"{questions_text} \n"
                                    f"{options_text}"
                        },
                    )
                    Answer.append(QA['Answer'])

                break
        return Question, Answer

    def __getitem__(self, idx):
        query_image_path = self.indexer[idx][0]
        conversation_id = self.indexer[idx][1]
        text_gt = self.data[query_image_path]
        if self.similar_template:
            one_shot = text_gt["similar_templates"][0]
        else:
            one_shot = text_gt["random_templates"][0]

        questions, answers = self.parse_conversation(text_gt)
        question_type = text_gt['conversation'][conversation_id]['type']
        question = questions[conversation_id]
        answer = answers[conversation_id]
        dataset_r = query_image_path.split(os.sep)[0]
        obj_r = query_image_path.split(os.sep)[1]
        if dataset_r == 'DS-MVTec':
            dataset_r = 'MVTec'
        
        
        if ("Defect" in question_type or question_type == "Anomaly Detection"):
            rag = self.domain_knowledge[dataset_r][obj_r]
            rag = 'The following provides domain knowledge regarding the characteristics of a normal product, as well as the various possible types of defects:'+'\n'.join(rag.values())
        else:
            rag = ''

        if self.is_one_shot:
            image_path_list = [one_shot, query_image_path]
            # llm_input = "Image-1: <image>\nImage-2: <image>\nI have uploaded two images for review, each featuring a product. The product shown in the first image appears to be in perfect condition, free of any defects. To answer a multiple-choice question, please inspect the product in the second image. There is only one correct option.\nSelect your answer by responding with the letter corresponding to the correct option, such as 'A'. {}\n\nQuestion: {}".format(self.instruct, question['text'])
            llm_input = "Image-1: <image>\nImage-2: <image>\nI have uploaded two images for review, each featuring a product. The product shown in the first image appears to be in perfect condition, free of any defects. To answer a multiple-choice question, please inspect the product in the second image. There is only one correct option.\n{}\nSelect your answer by responding with the letter corresponding to the correct option, such as 'A'. {}\n\nQuestion: {}".format(
                rag, self.instruct, question['text'])  # with introducing rag（domain knowledge）

        else:
            image_path_list = [query_image_path]
            llm_input = "Query image: <image>\nTo answer a multiple-choice question, please inspect the product in the given image. There is only one correct option.\n{}\nSelect your answer by responding with the letter corresponding to the correct option, such as 'A'. {}\n\nQuestion:{}".format(
                           rag, self.instruct, question['text'])

        image_paths = [os.path.join(self.root, image_path) for image_path in image_path_list]

        use_comp = False

        pixel_values_list = []
        num_patches_list = []

        comp_pixel_values = [] if use_comp else None
        for image_path in image_paths:
            if self.dynamic_image_size:
                pixel_values = load_image(image_path, input_size=self.input_size,
                                          use_thumbnail=True,
                                          max_num=self.max_num)
            else:
                image = Image.open(image_path).convert('RGB')
                pixel_values = self.transform(image).unsqueeze(0)

            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))
            if use_comp:
                comp_pixel_values.append(pixel_values[-1].unsqueeze(0))

        final_pixel_values = torch.cat(pixel_values_list, dim=0)
        if use_comp:
            comp_pixel_values = torch.cat(comp_pixel_values, dim=0)

        answer_entry = {
            "image": query_image_path,
            "question": question,
            "question_type": text_gt["conversation"][conversation_id]['type'],
            'options': text_gt["conversation"][conversation_id]['Options'],
            "correct_answer": answer,
        }

        return llm_input, final_pixel_values, comp_pixel_values, use_comp, num_patches_list, answer_entry


def custom_collate_fn(batch):
    llm_inputs = [item[0] for item in batch]
    pixel_values = torch.cat([item[1] for item in batch],0)
    comp_pixel_values = [item[2] for item in batch if item[2] is not None]

    if len(comp_pixel_values) > 0:
        comp_pixel_values = torch.stack(comp_pixel_values, 0)
    else:
        comp_pixel_values = None

    use_comps = [item[3] for item in batch]

    num_patches_tensor_list = [item[4] for item in batch]
    answer_entry = [item[5] for item in batch]

    return llm_inputs, pixel_values, comp_pixel_values, use_comps, num_patches_tensor_list, answer_entry


from torch.utils.data import Sampler

class CustomSampler(Sampler):
    def __init__(self, id_list):
        self.id_list = id_list

    def __iter__(self):
        return iter(self.id_list)

    def __len__(self):
        return len(self.id_list)

def parse_answer(response_text, options=None):
    # pattern = re.compile(r'\bAnswer:\s*([A-Za-z])[^A-Za-z]*')
    # pattern = re.compile(r'(?:Answer:\s*[^A-D]*)?([A-D])[^\w]*')
    pattern = re.compile(r'\b([A-E])\b')
    # 使用正则表达式提取答案
    answers = pattern.findall(response_text)

    if len(answers) == 0 and options is None:
        answers = [response_text[0].upper()]
        if answers[0] not in ['A','B','C','D','E']:
            answers = ['A']

    if len(answers) == 0 and options is not None:
        print(f"Failed to extract answer from response: {response_text}")
        # 模糊匹配options字典来得到答案
        options_values = list(options.values())
        # 使用difflib.get_close_matches来找到最接近的匹配项
        closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
        if closest_matches:
            # 如果有匹配项，找到对应的键
            closest_match = closest_matches[0]
            for key, value in options.items():
                if value == closest_match:
                    answers.append(key)
                    break
    return answers


def run(rank, world_size, args, ds_collection):
    random.seed(args.seed)

    device = torch.device(rank)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)


    model = CustomizedInternVLChatModel.from_pretrained(
        args.checkpoint,  low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map=device).eval()

    if rank==0:
        print(model)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    with open(args.system, 'r', encoding='utf-8') as file:
        model.system_message = file.read()


    total_params = sum(p.numel() for p in model.parameters()) / 1e9

    if rank == 0:
        print(f'[test] total_params: {total_params}B')
        print(f'[test] image_size: {image_size}')
        print(f'[test] template: {model.config.template}')
        print(f'[test] dynamic_image_size: {args.dynamic}')

    dataset = MMADDataset(
        root=ds_collection['root'],
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        max_num=args.max_num,
        think = args.think
    )

    import math
    split_length = math.ceil(len(dataset) / world_size)
    curr_indexes = list(range(int(rank * split_length), min(int((rank + 1) * split_length), len(dataset))))
    # print(len(dataset))
    # curr_indexes = curr_indexes[-79:]
    # split_length = math.ceil(256 / world_size)
    # curr_indexes = list(range(int(rank * split_length), min(int((rank + 1) * split_length), 256)))
    # print(curr_indexes)
    # curr_indexes = index_np[curr_indexes]
    # print(curr_indexes)
    print("Split Chunk Length:" + str(len(curr_indexes)))

    sampler = CustomSampler(curr_indexes)

    if args.think:
        generation_config = dict(
            num_beams=2,
            max_new_tokens=ds_collection['max_new_tokens'],
            min_new_tokens=ds_collection['min_new_tokens'],
            do_sample=True,
            temperature=0.5,
            no_repeat_ngram_size=4,
            top_p=0.5
        )
    else:
        generation_config = dict(
        num_beams=1,
        max_new_tokens=ds_collection['max_new_tokens'],
        min_new_tokens=ds_collection['min_new_tokens'],
        do_sample=False,
        temperature=0,
    )


    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=custom_collate_fn, sampler= sampler,num_workers=0)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    answer_entry_list = []
    with torch.no_grad():
        with tqdm(dataloader) as pbar:
            for llm_inputs, pixel_values, comp_pixel_values, use_comps, num_patches_list, answer_entry in pbar:
                # print(llm_inputs)

                # print(num_patches_list)
                pbar.set_postfix({'Current GPU': rank})

                # print(num_patches_list)
                # num_patches_list = [nii.tolist() for nii in num_patches_list_ten]

                pixel_values = pixel_values.to(torch.bfloat16).to(device)
                if comp_pixel_values is not None:
                    comp_pixel_values = comp_pixel_values.to(torch.bfloat16).to(device)

                # pixel_values = pixel_values.half().to(device)
                # comp_pixel_values = comp_pixel_values.half().to(device)

                gpt_answers = model.batch_chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    comp_pixel_values=comp_pixel_values,
                    questions=llm_inputs,
                    use_comps=use_comps,
                    num_patches_list=num_patches_list,
                    generation_config=generation_config,
                    verbose=True
                )

                if args.think:
                    gpt_answers_temp = []
                    for ith_a, gpt_output in enumerate(gpt_answers):
                        # print(gpt_output)
                        # print('-'*20)
                        # Extract answer from content if it has think/answer tags
                        content_match = re.findall(r'<\s*answer\s*>(.*?)<\s*/\s*answer\s*>', gpt_output, re.DOTALL)
                        gpt_answers_temp.append(content_match[-1].strip() if len(content_match)>0 else gpt_output.strip())
                    gpt_answers = gpt_answers_temp
                
                for ith_a, gpt_answer in enumerate(gpt_answers):
                    # print(f'{ith_a},| {gpt_answer}')
                    # print(f'parse_answer(gpt_answer): {parse_answer(gpt_answer)}')
                    # print(f'parse_answer(gpt_answer)[0]: {parse_answer(gpt_answer)[0]}')
                    # print(answer_entry[ith_a])
                    # print('--')
                    answer_entry[ith_a]['gpt_answer'] = parse_answer(gpt_answer, answer_entry[ith_a]['options'])[0]

                # print('gpt_answers {}'.format(gpt_answers))
                # print('Parsed gpt_answers {}'.format([ans_item['gpt_answer'] for ans_item in answer_entry]))
                # print('correct_answer {}'.format([ans_item['correct_answer'] for ans_item in answer_entry]))

                answer_entry_list.extend(answer_entry)

    return answer_entry_list


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help="Root directory for MLLM model")
    parser.add_argument("--data-root", type=str, help="Root directory for MMAD data")
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--dynamic', action='store_true', default=True)
    parser.add_argument('--system', type=str, default='./sysprompt.txt')
    parser.add_argument('--out-dir', type=str, default="./results/benchmark")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--auto', action='store_true')
    parser.add_argument("--think", type=bool, default=False, help="If think before giving answer")
    args = parser.parse_args()

    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    # answers_json_file_name = f'Answer_{ds_name}_{time_prefix}.json'
    # answers_json_path = os.path.join(args.out_dir, answers_json_file_name)

    ckpt_name = os.path.basename(args.checkpoint)
    if args.think:
        answers_json_path = os.path.join(args.out_dir, f'Answer_think_choice_{ckpt_name}.json')
    else:
        answers_json_path = os.path.join(args.out_dir, f'Answer_choice_{ckpt_name}.json')

    ds_collection = {
        'root': args.data_root,
        'metric': None,
        'max_new_tokens': 1024,
        'min_new_tokens': 1,
    }

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    world_size = n_gpus
    print(f'world_size: {world_size}')
    with Pool(world_size) as pool:
        func = functools.partial(run, world_size=world_size, args=args, ds_collection=ds_collection)
        result_lists = pool.map(func, range(world_size))

    answer_entry_list = []
    for i in range(world_size):
        answer_entry_list = answer_entry_list + result_lists[i]

    with open(answers_json_path, "w") as file:
        json.dump(answer_entry_list, file, indent=4)

    print('Answer saved to {}'.format(answers_json_path))

    caculate_accuracy_mmad(answers_json_path)


    caculate_accuracy_mmad_finer(answers_json_path)