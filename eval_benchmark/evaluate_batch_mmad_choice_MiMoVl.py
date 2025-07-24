from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Pool
import functools
import math
import argparse
import re
from difflib import get_close_matches
from summary import caculate_accuracy_mmad
from summary_finer import caculate_accuracy_mmad_finer


class MMADDataset(torch.utils.data.Dataset):
    def __init__(self, root, data, domain_knowledge_path, system,  is_one_shot=True, think = True):
        self.root = root
        with open(data, "r") as file:
            self.data = json.load(file)
            
        with open(domain_knowledge_path, "r") as file:
            self.domain_knowledge = json.load(file)

        self.indexer = []
        for img_path, v in self.data.items():
            total_question_num = len(v["conversation"])
            for i in range(total_question_num):
                self.indexer.append((img_path, i))

        # self.indexer = self.indexer[:50]


        self.is_one_shot = is_one_shot
        with open(system, 'r', encoding='utf-8') as file:
            self.system_message = file.read()

        if think:
            self.instruct = "Begin by documenting your analysis process within <think> </think> tags. Conclude with your final answer enclosed in <answer> </answer> tags. The final answer should be either 'A' or 'B' or 'C' or 'D'. The output format should be as follows: <think>...</think><answer>...</answer>. Please adhere strictly to the prescribed format."
        else:
            self.instruct = "Do not include any text in your response."

    def parse_conversation(self, text_gt):
        Question = []
        Answer = []
        keyword = "conversation"

        # 遍历字典中的所有键并寻找对话内容
        for key in text_gt.keys():
            if key.startswith(keyword):  # 如果键以关键字开头
                conversation = text_gt[key]
                for i, QA in enumerate(conversation):
                    options_items = list(QA["Options"].items())
                    options_text = ""
                    for j, (key, value) in enumerate(options_items):
                        options_text += f"{key}. {value}\n"
                    questions_text = QA["Question"]
                    Question.append(
                        {
                            "type": "text",
                            "text": f"{questions_text} \n{options_text}"
                        },
                    )
                    Answer.append(QA["Answer"])
                break  # 找到对话后退出循环
        return Question, Answer

    def __getitem__(self, idx):
        query_image_path, conversation_id = self.indexer[idx]
        text_gt = self.data[query_image_path]

        # 图像所属数据集和对象
        dataset_r = query_image_path.split(os.sep)[0]
        obj_r = query_image_path.split(os.sep)[1]
        if dataset_r == "DS-MVTec":
            dataset_r = "MVTec"

        questions, answers = self.parse_conversation(text_gt)
        question_type = text_gt["conversation"][conversation_id]["type"]
        question = questions[conversation_id]
        answer = answers[conversation_id]

        if ("Defect" in question_type or question_type == "Anomaly Detection"):
            rag = self.domain_knowledge[dataset_r][obj_r]
            rag = 'The following provides domain knowledge regarding the characteristics of a normal product, as well as the various possible types of defects:'+'\n'.join(rag.values())
        else:
            rag = ''

        # 如果是对比类型问题，使用领域知识
        if self.is_one_shot:
            image_paths = [
                f"file://{os.path.join(self.root, text_gt['similar_templates'][0])}",
                f"file://{os.path.join(self.root, query_image_path)}",
            ]
            llm_input =[ {"role": "system", "content": self.system_message},
                {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_paths[0]},
                    {"type": "image", "image": image_paths[1]},
                    {"type": "text", "text": (
                       "I have uploaded two images for review, each featuring a product. The product shown in the first image appears to be in perfect condition, free of any defects. To answer a multiple-choice question, please inspect the product in the second image. There is only one correct option.\n{}\nSelect your answer by responding with the letter corresponding to the correct option, such as 'A'. {}\n\nQuestion: {}".format(
                    rag, self.instruct, question['text']) 
                    )}
                ]
            }]
        else:
            image_paths = [f"file://{os.path.join(self.root, query_image_path)}"]
            llm_input =[  {"role": "system", "content": self.system_message},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_paths[0]},
                    {"type": "text", "text": (
                        "To answer a multiple-choice question, please inspect the product in the given image. There is only one correct option.\n{}\nSelect your answer by responding with the letter corresponding to the correct option, such as 'A'. {}\n\nQuestion: {}".format(
                           rag, self.instruct, question['text'])
                    )}
                ]
            }
            ]

        answer_entry = {
            "image": query_image_path,
            "question": question,
            "question_type": question_type,
            "options": text_gt["conversation"][conversation_id]["Options"],
            "correct_answer": answer,
        }

        return llm_input, answer_entry

    def __len__(self):
        return len(self.indexer)


# 定义collate函数用于DataLoader
def custom_collate_fn(batch):
    llm_inputs = [item[0] for item in batch]
    answer_entries = [item[1] for item in batch]
    return llm_inputs, answer_entries

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


# 定义生成和推理逻辑
def run_inference(rank, world_size, args):
    device = torch.device(f"cuda:{rank}")

    # 加载新模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )

    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    dataset = MMADDataset(
        root=args.data_root,
        data=args.data_file,
        domain_knowledge_path=args.domain_knowledge,
        system = args.system,
        is_one_shot= True,
        think = args.think
    )
    split_length = math.ceil(len(dataset) / world_size)
    curr_indexes = list(
        range(int(rank * split_length), min(int((rank + 1) * split_length), len(dataset)))
    )
    sampled_dataset = torch.utils.data.Subset(dataset, curr_indexes)
    dataloader = DataLoader(
        sampled_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, shuffle=False
    )

    if args.think:
        generation_config = dict(
        num_beams=2,
        max_new_tokens=1024,
        min_new_tokens=1,
        do_sample=True,
        temperature=0.5,
        no_repeat_ngram_size=4,
        top_p=0.5
    )
    else:
        generation_config = dict(
        num_beams=1,
        max_new_tokens=1024,
        min_new_tokens=0,
        do_sample=False,
        temperature=0,
    )

    
    generated_answers = []
    with torch.no_grad():
        with tqdm(dataloader) as pbar:
            for llm_inputs, answer_entry in pbar:
                pbar.set_postfix({'Current GPU': rank})
                texts = [
                    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in llm_inputs
                ]
                image_inputs, video_inputs = process_vision_info(llm_inputs)
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    padding_side='left',
                    return_tensors="pt",
                ).to(device)
                # print(llm_inputs)
                generated_ids = model.generate(**inputs, **generation_config)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ] 
                gpt_answers = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
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

                generated_answers.extend(answer_entry)

    return generated_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, help="Root directory for MLLM model")
    parser.add_argument("--data-root", type=str, help="Root directory for MMAD data")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--out-dir", type=str, default="./results/benchmark", help="Output directory for results")
    parser.add_argument('--system', type=str, default='./sysprompt.txt')
    parser.add_argument("--think", type=bool, default=False, help="If think before giving answer")
    args = parser.parse_args()

    args.data_file = os.path.join(args.data_root, 'mmad.json')
    args.domain_knowledge = os.path.join(args.data_root, 'domain_knowledge.json')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    ckpt_name = os.path.basename(args.checkpoint)
    if args.think:
        answers_json_path = os.path.join(args.out_dir, f'Answer_think_choice_{ckpt_name}.json')
    else:
        answers_json_path = os.path.join(args.out_dir, f'Answer_choice_{ckpt_name}.json')

    print(answers_json_path)

    torch.multiprocessing.set_start_method("spawn")
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus

    # run_inference(0,world_size,args)


    with Pool(world_size) as pool:
        func = functools.partial(run_inference, world_size=world_size, args=args)
        result_lists = pool.map(func, range(world_size))

    answer_entry_list = []
    for i in range(world_size):
        answer_entry_list = answer_entry_list + result_lists[i]

    with open(answers_json_path, "w") as file:
        json.dump(answer_entry_list, file, indent=4)

    print('Answer saved to {}'.format(answers_json_path))

    caculate_accuracy_mmad(answers_json_path)


    caculate_accuracy_mmad_finer(answers_json_path)