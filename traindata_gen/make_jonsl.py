import json
import os
import re
import cv2
from collections import Counter
import random
from tqdm import tqdm


def process_string(input_string):
    # 去掉数字
    no_digit_string = re.sub(r'\d+', '', input_string)
    # 替换下划线为空格
    no_underscore_string = no_digit_string.replace('_', ' ')
    # 转换成首字母大写的句子
    capitalized_sentence = no_underscore_string.capitalize().strip()+'.'
    return capitalized_sentence

def random_join(rag):
    vals = list(rag.values())
    random.shuffle(vals)
    return '\n'.join(vals)


def random_select_from_list(my_list, k):
    # 如果k超过列表长度，则直接返回整个列表
    if k >= len(my_list):
        return my_list

    # 使用random.sample从列表中随机选择k个不重复的元素
    selected_items = random.sample(my_list, k)
    return selected_items

def random_select_from_dict(my_dict, k):
    # 如果k超过字典长度，则直接返回整个字典
    if k >= len(my_dict):
        return my_dict

    # 从字典的键中随机选择k个
    selected_keys = random.sample(list(my_dict.keys()), k)

    # 根据选择的键获取对应的字典项
    selected_items = {key: my_dict[key] for key in selected_keys}
    return selected_items

def oversample_dict(data_list):
    # 统计每个类别的数量
    counter = Counter(d['adlabel'] for d in data_list)
    num_0 = counter[0]
    num_1 = counter[1]

    # 分割列表为两个类别
    dicts_0 = [d for d in data_list if d['adlabel'] == 0]
    dicts_1 = [d for d in data_list if d['adlabel'] == 1]

    # 进行过采样
    if num_0 > num_1:
        dicts_1_oversampled = dicts_1 + random.choices(dicts_1, k=num_0 - num_1)
        balanced_list = dicts_0 + dicts_1_oversampled
    else:
        dicts_0_oversampled = dicts_0 + random.choices(dicts_0, k=num_1 - num_0)
        balanced_list = dicts_1 + dicts_0_oversampled

    # 打乱列表顺序
    random.shuffle(balanced_list)

    return balanced_list

def get_white_areas(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 获取图像的尺寸
    height, width = image.shape

    # 将图像分为9个区域
    regions = {
        'top left': image[0:height // 3, 0:width // 3],
        'top': image[0:height // 3, width // 3:2 * width // 3],
        'top right': image[0:height // 3, 2 * width // 3:width],
        'left': image[height // 3:2 * height // 3, 0:width // 3],
        'center': image[height // 3:2 * height // 3, width // 3:2 * width // 3],
        'right': image[height // 3:2 * height // 3, 2 * width // 3:width],
        'bottom left': image[2 * height // 3:height, 0:width // 3],
        'bottom': image[2 * height // 3:height, width // 3:2 * width // 3],
        'bottom right': image[2 * height // 3:height, 2 * width // 3:width],
    }

    # 初始化一个字典来存储每个区域中白色像素的面积
    white_areas = {}

    # 计算每个区域中白色像素的数量
    for region_name, region in regions.items():
        white_area = cv2.countNonZero(region)
        # 仅记录白色像素数大于零的区域
        if white_area > 0:
            white_areas[region_name] = white_area

    # 如果没有白色区域，直接返回空列表
    if not white_areas:
        return []

    # 按照白色像素面积排序
    sorted_regions = sorted(white_areas.items(), key=lambda x: x[1], reverse=True)

    # 返回区域名称列表
    return [region[0] for region in sorted_regions]


regions = {
    'top left': ['Top left.', 'Left top.', 'Upper left.', 'Left upper.'],
    'top': ['Top.', 'Top center.', 'Summit.'],
    'top right': ['Top right.', 'Right top.', 'Upper right.', 'Right upper.'],
    'left': ['Left.', 'Leftward.', 'Left-hand.'],
    'center': ['Center.', 'Middle.'],
    'right': ['Right.', 'Rightward.', 'Right-hand.'],
    'bottom left': ['Bottom left.', 'Left bottom.', 'Lower left.', 'Left lower.'],
    'bottom': ['Bottom.', 'Bottom center.', 'Below.'],
    'bottom right': ['Bottom right.', 'Right bottom.', 'Lower right.', 'Right lower.']
}

anomaly_detection_choices = set(['No.', 'Yes.'])
defect_classification_choices = set()
defect_localization_choices = set(regions.keys())
object_classification_choices = set()


def select_and_format_elements(input_set, k, guaranteed_element):
    # 从集合中移除保证选取的元素（如果存在）
    if guaranteed_element in input_set:
        input_set = input_set - {guaranteed_element}

    # 从集合中随机选取 k-1 个元素
    selected_elements = random.sample(list(input_set), k - 1)

    # 将保证选取的元素添加到列表中
    selected_elements.append(guaranteed_element)

    # 打乱顺序以确保随机性，同时确保必选元素在其中
    random.shuffle(selected_elements)

    # 格式化成指定的字符串格式
    formatted_string = "\n".join([f"{chr(65 + i)}. {selected_elements[i]}" for i in range(k)])

    # 找出保证元素的位置
    position = selected_elements.index(guaranteed_element)
    element_position = f"{chr(65 + position)}"

    return formatted_string, element_position


def select_and_format_elements_loc(input_set, k, guaranteed_element,  exclude_set):
    filtered_set = input_set - exclude_set

    # 从集合中移除保证选取的元素（如果存在）
    if guaranteed_element in filtered_set:
        filtered_set = filtered_set - {guaranteed_element}

    selected_elements = [guaranteed_element]

    if k - 1 <= len(filtered_set):
        # 无放回采样，使用 random.sample()
        sampled_elements = random.sample(list(filtered_set), k - 1)
        selected_elements.extend(sampled_elements)
    else:
        # 可放回采样，使用 random.choices()
        sampled_elements = random.choices(list(filtered_set), k=k - 1)
        selected_elements.extend(sampled_elements)

    # 打乱顺序以确保随机性，同时确保必选元素在其中
    random.shuffle(selected_elements)

    # 找出保证元素的位置
    position = selected_elements.index(guaranteed_element)
    element_position = f"{chr(65 + position)}"

    replaced_list = [random.choice(regions[element]) for element in selected_elements]

    # 格式化成指定的字符串格式
    formatted_string = "\n".join([f"{chr(65 + i)}. {replaced_list[i]}" for i in range(k)])

    return formatted_string, element_position

def get_samples(id, image_list, adlabel, mask_path, chat_type, object_type, rag, anomaly_type=None):
    new_datas = []

    processed_object_type = process_string(object_type)
    if anomaly_type is not None:
        processed_anomaly_type = process_string(anomaly_type)

    if mask_path is not None:
        defect_location = get_white_areas(mask_path)
    else:
        defect_location = None

    if defect_location is not None and len(defect_location)==0: #Some pics are wrongly classified, e.g., /Real-IAD/bottle_cap/NG/QS/S0077/bottle_cap_0077_NG_QS_C5_20230926092821
        chat_type = chat_type.replace('anomalous', 'normal')
        adlabel = 0
        mask_path = None
        defect_location = None
        anomaly_type = None

    template = "Image-1: <image>\nImage-2: <image>\nI have uploaded two images for review, each featuring a product. The product shown in the first image appears to be in perfect condition, free of any defects. To answer a multiple-choice question, please inspect the product in the second image. There is only one correct option.\nThe following provides domain knowledge regarding the characteristics of a normal product, as well as the various possible types of defects: {}\nSelect your answer by responding with the letter corresponding to the correct option, such as 'A'. Begin by documenting your analysis process within <think> </think> tags. Conclude with your final answer enclosed in <answer> </answer> tags. The final answer should be either 'A' or 'B' or 'C' or 'D'. The output format should be as follows: <think>...</think><answer>...</answer>. Please adhere strictly to the prescribed format.\n\nQuestion: {}"


    if chat_type == 'one_shot_anomalous':
        ## anomaly detection
        formatted_string, solution = select_and_format_elements(anomaly_detection_choices, 2, 'Yes.')
        question = f"Is there any defect in the object? \n{formatted_string}\n"
        messages = [
            {"role": "user",
             "content": template.format(random_join(rag),question)},
        ]
        new_datas.append({
            "id": id,
            "defect_location": defect_location,
            "images": image_list,
            "object_type": object_type,
            "mask_path": mask_path,
            "adlabel": adlabel,
            "chat_type": chat_type,
            "anomaly_type": anomaly_type,
            "task_type": "Anomaly Detection",
            "messages": messages,
            "solution": solution
        })
        id += 1

        # defect classification
        if anomaly_type is not None:
            formatted_string, solution = select_and_format_elements(defect_classification_choices, 4, processed_anomaly_type)
            question = f"There is a defect in the object. What is the type of the defect? \n{formatted_string}\n"
            messages = [
                {"role": "user",
                 "content": template.format(random_join(rag), question)},
            ]
            new_datas.append({
                "id": id,
                "defect_location": defect_location,
                "images": image_list,
                "object_type": object_type,
                "mask_path": mask_path,
                "adlabel": adlabel,
                "chat_type": chat_type,
                "anomaly_type": anomaly_type,
                "task_type": "Defect Classification",
                "messages": messages,
                "solution": solution
            })
            id += 1



        ## defect localization
        if len(defect_location) < len(defect_localization_choices):
            formatted_string, solution = select_and_format_elements_loc(defect_localization_choices, 4, defect_location[0], exclude_set=set(defect_location[1:]))
            question = f"There is a defect in the object. Where is the defect? \n{formatted_string}\n"
            messages = [
                {"role": "user",
                 "content": template.format(random_join(rag), question)},
            ]
            new_datas.append({
                "id": id,
                "defect_location": defect_location,
                "images": image_list,
                "object_type": object_type,
                "mask_path": mask_path,
                "adlabel": adlabel,
                "chat_type": chat_type,
                "anomaly_type": anomaly_type,
                "task_type": "Defect Localization",
                "messages": messages,
                "solution": solution
            })
            id += 1

    elif chat_type == 'one_shot_normal':
        ## anomaly detection
        formatted_string, solution = select_and_format_elements(anomaly_detection_choices, 2, 'No.')
        question = f"Is there any defect in the object? \n{formatted_string}\n"
        messages = [
            {"role": "user",
             "content": template.format(random_join(rag), question)},
        ]
        new_datas.append({
            "id": id,
            "defect_location": defect_location,
            "images": image_list,
            "object_type": object_type,
            "mask_path": mask_path,
            "adlabel": adlabel,
            "chat_type": chat_type,
            "anomaly_type": anomaly_type,
            "task_type": "Anomaly Detection",
            "messages": messages,
            "solution": solution
        })
        id += 1

    if chat_type == 'one_shot_normal' or chat_type == 'obj_clas':
        # object classification
        template = "Query image: <image>\nTo answer a multiple-choice question, please inspect the product in the given image. There is only one correct option.\nSelect your answer by responding with the letter corresponding to the correct option, such as 'A'. Begin by documenting your analysis process within <think> </think> tags. Conclude with your final answer enclosed in <answer> </answer> tags. The final answer should be either 'A' or 'B' or 'C' or 'D'. The output format should be as follows: <think>...</think><answer>...</answer>. Please adhere strictly to the prescribed format.\n\nQuestion: {}"
        formatted_string, solution = select_and_format_elements(object_classification_choices, 4, processed_object_type)
        question = f"What kind of product is in the image? \n{formatted_string}\n"
        messages = [
                {"role": "user",
                 "content": template.format(question)},
        ]
        if len(image_list) == 2:
            image_list = [image_list[1]]
        new_datas.append({
                "id": id,
                "defect_location": defect_location,
                "images": image_list,
                "object_type": object_type,
                "mask_path": mask_path,
                "adlabel": adlabel,
                "chat_type": chat_type,
                "anomaly_type": anomaly_type,
                "task_type": "Object Classification",
                "messages": messages,
                "solution": solution
        })
        id += 1

    return new_datas


def make_jonsl(dataPath, number_per_type=100): # 每个类型随机取number_per_type个样本
    save_file_path = os.path.join(dataPath, 'grpo_train_data.jsonl')
    anomaly_image_pool_file = os.path.join(dataPath, 'anomaly_image_pool.json')
    normal_image_pool_file = os.path.join(dataPath, 'normal_image_pool.json')
    clas_image_pool_file = os.path.join(dataPath, 'clas_image_pool.json')
    domain_knowledge_file = os.path.join(dataPath, 'domain_knowledge_trainData.json')

    global defect_classification_choices
    global object_classification_choices

    with open(domain_knowledge_file, 'r', encoding='utf-8') as f:
        domain_knowledge = json.load(f)

    with open(anomaly_image_pool_file, 'r', encoding='utf-8') as file:
        anomaly_image_pool = json.load(file)

    with open(normal_image_pool_file, 'r', encoding='utf-8') as file:
        normal_image_pool = json.load(file)

    with open(clas_image_pool_file, 'r', encoding='utf-8') as file:
        clas_image_pool = json.load(file)


    for objtypeFolder in anomaly_image_pool.keys():   ##prepare choices
        defect_classification_choices.add(objtypeFolder.split(os.sep)[2])
        object_classification_choices.add(objtypeFolder.split(os.sep)[1])

    for objtypeFolder in normal_image_pool.keys():   ##prepare choices
        object_classification_choices.add(objtypeFolder.split(os.sep)[1])

    object_classification_choices = object_classification_choices.union(clas_image_pool.keys())
    defect_classification_choices.discard('None')

    object_classification_choices = set([process_string(obj_clas) for obj_clas in object_classification_choices])
    defect_classification_choices = set([process_string(def_clas) for def_clas in defect_classification_choices])

    id = 0
    data_list = []
    # 构建训练样本
    with tqdm(anomaly_image_pool.keys()) as pbar:
        for objtypeFolder in pbar:
            pbar.set_postfix({'Current item': objtypeFolder, 'Status': 'Processing'})

            anomaly_imgfiles = anomaly_image_pool[objtypeFolder]

            for img_file, v in random_select_from_dict(anomaly_imgfiles,number_per_type).items():
                ref_img = v['similar_templates'][0]
                img_mask_filePath = v['mask_path']

                dataset_r, object_type,_ = objtypeFolder.split(os.sep)

                rag = domain_knowledge[dataset_r][object_type]

                ref_img = os.path.join(dataPath, ref_img)
                img_mask_filePath = os.path.join(dataPath, img_mask_filePath)
                img_file = os.path.join(dataPath, img_file)

                data_dicts = get_samples(id, [ref_img, img_file], 1, img_mask_filePath, 'one_shot_anomalous',
                                         object_type=object_type, rag=rag, anomaly_type=v['anomaly_type'])
                data_list.extend(data_dicts)
                id += len(data_dicts)

    with tqdm(normal_image_pool.keys()) as pbar:
        for objtypeFolder in pbar:
            normal_imgfiles = normal_image_pool[objtypeFolder]

            for img_file, v in random_select_from_dict(normal_imgfiles,number_per_type).items():
                ref_img = v['similar_templates'][0]
                dataset_r, object_type, _ = objtypeFolder.split(os.sep)

                rag = domain_knowledge[dataset_r][object_type]

                ref_img = os.path.join(dataPath, ref_img)
                img_file = os.path.join(dataPath, img_file)

                data_dicts = get_samples(id, [ref_img, img_file], 0, None, 'one_shot_normal', object_type=object_type,rag=rag,
                                         anomaly_type=None)
                data_list.extend(data_dicts)
                id += len(data_dicts)

    with tqdm(clas_image_pool.keys()) as pbar:
        for object_type in pbar:
            img_files = clas_image_pool[object_type]
            for img_file in random_select_from_list(img_files,number_per_type):
                img_file = os.path.join(dataPath, img_file)
                data_dicts = get_samples(id, [img_file], 0, None, 'obj_clas', object_type=object_type, rag= None,
                                         anomaly_type=None)
                data_list.extend(data_dicts)
                id += len(data_dicts)


    with open(save_file_path, 'w') as file:
        for record in data_list:
            json_line = json.dumps(record)
            file.write(json_line + '\n')

