import os
from collections import defaultdict

from tqdm import tqdm

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import random


def get_sim_img_paths(dataPath, model, processor, target_image_relative_paths, ref_image_relative_paths, image_features=None, k=5, batch_size=256, device = "cuda" if torch.cuda.is_available() else "cpu"):
    image_tensors = []
    for target_image_relative_path in target_image_relative_paths:
        image = Image.open(os.path.join(dataPath, target_image_relative_path))
        inputs = processor(images=image, return_tensors="pt")
        image_tensors.append(inputs['pixel_values'])

    target_features = []
    for i in range(0, len(image_tensors), batch_size):
        batch = torch.cat(image_tensors[i:i + batch_size]).to(device)
        image_features_batch = model.get_image_features(pixel_values=batch)
        target_features.append(image_features_batch.detach().cpu())

    target_features = torch.cat(target_features, dim=0)

    target_features = target_features / target_features.norm(dim=-1, keepdim=True)

    if target_image_relative_paths == ref_image_relative_paths:
        image_features = target_features

    if image_features is None:
        image_tensors = []
        for ref_image_relative_path in ref_image_relative_paths:
            image = Image.open(os.path.join(dataPath, ref_image_relative_path))
            inputs = processor(images=image, return_tensors="pt")
            image_tensors.append(inputs['pixel_values'])

        image_features = []
        for i in range(0, len(image_tensors), batch_size):
            batch = torch.cat(image_tensors[i:i + batch_size]).to(device)
            image_features_batch = model.get_image_features(pixel_values=batch)
            image_features.append(image_features_batch.detach().cpu())
            # image_features.append(image_features_batch)

        image_features = torch.cat(image_features, dim=0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 计算查询图片与所有目标图片之间的相似度
    similarities = torch.mm(target_features, image_features.T)

    _, indices = torch.topk(similarities, k + 1, dim=1)

    indices = indices[:, 1:]

    import numpy as np
    ref_image_relative_paths = np.array(ref_image_relative_paths)
    sim_img_paths_list = ref_image_relative_paths[indices]

    return sim_img_paths_list, image_features


def get_subfolders(directory):
    # 使用 os.listdir() 获取指定目录下的所有文件和文件夹
    items = os.listdir(directory)

    # 使用列表推导式筛选出所有文件夹
    subfolders = [item for item in items if os.path.isdir(os.path.join(directory, item))]

    return subfolders


def get_image_files(folder_path):
    # 常见图片文件后缀
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'svg']

    # 检查输入的文件夹是否存在
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")

    # 获取文件夹中所有的文件和子文件夹
    all_files = os.listdir(folder_path)

    # 过滤文件列表，只保留指定后缀的图片文件
    image_files = [file for file in all_files if file.split('.')[-1].lower() in image_extensions]

    image_files.sort()

    return image_files


def get_root_folder(path):
    # 获取驱动器或根文件夹（在Linux/Unix中，这通常是空字符串）
    drive, path_without_drive = os.path.splitdrive(path)

    # 迭代获取路径中的最根部文件夹
    root_folder = path_without_drive
    while os.path.dirname(root_folder):
        root_folder = os.path.dirname(root_folder)

    return os.path.basename(root_folder)


def getNormal(rootpath, type_path, normal_indicator='good'):
    '''
    获取这一类别所有正常图片（路径）
    Args:
        rootpath:
        type_path:

    Returns:

    '''
    normal_img_files = []

    path1 = os.path.join(rootpath, type_path, 'train', normal_indicator)
    path2 = os.path.join(rootpath, type_path, 'test', normal_indicator)

    for img in get_image_files(path1):
        normal_img_files.append(os.path.join(type_path, 'train', normal_indicator, img))

    for img in get_image_files(path2):
        normal_img_files.append(os.path.join(type_path, 'test', normal_indicator, img))

    return normal_img_files


def getAnomalous(rootpath, type_path, normal_indicator='good'):
    '''
    获取这一类别所有异常图片（路径）
    Args:
        rootpath:
        type_path:

    Returns:

    '''
    anomaly_img_files = []
    mask_files = []
    anomaly_types = []

    path = os.path.join(rootpath, type_path, 'test')
    anomalytypes = get_subfolders(path)
    anomalytypes.remove(normal_indicator)

    for anomalytype in anomalytypes:
        img_names = get_image_files(os.path.join(path, anomalytype))
        for img in img_names:
            anomaly_img_files.append(os.path.join(type_path, 'test', anomalytype, img))
        mask_path = os.path.join(rootpath, type_path, 'ground_truth')
        mask_img_names = get_image_files(os.path.join(mask_path, anomalytype))
        for img in mask_img_names:
            mask_files.append(os.path.join(type_path, 'ground_truth', anomalytype, img))
        if anomalytype not in ['ko', 'anomalous']:
            anomaly_types.extend([anomalytype] * len(img_names))
        else:
            anomaly_types.extend([None] * len(img_names))

    return anomaly_img_files, anomaly_types, mask_files


def getnormalRealIAD(rootpath, type_path):
    '''
    获取RealIAD数据集中，类别type_path内所有正常图片路径，我们只考虑了OK中的文件，NG中的没有异常的视角并没有考虑。
    '''

    normal_img_files = []

    path1 = os.path.join(rootpath, type_path, 'OK')

    for img_subpath in get_subfolders(path1):
        for img in get_image_files(os.path.join(path1, img_subpath)):
            normal_img_files.append(os.path.join(type_path, 'OK', img_subpath, img))

    return normal_img_files


def getanomalousRealIAD(rootpath, type_path):
    '''
    获取RealIAD数据集中，类别type_path内所有异常图片路径，NG中的没有异常的视角不算异常。
    '''

    anomaly_img_files = []
    mask_files = []
    anomaly_types = []

    path = os.path.join(rootpath, type_path, 'NG')
    anomalytypes = get_subfolders(path)

    anomalytype_map = {'AK': 'pit', 'BX': 'deformation', 'CH': 'abrasion',
                       'HS': 'scratch', 'PS': 'damage', 'QS': 'missing parts', 'YW': 'foreign objects',
                       'ZW': 'contamination'
                       }


    for anomalytype in anomalytypes:
        for img_subpath in get_subfolders(os.path.join(path, anomalytype)):
            for img in get_image_files(os.path.join(path, anomalytype, img_subpath)):
                if os.path.splitext(img)[1] == '.png':
                    img_name = os.path.splitext(img)[0] + '.jpg'

                    anomaly_img_files.append(os.path.join(type_path, 'NG', anomalytype, img_subpath, img_name))
                    mask_files.append(os.path.join(type_path, 'NG', anomalytype, img_subpath, img))
                    anomaly_types.append(anomalytype_map[anomalytype])


    return anomaly_img_files, anomaly_types, mask_files


def safe_get(lst, index, default=None):
    try:
        return lst[index]
    except IndexError:
        return default


def gen_structured_data(dataPath, model_path, k=10, batch_size=256, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)

    subdatasetName_list = ['MPDD', 'Real-IAD']

    objtypeFolder_list = []
    for name in subdatasetName_list:
        objtypeFolder_list.extend(
            [os.path.join(name, obj_type) for obj_type in get_subfolders(os.path.join(dataPath, name))])

    # 正常图片池{类别:{所有图片路径：{}}}}; 异常图片池{类别: {所有图片路径：{}}}}
    print(objtypeFolder_list)

    normal_image_pool = {}
    anomaly_image_pool = {}

    with tqdm(objtypeFolder_list) as pbar:
        for objtypeFolder in pbar:
            pbar.set_postfix({'Current item': objtypeFolder, 'Status': 'Processing'})
            # if 'BTAD' == get_root_folder(objtypeFolder):
            #     normal_img_relative_paths = getNormal(dataPath, objtypeFolder, normal_indicator='ok')
            #     anomaly_img_relative_paths, anomaly_types, mask_relative_paths = getAnomalous(dataPath, objtypeFolder, normal_indicator='ok')

            if 'MPDD' == get_root_folder(objtypeFolder):
                normal_img_relative_paths = getNormal(dataPath, objtypeFolder, normal_indicator='good')
                anomaly_img_relative_paths, anomaly_types, mask_relative_paths = getAnomalous(dataPath, objtypeFolder,
                                                                                              normal_indicator='good')

            elif 'Real-IAD' == get_root_folder(objtypeFolder):
                normal_img_relative_paths = getnormalRealIAD(dataPath, objtypeFolder)
                anomaly_img_relative_paths, anomaly_types, mask_relative_paths = getanomalousRealIAD(dataPath,
                                                                                                     objtypeFolder)

            else:
                continue

            image_features = None

            normal_image_pool[os.path.join(objtypeFolder, 'good')] = defaultdict(dict)
            for anomaly_type in set(anomaly_types):
                anomaly_image_pool[os.path.join(objtypeFolder, str(anomaly_type))] = defaultdict(dict)

            similar_templates_list, image_features = get_sim_img_paths(dataPath,  model, processor, normal_img_relative_paths,
                                                                       normal_img_relative_paths, k=k,batch_size=batch_size,device=device)
            for img_path, similar_templates in zip(normal_img_relative_paths, similar_templates_list):
                similar_templates = list(similar_templates)
                random_templates = random.sample(normal_img_relative_paths, k)
                normal_image_pool[os.path.join(objtypeFolder, 'good')][img_path] = {
                    'similar_templates': similar_templates, 'random_templates': random_templates}

            similar_templates_list, _ = get_sim_img_paths(dataPath, model, processor, anomaly_img_relative_paths,
                                                          normal_img_relative_paths,
                                                          image_features=image_features, k=k,batch_size=batch_size,device=device)

            for img_path, similar_templates, anomaly_type,mask_relative_path in zip(anomaly_img_relative_paths, similar_templates_list,anomaly_types,mask_relative_paths):
                similar_templates = list(similar_templates)

                random_templates = random.sample(normal_img_relative_paths, k)

                anomaly_image_pool[os.path.join(objtypeFolder, str(anomaly_type))][img_path] = {
                    'similar_templates': similar_templates, 'random_templates': random_templates,
                    'mask_path': mask_relative_path, 'anomaly_type': anomaly_type}

            import json

            with open(os.path.join(dataPath, 'anomaly_image_pool.json'), 'w', encoding='utf-8') as json_file:
                json.dump(anomaly_image_pool, json_file, ensure_ascii=False, indent=4)
            with open(os.path.join(dataPath, 'normal_image_pool.json'), 'w', encoding='utf-8') as json_file:
                json.dump(normal_image_pool, json_file, ensure_ascii=False, indent=4)

    ## object type recognition pool
    subdatasetName_list = ['Vision']

    objtypeFolder_list = []
    for name in subdatasetName_list:
        objtypeFolder_list.extend(
            [os.path.join(name, obj_type) for obj_type in get_subfolders(os.path.join(dataPath, name))])

    clas_img_pool = {}   #{类别:所有图片路径}
    with tqdm(objtypeFolder_list) as pbar:
        for objtypeFolder in pbar:
            img_relative_paths = []
            img_names = get_image_files(os.path.join(dataPath, objtypeFolder, 'train'))
            for img in img_names:
                img_relative_paths.append(os.path.join(objtypeFolder, 'train', img))
            img_names = get_image_files(os.path.join(dataPath, objtypeFolder, 'val'))
            for img in img_names:
                img_relative_paths.append(os.path.join(objtypeFolder, 'val', img))
            clas_img_pool[objtypeFolder.split(os.sep)[1]] = []
            for img_relative_path in img_relative_paths:
                clas_img_pool[objtypeFolder.split(os.sep)[1]].append(img_relative_path)
            print(clas_img_pool)
    with open(os.path.join(dataPath, 'clas_image_pool.json'), 'w', encoding='utf-8') as json_file:
        json.dump(clas_img_pool, json_file, ensure_ascii=False, indent=4)
