import collections
import json
import os
from tqdm import tqdm
import random

from PIL import Image

from SFT.traindata_gen.utils import encode_image, request


def is_image_black(image_path):
    # 打开图像
    with Image.open(image_path) as img:
        # 将图像转换为灰阶模式
        gray_img = img.convert('L')
        # 获取图像的像素值
        pixels = list(gray_img.getdata())

        # 检查所有像素值是否为0
        return all(pixel == 0 for pixel in pixels)

def gen_domain_knowledge(dataPath):
    anomaly_image_pool_file = os.path.join(dataPath, 'anomaly_image_pool.json')
    normal_image_pool_file = os.path.join(dataPath, 'normal_image_pool.json')
    domain_knowledge_file = os.path.join(dataPath, 'domain_knowledge_trainData.json')


    with open(anomaly_image_pool_file, 'r', encoding='utf-8') as file:
        anomaly_image_pool = json.load(file)

    with open(normal_image_pool_file, 'r', encoding='utf-8') as file:
        normal_image_pool = json.load(file)


    domain_knowledge = collections.defaultdict(dict)

    with tqdm(anomaly_image_pool.keys()) as pbar:
        for objtypeFolder in pbar:
            pbar.set_postfix({'Current item': objtypeFolder, 'Status': 'Processing'})

            anomaly_imgfiles = anomaly_image_pool[objtypeFolder]

            dataset_name, object_name, _ = objtypeFolder.split(os.sep)

            while True:
                image_path = random.choice(list(anomaly_imgfiles.keys()))
                mask_path = os.path.join(dataPath, anomaly_imgfiles[image_path]['mask_path'])
                anomaly_type = anomaly_imgfiles[image_path]['anomaly_type']
                ref_image_path = os.path.join(dataPath, anomaly_imgfiles[image_path]['similar_templates'][0])
                image_path = os.path.join(dataPath, image_path)
                break
                # if not is_image_black(mask_path):
                #     break

            if anomaly_type is None:
                anomaly_type = 'General'


            content = []
            content.append \
                ({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ref_image_path)}"}})
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
            content.append \
                ({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(mask_path)}"}})
            content.append({"type": "text",
                            "text": f"I am providing three images. The first two images feature the object labeled \"{object_name}\". In the first image, the object is in good condition. The second image displays a \"{anomaly_type}\" defect, while the third image presents a mask highlighting the defect identified in the second image. Your task is to specify the visual characteristics of the \"{anomaly_type}\" defect occurring on the object. Instead of describing the defect in a specific image, generate descriptive text that enables others to identify whether \"{object_name}\" in other images exhibits this kind of defect."})

            description = request(content)
            description = description.replace('\n',' ')

            if object_name not in domain_knowledge[dataset_name]:
                domain_knowledge[dataset_name][object_name] = collections.defaultdict(dict)
                domain_knowledge[dataset_name][object_name][anomaly_type] = f"<{anomaly_type.replace('_', ' ').title()} Defect>\n Description: " + description
            else:
                domain_knowledge[dataset_name][object_name][anomaly_type] = f"<{anomaly_type.replace('_', ' ').title()} Defect>\n Description: " + description


    with tqdm(normal_image_pool.keys()) as pbar:
        for objtypeFolder in pbar:
            pbar.set_postfix({'Current item': objtypeFolder, 'Status': 'Processing'})

            anomaly_imgfiles = normal_image_pool[objtypeFolder]

            dataset_name, object_name, _ = objtypeFolder.split(os.sep)

            defect_des = domain_knowledge[dataset_name][object_name]

            image_path = os.path.join(dataPath,random.choice(list(anomaly_imgfiles.keys())))

            content = []
            content.append \
                ({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
            content.append({"type": "text",
                            "text": "This image depicts a \"{}\" in good condition. The following are some potential defects that objects may encounter: {}. Your task is to define the visual characteristics of a standard \"{}\" and briefly describe it. Do not output any other text, only the visual characteristics of the standard \"{}\", and try not to describe it with points.".format(
                                object_name,'\n'.join(defect_des.values()), object_name, object_name
                            )})

            description = request(content)
            description = description.replace('\n',' ')

            domain_knowledge[dataset_name][object_name]['good'] = f"<Normal Characteristics>\n Description: " + description

    with open(domain_knowledge_file, 'w', encoding='utf-8') as f:
        json.dump(domain_knowledge, f, ensure_ascii=False, indent=4)