import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def caculate_accuracy_mmad_finer(answers_json_path, normal_flag='good', show_overkill_miss=False):
    # 用于存储所有答案
    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    dataset_names = []
    type_list = []
    for answer in all_answers_json:
        # dataset_name = answer['image'].split('/')[0]
        dataset_name = '/'.join([answer['image'].split('/')[0], answer['image'].split('/')[1]])
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        if dataset_name not in dataset_names:
            dataset_names.append(dataset_name)
        if type not in type_list:
            type_list.append(question_type)

    # 初始化统计数据结构
    question_stats = {dataset_name: {} for dataset_name in dataset_names}
    detection_stats = {dataset_name: {} for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        detection_stats[dataset_name]['normal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        detection_stats[dataset_name]['abnormal'] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}
        for question_type in type_list:
            question_stats[dataset_name][question_type] = {'total': 0, 'correct': 0, 'correct_answers': {}, 'answers': {}}

    for answer in all_answers_json:
        # dataset_name = answer['image'].split('/')[0]
        dataset_name =  '/'.join([answer['image'].split('/')[0], answer['image'].split('/')[1]])
        question_type = answer['question_type']
        if question_type in ["Object Structure", "Object Details"]:
            question_type = "Object Analysis"
        gpt_answer = answer['gpt_answer']
        correct_answer = answer['correct_answer']
        if correct_answer not in ['A', 'B', 'C', 'D', 'E'] or gpt_answer not in ['A', 'B', 'C', 'D', 'E']:
            all_answers_json.remove(answer)
            print("Remove error:", "correct_answer:", correct_answer, "gpt_answer:", gpt_answer)
            continue

        question_stats[dataset_name][question_type]['total'] += 1
        if answer['correct_answer'] == answer['gpt_answer']:
            question_stats[dataset_name][question_type]['correct'] += 1

        if question_type == "Anomaly Detection":
            if normal_flag in answer['image']:
                detection_stats[dataset_name]['normal']['total'] += 1
                if answer['correct_answer'] == answer['gpt_answer']:
                    detection_stats[dataset_name]['normal']['correct'] += 1
            else:
                detection_stats[dataset_name]['abnormal']['total'] += 1
                if answer['correct_answer'] == answer['gpt_answer']:
                    detection_stats[dataset_name]['abnormal']['correct'] += 1


        answers_dict = question_stats[dataset_name][question_type]['answers']
        if gpt_answer not in answers_dict:
            answers_dict[gpt_answer] = 0
        answers_dict[gpt_answer] += 1
        correct_answers_dict = question_stats[dataset_name][question_type]['correct_answers']
        if correct_answer not in correct_answers_dict:
            correct_answers_dict[correct_answer] = 0
        correct_answers_dict[correct_answer] += 1

    # 创建准确率表格
    accuracy_df = pd.DataFrame(index=dataset_names)
    for dataset_name in dataset_names:
        for question_type in set(type_list):
            total = question_stats[dataset_name][question_type]['total']
            correct = question_stats[dataset_name][question_type]['correct']
            cls_accuracy = correct / total if total != 0 else 0
            accuracy_df.at[dataset_name, question_type] = cls_accuracy*100

            if question_type in ['Anomaly Detection']:
                TP = detection_stats[dataset_name]['abnormal']['correct']
                FP = detection_stats[dataset_name]['normal']['total'] - detection_stats[dataset_name]['normal']['correct']
                FN = detection_stats[dataset_name]['abnormal']['total'] - detection_stats[dataset_name]['abnormal']['correct']
                TN = detection_stats[dataset_name]['normal']['correct']
                Precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                Recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                TPR = Recall
                FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
                normal_acc = detection_stats[dataset_name]['normal']['correct'] / detection_stats[dataset_name]['normal']['total'] if detection_stats[dataset_name]['normal']['total'] != 0 else 0
                anomaly_acc = detection_stats[dataset_name]['abnormal']['correct'] / detection_stats[dataset_name]['abnormal']['total'] if detection_stats[dataset_name]['abnormal']['total'] != 0 else 0

                print(dataset_name)
                print(f"normal_total: {detection_stats[dataset_name]['normal']['total']}")
                print(f"normal_correct: {detection_stats[dataset_name]['normal']['correct']}")
                print(f"anomalous_total: {detection_stats[dataset_name]['abnormal']['total']}")
                print(f"anomalous_correct: {detection_stats[dataset_name]['abnormal']['correct']}")

                accuracy_df.at[dataset_name, 'normal_acc'] = normal_acc
                accuracy_df.at[dataset_name, 'anomaly_acc'] = anomaly_acc
                accuracy_df.at[dataset_name, 'Anomaly Detection'] = (normal_acc+anomaly_acc)/2*100

    # 计算每个问题的平均准确率
    accuracy_df['Average'] = accuracy_df[list(set(accuracy_df.columns)-set(['normal_acc','anomaly_acc']))].mean(axis=1)

    if show_overkill_miss:
        for dataset_name in dataset_names:
            normal_acc = detection_stats[dataset_name]['normal']['correct'] / detection_stats[dataset_name]['normal'][
                'total'] if detection_stats[dataset_name]['normal']['total'] != 0 else 0
            anomaly_acc = detection_stats[dataset_name]['abnormal']['correct'] / detection_stats[dataset_name]['abnormal'][
                'total'] if detection_stats[dataset_name]['abnormal']['total'] != 0 else 0
            accuracy_df.at[dataset_name, 'Overkill'] = (1 - normal_acc) * 100
            accuracy_df.at[dataset_name, 'Miss'] = (1 - anomaly_acc) * 100

    accuracy_df.loc['Average'] = accuracy_df.mean()

    # 数据可视化
    plt.figure(figsize=(10, 7))
    sns.heatmap(accuracy_df, annot=True, cmap='coolwarm', fmt=".1f", vmax=100, vmin=25)
    plt.title(f'Accuracy of {os.path.split(answers_json_path)[-1].replace(".json", "")}')
    # 旋转X轴标签
    plt.xticks(rotation=30, ha='right')  # ha='right'可以使标签稍微倾斜，以便更好地阅读

    # 自动调整边框，减少空白
    plt.tight_layout()
    plt.show()

    # 您想要的新列顺序，其中可能有一些不匹配的列名
    new_order = ['Anomaly Detection', 'Defect Classification', 'Defect Localization', 'Defect Description', 'Defect Analysis','Object Classification','Object Analysis', 'Average', 'normal_acc','anomaly_acc']

    # 从 new_order 中提取出与 df 中实际存在的列名匹配的列
    valid_columns = [col for col in new_order if col in accuracy_df.columns]

    # 根据符合的有效列来重新排列 DataFrame 的列顺序
    accuracy_df = accuracy_df[valid_columns]

    # 保存准确率表格
    accuracy_path = answers_json_path.replace('.json', '_accuracy_finer.csv')
    accuracy_df.to_csv(accuracy_path)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(accuracy_df)
    return question_stats

