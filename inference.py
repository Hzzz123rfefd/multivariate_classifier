import argparse
import json
from src import models
from src.utils import *
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
# data = {"msg_type": [0, 0, 1, 0, 0], "session_id": 0.01942690626517727, "iface_ver": [1], "proto_ver": [1], "retcode": [1, 0, 0, 0, 0, 0, 0], "ip_src": [0.0392156862745098, 0.0, 0.0, 0.00392156862745098], "ip_dst": [0.0392156862745098, 0.00392156862745098, 0.0, 0.0196078431372549], "proto": [1], "sport": [1, 0], "dport": [0, 1], "mac_src": [0.00784313725490196, 0.10196078431372549, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666], "mac_dst": [0.00784313725490196, 0.3568627450980392, 0.7333333333333333, 0.7333333333333333, 0.7333333333333333, 0.7333333333333333], "method_id": [0, 0, 1, 0, 0], "Client_id": 0.5405405405405405, "label": "3"}
# data = {"msg_type": [0, 0, 1, 0, 0], "session_id": 0.1651287032540068, "iface_ver": [1], "proto_ver": [1], "retcode": [1, 0, 0, 0, 0, 0, 0], "ip_src": [0.0392156862745098, 0.0, 0.0, 0.00784313725490196], "ip_dst": [0.0392156862745098, 0.00392156862745098, 0.0, 0.023529411764705882], "proto": [1], "sport": [1, 0], "dport": [0, 1], "mac_src": [0.00784313725490196, 0.16470588235294117, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666], "mac_dst": [0.00784313725490196, 0.4196078431372549, 0.7333333333333333, 0.7333333333333333, 0.7333333333333333, 0.7333333333333333], "method_id": [1, 0, 0, 0, 0], "Client_id": 0.6756756756756757, "label": "3"}
# # data = json.loads(data)

datasets = []
with open("data/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            json_obj = json.loads(line.strip())
            datasets.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON line: {line.strip()} - Error: {e}")

import numpy as np
def evaluate_binary_classification(y_true, y_pred):
    # 转为 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 转为二分类
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    
    # 计算混淆矩阵的统计值
    tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
    tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
    fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
    fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
    
    # 计算各项指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 打印结果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    return accuracy, precision, recall, f1_score


def main(args):
    config = load_config(args.model_config_path)
    """ get net struction"""
    net = models[config["model_type"]](**config["model"])
    net.load_pretrained(config["logging"]["save_dir"])
    labels = []
    predicts = []
    for data in datasets:
        feature = []
        for feature_name in feature_key:
            if feature_key[feature_name] == "one_hot" or feature_key[feature_name] == "mac_value"or feature_key[feature_name] == "ip_value":
                feature.extend(data[feature_name])
            elif feature_key[feature_name] == "value":
                feature.append(data[feature_name])
        feature = torch.tensor(feature)
        feature = feature.unsqueeze(0).unsqueeze(0)
        category = net.inference(feature).cpu().numpy().item()
        predicts.append(category)
        labels.append((int)(data['label']))
        if data["label"] != "0":
            print("label:",data["label"])
            print("category:", category)
    evaluate_binary_classification(labels, predicts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path",type=str,default = "config/intrusion_detection.yml")
    args = parser.parse_args()
    main(args)