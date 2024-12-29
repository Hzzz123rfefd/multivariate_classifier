import csv
import json
import os
import sys
sys.path.append(os.getcwd())
from src.utils import *

class OneHot:
    def __init__(self):
        self.dict = {}
        
    def _add(self, key, unique_values):
        self.dict[key] = unique_values
        
    def encode(self, key, value):
        unique_values = self.dict[key]
        one_hot_vector = [0] * len(unique_values)
        if value in unique_values:
            index = unique_values.index(value)
            one_hot_vector[index] = 1
        return one_hot_vector
    
    
class ValueNormalizer:
    def __init__(self, max_ratio = 1.05):
        self.max_ratio = max_ratio
        self.dict = {}
        

    def  _add(self, key, min_max_value:list):
        min = (float)(min_max_value[0])
        max = (float)(min_max_value[1])
        max = max * self.max_ratio
        self.dict[key] = [min, max]
        
    def encode(self, key, value):
        value = (float)(value)
        min_max_value = self.dict[key]
        return  (value - min_max_value[0])/(min_max_value[1] - min_max_value[0])
    
class IPV4Value:
    def __init__(self):
        pass
        
    def encode(self, value):
        ipv4 = list(map(int, value.split('.')))
        ipv4 = [each/255 for each in ipv4]
        return ipv4
    
class MACValue:
    def __init__(self):
        pass
        
    def encode(self, value):
        mac = [int(part, 16) for part in value.split(":")]
        mac = [each/255 for each in mac]
        return mac
        
class FeatureProcess:
    def __init__(self):
        self.one_hot = OneHot()
        self.nomalizer = ValueNormalizer(max_ratio = 1.05)
        self.ipv4 = IPV4Value()
        self.mac = MACValue()
        
    def fit(self, datasets):
        for feature_name in feature_key:
            if feature_key[feature_name] == "one_hot":
                unique_values = sorted(set(datasets[feature_name]))
                self.one_hot._add(feature_name, unique_values)
            elif feature_key[feature_name] == "value":
                max_value =  max((datasets[feature_name]))
                min_value = min((datasets[feature_name]))
                self.nomalizer._add(feature_name,[min_value, max_value])
                
    def transform(self, feature):
        feature_vector = {}
        for feature_name in feature_key:
            if feature_key[feature_name] == "one_hot":
                feature_vector[feature_name] = self.one_hot.encode(feature_name,feature[feature_name])
            elif feature_key[feature_name] == "value":
                feature_vector[feature_name] = self.nomalizer.encode(feature_name,feature[feature_name])
            elif feature_key[feature_name] == "ip_value":
                feature_vector[feature_name] = self.ipv4.encode(feature[feature_name])
            elif feature_key[feature_name] == "mac_value":
                feature_vector[feature_name] = self.mac.encode(feature[feature_name])
            else:
                feature_vector[feature_name] = feature[feature_name]
        return feature_vector
                
            

# 读取 CSV 文件
datasets = {
    "msg_type" : [],
    "session_id" : [],
    "iface_ver" : [],
    "proto_ver" : [],
    "retcode" : [],
    "ip_src" : [],
    "ip_dst" : [],
    "proto" : [],
    "sport" : [],
    "dport" : [],
    "mac_src" : [],
    "mac_dst" : [],
    "method_id" : [],
    "Client_id" : [],
    "length":[],
    "service_id":[],
    "Type":[],
    "timesensitive":[],
    "client_min":[],
    "client_max":[],
    "client_resendMin":[],
    "client_resendMax":[],
    "errorRate":[],
    "server_min":[],
    "server_max":[],
    "client_mac":[],
    "client_ip":[],
    "client_send_port":[],
    "client_rec_port":[],
    "server_mac":[],
    "server_ip":[],
    "server_send_port":[],
    "server_rec_port":[],
    "label":[]
}
with open("datasets/Train/all.csv", mode='r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)  
    for row in reader:
        for feature_name in feature_key:
            if feature_key[feature_name] == "value":
                # if feature_name == "session_id" and row["label"] == "1" or row["label"] == "3":
                #     datasets[feature_name].append(0.0)
                # else:
                #     datasets[feature_name].append((float)(row[feature_name]))
                datasets[feature_name].append((float)(row[feature_name]))
            else:
                datasets[feature_name].append(row[feature_name])
        # if row["label"] == "1":
        #     datasets["msg_type"].append("10")
        # elif row["label"] == "3":
        #     datasets["msg_type"].append("11")
        # else:
        #     datasets["msg_type"].append(row["msg_type"])
fp = FeatureProcess()
fp.fit(datasets)
data_len = len(datasets["msg_type"])

os.makedirs("data", exist_ok=True)
with open("data/train.jsonl", 'w', encoding='utf-8') as f:
    for i in range(data_len):
        feature = {}
        for key in datasets:
            feature[key] = datasets[key][i]
        feature_vector = fp.transform(feature)
        f.write(json.dumps(feature_vector, ensure_ascii=False) + '\n')




