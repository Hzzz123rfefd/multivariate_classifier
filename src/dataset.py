import datasets
from torch.utils.data import Dataset

from src.utils import *



class DatasetForIntrusionDetection(Dataset):
    def __init__(
            self,
            train_data_path:str = None,
            test_data_path:str = None,
            valid_data_path:str = None,
            data_type:str = "train"
    ):
        if data_type == "train":
            self.dataset = datasets.load_dataset('json', data_files = train_data_path,split = "train")
        elif data_type == "test":
            self.dataset = datasets.load_dataset('json', data_files = test_data_path,split = "train")
        elif data_type == "valid":
            self.dataset = datasets.load_dataset('json', data_files = valid_data_path,split = "train")
        self.dataset = self.dataset.shuffle(seed=42)
        self.total_len = len(self.dataset)
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        output = {
            "feature" : [],
            "label": [],
        }
        for feature_name in feature_key:
            if feature_key[feature_name] == "one_hot" or feature_key[feature_name] == "mac_value"or feature_key[feature_name] == "ip_value":
                output["feature"].extend(self.dataset[idx][feature_name])
            elif feature_key[feature_name] == "value":
                output["feature"].append(self.dataset[idx][feature_name])
        output["label"].append((int)(self.dataset[idx]['label']))
        output["feature"] = torch.tensor(output["feature"])
        output["label"] = torch.tensor(output["label"],dtype=torch.int64)
        return output
    
    def collate_fn(self,batch):
        return  recursive_collate_fn(batch)