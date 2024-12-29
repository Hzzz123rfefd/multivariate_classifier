import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from src.utils import *

class ModelForIntrusionDetection(nn.Module):
    def __init__(self, input_dim, num_classes, device = "cuda"):
        super(ModelForIntrusionDetection, self).__init__()
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1).to(self.device)
        # self.bn1 = nn.BatchNorm1d(64).to(self.device)
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2).to(self.device)
        
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1).to(self.device)
        # self.bn2 = nn.BatchNorm1d(128).to(self.device)
        # self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2).to(self.device)
        
        # self.fc1 = nn.Linear((input_dim // 4) * 128, 256).to(self.device)
        # self.fc2 = nn.Linear(256, num_classes).to(self.device)
        hidden1_size = 64,
        hidden2_size = 32,
        self.fc1 = nn.Linear(input_dim, 256).to(self.device)
        self.fc2 = nn.Linear(256, 32).to(self.device)
        self.fc3 = nn.Linear(32, num_classes).to(self.device)
        self.activation = nn.ReLU()
        
        
        
    def forward(self, input):
        feature = input["feature"].to(self.device)
        # x = feature.unsqueeze(1)  
        # x = self.conv1(x)  
        # # x = F.relu(self.bn1(x))
        # x = self.pool1(x) 
        # x = self.conv2(x) 
        # # x = F.relu(self.bn2(x))
        # x = self.pool2(x) 
        # x = x.flatten(start_dim=1)
        # x = F.relu(self.fc1(x))  
        # x = self.fc2(x)       
        # for i in range(16):
        #     if  input["label"][i,0] == 3:
        #         print(x[i,:])
        x = feature
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        output = {
            "label" :  input["label"].squeeze(1).to(self.device), 
            "predict" : x,
            "class_weights":torch.Tensor([1,1,1,1,1]).to(self.device)
        }
        return output

    def inference(self, feature):
        feature = feature.to(self.device)
        # x = feature
        # x = self.conv1(x)  
        # x = F.relu(self.bn1(x))
        # x = self.pool1(x) 
        # x = self.conv2(x) 
        # x = F.relu(self.bn2(x))
        # x = self.pool2(x) 
        # x = x.flatten(start_dim=1)
        # x = F.relu(self.fc1(x))  
        # x = self.fc2(x)        
        x = feature.squeeze(0)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        y = torch.argmax(x,dim = 1)
        return y.squeeze(0)
        

    def trainning(
        self,
        train_dataloader:DataLoader = None,
        test_dataloader:DataLoader = None,
        optimizer_name:str = "Adam",
        weight_decay:float = 1e-4,
        clip_max_norm:float = 0.5,
        factor:float = 0.3,
        patience:int = 15,
        lr:float = 1e-4,
        total_epoch:int = 1000,
        save_checkpoint_step:str = 10,
        save_model_dir:str = "models"
    ):
        first_trainning = True
        check_point_path = save_model_dir  + "/checkpoint.pth"
        log_path = save_model_dir + "/train.log"


        if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
            self.load_pretrained(save_model_dir)  
            first_trainning = False

        else:
            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)
            with open(log_path, "w") as file:
                pass


        if optimizer_name == "Adam":
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer, 
            mode = "min", 
            factor = factor, 
            patience = patience
        )

        if first_trainning:
            best_loss = float("inf")
            last_epoch = 0
        else:
            checkpoint = torch.load(check_point_path, map_location=self.device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint["loss"]
            last_epoch = checkpoint["epoch"] + 1

        try:
            for epoch in range(last_epoch,total_epoch):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                # test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                loss = train_loss 
                lr_scheduler.step(loss)
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                check_point_path = save_model_dir  + "/checkpoint.pth"
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": None,
                        "lr_scheduler": None
                    },
                    check_point_path
                )

                if epoch % save_checkpoint_step == 0:
                    os.makedirs(save_model_dir + "/" + "chaeckpoint-"+str(epoch))
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        save_model_dir + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
                    )
                if is_best:
                    self.save_pretrained(save_model_dir)

        except KeyboardInterrupt:
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict()
                    },
                    check_point_path
                )

    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir  + "/model.pth",map_location=torch.device(self.device)))
    
    def save_pretrained(self,  save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")
        
    def compute_loss(self, input):
        output = {}
        # batch_size = input["label"].shape[0]
        # labels_one_hot = torch.zeros(batch_size, self.num_classes).to(self.device).scatter_(1, input["label"].unsqueeze(1), 1)
        # criterion = nn.BCEWithLogitsLoss()
        # output["total_loss"] = criterion(input["predict"], labels_one_hot)
        if "class_weights" not in input:
            input["class_weights"] = None
        if "mask" not in input:
            criterion = nn.CrossEntropyLoss(weight = input["class_weights"])
            output["total_loss"] = criterion(input["predict"], input["label"])
        else:
            criterion = nn.CrossEntropyLoss(weight = input["class_weights"], reduction = 'none')
            loss = criterion(input["predict"],input["label"])
            masked_loss = loss * input["mask"]
            output["total_loss"] = masked_loss.sum() / input["mask"].sum()
        return output
    
    def train_one_epoch(self, epoch,train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train()
        self.to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())
            average_hit_rate.update(math.exp(-total_loss.avg))

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "total_loss: {:.4f},average_hit_rate:{:.4f},use_memory: {:.1f}G".format(
                total_loss.avg, 
                average_hit_rate.avg,
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg

    def test_epoch(self,epoch, test_dataloader,trainning_log_path = None):
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"])

            average_hit_rate.update(math.exp(-total_loss.avg))
            str = "Test Epoch: {:d}, total_loss: {:.4f},average_hit_rate:{:.4f}".format(
                epoch,
                total_loss.avg, 
                average_hit_rate.avg,
            )
        print(str)
        with open(trainning_log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg

                