from torch.utils.data.dataloader import default_collate
import torch

feature_key  = {
    "msg_type": "one_hot",
    "session_id": "one_hot",
    "iface_ver":"one_hot",
    "proto_ver":"one_hot",
    "retcode":"one_hot",
    "ip_src":"ip_value",
    "ip_dst":"ip_value",
    "proto":"one_hot",
    "sport":"one_hot",
    "dport":"one_hot",
    "mac_src":"mac_value",
    "mac_dst":"mac_value",
    "method_id":"one_hot",
    "Client_id":"one_hot",
    "length":"value",
    "service_id":"one_hot",
    "Type":"one_hot",
    "timesensitive":"one_hot",
    "client_min":"value",
    "client_max":"value",
    "client_resendMin":"value",
    "client_resendMax":"value",
    "errorRate":"value",
    "server_min":"value",
    "server_max":"value",
    "client_mac":"mac_value",
    "client_ip":"ip_value",
    "client_send_port":"value",
    "client_rec_port":"value",
    "server_mac":"mac_value",
    "server_ip":"ip_value",
    "server_send_port":"one_hot",
    "server_rec_port":"one_hot",
    "label":"label"
}

def recursive_collate_fn(batch):
    if isinstance(batch[0], dict):
        return {key: recursive_collate_fn([b[key] for b in batch]) for key in batch[0]}
    else:
        return default_collate(batch)
    
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
