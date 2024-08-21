import torch

import torch.nn as nn

def grad_conf(model, loss, optimizer):
    
    # 勾配をリセット
    optimizer.zero_grad()
    
    # lossを逆伝播
    loss.backward(retain_graph=True)
    
    #パラメータの勾配を保存するための構造を作成
    grad_dims = []
    for param in model.parameters():
        grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims))
    
    # 各パラメータに対する勾配を獲得
    parameters = model.parameters()
    
    # 勾配を獲得
    grads.fill_(0.0)
    count = 0
    for param in parameters:
        if param.grad is not None:
            beg = 0 if count == 0 else sum(grad_dims[:count])
            en = sum(grad_dims[:count+1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        count += 1
    
    
    return grads



def grad_conf_ver2(model, loss, optimizer):
    
    # 勾配をリセット
    optimizer.zero_grad()
    
    # lossを逆伝播
    loss.backward(retain_graph=True)
    
    # 勾配を獲得
    grads = {name: param.grad.clone() for name, param in model.named_parameters()}
    
    # 勾配の絶対値の総和を獲得
    grads_sum = sum(param.grad.abs().sum() for param in model.parameters() if param.grad is not None)

    return grads_sum
        
    
    
def grad_conf_amp(model, loss, optimizer, scaler, first=False):
    
    # 勾配をリセット
    optimizer.zero_grad()
    
    # lossを逆伝播
    scaler.scale(loss).backward(retain_graph=True)
    
    # クリップ時に正しくできるようにスケールを元に戻す
    #if first:
    #    scaler.unscale_(optimizer)
    
    # 勾配を獲得
    grads = {name: param.grad.clone() for name, param in model.named_parameters()}
    #grads = grads / scaler._scale
    
    # 勾配の絶対値の総和を獲得
    grads_sum = sum(param.grad.abs().sum() for param in model.parameters() if param.grad is not None)

    return grads_sum



class grad_sim:
    def __init__(self):

        self.new_grads = None
        self.old_grads = None
        self.CosineSimilarity = nn.CosineSimilarity(dim=0)

    def similarity(self, grads):

        self.old_grads = self.new_grads
        self.new_grads = grads

        #print("self.old_grads.shape : ", self.old_grads.shape)

        if self.old_grads is not None:

            grad_similarity = self.CosineSimilarity(self.old_grads, self.new_grads)
            #print("grad_similarity.shape : ", grad_similarity.shape)
            #print("grad_similarity : ", grad_similarity)
            
        else:
            grad_similarity = None
            
        

        return grad_similarity



    
    
    
    
    