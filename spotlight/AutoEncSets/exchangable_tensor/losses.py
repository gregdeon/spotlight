import torch

EPS = 1e-12

def mse(predicted, target, mask=None):
    if mask is not None:
        return torch.sum(torch.pow(predicted - target, 2) * mask) / torch.sum(mask)
    else:
        return torch.mean(torch.pow(predicted - target, 2))

def softmax(x, dim=-1):
    m = torch.clamp(torch.max(x, dim=dim, keepdim=True)[0], min=0.) 
    exps = torch.exp(x - m)
    return exps / (torch.sum(exps, dim=dim, keepdim=True))

def ce(predicted, target, mask=None):
    if mask is not None:
        #print(softmax(predicted))
        return torch.sum(-target * torch.log(EPS + softmax(predicted)) * mask)/ (torch.sum(mask))
    else:
        return torch.mean(target * torch.log(softmax(predicted)))

