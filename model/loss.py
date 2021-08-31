import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def l2_loss(embedding):
    return torch.mean(0.5 * torch.sum(embedding * embedding, axis=1))


def ae_loss(output, target, **kwargs):
    recon = output[0]
    input = target
    recon_loss = F.mse_loss(input, recon)
    loss_type = kwargs.get('cont_loss_type', 'mse')
    if (loss_type == 'rmse'):
        recon_loss = torch.sqrt(recon_loss)
    return recon_loss

def interp_loss(output, labels):
    mio = output[labels == 1.0]
    mao = output[labels == 0.0]
    idxs = torch.randperm(len(mio))
    mio = mio[idxs]
    idxs = torch.randperm(len(mao))
    mao = mao[idxs]

    ma = len(mao)
    mi = len(mio)
    if (ma % 2) != 0:
        mao = mao[:-1]

    if (mi % 2) != 0:
        mio = mio[:-1]
    loss = 0.0
    if mi >= 2:
        ip =  (1 - mio[:mi// 2] - mio[mi//2:])
        ip = torch.mul(ip, ip)
        loss += torch.mean(ip)
    if ma >= 2:
        ip =  (1 - mao[:ma//2] - mao[ma//2:])
        ip = torch.mul(ip, ip)
        loss += torch.mean(ip)
    return loss
