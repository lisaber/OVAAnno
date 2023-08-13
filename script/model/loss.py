import torch
import torch.nn as nn
import torch.nn.functional as F

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def binary_cross_entropy(recon_x, x):
    return -torch.sum(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1)
    # return F.binary_cross_entropy(recon_x,x)

def bce_loss(output, target):
    known_class_prob = 1. - output
    known_target = 1. - target
    result = torch.tensor(0.).to(output.device)
    loss = target * torch.log(output + 1e-6) + known_target * torch.log(known_class_prob + 1e-6)
    for i in range(loss.shape[1]):
        if len(output[output[:,i] >= 0.5]) == 0:
            result += torch.mean(loss[:,i][output[:,i] < 0.5])
        elif len(output[output[:,i] < 0.5]) == 0:
            result += torch.mean(loss[:,i][output[:,i] >= 0.5])
        else :
            result += (torch.mean(loss[:,i][output[:,i] >= 0.5]) + torch.mean(loss[:,i][output[:,i] < 0.5])) / 2
    result = result / loss.shape[1]
    return -result

def elbo(recon_x, x, z_params, binary=True):
    """
    elbo = likelihood - kl_divergence
    L = -elbo

    Params:
        recon_x:
        x:
    """
    mu, logvar = z_params
    kld = kl_divergence(mu, logvar)
    if binary:
        likelihood = -binary_cross_entropy(recon_x, x)
    else:
        likelihood = -F.mse_loss(recon_x, x,reduction="sum")
    return torch.sum(likelihood) / x.shape[0], torch.sum(kld) / x.shape[0]