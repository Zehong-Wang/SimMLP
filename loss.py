from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch.distributions import Beta


def cos_similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, *args, **kwargs)
        return loss
    

class Bootstrap(Loss):
    def __init__(self, eta, alpha=2, aux_pos_ratio=0):
        super(Bootstrap, self).__init__()
        self.eta = eta
        self.aux_pos_ratio = aux_pos_ratio
        self.beta = Beta(alpha, alpha)

    def sample_beta(self, num_samples):
        beta = self.beta.sample([num_samples])
        return beta

    def compute(self, anchor, sample, ):
        anchor = F.normalize(anchor, dim=-1, p=2) # N * D
        sample = F.normalize(sample, dim=-1, p=2) # N * D

        loss = (1 - (anchor * sample).sum(dim=-1)).pow_(self.eta)
        loss = loss.mean()

        if self.aux_pos_ratio > 0:
            aux_anchor, aux_sample = self.create_aux_pos_pairs(anchor, sample, self.aux_pos_ratio)
            aux_loss = (1 - (aux_anchor * aux_sample).sum(dim=-1)).pow_(self.eta)
            aux_loss = aux_loss.mean()
            
            loss = loss + aux_loss
            
        return loss

    def create_aux_pos_pairs(self, anchor, sample, aux_pos_ratio):
        assert type(aux_pos_ratio) is int

        device = anchor.device
        anchor = anchor.repeat([aux_pos_ratio, 1])
        sample = sample.repeat([aux_pos_ratio, 1])
        num_samples = anchor.shape[0]

        pos_lambda = self.sample_beta(num_samples).unsqueeze(-1).to(device)
        
        aux_anchor = pos_lambda * anchor + (1 - pos_lambda) * sample
        aux_sample = pos_lambda * sample + (1 - pos_lambda) * anchor

        return aux_anchor, aux_sample


    def regularize(self, z1, z2, reg_alpha=5e-3):
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)
        
        c = torch.matmul(z1_norm.t(), z2_norm) / z1_norm.shape[0]
        c_diff = (c - torch.eye(z1_norm.shape[1], device=z2_norm.device)).pow(2)
        reg = c_diff.diag().sum() + (c_diff.fill_diagonal_(0) * reg_alpha).sum()

        return reg


class InfoNCE(Loss):
    def __init__(self, tau, batch=0, pos_alpha=2, neg_alpha=2):
        super(InfoNCE, self).__init__()
        self.tau = tau
        self.batch = batch


    def compute(self, anchor, sample, pos_mask=None, neg_mask=None, interview=True):
        num_samples = anchor.shape[0]
        batch = self.batch
        device = anchor.device

        anchor = F.normalize(anchor)
        sample = F.normalize(sample)

        if pos_mask == None:
            inter_pos_mask = torch.eye(num_samples, device=device)
        else:
            inter_pos_mask = pos_mask

        if neg_mask == None:
            inter_neg_mask = torch.ones([num_samples, num_samples], device=device) - torch.eye(num_samples, device=device)
        else:
            inter_neg_mask = neg_mask

        if interview:
            intra_pos_mask = torch.zeros([num_samples, num_samples], device=device)
            pos_mask = torch.concat([inter_pos_mask, intra_pos_mask], dim=1)

            intra_neg_mask = torch.ones([num_samples, num_samples], device=device)
            intra_neg_mask.fill_diagonal_(0)
            neg_mask = torch.concat([inter_neg_mask, intra_neg_mask], dim=1)
        else:
            pos_mask = inter_pos_mask
            neg_mask = inter_neg_mask

        if interview:
            sample = torch.concat([sample, anchor], dim=0)

        if batch == 0:
            sim = anchor @ sample.t() / self.tau

            exp_sim = torch.exp(sim) * pos_mask + torch.exp(sim) * neg_mask
            log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
            loss = log_prob * pos_mask
            loss = loss.sum(dim=1)
            return -loss.mean()