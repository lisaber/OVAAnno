from ast import Tuple
from typing import Any, Optional,Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from utils.utils import label_to_membership

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

def bulid_mlp(layers,activation = nn.ReLU(),finally_activate = None,batch = False):
    net = []
    for i in range(1,len(layers) - 1):
        net.append(nn.Linear(layers[i-1],layers[i]))
        net.append(activation)
    net.append(nn.Linear(layers[len(layers) - 2],layers[len(layers) - 1]))
    if finally_activate != None:
        net.append(finally_activate)
    return nn.Sequential(*net)

def bulid_mlp_batch(layers,activation = nn.ReLU(), p = 0.2):
    net = []
    for i in range(1,len(layers)):
        net.append(nn.Linear(layers[i-1],layers[i]))
        if(p > 0):
            net.append(nn.Dropout(p = p))
        net.append(activation)
    return nn.Sequential(*net)

class Classifier(nn.Module):
    def __init__(self,dims,finally_activate = None,lambd = 1.):
        super(Classifier,self).__init__()
        self.revgrad = GradientReverseLayer()
        self.classifier = bulid_mlp(dims,finally_activate = finally_activate)
    # def set_lambda(self, lambd):
        self.lambd = lambd
    
    def forward(self,x,reverse=False):
        if reverse:
            # x_rev = self.revgrad(x)
            # x = grad_reverse(x, self.lambd)
            return self.revgrad(self.classifier(x))
        else : 
            return self.classifier(x)
    
    def weight_norm(self):
        w = self.classifier[0].weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier[0].weight.data = w.div(norm.expand_as(w))

class OpenClassifier(nn.Module):
    def __init__(self,dims,finally_activate = None,lambd = 1.):
        super(OpenClassifier,self).__init__()
        self.revgrad = GradientReverseLayer()
        self.hidden = nn.Linear(dims[0],dims[1])
        self.act = nn.Sigmoid()
        self.classifier = nn.Linear(dims[1],dims[2])
    # def set_lambda(self, lambd):
        self.lambd = lambd
    
    def forward(self,x,reverse=False):
        if reverse:
            x = self.act(self.hidden(x))
            # x_rev = self.revgrad(x)
            # x = grad_reverse(x, self.lambd)
            return x,self.classifier(x)
        else : 
            return self.classifier(self.act(self.hidden(x)))

class Domain_layer(nn.Module):
    def __init__(self,dims,finally_activate = None,lambd = 1.):
        super(Domain_layer,self).__init__()
        net = []
        for i in range(1,len(dims) - 1):
            net.append(nn.Linear(dims[i-1],dims[i]))
            # net.append(nn.BatchNorm1d(num_features=dims[i]))
            # net.append(nn.Dropout())
            net.append(nn.ReLU())
        self.backbone = nn.Sequential(*net)
        self.classifier = nn.Linear(dims[len(dims) - 2],dims[len(dims) - 1])
    
    def forward(self,x,reverse=False):
        feature = self.backbone(x)
        return self.classifier(feature),feature

class Encoder(nn.Module):
    def __init__(self,dims,p = 0.5):
        super(Encoder,self).__init__()

        [x_dim,h_dim,z_dim] = dims
        self.hidden = bulid_mlp_batch([x_dim] + h_dim,p = p)
        self.sample = GaussianSample(([x_dim] + h_dim)[-1],z_dim)

    def forward(self,x):
        # x = self.hidden(F.dropout(x,p=0.1))
        x = self.hidden(x)
        return self.sample(x)

class Decoder(nn.Module):
    def __init__(self,dims,output_activation=None,p = 0.5):
        super(Decoder,self).__init__()

        [z_dim,h_dim,x_dim] = dims
        self.hidden = bulid_mlp_batch([z_dim] + h_dim,p = p)
        self.reconstruction = nn.Linear(([z_dim] + h_dim)[-1],x_dim)
        self.output_activation = output_activation
    
    def freeze(self):
        for name,para in self.named_parameters():
            para.requires_grad = False

    def forward(self,x):
        x = self.hidden(x)
        if self.output_activation is not None:
            return self.output_activation(self.reconstruction(x))
        else:
            return self.reconstruction(x)


class GaussianSample(nn.Module):
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def reparametrize(self,mu,logvar):
        epsilon = torch.randn(mu.size(),requires_grad=False,device=mu.device)
        std = logvar.mul(0.5).exp_()
        z = mu.addcmul(std, epsilon)

        return z

    def forward(self,x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return self.reparametrize(mu,log_var),mu,log_var

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0,device = 'cpu',threshold = 0,celltype = None):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.register_buffer('label', torch.zeros(outputSize))
        self.register_buffer('prob', torch.ones(outputSize,inputSize) * 0.5)
        self.celltype = torch.tensor(celltype)
        self.threshold = torch.tensor(threshold)
        self.flag = 0
        self.T = T
        self.label =  self.label.to(device)
        self.celltype = self.celltype.to(device)
        self.prob = self.prob.to(device)
        self.threshold = self.threshold.to(device)
        self.device = device
        self.len = outputSize
        self.ratio = torch.ones(self.prob.shape[1],device=device)
    
    def forward(self, x, y = None):
        # out = torch.mm(x, self.memory.t())
        out = 1 / (1.+torch.cdist(x,self.memory))
        return out

    def update_weight(self, label, index,prob = None,epsilon = 0.8):
        weight_pos = self.label.index_select(0, index.data.view(-1)).resize_as_(label)
        weight_pos_prob = self.prob.index_select(0, index.data.view(-1)).resize_as_(prob)
        weight_pos.mul_(0)
        weight_pos_prob.mul_(0.0)
        weight_pos.add_(torch.mul(label.data, 1))
        weight_pos_prob.add_(torch.mul(prob.data, 1.0))

        # w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        # updated_weight = weight_pos.div(w_norm)
        self.label.index_copy_(0, index, weight_pos)
        self.prob.index_copy_(0,index,weight_pos_prob)
        # # self.memory = F.normalize(self.memory)#.cuda()

        self.label = self.label.long()
        # n_classes[n_classes == 0] = 1.

        prob_clone = self.prob.clone()
        label = torch.zeros_like(self.prob,dtype=torch.bool)
        label[torch.arange(0,self.label.shape[0]),self.label] = True
        prob_clone.masked_fill_(~label,2.)
        label = self.label.clone()
        label[prob_clone[torch.arange(0,self.label.shape[0]),self.label] < 0.5] = self.threshold.shape[1]
        prob_clone = torch.where(prob_clone < 0.5,2.,prob_clone)
        prob_clone = torch.msort(prob_clone)
        ones = torch.ones_like(self.label, dtype=torch.float)
        zeros = torch.zeros(self.prob.shape[1] + 1).to(self.device)
        n_classes = zeros.scatter_add(0, label, ones)
        n_classes = n_classes[0:self.prob.shape[1]]

        self.threshold = prob_clone[(n_classes * (1-epsilon)).long(),torch.arange(0,n_classes.shape[0])].repeat(self.threshold.shape[0],1)
        # self.threshold = self.len - n_classes
        # self.threshold = (n_classes / self.len).repeat(self.threshold.shape[0],1)
        n_classes[n_classes == 0] = 1
        self.ratio = (self.prob.shape[0] - n_classes) / n_classes
        
        self.threshold[self.threshold > 1.5] = 0.5

    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)

class Logit_Linear(nn.Module):
    def __init__(self, dataSize, label,T=0.05, momentum=0.0,device = 'cpu'):
        super(Logit_Linear, self).__init__()
        self.register_buffer('score', torch.zeros(dataSize))
        self.score = self.score.to(device)
        self.register_buffer('label',torch.from_numpy(label))
        self.label = self.label.to(device,dtype = torch.int64)
        self.class_num = len(torch.unique(self.label))
        self.class_id = torch.range(0,self.class_num - 1,device=device)
        zeros = torch.zeros(self.class_num,device=device)
        ones = torch.ones_like(self.label, dtype=torch.float)
        self.s_n_classes = zeros.scatter_add(0, self.label, ones)
        self.device = device

    def update_score(self, logit:torch.Tensor, index):
        score_mat = self.get_scores(logit_mat=logit)
        score = score_mat[torch.range(0,logit.shape[0] - 1,dtype=torch.int64),self.label[index]]
        self.score.index_copy_(0,index,score)
    
    def get_th(self,num = 0,epsilon = 0.95):
        # th = torch.zeros(num,device = self.device)
        zeros = torch.zeros(self.class_num,device=self.device)
        max_ele = torch.amax(self.score)
        label = self.label.reshape(-1,1)
        mat = label == self.class_id
        score_mat = self.score.reshape(-1,1).repeat(1,self.class_num)
        score_mat = score_mat.masked_fill(~mat,value = max_ele).msort()
        th = score_mat[(self.s_n_classes * (1-epsilon)).to(dtype=torch.int64),self.class_id.to(dtype=torch.int64)]
        
        # th = zeros.scatter_reduce(0,self.label,self.score,reduce='mean')
        # for i in range(num):
        #     th[i] = torch.msort(self.score[self.label == i])[int(sum(self.label == i) * (1-epsilon))]
        return th

    def get_scores(self,logit_mat:torch.Tensor):
        #collective decision score computation
        Nseen = logit_mat.shape[1] - 1
        score_sum = (logit_mat.sum(dim = 1).reshape(-1,1) - logit_mat) / Nseen
        score_mat = logit_mat - score_sum
        # score_mat = torch.zeros((logit_mat.shape[0], Nseen),device = self.device)
        # for k in range(Nseen):
        #     score_mat[:, k] = logit_mat[:, k] - (logit_mat.sum(dim = 1) - logit_mat[:, k])/(Nseen-1)
        return score_mat