import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import torch.nn.functional as F
plt.switch_backend('agg')

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot

def set_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        pass

def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.
    Parameters:
        targets (np.ndarray): matrix with one hot labels
    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)
    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi

def mixup(
    a: torch.FloatTensor,
    b: torch.FloatTensor,
    gamma: torch.FloatTensor,
) -> torch.FloatTensor:
    """Perform a MixUp operation.
    This is effectively just a weighted average, where
    `gamma = 0.5` yields the mean of `a` and `b`.

    Parameters
    ----------
    a : torch.FloatTensor
        [Batch, C] first sample matrix.
    b : torch.FloatTensor
        [Batch, C] second sample matrix.
    gamma : torch.FloatTensor
        [Batch,] MixUp coefficient.

    Returns
    -------
    m : torch.FloatTensor
        [Batch, C] mixed sample matrix.
    """
    return gamma * a + (1 - gamma) * b

class SampleMixUp(object):
    def __init__(
        self,
        alpha: float = 0.2,
        keep_dominant_obs: bool = True,
    ) -> None:
        """Perform a MixUp operation on a sample batch.

        Parameters
        ----------
        alpha : float
            alpha parameter of the Beta distribution.
        keep_dominant_obs : bool
            use max(gamma, 1-gamma) for each pair of samples
            so the identity of the dominant observation can be
            associated with the mixed sample.

        Returns
        -------
        None.

        References
        ----------
        mixup: Beyond Empirical Risk Minimization
        Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
        arXiv:1710.09412

        Notes
        -----
        Zhang et. al. note alpha [0.1, 0.4] improve performance on CIFAR-10,
        while larger values of alpha induce underfitting.
        """
        self.alpha = alpha
        if alpha > 0.0:
            self.beta = torch.distributions.beta.Beta(
                self.alpha,
                self.alpha,
            )
        self.keep_dominant_obs = keep_dominant_obs
        return

    def __call__(
        self,
        x,
        y 
    ) -> dict:
        """Perform a MixUp operation on the sample.

        Parameters
        ----------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label

        Returns
        -------
        sample : dict
            'input' - torch.FloatTensor, input vector
            'output' - torch.LongTensor, target label
        """
        if self.alpha == 0.0:
            # mixup is deactivated, return the original
            # sample without mixing
            return x,y

        input_ = x
        output = y

        # randomly permute the input and output
        ridx = torch.randperm(input_.size(0))
        r_input_ = input_[ridx]
        r_output = output[ridx]

        # perform the mixup operation between the source
        # data and the rearranged data -- random pairs
        gamma = self.beta.sample((input_.size(0),))
        if self.keep_dominant_obs:
            gamma, _ = torch.max(
                torch.stack(
                    [
                        gamma,
                        1 - gamma,
                    ],
                    dim=1,
                ),
                dim=1,
            )
        gamma = gamma.reshape(-1, 1)
        # move gamma weights to the same device as the
        # inputs
        gamma = gamma.to(device=input_.device)

        mix_input_ = mixup(input_, r_input_, gamma=gamma)

        x = mix_input_

        # if there are additional tensors in sample, also mix
        # them up
        # other_keys = [k for k in sample.keys() if k not in ("input", "output")]
        # for k in other_keys:
        #     if type(sample[k]) == torch.Tensor:
        #         sample[k] = mixup(sample[k], sample[k][ridx], gamma=gamma)

        # add the randomization index to the sample in case
        # it's useful downstream

        return x,y,ridx

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            x,label,domain,index = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            x,label,domain,index = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return x,label,domain,index

    def __len__(self):
        return len(self.data_loader)

class ForeverTestDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None, select_cell = None,CellType = None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device
        self.select_cell = select_cell
        self.celltype = np.array(['']*select_cell.shape[0],dtype=object)
        self.index = 0
        self.current_num = 0
        self.weight_max = torch.tensor(0.0)
        self.weight_min = torch.tensor(1.0)
        # for i in CellType:
        #     self.weight_max[i] = torch.tensor(0.0)
        #     self.weight_min[i] = torch.tensor(1.0)

    def __next__(self):
        # x_max,x_min = torch.max(self.select_cell),torch.min(self.select_cell)
        try:
            data,label,domain = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
                selected = send_to_device(selected,self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            self.index = 0
            data,label,domain = next(self.iter)
            # selected = self.select_cell[self.index:self.index + len(data)]
            if self.device is not None:
                data = send_to_device(data, self.device)
                selected = send_to_device(selected,self.device)
        self.index += len(data)
        self.current_num = len(data)
        return data,label,domain,self.weight_max,self.weight_min

    def __len__(self):
        return len(self.data_loader)
    
    def update(self,select_all):
        self.select_cell[self.index - self.current_num:self.index] = select_all
        self.weight_max = torch.max(self.select_cell)
        self.weight_min = torch.min(self.select_cell)
        # self.celltype[self.index - self.current_num:self.index] = pred_label
        # for i in np.unique(pred_label):
        #     self.weight_max[i] = torch.max(self.select_cell[self.celltype == i])
        #     self.weight_min[i] = torch.min(self.select_cell[self.celltype == i])

class ForeverTestDataIteratorExtension:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None,length = 0):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device
        self.index = 0
        self.current_num = 0
        self.frequency = torch.zeros(length)
        self.max_fre = torch.tensor(length)
        self.min_fre = torch.tensor(0)

    def __next__(self):
        # x_max,x_min = torch.max(self.select_cell),torch.min(self.select_cell)
        try:
            data,label,domain = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
                selected = send_to_device(selected,self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            self.index = 0
            data,label,domain = next(self.iter)
            # selected = self.select_cell[self.index:self.index + len(data)]
            if self.device is not None:
                data = send_to_device(data, self.device)
                selected = send_to_device(selected,self.device)
        self.index += len(data)
        self.current_num = len(data)
        frequency = self.frequency[self.index - self.current_num:self.index]
        return data,label,domain,frequency,self.max_fre,self.min_fre

    def __len__(self):
        return len(self.data_loader)

class ForeverDataIteratorExtension:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device
        self.index = 0
        self.current_num = 0

    def __next__(self):
        # x_max,x_min = torch.max(self.select_cell),torch.min(self.select_cell)
        try:
            data,label,domain,index,weight = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
                selected = send_to_device(selected,self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            self.index = 0
            data,label,domain,index,weight = next(self.iter)
            # selected = self.select_cell[self.index:self.index + len(data)]
            if self.device is not None:
                data = send_to_device(data, self.device)
                selected = send_to_device(selected,self.device)
        self.index += len(data)
        self.current_num = len(data)
        return data,label,domain,index,weight

    def __len__(self):
        return len(self.data_loader)
    
def get_th(logit,num = 0,epsilon = 0.95,label = None):
    score = get_scores(logit)
    th = torch.zeros(num)
    for i in range(num):
        th[i] = torch.msort(score[label == i, i])[int(sum(label == i) * (1-epsilon))]
    return th

def get_scores(logit_mat:torch.Tensor):
    #collective decision score computation
    Nseen = logit_mat.shape[1] - 1
    score_sum = (logit_mat.sum(dim = 1).reshape(-1,1) - logit_mat) / Nseen
    score_mat = logit_mat - score_sum
    return score_mat

def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.
    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to
    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

class Loss_Plot:
    def __init__(self) -> None:
        self.loss = []
    
    def add_loss(self,loss):
        self.loss.append(loss)

    def loss_plot(self,save_path):
        plt.figure(figsize=(12,8))
        plt.plot(list(range(len(self.loss))),self.loss)
        plt.savefig(save_path)

def EuclideanDistances(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

class Centroids(object):
    def __init__(self, class_num, dim,device = 'cpu'):
        self.class_num = class_num
        self.src_ctrs = torch.ones((class_num, dim)).to(device)
        self.tgt_ctrs = torch.ones((class_num, dim)).to(device)
        self.src_ctrs *= 1e-10
        self.tgt_ctrs *= 1e-10
            

    def get_centroids(self, domain=None, cid=None):
        if domain == 'source':
            return self.src_ctrs if cid is None else self.src_ctrs[cid, :]
        elif domain == 'target':
            return self.tgt_ctrs if cid is None else self.tgt_ctrs[cid, :]
        else:
            return self.src_ctrs, self.tgt_ctrs

    def update(self, feat_s, label_s, feat_t, pred_t):
        self.upd_src_centroids(feat_s, label_s)
        self.upd_tgt_centroids(feat_t, pred_t)
        
    def upd_src_centroids(self, feats, labels,cell_th = 1):
        # feats = to_np(feats)
        s_global_centroid = self.src_ctrs.clone().detach()
        batch,class_num = labels.shape
        embedding = feats.shape[1]
        device = feats.device
        labels = torch.argmax(labels,dim=1)
        zeros = torch.zeros(class_num,device = device)
        ones = torch.ones_like(labels, dtype=torch.float,device=device)
        s_n_classes = zeros.scatter_add(0, labels, ones)
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        zeros = torch.zeros(class_num, embedding).to(device)
        s_sum_feature = zeros.scatter_add(0, torch.transpose(labels.repeat(embedding, 1), 1, 0), feats)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(class_num, 1))
        current_s_centroid[s_n_classes < cell_th, ] = s_global_centroid[s_n_classes < cell_th, ]
        cs = (F.cosine_similarity(current_s_centroid, s_global_centroid).reshape(-1,1) + 1) / 2
        self.src_ctrs = (1-cs) * s_global_centroid + cs * current_s_centroid

        # last_centroids = to_np(self.src_ctrs)
        # probs = F.softmax(probs, dim=1)
        # src_ctrs = torch.ones(self.src_ctrs.shape).to(self.src_ctrs.device)

        # for i in range(self.class_num - 1):
        #     if torch.sum(labels == i) > 1:
        #         last_centroid = center[i, :]
        #         data_idx = torch.argwhere(labels == i)
        #         new_centroid = torch.mean(feats[data_idx, :], 0).squeeze()
        #         cs = cal_sim(new_centroid, last_centroid)
        #         # print(cs)
        #         new_centroid = cs * new_centroid + (1 - cs) * last_centroid
        #         src_ctrs[i, :] = new_centroid
        #     else :
        #         src_ctrs[i,:] = center[i,:]
        # self.src_ctrs = src_ctrs

    def upd_tgt_centroids(self, feats, probs,cell_th = 1):
        # feats = to_np(feats)
        # last_centroids = to_np(self.tgt_ctrs)
        # src_centroids = to_np(self.src_ctrs)
        p, labels = probs.max(1)
        # pseudo_label = to_np(pseudo_label)
        # probs = F.softmax(probs, dim=1)
        t_global_centroid = self.tgt_ctrs.clone().detach()
        embedding = feats.shape[1]
        batch,class_num = probs.shape
        device = feats.device
        # labels[p <= 0.5] = class_num
        zeros = torch.zeros(class_num,device = device)
        ones = torch.ones_like(labels, dtype=torch.float,device=device)
        t_n_classes = zeros.scatter_add(0, labels, ones)
        ones = torch.ones_like(t_n_classes)
        t_n_classes = torch.max(t_n_classes, ones)
        zeros = torch.zeros(class_num, embedding).to(device)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(labels.repeat(embedding, 1), 1, 0), feats)
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(class_num, 1))
        current_t_centroid[t_n_classes < cell_th, ] = t_global_centroid[t_n_classes < cell_th, ]
        cs = (F.cosine_similarity(current_t_centroid, t_global_centroid).reshape(-1,1) + 1) / 2
        self.tgt_ctrs = (1-cs) * t_global_centroid + cs * current_t_centroid

        # center = self.tgt_ctrs.clone().detach()
        # tgt_ctrs = torch.ones(self.tgt_ctrs.shape).to(self.tgt_ctrs.device)
        # for i in range(self.class_num):
        #     if torch.sum(pseudo_label == i) > 1:
        #         data_idx = torch.argwhere(pseudo_label == i)
        #         new_centroid = torch.mean(feats[data_idx, :], 0).squeeze()
        #         last_centroid = center[i, :]
        #         # if last_centroids[i] != np.zeros_like((1, feats.shape[0])):
        #         cs = cal_sim(new_centroid, last_centroid)
        #         # print(cs)
        #         new_centroid = cs * new_centroid + (1 - cs) * last_centroid
        #         tgt_ctrs[i, :] = new_centroid
        #     else :
        #         tgt_ctrs[i,:] = center[i,:]
        # self.tgt_ctrs = tgt_ctrs

def cal_sim(x1, x2, metric='cosine'):
    # x = x1.clone()
    if len(x1.shape) != 2:
        x1 = x1.reshape(-1, x1.shape[-1])
    if len(x2.shape) != 2:
        x2 = x2.reshape(-1, x2.shape[-1])

    if metric == 'cosine':
        sim = (F.cosine_similarity(x1, x2) + 1.) / 2.
    else:
        sim = F.pairwise_distance(x1, x2) / torch.norm(x2, dim=1)
    return sim

def crit_inter(center1, center2, lambd=1e-3):
    # dists = F.pairwise_distance(center1, center2)
    # loss = t.mean(dists)

    # dists = cal_cossim(center1.cpu().numpy(), center2.cpu().numpy())
    loss = F.mse_loss(center1, center2,reduction='mean')
    # loss = torch.mean(dists)
    # loss = 0
    # for i in range(center1.shape[0]):
    #     loss += dists[i]#[i]
    # loss /= center1.shape[0]
    # loss *= lambd
    return loss