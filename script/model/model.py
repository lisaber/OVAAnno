import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from kmeans_pytorch import kmeans

import math
import numpy as np
# from tqdm import tqdm, trange
from zmq import device
from module.layers import Encoder,Decoder,AE_Encoder,Classifier,Domain_layer,LinearAverage,Logit_Linear
from module.loss import elbo,binary_cross_entropy,CenterLoss
from tqdm import tqdm
import sys
from torch import Tensor
from torch.autograd import Variable
from utils.utils import ForeverDataIterator,Loss_Plot,ForeverDataIteratorExtension,ForeverTestDataIteratorExtension,Centroids

class VAE(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, binary=True):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VAE, self).__init__()
        [x_dim, z_dim,z_batch_dim, encode_dim, decode_dim] = dims
        self.binary = binary
        if binary:
            decode_activation = nn.Sigmoid()
        else:
            decode_activation = None

        self.encoder = Encoder([x_dim, encode_dim, z_dim],p = dropout)
        self.decoder = Decoder([z_batch_dim, decode_dim, x_dim],p = dropout,output_activation=decode_activation)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x

    def loss_function(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        likelihood, kl_loss = elbo(recon_x, x, (mu, logvar), binary=False)

        return (-likelihood, kl_loss)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def fit(self, dataloader,
            lr=0.002, 
            weight_decay=0,
            device='cpu',
            beta = 1,
            n = 200,
            max_iter=30000,
            verbose=True,
            patience=100,
            outdir='./'
       ):

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        iteration = 0
        n_epoch = int(np.ceil(max_iter/len(dataloader)))
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
#                 epoch_loss = 0
                epoch_recon_loss, epoch_kl_loss = 0, 0
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations')
                for i, x in tk0:
#                     epoch_lr = adjust_learning_rate(lr, optimizer, iteration)
                    x = x.float().to(device)
                    optimizer.zero_grad()
                    
                    recon_loss, kl_loss = self.loss_function(x)
#                     loss = (recon_loss + next(Beta) * kl_loss)/len(x);
                    loss = kl_loss/len(x) + recon_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 10) # clip
                    optimizer.step()
                    
                    epoch_kl_loss += kl_loss.item()
                    epoch_recon_loss += recon_loss.item()

                    tk0.set_postfix_str('loss={:.3f} recon_loss={:.3f} kl_loss={:.3f}'.format(
                            loss, recon_loss/len(x), kl_loss/len(x)))
                    tk0.update(1)
                    
                    iteration+=1
                tq.set_postfix_str('recon_loss {:.3f} kl_loss {:.3f}'.format(
                    epoch_recon_loss/((i+1)*len(x)), epoch_kl_loss/((i+1)*len(x))))
    
    def encodeBatch(self, dataloader, device='cpu', out='z', transforms=None):
        output = []
        for x in dataloader:
            x = x.view(x.size(0), -1).float().to(device)
            z, mu, logvar = self.encoder(x)

            if out == 'z':
                output.append(mu.detach().cpu())
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach().data)

        output = torch.cat(output).numpy()

        return output

class Class_VAE(nn.Module):
    def __init__(self, vae_dims, c_dim,domain_dim = None,bn=False, dropout=0, binary=True,finally_activate = nn.Sigmoid(),num_class = 0,device = 'cpu',temperature = 0.05):
        super(Class_VAE,self).__init__()
        self.vae_s = VAE(vae_dims, bn, dropout, binary)
        # self.vae_class = VAE(domain_dim, bn, dropout, True)
        self.classifier = Classifier(c_dim)
        self.closed_classifier = Classifier(c_dim)
        self.domain_ad = Classifier([32,1])
        self.class_dim = num_class
        self.binary = binary
        self.temp = temperature
        self.class_criterion = self.zeros_loss
        self.domain_criterion = self.zeros_loss
        self.sematic_criterion = self.zeros_loss
        self.class_t_criterion = self.zeros_loss
        self.index = torch.range(0,num_class - 1,device=device,dtype=torch.int64).reshape(-1,1)
        self.zero = torch.tensor(0.,device=device)
        self.device = device
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, y=None):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        class_result = self.classifier(z)

        return recon_x,class_result
        # return z,mu
    
    @torch.no_grad()
    def compute_logit(self,x):
        z, mu, logvar = self.vae_s.encoder(x)
        logit = self.classifier(z)
        return logit.detach().cpu()

    def hardest_class_loss(self,class_result,label,weight):
        neg_prob = 1. - class_result
        neg_label = 1. - label
        pos_loss = torch.mean(torch.sum(- torch.log(class_result + 1e-8) * label,dim=1))
        neg_loss = torch.mean(torch.max(- torch.log(neg_prob + 1e-8) * neg_label,dim=1)[0])
        class_loss = (pos_loss + neg_loss) * 0.5
        return class_loss

    def weight_class_loss(self,class_result,label,weight):
        neg_prob = 1 - class_result
        weight_prob = neg_prob.clone().detach()
        # weight.masked_fill_(1 - label,class_result.detach())
        weight_prob = torch.where(label == 1,weight_prob,class_result.clone().detach())
        # class_result.masked_fill_(1 - label,neg_prob)
        class_result = torch.where(label == 1,class_result,neg_prob)
        class_loss = -torch.mean(torch.sum(weight_prob * torch.log(class_result + 1e-8) * weight,dim=1))
        # class_loss = F.binary_cross_entropy(class_result,label,weight=weight,reduction='mean')
        return class_loss
    
    def ordinary_class_loss(self,class_result,label,weight):
        neg_prob = 1 - class_result
        class_result = torch.where(label == 1,class_result,neg_prob)
        class_loss = -torch.mean(torch.sum(torch.log(class_result + 1e-8),dim=1))
        return class_loss
    
    def zeros_loss(self,*args):
        return self.zero.clone().detach()

    def loss_source(self, x, label = None,domain = None,device = 'cpu',pre = False,weight = None,logit_save:Logit_Linear = None,index = None,weight_softmax = None):
        z, mu, logvar = self.vae_s.encoder(x)
        recon_x = self.vae_s.decoder(z)
        likelihood, kl_loss = elbo(recon_x, x, (mu, logvar),binary=self.binary)
        logit = self.classifier(z)
        c_logit = self.closed_classifier(z)
        c_label = torch.argmax(label,dim = 1)
        closed_class_loss = F.cross_entropy(c_logit,c_label,weight=weight_softmax)
        class_result = torch.sigmoid(logit)
        class_loss = self.class_criterion(class_result,label,weight)

        return (-likelihood,kl_loss,class_loss,z,closed_class_loss)

    def loss_target(self, x,domain = None,label_t = None,index = None,device = 'cpu',memory_bank:LinearAverage = None,th = None):
        z, mu, logvar = self.vae_s.encoder(x)
        recon_x = self.vae_s.decoder(z)
        likelihood, kl_loss = elbo(recon_x, x, (mu, logvar),binary=self.binary)

        close_pred_label = torch.argmax(F.softmax(self.closed_classifier(z)),dim=1)

        pred_logit = self.classifier(z,reverse = True)
        pred_prob = torch.sigmoid(pred_logit)
        class_loss = self.class_t_criterion(pred_prob,memory_bank.threshold)
        memory_bank.update_weight(label=close_pred_label.detach(),index=index.detach(),prob=pred_prob.detach(),epsilon=th)
        
        return (-likelihood,kl_loss,class_loss)

    def class_t_loss(self,pred_prob,threshold):
        # threshold = torch.ones_like(pred_prob) * 0.5
        pred_prob_neg = 1. - pred_prob
        index = torch.arange(0,pred_prob.shape[0])
        max_prob = torch.amax((1 - threshold[index,:]).detach(),dim=0)
        weight = torch.where(pred_prob.detach() > threshold[index,:],1 - pred_prob.detach(),pred_prob.detach())
        weight = torch.where(weight > max_prob,max_prob,weight)
        # weight = torch.abs(pred_prob.detach() - threshold[index,:])
        class_loss = torch.mean(torch.mean(weight*(-threshold[index,:] * torch.log(pred_prob + 1e-8) - (1-threshold)[index,:] * torch.log(pred_prob_neg + 1e-8)),dim=1))
        return class_loss
    
    def fit(self, args,train_iter : ForeverDataIteratorExtension = None,
            val_loader = None,
            test_iter : ForeverDataIterator = None,
            lr=0.002, 
            weight_decay=5e-4,
            beta = 1,
            sigma = 1,
            n = 200,
            max_iter=30000,
            verbose=True,
            patience=100,
            savepath='mymodel.pt',
            imgpath = None,
            device = 'cpu',
            iter = None,
            memory_bank : LinearAverage = None,
            memory_bank_s : LinearAverage = None,
            class_num = 0,
            embedding_size = 0,
            logit_save:Logit_Linear = None,
            weight = None,
            threshold = 0.95
       ):
        self.to(device)
        # self.classifier.to(device)
        self.train()
        if args.class_t:
            self.class_t_criterion = self.class_t_loss
        
        if args.hard_weight:
            self.class_criterion = self.hardest_class_loss
        elif args.sample_weight:
            self.class_criterion = self.weight_class_loss
        else :
            self.class_criterion = self.ordinary_class_loss

        if weight is not None:
            weight_softmax = torch.tensor(weight,dtype=torch.float32).to(device)
        else :
            weight_softmax = weight
        self.sematic_criterion = CenterLoss(num_classes=class_num, feat_dim=embedding_size, use_gpu=True,device=device)
        optimizer_vae = torch.optim.AdamW(self.parameters(),lr = lr,weight_decay=weight_decay)
        loss_plot = {'recon_loss':Loss_Plot(),'class_loss':Loss_Plot(),'kl_loss':Loss_Plot(),'adv_loss':Loss_Plot()}
        threshold_plot = {}
        for i in range(self.class_dim):
            threshold_plot['class_' + str(i)] = Loss_Plot()
        
        iteration = 0
        n_epoch = n

        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
#                 epoch_loss = 0
                self.train()
                epoch_recon_loss,epoch_kl_loss,epoch_class_loss, epoch_adv_loss,epoch_adv_loss_t,epoch_semantic_loss,epoch_class_loss_t,epoch_center_loss\
                                                    = self.zero.clone().detach(),self.zero.clone().detach(),self.zero.clone().detach(),self.zero.clone().detach(),self.zero.clone().detach(),self.zero.clone().detach(),self.zero.clone().detach(),self.zero.clone().detach()
                epoch_c_class_loss = self.zero.clone().detach()
                tk0 = tqdm(range(iter), total=iter, leave=False, desc='Iterations')

                for i in tk0:
                    x_s,label_s,domain_s,index_s,weight = next(train_iter)
                    x_s = x_s.to(device,non_blocking=True)
                    label_s = label_s.to(device,non_blocking=True)
                    domain_s = domain_s.to(device,non_blocking=True)
                    index_s = index_s.to(device,non_blocking=True)
                    weight = weight.to(device,non_blocking=True)
                    optimizer_vae.zero_grad()

                    x_t,label_t,domain_t,index_t = next(test_iter)
                    x_t = x_t.to(device,non_blocking=True)
                    domain_t = domain_t.to(device,non_blocking=True)
                    label_t = label_t.to(device,non_blocking=True)
                    index_t = index_t.to(device,non_blocking=True)

                    recon_loss_s , kl_loss_s,class_loss_s,z_s,c_class_loss = self.loss_source(x = x_s,label=label_s,device=device,domain=domain_s,index = index_s,weight=weight,weight_softmax = weight_softmax)
                    
                    
                    recon_loss_t , kl_loss_t,class_loss_t = self.loss_target(x = x_t,index = index_t,device=device,
                                                                                                    domain=domain_t,memory_bank=memory_bank,th=threshold)
                    loss = (recon_loss_t + kl_loss_t + recon_loss_s + kl_loss_s) / 2 + c_class_loss + class_loss_t * args.class_t_coff + class_loss_s
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 10) # clip
                    optimizer_vae.step()
                    optimizer_vae.zero_grad()
                    
                    epoch_kl_loss += (kl_loss_s.detach() + kl_loss_t.detach()) / 2
                    epoch_recon_loss += (recon_loss_s.detach() + recon_loss_t.detach()) / 2
                    epoch_class_loss += class_loss_s.detach() * args.class_coff
                    epoch_class_loss_t += class_loss_t.detach() * args.class_t_coff
                    epoch_c_class_loss += c_class_loss.detach() *  args.class_coff

                    tk0.update(1)
                    
                    iteration+=1
                loss_plot['recon_loss'].add_loss(epoch_recon_loss.cpu().float())
                loss_plot['class_loss'].add_loss(epoch_class_loss.cpu().float())
                loss_plot['adv_loss'].add_loss(epoch_class_loss_t.cpu().float())
                loss_plot['kl_loss'].add_loss(epoch_kl_loss.cpu().float())
                for i in range(self.class_dim):
                    threshold_plot['class_' + str(i)].add_loss(memory_bank.threshold[0][i].cpu().float())
                # self.eval()
                
                tq.set_postfix_str('recon {:.3f} kl {:.3f} o_class={:.3f} c_class={:.3f} class_t={:.3f} supLoss={:.3f} center={:.3f}'.format(
                    epoch_recon_loss.cpu().float() / iter, epoch_kl_loss.cpu().float() / iter, epoch_class_loss.cpu().float()/ iter,epoch_c_class_loss.cpu().float()/ iter,epoch_class_loss_t.cpu().float()/ iter,epoch_semantic_loss.cpu().float() / i,epoch_center_loss.cpu().float() / i))
        if val_loader == None:
            torch.save(self.state_dict(), savepath + '_vae_model.pt')
        loss_plot['recon_loss'].loss_plot(imgpath + '_recon_loss_plot.png')
        loss_plot['class_loss'].loss_plot(imgpath + '_class_loss_plot.png')
        loss_plot['adv_loss'].loss_plot(imgpath + '_adv_loss_plot.png')
        loss_plot['kl_loss'].loss_plot(imgpath + '_kl_loss_plot.png')
        for i in range(self.class_dim):
            threshold_plot['class_' + str(i)].loss_plot(imgpath + '_class_' + str(i) + '_plot.png')
    
    def predict_class(self,dataloader,device = "cpu",train = False):
        self.eval()
        latent = self.encodeBatch(dataloader=dataloader,device = device,train = train)
        latent = torch.tensor(latent).to(device)
        logit = self.classifier(latent)
        o_result = F.sigmoid(logit)
        c_logit = self.closed_classifier(latent)
        c_result = F.softmax(c_logit)
        # logit = logit.detach().cpu()
        o_result = o_result.detach().cpu()
        c_result = c_result.detach().cpu()
        pred_label = np.argmax(c_result,axis=1)
        
        return pred_label.numpy(),o_result.numpy(),latent.detach().cpu().numpy()

    def encodeBatch(self, dataloader, device='cpu', out='z',train = False):
        output = []
        output_label = []
        for x in dataloader:
            x,_ = x
            x = x.view(x.size(0), -1).float().to(device)
            if train:
                z,mu,logvar = self.vae_s.encoder(x)
            else :
                z,mu,logvar = self.vae_s.encoder(x)
            # recon_x = self.vae_s.decoder(z)

            if out == 'z':
                output.append(z.detach().cpu())
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach().data)

        output = torch.cat(output).numpy()

        return output
