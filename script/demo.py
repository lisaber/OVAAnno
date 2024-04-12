from types import CellType
import numpy as np
import os
import scanpy as sc
import argparse
parser = argparse.ArgumentParser(description='OVAAnno: Detecting novel cell type in single-cell chromatin accessibility data via open-set domain adaptation')
parser.add_argument('--train_data',type=str, help='train dataset path')
parser.add_argument('--test_data',type=str, help='test dataset path')
parser.add_argument('--epoch',default=4000,type=int, help='training epoch')
parser.add_argument('--class_coff', default=1, type=int, help='classifier coefficients')
parser.add_argument('--class_t_coff', default=1, type=int, help='classifier coefficients')
parser.add_argument('--adv_coff', default=1, type=int, help='adversial coefficients')
parser.add_argument('--binary', action="store_true", help='Flag to set whether to use binary cross-entropy loss for reconstruction')
parser.add_argument('--hard_weight', action="store_true", help='using hardest weight')
parser.add_argument('--sample_weight', action="store_true", help='using sample weight')
parser.add_argument('--batch',default=64,type=int,help='batch size')
parser.add_argument('--lr',default=0.0002,type=float,help='learning rate')
parser.add_argument('--device',default=0,type=int,help='GPU device')
parser.add_argument('--threshold','-t',default=0.95,type=float,help='top inteval(0.95 as a better choice)')
parser.add_argument('--class_t',action="store_true",help='using hardest loss')
parser.add_argument('--flag',default='_',type=str, help='save path flag')
args = parser.parse_args()
epoch = args.epoch
# domain = args.domain
binary = args.binary
batch_size = args.batch
lr = args.lr
threshold = args.threshold
# k = args.k
# time = args.time
device = args.device
flag = args.flag
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
# memory_gpu = 1
memory_gpu = np.argmax(memory_gpu) if device == 5 else device
os.environ['CUDA_VISIBLE_DEVICES'] = str(memory_gpu)
import torch
device = torch.device(0)
from model.dataset import load_train_dataset,load_test_dataset
from model.model import Class_VAE
from model.layers import LinearAverage,Logit_Linear
from model.process_data import select_peak
from utils.metric_compute import compute_metric,compute_mean_f1,compute_EAS,compute_EAS_EpiAnno
from sklearn import preprocessing
from anndata import AnnData
from utils.utils import set_seed,ForeverDataIterator,ForeverDataIteratorExtension

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

set_seed()

train_adata = sc.read_h5ad(args.train_data)
test_adata = sc.read_h5ad(args.test_data)
train_adata.X[train_adata.X > 0] = 1
test_adata.X[test_adata.X > 0] = 1
train_adata.obs['domain'] = 0
test_adata.obs['domain'] = 1

one_hot = preprocessing.OneHotEncoder(sparse=False)
le = preprocessing.LabelEncoder().fit(np.append(np.unique(train_adata.obs['CellType'].values),'unknown'))
le_domain = one_hot.fit([[0],[1]])

train_adata,test_adata = select_peak(train_adata,test_adata,peak_rate=0.001)
# train_adata.write("embedding/" + np.unique(test_adata.obs['tissue'])[0] +"_forebrain_train_pre_embedding.h5ad")
# test_adata.write("embedding/" + np.unique(test_adata.obs['tissue'])[0] +"_forebrain_test_pre_embedding.h5ad")
train_adata.obs['Label'] = le.transform(np.array(train_adata.obs['CellType'].values).reshape(-1,1))
test_adata.obs['Label'] = le.transform(np.array(test_adata.obs['CellType'].values).reshape(-1,1))
# train_adata = balance_populations(train_adata)

class_sample_counts = np.array(list(range(len(np.unique(train_adata.obs['Label'].values)))))
for i in np.unique(train_adata.obs['Label'].values):
    class_sample_counts[i] = len(train_adata[train_adata.obs['Label'].values == i])

# weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)
# sample_weight = weights[train_adata.obs['Label'].values]
# sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weight, num_samples=len(sample_weight), replacement=True)

cell_num = train_adata.shape[0] 
input_dim = train_adata.shape[1]
alpha = test_adata.shape[0] / cell_num

train_onehot = np.zeros((cell_num,len(np.unique(train_adata.obs['Label'].values))))
for i in np.unique(train_adata.obs['Label'].values):
    train_onehot[train_adata.obs['Label'].values == i,i] = 1

weight = np.zeros(len(np.unique(train_adata.obs['Label'].values)))
class_weight = np.zeros((cell_num,len(np.unique(train_adata.obs['Label'].values))))
for i in np.unique(train_adata.obs['Label'].values):
    class_weight[train_adata.obs['Label'].values != i,i] = 1
    class_weight[train_adata.obs['Label'].values == i,i] = np.sum(class_sample_counts[list(range(len(np.unique(train_adata.obs['Label'].values)))) != i]) / class_sample_counts[i]
    weight[i] = max(class_sample_counts) / class_sample_counts[i]
# weight = None

con_select = np.zeros((len(np.unique(train_adata.obs['Label'].values)),cell_num),dtype=bool)
for i in np.unique(train_adata.obs['Label'].values):
    con_select[i,train_adata.obs['Label'].values == i] = True

# train_sampler = ImBalanceSampler(train_adata)
# if args.select:
#     sc.pp.neighbors(test_adata)
#     sc.tl.leiden(test_adata)
#     test_adata.obs['leiden'] = np.array(test_adata.obs['leiden'].values,dtype='i')
#     train_adata.obs['leiden'] = train_adata.obs['Label'].values
#     test_sampler = ImBalanceSampler(test_adata)
# else :
#     test_sampler = None
# device = torch.device(int(np.argmax(memory_gpu)))
# device = torch.device(0)

train_loader = load_train_dataset(train_adata,weight=class_weight,onehotlabel=train_onehot,batch_size=batch_size,drop_last=False,num_workers=4,shuffle=True,sample=None)
test_adata_loader_1 = load_train_dataset(test_adata,batch_size = batch_size,shuffle=True,drop_last=False,num_workers=4,sample=None)

# print('test batch size {}'.format(round(batch_size * alpha)))

latent_dim = 32
encode_dim = [3200,1600,800,400]
# encode_dim = [1024,128]
class_dim = len(np.unique(train_adata.obs["CellType"]))
decode_dim = []
# class_encode_dim = [3200,1600,800,400]
# class_AE_dim = [class_dim,latent_dim,latent_dim,class_encode_dim,decode_dim]
# dims = [input_dim, latent_dim,latent_dim + len(np.unique(train_data.obs["batch"])), encode_dim,decode_dim]
dims = [input_dim, latent_dim,latent_dim, encode_dim,decode_dim]
# c_dim = [latent_dim,latent_dim,class_dim]
c_dim = [latent_dim,class_dim]
# domain_dim = [latent_dim,int(latent_dim / 2),2]
domain_dim = [latent_dim,class_dim]

train_iter = ForeverDataIteratorExtension(train_loader)
test_iter = ForeverDataIterator(test_adata_loader_1)#,length=test_adata.shape[0]
memory_bank = LinearAverage(inputSize=class_dim,outputSize=len(test_adata),device=device,threshold=np.ones((batch_size,class_dim)),celltype = test_adata.obs['Label'].values)
# memory_bank_s = LinearAverage(inputSize=latent_dim,outputSize=len(train_adata),device=device,label = con_select)
# logit_save = Logit_Linear(train_adata.shape[0],train_adata.obs['Label'].values,device=device)
model = Class_VAE(dims,c_dim,domain_dim=domain_dim,dropout=0,binary = binary,finally_activate=None,num_class=class_dim,device=device)
# save_path = ('hardest_class_' if args.hard_weight else 'class_') + str(args.class_coff) + ('_class_adv_' if args.class_adv else '_full_adv_') + str(args.adv_coff) + ('_sematic' if args.sematic else '') + ('_class_t' if args.class_t else '') + '_n_' + str(epoch) + '_' + str(batch_size) + '_' + np.unique(test_adata.obs['tissue'])[0] +'_forebrain_onehot'
save_path = 'n_' + str(epoch) + '_' + str(batch_size) + '_' + flag

model.fit(args,train_iter = train_iter,
            # val_loader=val_loader,
            test_iter = test_iter,
            lr=lr, 
            n = epoch,
            weight_decay=5e-4,
            savepath='save_model/' + save_path,
            imgpath = 'img/' + save_path,
            device = device,
            iter = len(train_loader),
            memory_bank = memory_bank,
            # memory_bank_s = memory_bank_s,
            class_num = class_dim,
            embedding_size = latent_dim,
            # logit_save = logit_save
            weight=weight
            )

# model.load_state_dict(torch.load('save_model/'+ save_path + '_vae_model.pt'))

del train_loader
del test_adata_loader_1
# for i in range(len(train_loader)):
#     x_s,label_s,domain_s,_,th = next(train_iter)
#     x_s = x_s.to(device)
#     logit_s = model.compute_logit(x = x_s)
#     train_iter.update_logit(logit_s)
# model.classifier.load_state_dict(torch.load('save_model/forebrain_classifier_model.pt'))
test_adata_loader = load_test_dataset(test_adata,shuffle=False,drop_last=False)
train_all_loader = load_test_dataset(train_adata,shuffle=False,drop_last=False)

# train_part_pred_label,train_part_prob,_ = model.predict_class(train_loader,device=device,data_type='train')
# val_pred_label,val_true_label,val_prob,_ = model.predict_class(val_loader,device=device)
train_pred_label,train_prob,train_all_embedding = model.predict_class(train_all_loader,device=device,train = True)
test_pred_label,test_prob,test_embedding = model.predict_class(test_adata_loader,device=device)

print("train sigmoid accurary:{},mf1:{},cohen:{}".format(*compute_metric(train_pred_label,train_adata.obs['Label'].values)))
# print("val accurary:{},mf1:{},cohen:{}".format(*compute_metric(val_pred_label,val_true_label)))
# test_adata.obs["score"] = test_prob
# test_adata.obs["Label"] = test_adata.obs["CellType"]
origin_label = np.unique(train_adata.obs["Label"].values)
test_known = test_adata[test_adata.obs["Label"]!=class_dim]
test_pred_known = test_pred_label[test_adata.obs["Label"]!=class_dim]
# tissue = np.unique(test_adata.obs['tissue'].values)

# test_all_label = copy.deepcopy(test_pred_label)
print("------------------------------------------------")
print("{} pred_label:{},length:{},known_length:{}".format(flag,np.unique(test_adata.obs["Label"]),test_adata.shape[0],test_known.shape[0]))
print("accurary:{},mf1:{},cohen:{},mean_f1:{}".format(*compute_metric(test_pred_known,test_known.obs["Label"].values),compute_mean_f1(y_pred=test_pred_known,y_true=test_known.obs["Label"].values)))

test_adata.obs['pred_label_num'] = test_pred_label
test_adata.obs['pred_label'] = le.inverse_transform(test_pred_label)
train_embedding = AnnData(train_all_embedding,obs=train_adata.obs)
test_embedding = AnnData(test_embedding,obs=test_adata.obs)
# test_o_result = np.argmax(test_prob,axis = 1)
test_prob_select = test_prob[np.arange(0,len(test_pred_label)),test_pred_label]
# test_pred_label[test_pred_label != test_o_result] = class_dim
test_embedding.obs["score"] = test_prob_select

# for i in [0.5,0.7,0.8,0.9]:
test_pred_label[test_prob_select < 0.5] = class_dim
print("{} EAS_EpiAnno:{} , same:{} , diff:{}".format(i,*compute_EAS_EpiAnno(y_pred=test_pred_label,origin_label=origin_label,y_true = test_adata.obs["Label"].values)))
print("{} EAS:{}".format(i,compute_EAS(y_pred=test_pred_label,y_true = test_adata.obs["Label"].values,unknown=class_dim)))

# test_o_result = np.argmax(test_prob,axis = 1)
# test_prob_max = np.max(test_prob,axis = 1)
# test_pred_label = test_adata.obs['pred_label_num'].values
# test_pred_label[test_pred_label != test_o_result] = class_dim
# test_pred_label[test_prob_max < 0.5] = class_dim
# print("test prob max EAS_EpiAnno:{} , same:{} , diff:{}".format(*compute_EAS_EpiAnno(y_pred=test_pred_label,origin_label=origin_label,y_true = test_adata.obs["Label"].values)))
# print("test prob max EAS:{}".format(compute_EAS(y_pred=test_pred_label,y_true = test_adata.obs["Label"].values,unknown=class_dim)))
# test_embedding.obs["score_max"] = test_prob_max
# test_embedding.obs["binary_label"] = test_o_result
# print("------------------------------------------------")
# logit_test = torch.from_numpy(logit_test)
# test_score = get_scores(logit_mat=logit_test)
# test_score = test_score.numpy()
# th_pred = np.argmax(test_score,axis=1)
# test_score = np.max(test_score,axis=1)
# test_embedding.obs['logit_score'] = test_score
# test_pred_known = th_pred[test_adata.obs["Label"]!=class_dim]
# print("accurary:{},mf1:{},cohen:{}".format(*compute_metric(test_pred_known,test_known.obs["Label"].values)))

# th_first_pred = copy.deepcopy(th_pred)
# for i in [1,0.95,0.9,0.8]:
#     th = get_th(logit=torch.from_numpy(logit_train),num = class_dim,label=train_adata.obs['Label'].values,epsilon=i)
#     th = th.numpy()
#     th = th[th_first_pred]
#     test_embedding.obs['th_' + str(i)] = th
#     th_pred[test_score < th] = class_dim
#     th_pred = th_pred.astype(np.int)
#     print("{} EAS_EpiAnno:{} , same:{} , diff:{}".format(i,*compute_EAS_EpiAnno(y_pred=th_pred,origin_label=origin_label,y_true = test_adata.obs["Label"].values)))

train_embedding.write('embedding/'+ save_path +"_train_embedding.h5ad")
test_embedding.write('embedding/' + save_path +"_test_embedding.h5ad")