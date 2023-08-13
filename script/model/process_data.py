from doctest import testfile
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix,csc_matrix,csc_array

from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.feature_extraction.text import TfidfTransformer

from episcanpy.api.pp import select_var_feature
from pybedtools import bedtool
import random
from typing import Literal
from anndata import AnnData
import anndata as ad
import scanpy as sc
from xgboost import train
import math

def CreateOneHotEncode(num):
    onehot = OneHotEncoder(categories = np.array(range(0,num)).reshape(1,-1),sparse = False).fit(np.array(range(0,num)).reshape(-1,1))
    return onehot

def CreateLabelEncode(label):
    enc = LabelEncoder()
    enc.fit(np.append(label,"unknown"))
    return enc

def LabelToNum(adata,enc):
    adata.obs["Label"] = enc.transform(adata.obs["CellType"])
    max_label = np.max(adata.obs["Label"])
    adata.obs["Label_binary"] = [0 if x == max_label else 1 for x in adata.obs["Label"]]
    return adata

def select_peak(train_data,test_data,peak_rate = 0.01):
    print("raw train data:{}".format(train_data))
    print("raw test data:{}".format(test_data))
    # eps = 1e-6
    sc.pp.filter_cells(train_data,min_genes=1)
    sc.pp.filter_genes(train_data,min_cells=int(train_data.shape[0] * peak_rate))
    test_data = test_data[:,train_data.var.index.values]
    sc.pp.filter_genes(test_data,min_cells=int(test_data.shape[0] * peak_rate))
    train_data = train_data[:,test_data.var.index.values]
    select_var_feature(train_data,min_score=-1e4,show = False)
    var_annot_train = train_data.var.sort_values(ascending=False, by ='variability_score')
    max_len = len(var_annot_train)
    train_index = 20000
    var_annot_train = var_annot_train.iloc[0:train_index,:]
    var_annot = var_annot_train.index.values
    
    train_data = train_data[:,var_annot.tolist()]
    test_data = test_data[:,train_data.var.index.values]
    sc.pp.filter_cells(train_data,min_genes=1)
    sc.pp.filter_cells(test_data,min_genes=1)
    if not isinstance(train_data.X,np.ndarray):
        train_data = AnnData(train_data.X.A,obs=train_data.obs,var=train_data.var)
    if not isinstance(test_data.X,np.ndarray):
        test_data = AnnData(test_data.X.A,obs=test_data.obs,var=test_data.var)
    # train_data.X = tfidf_transform(train_data.X.tocsr(),norm='l1').tocsc()
    # test_data.X = tfidf_transform(test_data.X.tocsr(),norm='l1').tocsc()
    # sc.pp.scale(train_data)
    # sc.pp.scale(test_data)

    print("processed_atac:{}".format(train_data))
    print("process test data:{}".format(test_data))
    return train_data,test_data

def RNA_Process(train_data,test_data,peak_rate = 0.001):
    print("raw train data:{}".format(train_data))
    print("raw test data:{}".format(test_data))
    sc.pp.filter_cells(train_data,min_genes=1)
    sc.pp.filter_genes(train_data,min_counts=round(peak_rate * train_data.shape[0]))
    test_data = test_data[:,train_data.var.index.values]
    sc.pp.filter_genes(test_data,min_counts=round(peak_rate * test_data.shape[0]))
    train_data = train_data[:,test_data.var.index.values]
    sc.pp.normalize_total(train_data, target_sum=1e4)
    sc.pp.log1p(train_data)
    sc.pp.normalize_total(test_data, target_sum=1e4)
    sc.pp.log1p(test_data)
    sc.pp.highly_variable_genes(train_data,n_top_genes = 3000)
    train_gene_index = train_data[:,train_data.var['highly_variable'].values == True].var.index.values
    
    train_data=train_data[:,train_gene_index.tolist()]
    test_data=test_data[:,train_gene_index.tolist()]

    if not isinstance(train_data.X,np.ndarray):
        train_data = AnnData(train_data.X.A,obs=train_data.obs,var=train_data.var)
    if not isinstance(test_data.X,np.ndarray):
        test_data = AnnData(test_data.X.A,obs=test_data.obs,var=test_data.var)

    print("processed_atac:{}".format(train_data))
    print("process test data:{}".format(test_data))
    return train_data,test_data

