import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,cohen_kappa_score,precision_score,recall_score
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.preprocessing import OneHotEncoder

def compute_mf1(y_pred,y_true,max_label = None):
    unique = np.unique(y_true)
    if max_label is not None:
        unique = unique[unique != np.max(unique)]
    f1_score_list = []
    for label in unique:
        f1 = f1_score(np.array(y_true)==label,np.array(y_pred)==label)
        f1_score_list.append(f1)
    return np.median(f1_score_list)
    # return np.median(f1_score(y_true=y_true,y_pred=y_pred,average=None))

def compute_mean_f1(y_pred,y_true,max_label = None):
    unique = np.unique(y_true)
    if max_label is not None:
        unique = unique[unique != np.max(unique)]
    f1_score_list = []
    for label in unique:
        f1 = f1_score(np.array(y_true)==label,np.array(y_pred)==label)
        f1_score_list.append(f1)
    return np.mean(f1_score_list)

def compute_mf1_v2(y_pred,y_true):
    return np.median(f1_score(y_true=y_true,y_pred=y_pred,average=None)[0:len(np.unique(y_true)) - 1].tolist())

def compute_accu(y_pred,y_true):
    return accuracy_score(y_true=y_true,y_pred=y_pred)

def compute_cohen(y_pred,y_true):
    return cohen_kappa_score(y1=y_true,y2=y_pred)

def compute_metric(y_pred,y_true,max_label = None):
    return compute_accu(y_pred,y_true),compute_mf1(y_pred,y_true,max_label=max_label),compute_cohen(y_pred,y_true)

def compute_ari(y_pred,y_true):
    return np.median(ARI(labels_pred=y_pred,labels_true=y_true))
    
def compute_auc(y_pred,y_true,max_label = None):
    unique = np.unique(y_true)
    if max_label is not None:
        unique = unique[unique != np.max(unique)]
    auc_score_list = []
    for label in unique:
        auc = roc_auc_score(np.array(y_true)==label,np.array(y_pred)==label)
        auc_score_list.append(auc)
    return np.median(auc_score_list)
    # y_true = OneHotEncoder().fit_transform(y_true.reshape(-1,1)).toarray()
    # return np.median(roc_auc_score(y_true = y_true,y_score=y_pred,average=None))

def compute_binary_auc(y_prob,y_true):
    return roc_auc_score(y_true=y_true,y_score=y_prob)

def compute_EAS(y_pred,y_true,unknown):
    y_true_known = y_true[y_true != unknown]
    y_true_unknown = y_true[y_true == unknown]
    share_assign = len(y_true_known[y_pred[y_true != unknown] == y_true_known]) / len(y_true_known)
    unique_assign = len(y_true_unknown[y_pred[y_true == unknown] != y_true_unknown]) / len(y_true_unknown)
    return share_assign - unique_assign

def compute_precision_recall(y_pred,y_true):
    unique = np.unique(y_true)
    # unique = unique[unique != np.max(unique)]
    for label in unique:
        print("label {} precision score :{},recall score:{}".format(label,precision_score(np.array(y_true)==label,np.array(y_pred)==label),recall_score(np.array(y_true)==label,np.array(y_pred)==label)))

def compute_EAS_EpiAnno(y_pred,origin_label,y_true):
    pred_same_label = []
    true_same_label = []
    true_diff_label = []
    pred_diff_label = []
    same_assigned = []
    diff_assigned = []


    same_true_assigned = 0
    diff_true_assigned = 0
    for i in range(len(y_true)):
        if y_true[i] in origin_label:
            true_same_label.append(y_true[i])
            pred_same_label.append(y_true[i])
            # if y_pred[i] in origin_label:
            if y_pred[i] in origin_label:
    #             print('bingo')
                same_true_assigned += 1
                same_assigned.append('assigned')
            else:
    #             print(1)
                same_assigned.append('unassigned')
        else:
            true_diff_label.append(y_true[i])
            pred_diff_label.append(y_true[i])
            # if y_pred[i] in origin_label:
            if y_pred[i] in origin_label:
                diff_true_assigned += 1
                diff_assigned.append('assigned')
            else:
                diff_assigned.append('unassigned')
    # print("same:{},diff:{}".format(same_true_assigned,diff_true_assigned))
    same_assigned_rate = same_true_assigned/len(true_same_label)
    diff_assigned_rate = diff_true_assigned/len(true_diff_label)

    return same_assigned_rate-diff_assigned_rate,same_true_assigned,diff_true_assigned