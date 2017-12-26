#!/usr/bin/python
#-*- coding: utf-8 -*-

import time
import cv_eval
from functions import *
from grlmf import GRLMF

cvs, sp_arg, model_settings, predict_num = "a", 1, [], 0
method = "grlmf"
seeds = [7771, 8367, 22, 1812, 4659]
args = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, 'lambda_t': 0.125, 'alpha': 0.25, 'beta': 0.125,
        'theta': 0.5, 'max_iter': 100}

for key, val in model_settings:
    args[key] = val

#-- train HMDD
intMat = np.loadtxt('mirna_dis_assM.txt', dtype=np.float64)  # mirna-disease interaction matrix
miRNAMat = np.loadtxt('MiSiMat_new.txt', dtype=np.float64)  # mirna similarity matrix
disMat = np.loadtxt('DiSiMat_new.txt', dtype=np.float64)  # disease similarity matrix


if predict_num == 0:
    if cvs == "a":  # CV setting CV_a
        X, D, T, cv = intMat, miRNAMat, disMat, 1
    if cvs == "r":  # CV setting CV_r
        X, D, T, cv = intMat, miRNAMat, disMat, 0
    if cvs == "d":  # CV setting CV_d
        X, D, T, cv = intMat.T, disMat, miRNAMat, 0

    cv_data = cross_validation(X, seeds, cv, num=5)

if sp_arg == 1 or predict_num > 0:
    tic = time.clock()
    model = GRLMF(K1=7, K2=8, num_factors=2, lambda_d=0.125,
                  lambda_t=0.15, alpha=0.25, beta=0.25, theta=0.5,
                  max_iter=150)
cmd = str(model)
if predict_num == 0:
    print (" CVS:"+ cvs+"\n"+cmd)
    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
    print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))

elif predict_num > 0:
    f1 = open('mirna_name.txt', 'r')
    f2 = open('dis_name.txt', 'r')
    mirna_names = []
    dis_names = []
    for ele in f1.readlines():
        mirna_names.append(ele.strip('\n'))

    for ele in f2.readlines():
        dis_names.append(ele.strip('\n'))

    f1.close()
    f2.close()

    seed = 22
    model.fix_model(intMat, intMat, miRNAMat, disMat, seed)
    x, y = np.where(intMat == 0)
    scores = model.predict_scores(zip(x, y), 5)
    ii = np.argsort(scores)[::-1]

    predict_pairs = [(mirna_names[x[i]], dis_names[y[i]], scores[i]) for i in ii]

    with open("novel_predict.txt", 'w') as f:
        for ele in predict_pairs:
            x = list(ele)
            f.write(x[0] + '\t' + x[1] + '\t' + str(x[2]) + '\n')