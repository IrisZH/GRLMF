import numpy as np
from collections import defaultdict

def cross_validation(intMat, seeds, cv=0, num=5):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_mirs, num_dis = intMat.shape
        prng = np.random.RandomState(seed)

        if cv == 0:
            index = prng.permutation(num_mirs)
        elif cv == 1:
            index = prng.permutation(intMat.size)
        else:
            index = prng.permutation(num_dis)

        step = int(index.size/num)
        for i in range(num):
            if i < num-1:
                ii = index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_dis)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_dis, k % num_dis] for k in ii], dtype=np.int32)
            else:
                test_data = np.array([[k, j] for j in ii for k in range(num_mirs)], dtype = np.int32)

            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data


def train(model, cv_data, intMat, mirnaMat, disMat):
    aupr, auc = [], []

    for seed in cv_data.keys():
        fpr, tpr = [], []
        c = 1
        for W, test_data, test_label in cv_data[seed]:
            model.fix_model(W, intMat, mirnaMat, disMat, seed)
            aupr_val, auc_val, fpr_val, tpr_val = model.evaluation(test_data, test_label)
            aupr.append(aupr_val)
            auc.append(auc_val)
            fpr.extend(fpr_val)
            tpr.extend(tpr_val)

            # with open('/Users/iriszhang/Code/MiDiAss_PyDTI/Result/' + "fpr_nrlmf_5fold" + str(c) + '.txt', 'w') as f:
            #     for ele in fpr:
            #         f.write(str(ele) + '\n')
            # with open('/Users/iriszhang/Code/MiDiAss_PyDTI/Result/' + "tpr_nrlmf_5fold" + str(c) + '.txt', 'w') as f:
            #     for ele in tpr:
            #         f.write(str(ele) + '\n')
            # c += 1



    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


