import time
from functions import *
from grlmf import GRLMF

def grlmf_cv_eval(dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []

    for r in [50, 100]:
        for x in np.logspace(-1, 0, 10):
            for y in np.logspace(-1, 0, 8):
                for z in np.logspace(-2, 0, 10):
                    for t in np.logspace(-1, 0, 8):
                        for k1 in [6, 8, 10]:
                            for k2 in[6, 8, 10]:
                                tic = time.clock()
                                model = GRLMF(cfix=para['c'], K1=para['K1'], K2=para['K2'], num_factors=r, lambda_d=x, lambda_t=x, alpha=y, beta=z, theta=t, max_iter=150)
                                cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                                print (cmd)
                                aupr_vec, auc_vec, fpr, tpr = train(model, cv_data, X, D, T)
                                aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                                auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                                print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
                                if auc_avg > max_auc:
                                    max_auc = auc_avg
                                    auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print (cmd)

