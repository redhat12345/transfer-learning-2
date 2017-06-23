import argparse
import logging
import numpy as np

from sklearn.svm import SVC, LinearSVC

import GPy
import GPyOpt


def bayesian_opt(linear_or_kernel, prefix):
    logging.info('Loading CNN codes...')
    data_train = np.load('cnncodes_train.npz')
    data_test = np.load('cnncodes_test.npz')
    X_train, y_train = data_train['cnn_codes'], data_train['y']
    X_test, y_test = data_test['cnn_codes'], data_test['y']
    
    
    logging.info('Bayesian optimisation...')
    # Bayesian opt of the C for linear kernel and C, gamma for rbf kernel.
    if linear_or_kernel == 'linear':
        domain = {'name': 'C',      'type': 'continuous', 'domain': (-10.,3.)},
    else:
        domain = [{'name': 'C',      'type': 'continuous', 'domain': (0.,7.)},
                 {'name': 'gamma',  'type': 'continuous', 'domain': (-12.,-2.)}]
    batch_size = 8
    num_cores = 8
    np.random.seed(123)
    opt = GPyOpt.methods.BayesianOptimization(f = lambda x: fit_svc_val(x, X_train, y_train, linear_or_kernel), # function to optimize       
                                          domain = domain,
                                          acquisition_type ='EI',
                                          normalize_Y = True,
                                          initial_design_numdata = 10,
                                          evaluator_type = 'local_penalization',
                                          batch_size = batch_size,
                                          num_cores = num_cores,
                                          acquisition_jitter = 0,
                                        )
    opt.run_optimization(max_iter=5)
    opt.plot_convergence('{}_convergence.png'.format(prefix))
    opt.plot_acquisition('{}_acquisition.png'.format(prefix))

    x_best = np.exp(opt.X[np.argmin(opt.Y)])
    if linear_or_kernel == 'linear':
        logging.info('The best parameters obtained: C={}'.format(x_best[0]))
        svc = LinearSVC(C=x_best[0], verbose=True)
    else:
        logging.info('The best parameters obtained: C={}, gamma={}'.format(x_best[0], x_best[1]))
        svc = SVC(C=x_best[0], gamma=x_best[1])
    
    svc.fit(X_train,y_train)
    y_train_pred = svc.predict(X_train)
    y_test_pred = svc.predict(X_test)
    logging.info('Accuracy on the training data: {}'.format(np.mean(y_train_pred == y_train)))
    logging.info('Accuracy on the test data: {}'.format(np.mean(y_test_pred == y_test)))
    

def fit_svc_val(x, X_train, Y_train, linear_or_kernel):
    nfold = 3 # nfold cross validation with different split of training data
    x = np.atleast_2d(np.exp(x))
    fs = np.zeros((x.shape[0],1))
    for i in range(x.shape[0]):
        fs[i] = 0
        for n in range(nfold):
            idx = np.array(range(X_train.shape[0]))
            idx_valid = np.logical_and(idx>=X_train.shape[0]/nfold*n, idx<X_train.shape[0]/nfold*(n+1))
            idx_train = np.logical_not(idx_valid)
            if linear_or_kernel == 'linear':
                svc = LinearSVC(verbose=True, C=x[i,0])
            else:
                svc = SVC(verbose=True, C=x[i,0], gamma=x[i,1])
            svc.fit(X_train[idx_train],Y_train[idx_train])
            fs[i] += np.mean(svc.predict(X_train[idx_valid]) == Y_train[idx_valid])
            logging.info('Completed single cross-val with acc {}'.format(fs[i] * 1./nfold))
        fs[i] *= 1./nfold
    return 1 - fs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    modes = ['linear', 'kernel']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=modes[0], choices=modes,
                      help='Operation to perform. Possible values: {}'.format(', '.join(modes)))
    parser.add_argument('--prefix', type=str, default='bayes_opt',
                      help='Prefix to add before the names of the output files')
    args = parser.parse_args()

    bayesian_opt(args.mode, args.prefix)