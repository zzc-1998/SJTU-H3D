import pandas as pd 
import scipy
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
import argparse
import pprint
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import itertools,os

def str2float_list(strings):
    strings = strings.split(',')
    return [float(string) for string in strings]

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def agg_projections(x):
    global config
    selected_projections = str2float_list(config.selected_projections)
    return (selected_projections * x).sum()

def scale(x,y): # rescale the probability sum as 1 (x/(x+y) + y/(x+y) = 1)
    return (x-y) / (x+y)

def main(config):
    c1 = 100
    c2 = np.pi
    df = pd.read_csv(config.saqm_csv_path)
    numerical_cols = ['text'+ str(i) for i in range(config.num_texts)]
    df = df[numerical_cols]
    #choose the used projections, using all projections by default
    agg_weights = {f'text{i}': agg_projections for i in range(config.num_texts)}
    avg_df = df.groupby(df.index // config.num_projection).agg(agg_weights)  

    # acquire the semantic affinity score  
    QA = scale(avg_df['text0'],avg_df['text1']) + scale(avg_df['text2'],avg_df['text3']) + scale(avg_df['text4'],avg_df['text5'])
    QA = 1 / (1 + np.exp(-QA.values/3))
    

    # acquire the niqe score
    df = pd.read_csv(config.snqm_csv_path)
    numerical_cols = ['niqe']
    df = df[numerical_cols]
    #choose the used projections, using all projections by default
    QN = df.groupby(df.index // config.num_projection).agg({'niqe': agg_projections})
    niqe_values = QN.values
    QN = -1 / (1 + np.exp(-niqe_values/c1)).T[0]
  
    # acquire the geometry loss score  
    df = pd.read_csv(config.glqm_csv_path)
    QG = df['dihedral_angle_mean']
    QG = -1 / (1 + np.exp(-QG.values/c2))

    # acquire the final score
    scores = QA + QN + QG
    
    #begin evaluation, zeroshot by default
    label = pd.read_csv(config.info_path)['MOS'].values/config.mos_scale
    k_fold = config.k_fold
    total_num = config.total_num
    fold_num = int(total_num/k_fold)
    best_all = np.zeros([k_fold, 4])
    features = np.array([QA, QN, QG]).T

    for i in range(k_fold): 
        # do SVR training
        if config.supervised:
            scaler = StandardScaler()
            #scaler = MinMaxScaler()
            X_train = scaler.fit_transform(np.concatenate((features[:i*fold_num,:], features[(i+1)*fold_num:,:]), axis=0))
            y_train = np.concatenate([label[:i*fold_num], label[(i+1)*fold_num:]])
            X_test = scaler.transform(features[i*fold_num:(i+1)*fold_num,:])
            svr = SVR(kernel='rbf')
            svr.fit(X_train, y_train)
            y_output = svr.predict(X_test)
        else:
            y_output = scores[i*fold_num:(i+1)*fold_num]

        y_test = label[i*fold_num:(i+1)*fold_num]
        y_output= y_output * config.mos_scale
        y_test = y_test * config.mos_scale
        y_output_logistic = fit_function(y_test, y_output)
        test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
        test_SROCC = stats.spearmanr(y_output, y_test)[0]
        test_RMSE = np.sqrt(((y_output_logistic-y_test) ** 2).mean())
        test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]
        best_all[i, :] = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
        #print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC, test_PLCC, test_RMSE))
    best_mean = np.mean(best_all, 0)
    print('*************************************************************************************************************************')
    print("The mean performance: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_mean[0], best_mean[1], best_mean[2], best_mean[3]))
    print('*************************************************************************************************************************')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default = 'H3D')
    parser.add_argument('--saqm_csv_path', type=str, default = '')
    parser.add_argument('--glqm_csv_path', type=str, default='')
    parser.add_argument('--info_path', type=str, default='')
    parser.add_argument('--snqm_csv_path', type=str, default='')
    parser.add_argument('--num_texts', type=int, default = 6)
    parser.add_argument('--selected_projections', type=str, default = '1,1,1,1,1,1')
    parser.add_argument('--num_projection', type=int, default = 6 )
    parser.add_argument('--k_fold', type=int, default = 5 )
    parser.add_argument('--total_num', type=int, default = 1120 )
    parser.add_argument('--mos_scale', type=int, default = 1)
    parser.add_argument('--supervised', type=bool, default = False)
    


    config = parser.parse_args()
    pprint.pprint(config.__dict__)
    main(config)

