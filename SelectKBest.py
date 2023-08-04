import pandas as pd
from lightgbm import LGBMClassifier
import lightgbm as lgb
import numpy as np
import os
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.svm import SVR,SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,SpectralEmbedding
from sklearn.linear_model import LogisticRegression,LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.cluster import DBSCAN,KMeans
from xgboost import XGBClassifier
from keras.utils import to_categorical
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from scipy import stats
import argparse
import palettable

# args = argparse.
warnings.filterwarnings("ignore")

# 读取文件并合并
def read_and_preprocess_data():
    rna_seq_file_path = r'./RNA_data_sorted_tpm.csv'
    RNA_seq_data = pd.read_csv(rna_seq_file_path,header=0,index_col=0).T
    print(RNA_seq_data.shape)

    SSB_file_path = r'./New_SSB_data_sorted_tpm.csv'
    SSB_data = pd.read_csv(SSB_file_path,header=0,index_col=0).T
    SSB_columns = SSB_data.columns.tolist()
    AP_file_path = r'./New_AP_data_sorted_tpm.csv'

    AP_data = pd.read_csv(AP_file_path,header=0,index_col=0).T
    AP_columns = AP_data.columns.tolist()

    print(SSB_data.shape,AP_data.shape,RNA_seq_data.shape)
    # AP_data = AP_data.T[AP_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=False).index].T
    # SSB_data = SSB_data.T[SSB_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=False).index].T
    # RNA_seq_data = RNA_seq_data.T[RNA_seq_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=False).index].T
    # AP_data_topN_exper_num_ = int(AP_data.shape[0]*0.02)
    # SSB_data_topN_exper_num_ = int(SSB_data.shape[0]*0.02)
    # RNA_seq_data_topN_exper_num_ = int(RNA_seq_data.shape[0]*0.02)
    # AP_data_topN_exper_num = int(AP_data.shape[0]*1.0)
    # SSB_data_topN_exper_num = int(SSB_data.shape[0]*1.0)
    # RNA_seq_data_topN_exper_num = int(RNA_seq_data.shape[0]*1.0)
    # AP_data = AP_data_topN = AP_data.iloc[AP_data_topN_exper_num_:AP_data_topN_exper_num,:]
    # SSB_data = SSB_data_topN = SSB_data.iloc[SSB_data_topN_exper_num_:SSB_data_topN_exper_num,:]
    # RNA_seq_data = RNA_seq_data_topN = RNA_seq_data.iloc[RNA_seq_data_topN_exper_num_:RNA_seq_data_topN_exper_num,:]
    # AP_data = AP_data.T.apply(lambda row: row[row < row.max()] if row.max() < 100 else row, axis=1).dropna()
    # print(SSB_data.shape,AP_data.shape,RNA_seq_data.shape)
    # print(SSB_data.shape,AP_data.shape,RNA_seq_data.shape)
    # AP_data = AP_data.apply(lambda row: row[row < row.max()] if row.max() < 5 else row, axis=1).dropna()
    # print(AP_data.shape)
    # AP_data = AP_data.apply(lambda row: row[row < row.max()] if row.max() >400 else row, axis=1).dropna()
    # print(AP_data.shape)

    # SSB_data = SSB_data.apply(lambda row: row[row < row.max()] if row.max() < 5 else row, axis=1).dropna()
    # print(SSB_data.shape)
    # SSB_data = SSB_data.apply(lambda row: row[row < row.max()] if row.max() >400 else row, axis=1).dropna()
    # print(SSB_data.shape)
    # rna_co = set(RNA_seq_data.T.columns.to_list()) 
    # RNA_seq_data = RNA_seq_data.apply(lambda row: row[row < row.max()] if row.max() < 0.1 else row, axis=1).dropna()
    # RNA_seq_data_mean = RNA_seq_data.values.mean()
    # RNA_seq_data_medium = RNA_seq_data.values.
    RNA_seq_data = RNA_seq_data[RNA_seq_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
    AP_data = AP_data[AP_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
    SSB_data = SSB_data[SSB_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
    # RNA_seq_data = RNA_seq_data[RNA_seq_data.apply(lambda x:  (x == 0).sum()/len(x) <= 0.8, axis=1)].dropna()
    # AP_data = AP_data[AP_data.apply(lambda x: (x == 0).sum()/len(x) <= 0.8, axis=1)].dropna()
    # SSB_data = SSB_data[SSB_data.apply(lambda x: (x == 0).sum()/len(x) <= 0.8, axis=1)].dropna() 
    # rna_cob = set(RNA_seq_data.T.columns.to_list()) 

    # print(set(rna_co) - set(rna_cob))
    # RNA_seq_data = RNA_seq_data.apply(lambda row: row[row > row.min()] if row.max() > 800 else row, axis=1).dropna()
    # SSB_data = SSB_data.apply(lambda x: x[~((x - x.mean()) > 1 * x.std())]).dropna()
    # RNA_seq_data = RNA_seq_data.apply(lambda x: x[~((x - x.mean()) > 1 * x.std())]).dropna()

    # AP_data = (AP_data - AP_data.min(axis=0))/(AP_data.max(axis=0)-AP_data.min(axis=0))
    # SSB_data = (SSB_data - SSB_data.min(axis=0))/(SSB_data.max(axis=0)-SSB_data.min(axis=0))
    # RNA_seq_data = (RNA_seq_data - RNA_seq_data.min(axis=0))/(RNA_seq_data.max(axis=0)-RNA_seq_data.min(axis=0))
    # AP_data = AP_data.applymap(lambda x :np.log2(x+1))
    # SSB_data = SSB_data.applymap(lambda x :np.log2(x+1))
    # RNA_seq_data = RNA_seq_data.applymap(lambda x :np.log2(x+1))

    # AP_data_mean = AP_data.values.mean()
    # AP_data_std = AP_data.values.std()
    # SSB_data_mean = SSB_data.values.mean()
    # SSB_data_std = SSB_data.values.std()
    # RNA_seq_data_mean = RNA_seq_data.values.mean()
    # RNA_seq_data_std = RNA_seq_data.values.std()

    # AP_data = AP_data.apply(lambda x: x[~((x - x.mean()).abs() > 3 * x.std())]).dropna()
    # SSB_data = SSB_data.apply(lambda x: x[~((x - x.mean()).abs() > 3 * x.std())]).dropna()
    # RNA_seq_data = RNA_seq_data.apply(lambda x: x[~((x - x.mean()).abs() > 3 * x.std())]).dropna()
    # AP_data = AP_data.apply(lambda x: x[((x - AP_data_mean)> 1 * AP_data_std)])
    # SSB_data = SSB_data.apply(lambda x: x[((x - SSB_data_mean)> 1 * SSB_data_std)])
    # RNA_seq_data = RNA_seq_data.apply(lambda x: x[((x - RNA_seq_data_mean)> 1 * RNA_seq_data_std)])
    # AP_data = AP_data.fillna(0)
    # SSB_data =SSB_data.fillna(0)
    # RNA_seq_data=RNA_seq_data.fillna(0)
    # print(SSB_data.shape,AP_data.shape,RNA_seq_data.shape)
    # AP_data = AP_data.T[AP_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=False).index]
    # SSB_data = SSB_data.T[SSB_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=False).index]
    # RNA_seq_data = RNA_seq_data.T[RNA_seq_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=False).index]
    new_ssb_columns = []
    for column in AP_columns:
        column = column.split('.')[0]
        new_ssb_columns.append(column)
    RNA_seq_data.columns = new_ssb_columns
    AP_data.columns =new_ssb_columns
    SSB_data.columns = new_ssb_columns
    # AP_data = AP_data.T[AP_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=False).index].T
    # SSB_data = SSB_data.T[SSB_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=False).index].T
    # RNA_seq_data = RNA_seq_data.T[RNA_seq_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=False).index].T
    # AP_data_topN_exper_num_ = int(AP_data.shape[0]*0.20)
    # SSB_data_topN_exper_num_ = int(SSB_data.shape[0]*0.20)
    # RNA_seq_data_topN_exper_num_ = int(RNA_seq_data.shape[0]*0.02)
    # AP_data_topN_exper_num = int(AP_data.shape[0]*0.80)
    # SSB_data_topN_exper_num = int(SSB_data.shape[0]*0.80)
    # RNA_seq_data_topN_exper_num = int(RNA_seq_data.shape[0]*0.80)
    # AP_data = AP_data_topN = AP_data.iloc[AP_data_topN_exper_num_:AP_data_topN_exper_num,:]
    # SSB_data = SSB_data_topN = SSB_data.iloc[SSB_data_topN_exper_num_:SSB_data_topN_exper_num,:]
    # RNA_seq_data = RNA_seq_data_topN = RNA_seq_data.iloc[RNA_seq_data_topN_exper_num_:RNA_seq_data_topN_exper_num,:]
    # AP_data = AP_data.T[AP_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=True).index].T
    # SSB_data = SSB_data.T[SSB_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=True).index].T
    # RNA_seq_data = RNA_seq_data.T[RNA_seq_data.iloc[0:,0:].apply(lambda x: x.max(), axis=1).sort_values(ascending=True).index].T
    # AP_data_topN_exper_num = int(AP_data.shape[0]*0.10)
    # SSB_data_topN_exper_num = int(SSB_data.shape[0]*0.10)
    # RNA_seq_data_topN_exper_num = int(RNA_seq_data.shape[0]*0.10)
    # AP_data_lastN = AP_data.iloc[:AP_data_topN_exper_num,:]
    # SSB_data_lastN = SSB_data.iloc[:SSB_data_topN_exper_num,:]
    # RNA_seq_data_lastN = RNA_seq_data.iloc[:RNA_seq_data_topN_exper_num,:]

    # AP_data = pd.merge(AP_data_lastN,AP_data_topN, how='outer', on=ssb_columns)
    # SSB_data =pd.merge(SSB_data_lastN,SSB_data_topN, how='outer', on=ssb_columns)
    # RNA_seq_data =pd.merge(RNA_seq_data_lastN,RNA_seq_data_topN, how='outer', on=ssb_columns)



    #筛选相同的基因
    # RNA_seq_data = RNA_seq_data.T[SSB_data.T.columns.intersection(RNA_seq_data.T.columns)].T
    # AP_data = AP_data.T[RNA_seq_data.T.columns.intersection(AP_data.T.columns)].T
    # SSB_data = SSB_data.T[AP_data.T.columns.intersection(SSB_data.T.columns)].T
    # RNA_seq_data = RNA_seq_data.T[AP_data.T.columns.intersection(RNA_seq_data.T.columns)].T


    # # print(AP_data['gene_id'])
    y_age = np.zeros(len(AP_columns))
    for j,age in enumerate(['3-','12-','19-','22-','24-']):
        index =[]
        for i,label in enumerate(AP_columns):
                if age in str(label):
                    index.append(i)

        # if j < 2:        
        #     y_age[index] = 0
        # else :
        #     y_age[index] = 1 
        y_age[index] = j
    y_tissue = np.zeros(len(AP_columns))

    for j,tissue in enumerate(['B-','H-','L-','M-','P-','S-']):
        index =[]
        for i,label in enumerate(AP_columns):
                if tissue in str(label):
                    index.append(i)

        # if j < 2:        
        #     y_age[index] = 0
        # else :
        #     y_age[index] = 1 
        y_tissue[index] = j

    print(y_age,y_tissue,SSB_data.shape,AP_data.shape,RNA_seq_data.shape)
    return SSB_data,AP_data,RNA_seq_data,y_age,y_tissue

def caculte_corrs(SSB_data,AP_data,RNA_seq_data,label):
    label = label
    SSB_data = np.array(SSB_data.T)
    RNA_seq_data = np.array(RNA_seq_data.T)
    AP_data = np.array(AP_data.T)
    inv_sum = 0    
    print(RNA_seq_data.shape,AP_data.shape,SSB_data.shape)
    corr_ssb_RNA = np.zeros((SSB_data.shape[0],SSB_data.shape[0]))
    for i in range(SSB_data.shape[0]):
        for j in range(RNA_seq_data.shape[0]):
            corr_ssb_RNA[i][j] = corr_ssb_RNA[i][j] = stats.pearsonr(SSB_data[i],RNA_seq_data[j])[0]
            if corr_ssb_RNA[i][j] <0.2:
                 inv_sum+=1
    # corr_ssb_RNA = np.corrcoef(SSB_data,RNA_seq_data)
    print(corr_ssb_RNA)
    print(corr_ssb_RNA.shape)
    print(inv_sum)
    plt.subplots(figsize=(18,18),dpi=1080,facecolor='w')# 设置画布大小，分辨率，和底色
    plt.subplot(2,2,1)
    fig=sns.heatmap(corr_ssb_RNA,annot=True, cbar=True,vmax=1, square=True, cmap="Reds", fmt='.2g',xticklabels = label,yticklabels = label,annot_kws={'size':0.05,'weight':'normal', 'color':'blue'})#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1  
    corr_AP_RNA = np.zeros((SSB_data.shape[0],SSB_data.shape[0]))
    for i in range(AP_data.shape[0]):
        for j in range(RNA_seq_data.shape[0]):
            corr_AP_RNA[i][j] = corr_AP_RNA[i][j] = stats.pearsonr(AP_data[i],RNA_seq_data[j])[0]        
    plt.subplot(2,2,2)
    fig=sns.heatmap(corr_AP_RNA,annot=True, cbar=True,vmax=1, square=True, cmap="Reds", fmt='.2g',xticklabels = label,yticklabels = label,annot_kws={'size':0.05,'weight':'normal', 'color':'blue'})#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1  
    corr_ssb_AP = np.zeros((SSB_data.shape[0],SSB_data.shape[0]))
    for i in range(SSB_data.shape[0]):
        for j in range(AP_data.shape[0]):
            corr_ssb_AP[i][j] = corr_ssb_AP[i][j] = stats.pearsonr(AP_data[i],SSB_data[j])[0]
    plt.subplot(2,2,3)
    fig=sns.heatmap(corr_ssb_AP,annot=True, cbar=True,vmax=1, square=True, cmap="Reds", fmt='.2g',xticklabels = label,yticklabels = label,annot_kws={'size':0.05,'weight':'normal', 'color':'blue'})#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1  
    plt.savefig('AP^SSB^RNA_corrs_heatmap_caculate_all'+'.png',format = 'png')
    # plt.show()
def caculte_topN_genes_corrs(SSB_data,AP_data,RNA_seq_data,selected_gene_num,label):
    y_label = label
    x_label = SSB_data.columns.tolist()
    SSB_data = np.array(SSB_data)
    RNA_seq_data = np.array(RNA_seq_data)
    AP_data = np.array(AP_data)
    inv_sum = 0    
    plt.subplots(figsize=(18,18),dpi=1080,facecolor='w')# 设置画布大小，分辨率，和底色
    plt.subplot(2,2,1)
    fig=sns.heatmap(SSB_data,annot=True, cbar=True,vmax=1, square=True, cmap="Reds", fmt='.2g',xticklabels = x_label,yticklabels = label,annot_kws={'size':0.05,'weight':'normal', 'color':'blue'})#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1  
    plt.subplot(2,2,2)
    fig=sns.heatmap(AP_data,annot=True, cbar=True,vmax=1, square=True, cmap="Reds", fmt='.2g',xticklabels = x_label,yticklabels = label,annot_kws={'size':0.05,'weight':'normal', 'color':'blue'})#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1  

    plt.subplot(2,2,3)
    fig=sns.heatmap(RNA_seq_data,annot=True, cbar=True,vmax=1, square=True, cmap="Reds", fmt='.2g',xticklabels = label,yticklabels = label,annot_kws={'size':0.05,'weight':'normal', 'color':'blue'})#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1  
    plt.savefig('AP^SSB^RNA_genes_corrs_heatmap_caculate_'+str(selected_gene_num)+'.png',format = 'png')
    # plt.show()

def cluster_compare(methods,SSB_data,AP_data,RNA_seq_data,y_age,y_tissue):

    # AP_data = AP_data.apply(lambda row: row[row < row.max()] if row.max() < 100 else row, axis=1).dropna()
    # SSB_data = SSB_data.apply(lambda row: row[row < row.max()] if row.max() < 100 else row, axis=1).dropna()
    # RNA_seq_data = RNA_seq_data.apply(lambda row: row[row < row.max()] if row.max() < 100 else row, axis=1).dropna()

    # RNA_seq_data = RNA_seq_data.T[SSB_data.T.columns.intersection(RNA_seq_data.T.columns)].T
    # AP_data = AP_data.T[RNA_seq_data.T.columns.intersection(AP_data.T.columns)].T
    # SSB_data = SSB_data.T[AP_data.T.columns.intersection(SSB_data.T.columns)].T
    # RNA_seq_data = RNA_seq_data.T[AP_data.T.columns.intersection(RNA_seq_data.T.columns)].T
    SSB_data = (SSB_data - SSB_data.mean(axis=0))/SSB_data.std(axis=0)
    AP_data =  (AP_data - AP_data.mean(axis=0))/AP_data.std(axis=0)
    RNA_seq_data =  (RNA_seq_data - RNA_seq_data.mean(axis=0))/RNA_seq_data.std(axis=0)

    print(SSB_data.shape,AP_data.shape,RNA_seq_data.shape)

    SSB_data = SSB_data.T
    AP_data = AP_data.T
    RNA_seq_data = RNA_seq_data.T
    AP_SSB_merge_data = SSB_data.add(AP_data,fill_value=0)
    SSB_RNA_seq_merge_data = SSB_data.add(RNA_seq_data,fill_value=0)
    AP_RNA_seq_merge_data = AP_data.add(RNA_seq_data,fill_value=0)
    ALL_merge_data = AP_RNA_seq_merge_data.add(SSB_data,fill_value=0)
    print(AP_SSB_merge_data.shape,SSB_RNA_seq_merge_data.shape,AP_RNA_seq_merge_data.shape,ALL_merge_data.shape)
    AP_SSB_merge_data = (AP_SSB_merge_data - AP_SSB_merge_data.min(axis=0))/(AP_SSB_merge_data.max(axis=0)-AP_SSB_merge_data.min(axis=0))
    SSB_RNA_seq_merge_data = (SSB_RNA_seq_merge_data - SSB_RNA_seq_merge_data.min(axis=0))/(SSB_RNA_seq_merge_data.max(axis=0)-SSB_RNA_seq_merge_data.min(axis=0))
    AP_RNA_seq_merge_data = (AP_RNA_seq_merge_data - AP_RNA_seq_merge_data.min(axis=0))/(AP_RNA_seq_merge_data.max(axis=0)-AP_RNA_seq_merge_data.min(axis=0))
    ALL_merge_data = (ALL_merge_data - ALL_merge_data.min(axis=0))/(ALL_merge_data.max(axis=0)-ALL_merge_data.min(axis=0))
    to_plot_multi_SSB_aucs = []
    to_plot_multi_AP_aucs = []
    to_plot_multi_RNA_seq_aucs = []
    to_plot_multi_SSB_AP_seq_aucs = []        
    to_plot_multi_SSB_RNA_seq_aucs = []
    to_plot_multi_AP_RNA_seq_aucs = []
    to_plot_multi_all_aucs = []
    for method in methods:    
        age_info = ['3','12','19','22','24']
        tissue_info = ['B-','H-','L-','M-','P-','S-']
        for idx in range(0,2):
            to_save_results = []
            if idx == 0:
                y_label = y_age
                label_info =age_info
            else :
                label_info =tissue_info
                y_label = y_tissue
            if method == 'LDA':            
                lda = LinearDiscriminantAnalysis(n_components = 2)         
                clf = SVC(kernel = 'rbf',gamma=0.15,C=1)   
                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                lda.fit(x_train,y_train)
                X_train_ida = lda.transform(x_train)
                X_test_ida = lda.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                lda.fit(x_train,y_train)
                X_train_ida = lda.transform(x_train)
                X_test_ida = lda.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(multi_auc)+'\n'

                to_save_results.append(to_save_result)
                x_train,x_test,y_train,y_test = train_test_split(np.array(RNA_seq_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                lda.fit(x_train,y_train)
                X_train_ida = lda.transform(x_train)
                X_test_ida = lda.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_RNA_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_SSB_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                lda = LinearDiscriminantAnalysis(n_components = 2)         
                lda.fit(x_train,y_train)
                X_train_ida = lda.transform(x_train)
                X_test_ida = lda.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_AP_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP_SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                lda.fit(x_train,y_train)
                X_train_ida = lda.transform(x_train)
                X_test_ida = lda.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
   
                to_plot_multi_SSB_RNA_seq_aucs.append(multi_auc)             
                to_save_result = 'cluster method :'+method+' SSB_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(multi_auc)+'\n'
                to_save_results.append(to_save_result)
                
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                lda.fit(x_train,y_train)
                X_train_ida = lda.transform(x_train)
                X_test_ida = lda.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *5
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
  
                to_plot_multi_AP_RNA_seq_aucs.append(multi_auc)              
                to_save_result = 'cluster method :'+method+' AP_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(ALL_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                lda.fit(x_train,y_train)
                X_train_ida = lda.transform(x_train)
                X_test_ida = lda.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_all_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' ALL_merge :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

            elif method == 'PCA':
                clf = SVC(kernel = 'rbf',gamma=0.15,C=1)   
                pca = PCA(n_components = 5)
                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                pca.fit(x_train,y_train)
                X_train_ida = pca.transform(x_train)
                X_test_ida = pca.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                print(pred_label)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                pca = PCA(n_components = 5)
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                pca.fit(x_train,y_train)
                X_train_ida = pca.transform(x_train)
                X_test_ida = pca.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'

                to_save_results.append(to_save_result)
                x_train,x_test,y_train,y_test = train_test_split(np.array(RNA_seq_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                pca.fit(x_train,y_train)
                X_train_ida = pca.transform(x_train)
                X_test_ida = pca.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_RNA_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_SSB_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                pca.fit(x_train,y_train)
                X_train_ida = pca.transform(x_train)
                X_test_ida = pca.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_AP_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP_SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                pca.fit(x_train,y_train)
                X_train_ida = pca.transform(x_train)
                X_test_ida = pca.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
   
                to_plot_multi_SSB_RNA_seq_aucs.append(multi_auc)             
                to_save_result = 'cluster method :'+method+' SSB_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)
                
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                pca.fit(x_train,y_train)
                X_train_ida = pca.transform(x_train)
                X_test_ida = pca.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_RNA_seq_aucs.append(multi_auc)              
                to_save_result = 'cluster method :'+method+' AP_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(ALL_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                pca.fit(x_train,y_train)
                X_train_ida = pca.transform(x_train)
                X_test_ida = pca.transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_all_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' ALL_merge :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

            elif method == 'TSNE':
                tsne = TSNE(n_components=3, verbose=0, perplexity=5)
                clf = SVC(kernel = 'rbf',gamma=0.15,C=1)   
                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                tsne.fit(x_train,y_train)
                X_train_ida = tsne.fit_transform(x_train)
                X_test_ida = tsne.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                # print(to_save_result)
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                tsne.fit(x_train,y_train)
                X_train_ida = tsne.fit_transform(x_train)
                X_test_ida = tsne.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                # print(to_save_result)

                to_save_results.append(to_save_result)
                x_train,x_test,y_train,y_test = train_test_split(np.array(RNA_seq_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                tsne.fit(x_train,y_train)
                X_train_ida = tsne.fit_transform(x_train)
                X_test_ida = tsne.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                # print(pred)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_RNA_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                # print(to_save_result)
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_SSB_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                tsne.fit(x_train,y_train)
                X_train_ida = tsne.fit_transform(x_train)
                X_test_ida = tsne.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_AP_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP_SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                tsne.fit(x_train,y_train)
                X_train_ida = tsne.fit_transform(x_train)
                X_test_ida = tsne.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
   
                to_plot_multi_SSB_RNA_seq_aucs.append(multi_auc)            
                to_save_result = 'cluster method :'+method+' SSB_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)
                
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                tsne.fit(x_train,y_train)
                X_train_ida = tsne.fit_transform(x_train)
                X_test_ida = tsne.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
  
                to_plot_multi_AP_RNA_seq_aucs.append(multi_auc)              
                to_save_result = 'cluster method :'+method+' AP_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(ALL_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                tsne.fit(x_train,y_train)
                X_train_ida = tsne.fit_transform(x_train)
                X_test_ida = tsne.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_all_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' ALL_merge :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

            elif method == 'SpectralEmbedding':
                embedding = SpectralEmbedding(n_components=5)
                clf = SVC(kernel = 'rbf',gamma=0.15,C=1)   
                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                embedding .fit(x_train,y_train)
                X_train_ida = embedding .fit_transform(x_train)
                X_test_ida = embedding .fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                embedding .fit(x_train,y_train)
                X_train_ida = embedding .fit_transform(x_train)
                X_test_ida = embedding .fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'

                to_save_results.append(to_save_result)
                x_train,x_test,y_train,y_test = train_test_split(np.array(RNA_seq_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                embedding .fit(x_train,y_train)
                X_train_ida = embedding .fit_transform(x_train)
                X_test_ida = embedding .fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_RNA_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_SSB_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                embedding.fit(x_train,y_train)
                X_train_ida = embedding.fit_transform(x_train)
                X_test_ida = embedding.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_AP_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP_SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                embedding.fit(x_train,y_train)
                X_train_ida = embedding.fit_transform(x_train)
                X_test_ida = embedding.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_RNA_seq_aucs.append(multi_auc)             
                to_save_result = 'cluster method :'+method+' SSB_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)
                
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                embedding.fit(x_train,y_train)
                X_train_ida = embedding.fit_transform(x_train)
                X_test_ida = embedding.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_RNA_seq_aucs.append(multi_auc)              
                to_save_result = 'cluster method :'+method+' AP_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(ALL_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                embedding.fit(x_train,y_train)
                X_train_ida = embedding.fit_transform(x_train)
                X_test_ida = embedding.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_all_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' ALL_merge :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

            elif method == 'KMeans':
                if idx == 0:
                    knn  = KMeans(n_clusters=5)
                else :
                    knn  = KMeans(n_clusters=6)
                clf = SVC(kernel = 'rbf',gamma=0.15,C=1)   
                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                knn.fit(x_train,y_train)
                X_train_ida = knn.fit_transform(x_train)
                X_test_ida = knn.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                # print(to_save_result)
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                knn.fit(x_train,y_train)
                X_train_ida = knn.fit_transform(x_train)
                X_test_ida = knn.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                # print(to_save_result)

                to_save_results.append(to_save_result)
                x_train,x_test,y_train,y_test = train_test_split(np.array(RNA_seq_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                knn.fit(x_train,y_train)
                X_train_ida = knn.fit_transform(x_train)
                X_test_ida = knn.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_RNA_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                # print(to_save_result)
                to_save_results.append(to_save_result)
        
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_SSB_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                knn.fit(x_train,y_train)
                X_train_ida = knn.fit_transform(x_train)
                X_test_ida = knn.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_AP_seq_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' AP_SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                knn.fit(x_train,y_train)
                X_train_ida = knn.fit_transform(x_train)
                X_test_ida = knn.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_RNA_seq_aucs.append(multi_auc)             
                to_save_result = 'cluster method :'+method+' SSB_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)
                
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                knn.fit(x_train,y_train)
                X_train_ida = knn.fit_transform(x_train)
                X_test_ida = knn.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_RNA_seq_aucs.append(multi_auc)              
                to_save_result = 'cluster method :'+method+' AP_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(ALL_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                knn.fit(x_train,y_train)
                X_train_ida = knn.fit_transform(x_train)
                X_test_ida = knn.fit_transform(x_test)
                clf.fit(X_train_ida,y_train) 
                pred_label = clf.predict(X_test_ida)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_all_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' ALL_merge :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)
            elif method == 'KNeighborsClassifier':
                if idx == 0:
                    neighbors = KNeighborsClassifier(n_neighbors=5)
                else :
                    neighbors = KNeighborsClassifier(n_neighbors=6)
                clf = SVC(kernel = 'rbf',gamma=0.15,C=1)   
                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                neighbors.fit(x_train,y_train)
                pred_label = neighbors.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                neighbors.fit(x_train,y_train)
                pred_label = neighbors.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_aucs.append(multi_auc)   
                to_save_result = 'cluster method :'+method+' AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)
                x_train,x_test,y_train,y_test = train_test_split(np.array(RNA_seq_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                neighbors.fit(x_train,y_train)
                pred_label = neighbors.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_RNA_seq_aucs.append(multi_auc)    
                to_save_result = 'cluster method :'+method+' RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_SSB_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                neighbors.fit(x_train,y_train)
                pred_label = neighbors.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_AP_seq_aucs.append(multi_auc)    
                to_save_result = 'cluster method :'+method+' AP_SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                neighbors.fit(x_train,y_train)
                pred_label = neighbors.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_RNA_seq_aucs.append(multi_auc)    
                to_save_result = 'cluster method :'+method+' SSB_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)
                
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                neighbors.fit(x_train,y_train)
                pred_label = neighbors.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_RNA_seq_aucs.append(multi_auc)    
                to_save_result = 'cluster method :'+method+' AP_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(ALL_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                neighbors.fit(x_train,y_train)
                pred_label = neighbors.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_all_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' ALL_merge :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

            elif method == 'DBSCAN':
                DB = DBSCAN(eps=5, min_samples=20)
                clf = SVC(kernel = 'rbf',gamma=0.15,C=1)   
                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                DB.fit(x_train,y_train)
                print(DB.labels_)
                clf.fit(x_train,DB.labels_) 
                to_plot_multi_SSB_aucs.append(multi_auc)
                pred_label = clf.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')

                to_save_result = 'cluster method :'+method+' SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                # print(to_save_result)
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                DB.fit(x_train,y_train)
                clf.fit(x_train,DB.labels_) 
                pred_label = clf.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
             
                to_plot_multi_AP_aucs.append(multi_auc)   
                to_save_result = 'cluster method :'+method+' AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                # print(to_save_result)

                to_save_results.append(to_save_result)
                x_train,x_test,y_train,y_test = train_test_split(np.array(RNA_seq_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                DB.fit(x_train,y_train)
                clf.fit(x_train,DB.labels_) 
                pred_label = clf.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
            
                to_plot_multi_RNA_seq_aucs.append(multi_auc)    
                to_save_result = 'cluster method :'+method+' RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                # print(to_save_result)
                to_save_results.append(to_save_result)
                
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_SSB_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                DB.fit(x_train,y_train)
                clf.fit(x_train,DB.labels_) 
                pred_label = clf.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_AP_seq_aucs.append(multi_auc)    
                to_save_result = 'cluster method :'+method+' AP_SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                DB.fit(x_train,y_train)
                clf.fit(x_train,DB.labels_) 
                pred_label = clf.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_SSB_RNA_seq_aucs.append(multi_auc)    
                to_save_result = 'cluster method :'+method+' SSB_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)
                
                x_train,x_test,y_train,y_test = train_test_split(np.array(AP_RNA_seq_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                DB.fit(x_train,y_train)
                clf.fit(x_train,DB.labels_) 
                pred_label = clf.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_AP_RNA_seq_aucs.append(multi_auc)    
                to_save_result = 'cluster method :'+method+' AP_RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)

                x_train,x_test,y_train,y_test = train_test_split(np.array(ALL_merge_data),y_label,test_size=0.15,random_state=44,stratify=y_label)
                DB.fit(x_train,y_train)
                clf.fit(x_train,DB.labels_) 
                pred_label = clf.predict(x_test)
                if idx == 0:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0 :
                        pred = np.array([[1,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
                    pred_label = np.nan_to_num(pred_label)
                    if np.max(pred_label) == 0:
                        pred = np.array([[1,0,0,0,0,0] for i in range(len(pred_label))],dtype = np.float32)
                    else:
                        pred = to_categorical(pred_label)
                pred_label = np.array(pred_label,dtype=np.int64)
                multi_auc = roc_auc_score(y_test, pred, multi_class='ovo')
                to_plot_multi_all_aucs.append(multi_auc)
                to_save_result = 'cluster method :'+method+' ALL_merge :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) + 'multi_AUC_result:' +str(multi_auc)+'\n'
                to_save_results.append(to_save_result)  
            if idx == 0: 
                cluster_output_file_path = './result/'+'cluster_age/'
            else :
                cluster_output_file_path = './result/'+'cluster_tissue/'
            if not os.path.exists(cluster_output_file_path):
                os.makedirs(cluster_output_file_path)
            cluster_output_file = open(cluster_output_file_path+method+'.txt','w')
            for  i in to_save_results:
                cluster_output_file.write(i)
            
# 训练 LightGBM 模型
def select_topN_genes(svc,SSB_data, AP_data,RNA_seq_data, y_label,saveFlag = 0,feature_numbers=500):
  
    from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif,chi2,f_regression,r_regression,mutual_info_regression
    model = SelectKBest(f_classif, k=feature_numbers)
    # AP_data = model.fit_transform(AP_data.T, y_label)
    # SSB_data = model.fit_transform(SSB_data.T , y_label)
    # RNA_seq_data = model.fit_transform(RNA_seq_data.T, y_label)
    # return AP_data,SSB_data,RNA_seq_data
    selected_AP_data = model.fit_transform(AP_data.T, y_label)
    # print(AP_data.T['0610007P14Rik'])
    AP_scores = model.scores_
    AP_p_values = model.pvalues_
    indices = np.argsort(AP_scores)[::-1]
    # k_best_features = list(AP_data.T.columns.values[indices[0:feature_numbers]])
    AP_names = set(model.get_feature_names_out(AP_data.T.columns.tolist()))
    # k_best_features = set(k_best_features)
    # print(names == k_best_features)
    if feature_numbers == 'all':
        selected_AP_data = AP_data.T.iloc[:,indices]
    else:
        selected_AP_data = AP_data.T.iloc[:,indices[:feature_numbers]]
    # selected_AP_data = AP_data.T[model.get_feature_names_out(AP_data.T.columns.tolist())]

    # print(k_best_features,selected_AP_data.columns)

    selected_SSB_data = model.fit_transform(SSB_data.T , y_label)
    SSB_scores = model.scores_

    SSB_p_values = model.pvalues_
    indices = np.argsort(SSB_scores)[::-1]
    k_best_features = list(SSB_data.T.columns.values[indices])
    SSB_names = set(model.get_feature_names_out(SSB_data.T.columns.tolist()))
    k_best_features = set(k_best_features)
    if feature_numbers == 'all':
        selected_SSB_data = SSB_data.T.iloc[:,indices]
    else:
        selected_SSB_data = SSB_data.T.iloc[:,indices[:feature_numbers]]
    # selected_SSB_data = SSB_data.T[model.get_feature_names_out(SSB_data.T.columns.tolist())]

    model = SelectKBest(f_classif, k=feature_numbers)

    selected_RNA_seq_data = model.fit_transform(RNA_seq_data.T, y_label)

    RNA_scores = model.scores_
    RNA_p_values = model.pvalues_
    indices = np.argsort(RNA_p_values)
    indices = np.argsort(RNA_scores)[::-1]

    k_best_features = list(RNA_seq_data.T.columns.values[indices])
    RNA_names = set(model.get_feature_names_out(RNA_seq_data.T.columns.tolist()))
    k_best_features = set(k_best_features)
    # print(RNA_names == k_best_features)
    if feature_numbers == 'all':
        selected_RNA_seq_data = RNA_seq_data.T.iloc[:,indices]
    else:
        selected_RNA_seq_data = RNA_seq_data.T.iloc[:,indices[:feature_numbers]]
    # selected_RNA_seq_data = RNA_seq_data.T[model.get_feature_names_out(RNA_seq_data.T.columns.tolist())]

    if feature_numbers == 'all':
        AP_feature_importance_df = pd.DataFrame({'scores': sorted(AP_scores,reverse=1),'alogrithm-pvalue': sorted(AP_p_values)},index =selected_AP_data.columns.tolist())
        SSB_feature_importance_df = pd.DataFrame({'scores': sorted(SSB_scores,reverse=1), 'alogrithm-pvalue': sorted(SSB_p_values)},index =selected_SSB_data.columns.tolist())
        RNA_feature_importance_df = pd.DataFrame({'scores': sorted(RNA_scores,reverse=1), 'alogrithm-pvalue': sorted(RNA_p_values)},index =selected_RNA_seq_data.columns.tolist() )
    else:
        AP_feature_importance_df = pd.DataFrame({'scores': sorted(AP_scores,reverse=1)[:feature_numbers],'alogrithm-pvalue': sorted(AP_p_values)[:feature_numbers]},index =selected_AP_data.columns.tolist())
        SSB_feature_importance_df = pd.DataFrame({'scores': sorted(SSB_scores,reverse=1)[:feature_numbers], 'alogrithm-pvalue': sorted(SSB_p_values)[:feature_numbers]},index =selected_SSB_data.columns.tolist())
        RNA_feature_importance_df = pd.DataFrame({'scores': sorted(RNA_scores,reverse=1)[:feature_numbers], 'alogrithm-pvalue': sorted(RNA_p_values)[:feature_numbers]},index =selected_RNA_seq_data.columns.tolist() )

    topN_save_path = './data_topN_features/genes_metrics'
    # print(AP_feature_importance_df)
    # if not os.path.exists(topN_save_path):
    #     os.makedirs(topN_save_path)
    # if np.max(y_label) >4:
    #     print(111)
    #     AP_feature_importance_df.to_csv(topN_save_path+'/selected_AP_data_tissue_metrics'+str(feature_numbers)+'.csv')
    #     SSB_feature_importance_df.to_csv(topN_save_path+'/selected_SSB_data_tissue_metrics'+str(feature_numbers)+'.csv')
    #     RNA_feature_importance_df.to_csv(topN_save_path+'/selected_RNA_seq_data_tissue_metrics'+str(feature_numbers)+'.csv')
    # else :
    #     AP_feature_importance_df.to_csv(topN_save_path+'/selected_AP_data_metrics'+str(feature_numbers)+'.csv')
    #     SSB_feature_importance_df.to_csv(topN_save_path+'/selected_SSB_data_metrics'+str(feature_numbers)+'.csv')
    #     RNA_feature_importance_df.to_csv(topN_save_path+'/selected_RNA_seq_data_metrics'+str(feature_numbers)+'.csv')

    if saveFlag:
        # topN_save_path = './data_topN_features/'
        # if not os.path.exists(topN_save_path):
        #     os.makedirs(topN_save_path)
        # if np.max(y_label) >4:
        #     if selected_AP_data.shape[0]==83:
        #         selected_AP_data.to_csv(topN_save_path+'selected_AP_data_tissue_filter_'+str(feature_numbers)+'.csv')
        #         selected_SSB_data.to_csv(topN_save_path+'selected_SSB_data_tissue_filter_'+str(feature_numbers)+'.csv')
        #         selected_RNA_seq_data.to_csv(topN_save_path+'selected_RNA_seq_data_tissue_filter_'+str(feature_numbers)+'.csv')
        #     else:
        #         selected_AP_data.to_csv(topN_save_path+'selected_AP_data_tissue_filter_71_'+str(feature_numbers)+'.csv')
        #         selected_SSB_data.to_csv(topN_save_path+'selected_SSB_data_tissue_filter_71_'+str(feature_numbers)+'.csv')
        #         selected_RNA_seq_data.to_csv(topN_save_path+'selected_RNA_seq_data_tissue_filter_71_'+str(feature_numbers)+'.csv')
        # else :
        #     if selected_AP_data.shape[0]==83:
        #         selected_AP_data.to_csv(topN_save_path+'selected_AP_data_filter_'+str(feature_numbers)+'.csv')
        #         selected_SSB_data.to_csv(topN_save_path+'selected_SSB_data_filter_'+str(feature_numbers)+'.csv')
        #         selected_RNA_seq_data.to_csv(topN_save_path+'selected_RNA_seq_data_filter_'+str(feature_numbers)+'.csv')
        #     else:
        #         selected_AP_data.to_csv(topN_save_path+'selected_AP_data_filter_71_'+str(feature_numbers)+'.csv')
        #         selected_SSB_data.to_csv(topN_save_path+'selected_SSB_data_filter_71_'+str(feature_numbers)+'.csv')
        #         selected_RNA_seq_data.to_csv(topN_save_path+'selected_RNA_seq_data_filter_71_'+str(feature_numbers)+'.csv')

        # selected_AP_data_pwithlabel = []
        # selected_AP_data_corrwithlabel = []
        # selected_SSB_data_pwithlabel = []
        # selected_SSB_data_corrwithlabel = []
        # selected_RNA_seq_data_pwithlabel = []
        # selected_RNA_seq_data_corrwithlabel = []   

        if np.max(y_label) <=4:
            y_labelto_age = np.array(y_label)
            for i in range(y_label.shape[0]):
                if y_label[i] == 0:
                    y_labelto_age[i] =3
                elif y_label[i] == 1:
                    y_labelto_age[i] =12
                elif y_label[i] == 2:
                    y_labelto_age[i] =19
                elif y_label[i] == 3:
                    y_labelto_age[i] =22
                elif y_label[i] == 4:
                    y_labelto_age[i] =24
        #     num_sig_ap = 0
        #     num_sig_ssb = 0
        #     num_sig_rna = 0
        #     for idx in range(selected_AP_data.shape[1]):
        #         data_ = selected_AP_data.iloc[:,idx]
        #         sta = stats.spearmanr(np.array(data_),y_labelto_age)
        #         if sta[1] < 0.05:
        #             num_sig_ap +=1
        #         selected_AP_data_pwithlabel.append(sta[1])
        #         selected_AP_data_corrwithlabel.append(sta[0])  
        #     for idx in range(selected_SSB_data.shape[1]):
        #         data_ = selected_SSB_data.iloc[:,idx]
        #         sta = stats.spearmanr(np.array(data_),y_labelto_age)
        #         if sta[1] < 0.05:
        #             num_sig_ssb +=1
        #         selected_SSB_data_pwithlabel.append(sta[1])
        #         selected_SSB_data_corrwithlabel.append(sta[0])
        #     for idx in range(selected_RNA_seq_data.shape[1]):
        #         data_ = selected_RNA_seq_data.iloc[:,idx]
        #         sta = stats.spearmanr(np.array(data_),y_labelto_age)
        #         if sta[1] < 0.05:
        #             num_sig_rna +=1
        #         selected_RNA_seq_data_pwithlabel.append(sta[1])
        #         selected_RNA_seq_data_corrwithlabel.append(sta[0])
        #     print(selected_AP_data.shape[1],num_sig_ap)
        #     print(selected_SSB_data.shape[1],num_sig_ssb)
        #     print(selected_RNA_seq_data.shape[1],num_sig_rna)

            # RNA_pvalue_age_df = pd.DataFrame({'P-VALUEwithAge': selected_RNA_seq_data_pwithlabel,'correlationValue': selected_RNA_seq_data_corrwithlabel},index =selected_RNA_seq_data.columns.tolist() )
            # SSB_pvalue_age_df = pd.DataFrame({'P-VALUEwithAge': selected_SSB_data_pwithlabel, 'correlationValue': selected_SSB_data_corrwithlabel},index =selected_SSB_data.columns.tolist() )
            # AP_pvalue_age_df = pd.DataFrame({'P-VALUEwithAge':selected_AP_data_pwithlabel, 'correlationValue': selected_AP_data_corrwithlabel},index =selected_AP_data.columns.tolist() )
            # RNA_pvalue_age_df.to_csv(topN_save_path+'selected_RNA_data_spearman_metric'+str(feature_numbers)+'.csv')
            # SSB_pvalue_age_df.to_csv(topN_save_path+'selected_SSB_data_spearman_metric'+str(feature_numbers)+'.csv')
            # AP_pvalue_age_df.to_csv(topN_save_path+'selected_AP_seq_data_spearman_metric'+str(feature_numbers)+'.csv')

#ap-tissue-calculate
            ap_heart_corr =[]
            ap_brain_corr =[]
            ap_bone_corr =[]
            ap_prem_corr =[]
            ap_s_corr =[]
            ap_liver_corr =[]
            ap_heart_pvalue =[]
            ap_brain_pvalue =[]
            ap_bone_pvalue =[]
            ap_prem_pvalue =[]
            ap_s_pvalue =[]
            ap_liver_pvalue =[]
            for idx in range(selected_AP_data.shape[1]):
                data_ = selected_AP_data.iloc[:,idx]
                heat_list_c = []
                brain_list_c = []
                bone_list_c = []
                prem_list_c = []
                s_list_c = []
                liver_list_c = []
                indexs = data_.index.tolist()
                for i,index_ in enumerate(indexs):
                    if 'B-' in index_:
                        brain_list_c.append(i)
                    elif 'H-' in index_:
                        heat_list_c.append(i)
                    elif 'M-' in index_:
                        bone_list_c.append(i)
                    elif 'P-' in index_:
                        prem_list_c.append(i)
                    elif 'S-' in index_:
                        s_list_c.append(i)
                    elif 'L-' in index_:
                        liver_list_c.append(i)
                heat_list_index = [indexs[i] for i in heat_list_c]
                brain_list_index =[indexs[i] for i in brain_list_c]
                bone_list_index = [indexs[i] for i in bone_list_c]
                prem_list_index =[indexs[i] for i in prem_list_c] 
                s_list_index = [indexs[i] for i in s_list_c]
                liver_list_index = [indexs[i] for i in liver_list_c]

                heat_data_ = data_.loc[heat_list_index]
                brain_data_ = data_.loc[brain_list_index]
                bone_data_ = data_.loc[bone_list_index]
                prem_data_ = data_.loc[prem_list_index]
                s_data_ = data_.loc[s_list_index]
                liver_data_ = data_.loc[liver_list_index]

                heat_data_ylabel =y_labelto_age[heat_list_c]
                brain_data_ylabel =y_labelto_age[brain_list_c]
                bone_data_ylabel =y_labelto_age[bone_list_c]
                prem_data_ylabel =y_labelto_age[prem_list_c]
                s_data_ylabel =y_labelto_age[s_list_c]
                liver_data_ylabel =y_labelto_age[liver_list_c]
                ap_heart_corr.append(stats.spearmanr(np.array(heat_data_),heat_data_ylabel)[0])
                ap_brain_corr.append(stats.spearmanr(np.array(brain_data_),brain_data_ylabel)[0])
                ap_bone_corr.append(stats.spearmanr(np.array(bone_data_),bone_data_ylabel)[0])
                ap_prem_corr.append(stats.spearmanr(np.array(prem_data_),prem_data_ylabel)[0])
                ap_s_corr.append(stats.spearmanr(np.array(s_data_),s_data_ylabel)[0])
                ap_liver_corr.append(stats.spearmanr(np.array(liver_data_),liver_data_ylabel)[0])
                ap_heart_pvalue.append(stats.spearmanr(np.array(heat_data_),heat_data_ylabel)[1])
                ap_brain_pvalue.append(stats.spearmanr(np.array(brain_data_),brain_data_ylabel)[1])
                ap_bone_pvalue.append(stats.spearmanr(np.array(bone_data_),bone_data_ylabel)[1])
                ap_prem_pvalue.append(stats.spearmanr(np.array(prem_data_),prem_data_ylabel)[1])
                ap_s_pvalue.append(stats.spearmanr(np.array(s_data_),s_data_ylabel)[1])
                ap_liver_pvalue.append(stats.spearmanr(np.array(liver_data_),liver_data_ylabel)[1])
                sta = stats.spearmanr(np.array(data_),y_labelto_age)

# #ssb-tissue-calculate
            ssb_heart_corr =[]
            ssb_brain_corr =[]
            ssb_bone_corr =[]
            ssb_prem_corr =[]
            ssb_s_corr =[]
            ssb_liver_corr =[]
            ssb_heart_pvalue =[]
            ssb_brain_pvalue =[]
            ssb_bone_pvalue =[]
            ssb_prem_pvalue =[]
            ssb_s_pvalue =[]
            ssb_liver_pvalue =[]
            for idx in range(selected_SSB_data.shape[1]):
                heat_list_c = []
                brain_list_c = []
                bone_list_c = []
                prem_list_c = []
                s_list_c = []
                liver_list_c = []
                data_ = selected_SSB_data.iloc[:,idx]
                for i,index_ in enumerate(selected_SSB_data.index.tolist()):
                    if 'B-' in index_:
                        brain_list_c.append(i)
                    elif 'H-' in index_:
                        heat_list_c.append(i)
                    elif 'M-' in index_:
                        bone_list_c.append(i)
                    elif 'P-' in index_:
                        prem_list_c.append(i)
                    elif 'S-' in index_:
                        s_list_c.append(i)
                    elif 'L-' in index_:
                        liver_list_c.append(i)
                heat_list_index = [indexs[i] for i in heat_list_c]
                brain_list_index =[indexs[i] for i in brain_list_c]
                bone_list_index = [indexs[i] for i in bone_list_c]
                prem_list_index =[indexs[i] for i in prem_list_c] 
                s_list_index = [indexs[i] for i in s_list_c]
                liver_list_index = [indexs[i] for i in liver_list_c]

                heat_data_ = data_.loc[heat_list_index]
                brain_data_ = data_.loc[brain_list_index]
                bone_data_ = data_.loc[bone_list_index]
                prem_data_ = data_.loc[prem_list_index]
                s_data_ = data_.loc[s_list_index]
                liver_data_ = data_.loc[liver_list_index]

                heat_data_ylabel =y_labelto_age[heat_list_c]
                brain_data_ylabel =y_labelto_age[brain_list_c]
                bone_data_ylabel =y_labelto_age[bone_list_c]
                prem_data_ylabel =y_labelto_age[prem_list_c]
                s_data_ylabel =y_labelto_age[s_list_c]
                liver_data_ylabel =y_labelto_age[liver_list_c]
                ssb_heart_corr.append(stats.spearmanr(np.array(heat_data_),heat_data_ylabel)[0])
                ssb_brain_corr.append(stats.spearmanr(np.array(brain_data_),brain_data_ylabel)[0])
                ssb_bone_corr.append(stats.spearmanr(np.array(bone_data_),bone_data_ylabel)[0])
                ssb_prem_corr.append(stats.spearmanr(np.array(prem_data_),prem_data_ylabel)[0])
                ssb_s_corr.append(stats.spearmanr(np.array(s_data_),s_data_ylabel)[0])
                ssb_liver_corr.append(stats.spearmanr(np.array(liver_data_),liver_data_ylabel)[0])

                ssb_heart_pvalue.append(stats.spearmanr(np.array(heat_data_),heat_data_ylabel)[1])
                ssb_brain_pvalue.append(stats.spearmanr(np.array(brain_data_),brain_data_ylabel)[1])
                ssb_bone_pvalue.append(stats.spearmanr(np.array(bone_data_),bone_data_ylabel)[1])
                ssb_prem_pvalue.append(stats.spearmanr(np.array(prem_data_),prem_data_ylabel)[1])
                ssb_s_pvalue.append(stats.spearmanr(np.array(s_data_),s_data_ylabel)[1])
                ssb_liver_pvalue.append(stats.spearmanr(np.array(liver_data_),liver_data_ylabel)[1])

                sta = stats.spearmanr(np.array(data_),y_labelto_age)

            rna_heart_corr =[]
            rna_brain_corr =[]
            rna_bone_corr =[]
            rna_prem_corr =[]
            rna_s_corr =[]
            rna_liver_corr =[] 
            rna_heart_pvalue =[]
            rna_brain_pvalue =[]
            rna_bone_pvalue =[]
            rna_prem_pvalue =[]
            rna_s_pvalue =[]
            rna_liver_pvalue =[] 
            for idx in range(selected_RNA_seq_data.shape[1]):
                heat_list_c = []
                brain_list_c = []
                bone_list_c = []
                prem_list_c = []
                s_list_c = []
                liver_list_c = []
                data_ = selected_RNA_seq_data.iloc[:,idx]   
                for i,index_ in enumerate(selected_RNA_seq_data.index.tolist()):
                    if 'B-' in index_:
                        brain_list_c.append(i)
                    elif 'H-' in index_:
                        heat_list_c.append(i)
                    elif 'M-' in index_:
                        bone_list_c.append(i)
                    elif 'P-' in index_:
                        prem_list_c.append(i)
                    elif 'S-' in index_:
                        s_list_c.append(i)
                    elif 'L-' in index_:
                        liver_list_c.append(i)
                heat_list_index = [indexs[i] for i in heat_list_c]
                brain_list_index =[indexs[i] for i in brain_list_c]
                bone_list_index = [indexs[i] for i in bone_list_c]
                prem_list_index =[indexs[i] for i in prem_list_c] 
                s_list_index = [indexs[i] for i in s_list_c]
                liver_list_index = [indexs[i] for i in liver_list_c]

                heat_data_ = data_.loc[heat_list_index]
                brain_data_ = data_.loc[brain_list_index]
                bone_data_ = data_.loc[bone_list_index]
                prem_data_ = data_.loc[prem_list_index]
                s_data_ = data_.loc[s_list_index]
                liver_data_ = data_.loc[liver_list_index]

                heat_data_ylabel =y_labelto_age[heat_list_c]
                brain_data_ylabel =y_labelto_age[brain_list_c]
                bone_data_ylabel =y_labelto_age[bone_list_c]
                prem_data_ylabel =y_labelto_age[prem_list_c]
                s_data_ylabel =y_labelto_age[s_list_c]
                liver_data_ylabel =y_labelto_age[liver_list_c]
                rna_heart_corr.append(stats.spearmanr(np.array(heat_data_),heat_data_ylabel)[0])
                rna_brain_corr.append(stats.spearmanr(np.array(brain_data_),brain_data_ylabel)[0])
                rna_bone_corr.append(stats.spearmanr(np.array(bone_data_),bone_data_ylabel)[0])
                rna_prem_corr.append(stats.spearmanr(np.array(prem_data_),prem_data_ylabel)[0])
                rna_s_corr.append(stats.spearmanr(np.array(s_data_),s_data_ylabel)[0])
                rna_liver_corr.append(stats.spearmanr(np.array(liver_data_),liver_data_ylabel)[0])   

                rna_heart_pvalue.append(stats.spearmanr(np.array(heat_data_),heat_data_ylabel)[1])
                rna_brain_pvalue.append(stats.spearmanr(np.array(brain_data_),brain_data_ylabel)[1])
                rna_bone_pvalue.append(stats.spearmanr(np.array(bone_data_),bone_data_ylabel)[1])
                rna_prem_pvalue.append(stats.spearmanr(np.array(prem_data_),prem_data_ylabel)[1])
                rna_s_pvalue.append(stats.spearmanr(np.array(s_data_),s_data_ylabel)[1])
                rna_liver_pvalue.append(stats.spearmanr(np.array(liver_data_),liver_data_ylabel)[1])  

                sta = stats.spearmanr(np.array(data_),y_labelto_age)

            font_dict=dict(size=12,
              family='Arial',
              weight='light',
            #   style='italic',
              )
            plt.rc('font',**font_dict)
# #plot_ap_tissue
            plt_labels = ['B','H','P','M','L','S']
            width = 0.5
            bins = np.linspace(-1, 1, 500)
            plt.figure(figsize=(40, 20))
            plt.subplot(3,6,1)
            ap_heart_hist, bin_edges = np.histogram(ap_heart_corr, bins=bins, range=(-1,1))
            # plt.hist(ap_heart_corr,bins = bins,color="#8ECFC9",alpha=1.0)
#             plt.title("AP heart")
#             plt.xlim(-1,1)
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")
#             plt.subplot(3,6,2)
            ap_brain_hist, bin_edges = np.histogram(ap_brain_corr, bins=bins, range=(-1,1))
#             plt.hist(ap_brain_corr,bins = bins,color="#F27970",alpha=1.0)
#             plt.title("AP brain")
#             plt.xlim(-1,1)
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")
#             plt.subplot(3,6,3)
            ap_liver_hist, bin_edges = np.histogram(ap_liver_corr, bins=bins, range=(-1,1))
#             plt.hist(ap_liver_corr,bins = bins,color="#BB9727",alpha=1.0)
#             plt.title("AP liver")
#             plt.xlim(-1,1)    
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")        
#             plt.subplot(3,6,4)
            ap_bone_hist, bin_edges = np.histogram(ap_bone_corr, bins=bins, range=(-1,1))
#             plt.hist(ap_bone_corr,bins = bins,color="#54B345",alpha=1.0)
#             plt.title("AP bone marrow")
#             plt.xlim(-1,1)     
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")       
#             plt.subplot(3,6,5)
            ap_s_hist, bin_edges = np.histogram(ap_s_corr, bins=bins, range=(-1,1))
#             plt.hist(ap_s_corr,bins = bins,color="#C76DA2",alpha=1.0)
#             plt.title("AP sperm")
#             plt.xlim(-1,1)    
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")        
#             plt.subplot(3,6,6)
            ap_prem_hist, bin_edges = np.histogram(ap_prem_corr, bins=bins, range=(-1,1))
#             plt.hist(ap_s_corr,bins = bins,color="#A1A9D0",alpha=1.0)
#             plt.title("AP PBMC")            
#             plt.xlim(-1,1)    
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")        
# # plot_ssb_tissue
#             plt.subplot(3,6,7)
            ssb_heart_hist, bin_edges = np.histogram(ssb_heart_corr, bins=bins, range=(-1,1))
#             plt.hist(ssb_heart_corr,bins = bins,color="#8ECFC9",alpha=1.0)
#             plt.title("SSB heart")
#             plt.xlim(-1,1)
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")
#             plt.subplot(3,6,8)
            ssb_brain_hist, bin_edges = np.histogram(ssb_brain_corr, bins=bins, range=(-1,1))
#             plt.hist(ssb_brain_corr,bins = bins,color="#F27970",alpha=1.0)
#             plt.title("SSB brain")
#             plt.xlim(-1,1)
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")
#             plt.subplot(3,6,9)
            ssb_liver_hist, bin_edges = np.histogram(ssb_liver_corr, bins=bins, range=(-1,1))
#             plt.hist(ssb_liver_corr,bins = bins,color="#BB9727",alpha=1.0)
#             plt.title("SSB liver")
#             plt.xlim(-1,1)
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")            
#             plt.subplot(3,6,10)
            ssb_bone_hist, bin_edges = np.histogram(ssb_bone_corr, bins=bins, range=(-1,1))
#             plt.hist(ssb_bone_corr,bins = bins,color="#54B345",alpha=1.0)
#             plt.title("SSB bone marrow")
#             plt.xlim(-1,1)   
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")         
#             plt.subplot(3,6,11)
            ssb_s_hist, bin_edges = np.histogram(ssb_s_corr, bins=bins, range=(-1,1))
#             plt.hist(ssb_s_corr,bins = bins,color="#C76DA2",alpha=1.0)
#             plt.title("SSB sperm")
#             plt.xlim(-1,1) 
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")           
#             plt.subplot(3,6,12)
            ssb_prem_hist, bin_edges = np.histogram(ssb_prem_corr, bins=bins, range=(-1,1))
#             plt.hist(ssb_s_corr,bins = bins,color="#A1A9D0",alpha=1.0)
#             plt.title("SSB PBMC")            
#             plt.xlim(-1,1) 
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")
# #plot_rna_tissue
#             print(np.max(rna_heart_corr),np.min(rna_heart_corr))
#             plt.subplot(3,6,13)
            rna_heart_hist, bin_edges = np.histogram(rna_heart_corr, bins=bins, range=(-1,1))
#             plt.hist(rna_heart_corr,bins = bins,color="#8ECFC9",alpha=1.0)
#             plt.title("RNA heart")
#             plt.xlim(-1,1)
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")
#             plt.subplot(3,6,14)
            rna_brain_hist, bin_edges = np.histogram(rna_brain_corr, bins=bins, range=(-1,1))
#             plt.hist(rna_brain_corr,bins = bins,color="#F27970",alpha=1.0)
#             plt.title("RNA brain")
#             plt.xlim(-1,1)
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")
#             plt.subplot(3,6,15)
            rna_liver_hist, bin_edges = np.histogram(rna_liver_corr, bins=bins, range=(-1,1))
#             plt.hist(rna_liver_corr,bins = bins,color="#BB9727",alpha=1.0)
#             plt.title("RNA liver")
#             plt.xlim(-1,1)  
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")          
#             plt.subplot(3,6,16)
            rna_bone_hist, bin_edges = np.histogram(rna_bone_corr, bins=bins, range=(-1,1))
#             plt.hist(rna_bone_corr,bins = bins,color="#54B345",alpha=1.0)
#             plt.title("RNA bone marrow")
#             plt.xlim(-1,1)  
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")          
#             plt.subplot(3,6,17)
            rna_s_hist, bin_edges = np.histogram(rna_s_corr, bins=bins, range=(-1,1))
#             plt.hist(rna_s_corr,bins = bins,color="#C76DA2",alpha=1.0)
#             plt.title("RNA sperm")
#             plt.xlim(-1,1)   
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")         
#             plt.subplot(3,6,18)
            rna_prem_hist, bin_edges = np.histogram(rna_prem_corr, bins=bins, range=(-1,1))
#             plt.hist(rna_s_corr,bins = bins,color="#A1A9D0",alpha=1.0)
#             plt.title("RNA PBMC")            
#             plt.xlim(-1,1) 
#             plt.xlabel("Spearman correlation value")
#             plt.ylabel("Frequency")
#             plt.savefig("tissue_83_ALL_samples_corr_hist.pdf",dpi=1080,format='pdf')
#         #     colors = ['#71ae46']
#         #             # - ce2626 - ac2026 - 71ae46 - 96b744 - c4cc38]
            ap_CORR = ap_heart_corr+ap_brain_corr+ap_liver_corr+ap_bone_corr+ap_s_corr+ap_prem_corr
            ssb_CORR = ssb_heart_corr+ssb_brain_corr+ssb_liver_corr+ssb_bone_corr+ssb_s_corr+ssb_prem_corr
            rna_CORR = rna_heart_corr+rna_brain_corr+rna_liver_corr+rna_bone_corr+rna_s_corr+rna_prem_corr

            ap_pvalues = ap_heart_pvalue+ap_brain_pvalue+ap_liver_pvalue+ap_bone_pvalue+ap_s_pvalue+ap_prem_pvalue
            ssb_pvalues = ssb_heart_pvalue+ssb_brain_pvalue+ssb_liver_pvalue+ssb_bone_pvalue+ssb_s_pvalue+ssb_prem_pvalue
            rna_pvalues = rna_heart_pvalue+rna_brain_pvalue+rna_liver_pvalue+rna_bone_pvalue+rna_s_pvalue+rna_prem_pvalue
            label_ap_l = selected_AP_data.shape[1]
            AP_labels = ["heart"]*label_ap_l+["brain"]*label_ap_l+["liver"]*label_ap_l+["bone"]*label_ap_l+["sperm"]*label_ap_l+["pbmc"]*label_ap_l
            label_ssb_l = selected_SSB_data.shape[1]
            SSB_labels = ["heart"]*label_ssb_l+["brain"]*label_ssb_l+["liver"]*label_ssb_l+["bone"]*label_ssb_l+["sperm"]*label_ssb_l+["pbmc"]*label_ssb_l
            label_rna_l = selected_RNA_seq_data.shape[1]
            RNA_labels = ["heart"]*label_rna_l+["brain"]*label_rna_l+["liver"]*label_rna_l+["bone"]*label_rna_l+["sperm"]*label_rna_l+["pbmc"]*label_rna_l
            # print(len(rna_CORR),len([str(label_ap_l)*label_ap_l*6]),len([str(label_ap_l)]*(label_ap_l*6)),[str(label_ap_l)]*label_ap_l*6)
            ap_tissue_correlation_df = pd.DataFrame({'rho': ap_CORR, 'pvalue': ap_pvalues,"tissue":AP_labels,"data_type":['LPKM-AP']*(label_ap_l*6),"topN":[str(label_ap_l)]*(label_ap_l*6)},index = selected_AP_data.columns.tolist()*6 )
            ssb_tissue_correlation_df = pd.DataFrame({'rho': ssb_CORR, 'pvalue': ssb_pvalues,"tissue":SSB_labels,"data_type":['LPKM-SSB']*(label_ssb_l*6),"topN":[str(label_ssb_l)]*(label_ssb_l*6)},index = selected_SSB_data.columns.tolist()*6 )
            rna_tissue_correlation_df = pd.DataFrame({'rho': rna_CORR, 'pvalue': rna_pvalues,"tissue":RNA_labels,"data_type":['TPM']*(label_rna_l*6),"topN":[str(label_rna_l)]*(label_rna_l*6)},index = selected_RNA_seq_data.columns.tolist()*6 )

            bin_edges = range(len(ap_bone_corr))
            # ap_tissue_correlation_df = pd.DataFrame({'ap_heart_rho': ap_heart_corr, 'ap_heart_pvalue': ap_heart_pvalue,'ap_brain_rho':ap_brain_corr,'ap_brain_pvalue': ap_brain_pvalue,'ap_liver_rho':ap_liver_corr,'ap_liver_pvalue':ap_liver_pvalue,'ap_bone_rho':ap_bone_corr,'ap_bone_pvalue':ap_bone_pvalue,'ap_sperm_rho':ap_s_corr,'ap_sperm_pvalue':ap_s_pvalue,'ap_pbmc_rho':ap_prem_corr,'ap_pbmc_pvalue':ap_prem_pvalue},index = selected_AP_data.columns.tolist() )
            # ssb_tissue_correlation_df = pd.DataFrame({'ssb_heart_rho': ssb_heart_corr, 'ssb_heart_pvalue': ssb_heart_pvalue,'ssb_brain_rho':ssb_brain_corr,'ssb_brain_pvalue': ssb_brain_pvalue,'ssb_liver_rho':ssb_liver_corr,'ssb_liver_pvalue':ssb_liver_pvalue,'ssb_bone_rho':ssb_bone_corr,'ssb_bone_pvalue':ssb_bone_pvalue,'ssb_sperm_rho':ssb_s_corr,'ssb_sperm_pvalue':ssb_s_pvalue,'ssb_pbmc_rho':ssb_prem_corr,'ssb_pbmc_pvalue':ssb_prem_pvalue},index =selected_SSB_data.columns.tolist() )
            # rna_tissue_correlation_df = pd.DataFrame({'rna_heart_rho': rna_heart_corr, 'rna_heart_pvalue': rna_heart_pvalue,'rna_brain_rho':rna_brain_corr,'rna_brain_pvalue': rna_brain_pvalue,'rna_liver_rho':rna_liver_corr,'rna_liver_pvalue':rna_liver_pvalue,'rna_bone_rho':rna_bone_corr,'rna_bone_pvalue':rna_bone_pvalue,'rna_sperm_rho':rna_s_corr,'rna_sperm_pvalue':rna_s_pvalue,'rna_pbmc_rho':rna_prem_corr,'rna_pbmc_pvalue':rna_prem_pvalue},index =selected_RNA_seq_data.columns.tolist() )
            ap_tissue_correlation_df.to_csv(topN_save_path+'/selected_AP_tissue_correlation_metrics'+str(feature_numbers)+'.csv')
            ssb_tissue_correlation_df.to_csv(topN_save_path+'/selected_SSB_tissue_correlation_metrics'+str(feature_numbers)+'.csv')
            rna_tissue_correlation_df.to_csv(topN_save_path+'/selected_RNA_tissue_correlation_metrics'+str(feature_numbers)+'.csv')
#             y = selected_SSB_data_pwithlabel
#             #y = -np.log10(np.array(selected_SSB_data_pwithlabel))
#             num_bins = 500
#             cmap = plt.cm.get_cmap('YlOrRd')
#             hist, bin_edges = np.histogram(selected_SSB_data_pwithlabel, bins=num_bins, range=(0,1))
#             bin_colors = (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
#             right_pos = 0
#             idx_=0
#             for i in bin_edges:
#                 idx_ +=1
#                 if i >0.05:
#                     right_pos = idx_
#                     break
#             print(hist,bin_edges,right_pos)
#             fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
#             axs[0].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.4, color=cmap(bin_colors))
#             axs[0].bar(bin_edges[:right_pos], hist[:right_pos], width=np.diff(bin_edges[:right_pos+1]),alpha=0.8, color="purple")
#             axs[0].axvline(0.05)
#             axs[0].set_ylabel('Count')
#             axs[0].set_xlabel('p-value')
#             if feature_numbers=='all':
#                 axs[0].set_title('ALL SSB genes')
#             else:
#                 axs[0].set_title("Top 500 SSB genes")
#             axs[0].set_xlim(-0.05,1)
#             sig_counts ,_= np.histogram(y, bins=[0,0.05])
#             axs[0].text(-0.05,np.max(hist[:right_pos]),s="%.2f"%(sig_counts*100.0/len(y))+"%")
#             # x = selected_AP_data.columns.tolist()
#             y = selected_AP_data_pwithlabel
#             hist, bin_edges = np.histogram(selected_AP_data_pwithlabel, bins=num_bins, range=(0,1))
#             axs[1].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.4, color=cmap(bin_colors))
#             axs[1].bar(bin_edges[:right_pos], hist[:right_pos], width=np.diff(bin_edges[:right_pos+1]),alpha=0.8, color="purple")

#             axs[1].set_ylabel('Count')
#             axs[1].set_xlabel('p-value')
#             if feature_numbers=='all':
#                 axs[1].set_title('ALL AP genes')
#             else:
#                 axs[1].set_title("Top 500 AP genes")
#             axs[1].axvline(0.05)

#             sig_counts ,_= np.histogram(y, bins=[0,0.05])
#             axs[1].text(-0.05,np.max(hist[:right_pos]),s="%.2f"%(sig_counts*100.0/len(y))+"%")
#             y = selected_RNA_seq_data_pwithlabel
#             print(y)
#             hist, bin_edges = np.histogram(selected_RNA_seq_data_pwithlabel, bins=num_bins, range=(0,1))
#             axs[2].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.4, color=cmap(bin_colors))
#             axs[2].bar(bin_edges[:right_pos], hist[:right_pos], width=np.diff(bin_edges[:right_pos+1]),alpha=0.8, color="purple")
#             axs[2].set_ylabel('Count')
#             axs[2].set_xlabel('p-value')
#             if feature_numbers=='all':
#                 axs[2].set_title('ALL RNA seq genes')
#             else:
#                 axs[2].set_title("Top 500 RNA seq genes")
#             sig_counts ,_= np.histogram(y, bins=[0,0.05])
#             axs[2].axvline(0.05)
#             axs[2].text(-0.05,np.max(hist[:right_pos]),s="%.2f"%(sig_counts*100.0/len(y))+"%")
#             # from mpl_toolkits.axes_grid1 import make_axes_locatable
#             # divider = make_axes_locatable(axs[-1])
#             # cax = divider.append_axes("right", size="5%", pad=0.1)
#             cax = fig.add_axes([0.93, 0.3, 0.01, 0.4])
#             sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(bin_colors), vmax=max(bin_colors)))
#             sm.set_array([])
#             cbar = plt.colorbar(sm, cax=cax)
#             cbar.set_label('Normalized frequency')
#             plt.subplots_adjust(wspace=0.3)
#             plt.savefig(topN_save_path+'hist_threeData_of_PvaluewithAgeLabel_500.png',dpi = 1080)
#             plt.clf()
            
#             bins = np.linspace(-1, 1, 500)
#             colors = ['gray' if bins[i] >= -0.2 and bins[i+1] <= 0.2 else "red" for i in range(len(bins)-1)]
#             gray_l = 0
#             gray_r = 0
#             for idx,i in enumerate(colors):
#                 if i == 'gray' and gray_l ==0:
#                     gray_l = idx
#                 if i == 'gray' and colors[idx+1] == 'red':
#                     gray_r = idx+2
#                     break  
#             print(gray_l,gray_r)  
#             y = selected_SSB_data_corrwithlabel

#             range_bin = [-0.2,0.2]
#             hist, _ = np.histogram(y, bins=500,range=(-1,1))
#             valid_length = np.sum(hist)
#             hist =  np.sum(hist[gray_l:gray_r])
#             print(hist,len(y))
#             plt.title("ALL 71")
#             plt.subplot(3,1,1)
#             plt.hist(y,bins = bins,color="red",alpha=1.0) 
#             plt.hist(y,bins = bins[gray_l:gray_r],color="gray",alpha=1.0) 
#             # for c, p in zip(n[], patches):
#             #     plt.setp(p, 'facecolor', "gray")
#             plt.ylabel('Count')
#             plt.xlabel('Rho- values')
#             plt.axvline(-0.2)
#             plt.axvline(0.2)
#             plt.xlim(-1,1)
#             # plt.ylim(0,8)
#             if feature_numbers=='all':
#                 plt.title('ALL SSB genes')
#             else :
#                 plt.title('Top 500 SSB genes')

#             plt.text(-0.03,6,s="%.2f"%(hist*100.0/valid_length)+"%")
#             # x = selected_AP_data.columns.tolist()
#             y = selected_AP_data_corrwithlabel
#             hist, _ = np.histogram(y, bins=500,range=(-1,1))
#             valid_length = np.sum(hist)
#             hist =  np.sum(hist[gray_l:gray_r])
#             print(hist,len(y))
#             plt.subplot(3,1,2)
#             plt.hist(y,bins = bins,color="red",alpha=1.0) 
#             plt.hist(y,bins = bins[gray_l:gray_r],color="gray",alpha=1.0) 
#             plt.ylabel('Count')
#             plt.xlabel('Rho- values')
#             plt.xlim(-1,1)
#             # plt.ylim(0,8)
#             plt.axvline(-0.2)
#             plt.axvline(0.2)
#             if feature_numbers=='all':
#                 plt.title('ALL AP genes')
#             else :
#                 plt.title('Top 500 AP genes')
#             plt.text(-0.03,6,s="%.2f"%(hist*100.0/valid_length)+"%")
#             y = selected_RNA_seq_data_corrwithlabel
#             print(y,np.min(y),np.max(y))
#             hist, _ = np.histogram(y, bins=500,range=(-1,1))
#             valid_length = np.sum(hist)
#             hist =  np.sum(hist[gray_l:gray_r])
#             print(hist,len(y))
#             plt.subplot(3,1,3)
#             plt.hist(y,bins = bins,color="red",alpha=1.0) 
#             plt.hist(y,bins = bins[gray_l:gray_r],color="gray",alpha=1.0) 
#             plt.ylabel('Count')
#             plt.xlabel('Rho- values')
#             plt.xlim(-1,1)
#             # plt.ylim(0,8)
#             plt.axvline(-0.2)
#             plt.axvline(0.2)
#             if feature_numbers=='all':
#                 plt.title('ALL RNA_seq genes')
#             else :
#                 plt.title('Top 500 RNA_seq genes')
#             plt.text(-0.03,6,s="%.2f"%(hist*100.0/valid_length)+"%")

#             plt.savefig(topN_save_path+'hist_threeData_otestwithlabelWithAgeLabel_500.png',dpi = 1080)




#71 sample
        # selected_AP_data_pwithlabel = []
        # selected_AP_data_corrwithlabel = []
        # selected_SSB_data_pwithlabel = []
        # selected_SSB_data_corrwithlabel = []
        # selected_RNA_seq_data_pwithlabel = []
        # selected_RNA_seq_data_corrwithlabel = []   
        # if np.max(y_label) <=4:
        #     y_labelto_age = np.array(y_label)[:-12]
        #     print(y_labelto_age.shape)
        #     for i in range(y_label.shape[0]):
        #         if y_label[i] == 0:
        #             y_labelto_age[i] =3
        #         elif y_label[i] == 1:
        #             y_labelto_age[i] =12
        #         elif y_label[i] == 2:
        #             y_labelto_age[i] =19
        #         elif y_label[i] == 3:
        #             y_labelto_age[i] =22
        #         # elif y_label[i] == 4:
        #         #     y_labelto_age[i] =24
        #     for idx in range(selected_AP_data.shape[1]):
        #         data_ = selected_AP_data.iloc[:,idx]
        #         sta = stats.spearmanr(np.array(data_)[:-12],y_labelto_age)
        #         selected_AP_data_pwithlabel.append(sta[1])
        #         selected_AP_data_corrwithlabel.append(sta[0])
        #     for idx in range(selected_SSB_data.shape[1]):
        #         data_ = selected_SSB_data.iloc[:,idx]
        #         sta = stats.spearmanr(np.array(data_)[:-12],y_labelto_age)
        #         selected_SSB_data_pwithlabel.append(sta[1])
        #         selected_SSB_data_corrwithlabel.append(sta[0])
        #     for idx in range(selected_RNA_seq_data.shape[1]):
        #         data_ = selected_RNA_seq_data.iloc[:,idx]          
        #         sta = stats.spearmanr(np.array(data_)[:-12],y_labelto_age)
        #         selected_RNA_seq_data_pwithlabel.append(sta[1])
        #         selected_RNA_seq_data_corrwithlabel.append(sta[0])
        #     # xxx = pd.DataFrame({'sampleId:':selected_RNA_seq_data.index.tolist()[:-12],'age_label':y_labelto_age})
        #     # xxx.to_csv("xxx.csv")
        #     y = selected_SSB_data_pwithlabel
        #     #y = -np.log10(np.array(selected_SSB_data_pwithlabel))
        #     num_bins = 500
        #     cmap = plt.cm.get_cmap('YlOrRd')
        #     hist, bin_edges = np.histogram(selected_SSB_data_pwithlabel, bins=num_bins, range=(0,1))
        #     bin_colors = (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
        #     right_pos = 0
        #     idx_=0
        #     for i in bin_edges:
        #         idx_ +=1
        #         if i >0.05:
        #             right_pos = idx_
        #             break
        #     # print(hist,bin_edges,right_pos)
        #     fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
        #     axs[0].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.4, color=cmap(bin_colors))
        #     axs[0].bar(bin_edges[:right_pos], hist[:right_pos], width=np.diff(bin_edges[:right_pos+1]),alpha=0.8, color="purple")
        #     axs[0].axvline(0.05)
        #     axs[0].set_ylabel('Count')
        #     axs[0].set_xlabel('p-value')
        #     if feature_numbers=='all':
        #         axs[0].set_title('ALL SSB genes')
        #     else :
        #         axs[0].set_title('Top 500 SSB genes')
        #     axs[0].set_xlim(-0.05,1)
        #     sig_counts ,_= np.histogram(y, bins=[0,0.1])
        #     axs[0].text(-0.05,np.max(hist[:right_pos]),s="%.2f"%(sig_counts*100.0/len(y))+"%")
        #     # x = selected_AP_data.columns.tolist()
        #     y = selected_AP_data_pwithlabel
        #     hist, bin_edges = np.histogram(selected_AP_data_pwithlabel, bins=num_bins, range=(0,1))
        #     axs[1].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.4, color=cmap(bin_colors))
        #     axs[1].bar(bin_edges[:right_pos], hist[:right_pos], width=np.diff(bin_edges[:right_pos+1]),alpha=0.8, color="purple")

        #     axs[1].set_ylabel('Count')
        #     axs[1].set_xlabel('p-value')
        #     if feature_numbers=='all':
        #         axs[1].set_title('ALL AP genes')
        #     else :
        #         axs[1].set_title('Top 500 AP genes')
        #     axs[1].axvline(0.05)

        #     sig_counts ,_= np.histogram(y, bins=[0,0.1])
        #     axs[1].text(-0.05,np.max(hist[:right_pos]),s="%.2f"%(sig_counts*100.0/len(y))+"%")
        #     y = selected_RNA_seq_data_pwithlabel
        #     print(y)
        #     hist, bin_edges = np.histogram(selected_RNA_seq_data_pwithlabel, bins=num_bins, range=(0,1))
        #     axs[2].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.4, color=cmap(bin_colors))
        #     axs[2].bar(bin_edges[:right_pos], hist[:right_pos], width=np.diff(bin_edges[:right_pos+1]),alpha=0.8, color="purple")
        #     axs[2].set_ylabel('Count')
        #     axs[2].set_xlabel('p-value')
        #     if feature_numbers=='all':
        #         axs[2].set_title('ALL RNA_seq genes')
        #     else :
        #         axs[2].set_title('Top 500 RNA_seq genes')
        #     sig_counts ,_= np.histogram(y, bins=[0,0.1])
        #     axs[2].axvline(0.05)
        #     axs[2].text(-0.05,np.max(hist[:right_pos]),s="%.2f"%(sig_counts*100.0/len(y))+"%")
        #     # from mpl_toolkits.axes_grid1 import make_axes_locatable
        #     # divider = make_axes_locatable(axs[-1])
        #     # cax = divider.append_axes("right", size="5%", pad=0.1)
        #     cax = fig.add_axes([0.93, 0.3, 0.01, 0.4])
        #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(bin_colors), vmax=max(bin_colors)))
        #     sm.set_array([])
        #     cbar = plt.colorbar(sm, cax=cax)
        #     cbar.set_label('Normalized frequency')
        #     plt.subplots_adjust(wspace=0.3)
        #     plt.savefig(topN_save_path+'hist_threeData_of_PvaluewithAgeLabel_500_71.png',dpi = 1080)
        #     plt.clf()

        #     bins = np.linspace(-1, 1, 500)
        #     colors = ['gray' if bins[i] >= -0.2 and bins[i+1] <= 0.2 else "red" for i in range(len(bins)-1)]
        #     gray_l = 0
        #     gray_r = 0
        #     for idx,i in enumerate(colors):
        #         if i == 'gray' and gray_l ==0:
        #             gray_l = idx
        #         if i == 'gray' and colors[idx+1] == 'red':
        #             gray_r = idx+2
        #             break  
        #     print(gray_l,gray_r)  
        #     y = selected_SSB_data_corrwithlabel

        #     idx = np.digitize(y, bins) 
        #     print(len(colors),max(idx) )
        #     range_bin = [-0.2,0.2]
        #     hist, _ = np.histogram(y, bins=500,range=(-1,1))
        #     valid_length = np.sum(hist)
        #     hist =  np.sum(hist[gray_l:gray_r])
        #     fig = plt.figure(figsize=(10, 15)) 
        #     plt.title("ALL 71")
        #     plt.subplot(3,1,1)
        #     plt.hist(y,bins = bins,color="red",alpha=1.0) 
        #     plt.hist(y,bins = bins[gray_l:gray_r],color="gray",alpha=1.0) 
        #     # for c, p in zip(n[], patches):
        #     #     plt.setp(p, 'facecolor', "gray")
        #     plt.ylabel('Count')
        #     plt.xlabel('Rho- values')
        #     plt.axvline(-0.2)
        #     plt.axvline(0.2)
        #     plt.xlim(-1,1)
        #     # plt.ylim(0,8)
        #     if feature_numbers=='all':
        #         plt.title('ALL SSB genes')
        #     else :
        #         plt.title('Top 500 SSB genes')
        #     plt.text(-0.03,6,s="%.2f"%(hist*100.0/valid_length)+"%")
        #     # x = selected_AP_data.columns.tolist()
        #     y = selected_AP_data_corrwithlabel
        #     hist, _ = np.histogram(y, bins=500,range=(-1,1))
        #     valid_length = np.sum(hist)
        #     hist =  np.sum(hist[gray_l:gray_r])
        #     plt.subplot(3,1,2)
        #     plt.hist(y,bins = bins,color="red",alpha=1.0) 
        #     plt.hist(y,bins = bins[gray_l:gray_r],color="gray",alpha=1.0) 
        #     plt.ylabel('Count')
        #     plt.xlabel('Rho- values')
        #     plt.xlim(-1,1)
        #     # plt.ylim(0,8)
        #     plt.axvline(-0.2)
        #     plt.axvline(0.2)
        #     if feature_numbers=='all':
        #         plt.title('ALL AP genes')
        #     else :
        #         plt.title('Top 500 AP genes')
        #     plt.text(-0.03,6,s="%.2f"%(hist*100.0/valid_length)+"%")
        #     y = selected_RNA_seq_data_corrwithlabel
        #     hist, _ = np.histogram(y, bins=500,range=(-1,1))
        #     valid_length = np.sum(hist)
        #     hist =  np.sum(hist[gray_l:gray_r])
        #     plt.subplot(3,1,3)
        #     plt.hist(y,bins = bins,color="red",alpha=1.0) 
        #     plt.hist(y,bins = bins[gray_l:gray_r],color="gray",alpha=1.0) 
        #     plt.ylabel('Count')
        #     plt.xlabel('Rho- values')
        #     plt.xlim(-1,1)
        #     # plt.ylim(0,8)
        #     plt.axvline(-0.2)
        #     plt.axvline(0.2)
        #     if feature_numbers=='all':
        #         plt.title('ALL RNA_seq genes')
        #     else :
        #         plt.title('Top 500 RNA_seq genes')
        #     plt.text(-0.03,6,s="%.2f"%(hist*100.0/valid_length)+"%")

        #     plt.savefig(topN_save_path+'hist_threeData_otestwithlabelWithAgeLabel_500_71.png',dpi = 1080)
            # if feature_numbers == 'all':
            #     AP_feature_importance_df = pd.DataFrame({'scores': sorted(AP_scores,reverse=1),'alogrithm-pvalue': sorted(AP_p_values),'CorrWithLabel':selected_AP_data_corrwithlabel,'pvalueWithLabel':selected_AP_data_pwithlabel},index =selected_AP_data.columns.tolist())
            #     SSB_feature_importance_df = pd.DataFrame({'scores': sorted(SSB_scores,reverse=1), 'alogrithm-pvalue': sorted(SSB_p_values),'CorrWithLabel':selected_SSB_data_corrwithlabel,'pvalueWithLabel':selected_SSB_data_pwithlabel},index =selected_SSB_data.columns.tolist())
            #     RNA_feature_importance_df = pd.DataFrame({'scores': sorted(RNA_scores,reverse=1), 'alogrithm-pvalue': sorted(RNA_p_values),'CorrWithLabel':selected_RNA_seq_data_corrwithlabel,'pvalueWithLabel':selected_RNA_seq_data_pwithlabel},index =selected_RNA_seq_data.columns.tolist() )
            # else:
            #     AP_feature_importance_df = pd.DataFrame({'scores': sorted(AP_scores,reverse=1)[:feature_numbers],'alogrithm-pvalue': sorted(AP_p_values)[:feature_numbers],'CorrWithLabel':selected_AP_data_corrwithlabel,'pvalueWithLabel':selected_AP_data_pwithlabel},index =selected_AP_data.columns.tolist())
            #     SSB_feature_importance_df = pd.DataFrame({'scores': sorted(SSB_scores,reverse=1)[:feature_numbers], 'alogrithm-pvalue': sorted(SSB_p_values)[:feature_numbers],'CorrWithLabel':selected_SSB_data_corrwithlabel,'pvalueWithLabel':selected_SSB_data_pwithlabel},index =selected_SSB_data.columns.tolist())
            #     RNA_feature_importance_df = pd.DataFrame({'scores': sorted(RNA_scores,reverse=1)[:feature_numbers], 'alogrithm-pvalue': sorted(RNA_p_values)[:feature_numbers],'CorrWithLabel':selected_RNA_seq_data_corrwithlabel,'pvalueWithLabel':selected_RNA_seq_data_pwithlabel},index =selected_RNA_seq_data.columns.tolist() )

            # topN_save_path = './data_topN_features/genes_metrics'
            # if not os.path.exists(topN_save_path):
            #     os.makedirs(topN_save_path)
            # if np.max(y_label) >4:
            #     AP_feature_importance_df.to_csv(topN_save_path+'/selected_AP_data_tissue_metrics'+str(feature_numbers)+'.csv')
            #     SSB_feature_importance_df.to_csv(topN_save_path+'/selected_SSB_data_tissue_metrics'+str(feature_numbers)+'.csv')
            #     RNA_feature_importance_df.to_csv(topN_save_path+'/selected_RNA_seq_data_tissue_metrics'+str(feature_numbers)+'.csv')
            # else :
            #     AP_feature_importance_df.to_csv(topN_save_path+'/selected_AP_data_metrics'+str(feature_numbers)+'.csv')
            #     SSB_feature_importance_df.to_csv(topN_save_path+'/selected_SSB_data_metrics'+str(feature_numbers)+'.csv')
            #     RNA_feature_importance_df.to_csv(topN_save_path+'/selected_RNA_seq_data_metrics'+str(feature_numbers)+'.csv')
            # AP_feature_importance_df = AP_feature_importance_df.sort_values(by=['pvalueWithLabel'])
            # SSB_feature_importance_df = SSB_feature_importance_df.sort_values(by=['pvalueWithLabel'])
            # RNA_feature_importance_df = RNA_feature_importance_df.sort_values(by=['pvalueWithLabel'])
        #     if feature_numbers == 'all':
        #         selected_AP_data = selected_AP_data[AP_feature_importance_df.index.tolist()]
        #         selected_SSB_data = selected_SSB_data[SSB_feature_importance_df.index.tolist()]
        #         selected_RNA_seq_data = selected_RNA_seq_data[RNA_feature_importance_df.index.tolist()]
                
        #     else :
        #         selected_AP_data = selected_AP_data[AP_feature_importance_df.index.tolist()[:feature_numbers]]
        #         selected_SSB_data = selected_SSB_data[SSB_feature_importance_df.index.tolist()[:feature_numbers]]
        #         selected_RNA_seq_data = selected_RNA_seq_data[RNA_feature_importance_df.index.tolist()[:feature_numbers]]
        #     topN_save_path = './data_topN_features/genes_metrics'
        #     if not os.path.exists(topN_save_path):
        #         os.makedirs(topN_save_path)
        #     AP_feature_importance_df.to_csv(topN_save_path+'/selected_AP_data_metricsByPvaluewithlabel'+str(feature_numbers)+'.csv')
        #     SSB_feature_importance_df.to_csv(topN_save_path+'/selected_SSB_data_metricsByPvaluewithlabel'+str(feature_numbers)+'.csv')
        #     RNA_feature_importance_df.to_csv(topN_save_path+'/selected_RNA_seq_data_metricsByPvaluewithlabel'+str(feature_numbers)+'.csv')
        # print(selected_AP_data.shape,selected_SSB_data.shape,selected_RNA_seq_data.shape)
    return selected_AP_data,selected_SSB_data,selected_RNA_seq_data




#using lgb to test the acc of three type data individually
def test_the_topN_features(svc,selected_SSB_data,selected_AP_data,selected_RNA_seq_data,y_label):
    accuracies = []
    max_acc = 0
    if svc == 'Lasso' or svc == 'Ridge' or svc == 'ElasticNet' or svc == 'LinearRegression' or svc == 'LogisticRegression':
        if svc == 'LinearRegression':
            model = LinearRegression(fit_intercept=True,  copy_X=True, n_jobs=-1)
        elif svc == 'Ridge':
            model = Ridge(random_state = 42)
        elif svc == 'Lasso':
            model = Lasso(random_state = 42)
        elif svc == 'ElasticNet':
            model = ElasticNet(random_state = 42)
        elif svc == 'LogisticRegression':
            model = LogisticRegression(multi_class='multinomial',verbose=0,random_state=42)
    else :
        if svc == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=42)
        elif svc == 'MultinomialNB':
            model = GaussianNB()
        elif svc == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(criterion='gini', splitter='best',min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=42,min_impurity_decrease=0.0)
        elif svc == 'RandomForestClassifier':
            model = RandomForestClassifier(n_estimators=1000,max_depth=2, random_state=0)
        elif svc == 'XGBClassifier':
            model = XGBClassifier( booster='gbtree',objective='multi:softmax',random_state=42,learning_rate=0.1,n_jobs=-1)
        elif svc == 'LGBMClassifier':
            model = LGBMClassifier(n_estimators=1000,objective="multiclass")
        elif svc == 'AdaBoostClassifier':
            model = AdaBoostClassifier(n_estimators=50, learning_rate=0.1, algorithm='SAMME.R', random_state=42)
    x_train,x_test,y_train,y_test = train_test_split(np.array(selected_SSB_data),y_label,test_size=0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    print(model.predict(x_test))
    y_pred = np.array(model.predict(x_test),dtype = np.int64)
    print(y_pred)
    acc = accuracy_score(y_test,y_pred)
    print(f"SSB_data _ Age label total accuracy: {acc}")

    x_train,x_test,y_train,y_test = train_test_split(np.array(selected_AP_data),y_label,test_size=0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    y_pred = np.array(model.predict(x_test),dtype = np.int64)
    acc = accuracy_score(y_test,y_pred)
    print(f"AP_data _ Age label total accuracy: {acc}")

    x_train,x_test,y_train,y_test = train_test_split(np.array(selected_RNA_seq_data),y_label,test_size=0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    y_pred = np.array(model.predict(x_test),dtype = np.int64)
    acc = accuracy_score(y_test,y_pred)
    print(f"RNA_seq_data _ Age label total accuracy: {acc}")

    kf = KFold(n_splits=10, shuffle=True,random_state=42)  # 使用10折交叉验证，打乱数据
    print("Kfold acc:\n")
    data_norm = np.array(selected_SSB_data)
    for train_index, test_index in kf.split(data_norm):

        # 4. 使用训练集特征向量训练模型
        model.fit(data_norm[train_index], y_label[train_index])

        y_pred = model.predict(data_norm[test_index])
        y_pred = np.array(y_pred,dtype = np.int64)
        accuracy = accuracy_score(y_label[test_index], y_pred)
        # loss, accuracy = rf.evaluate(data_norm[test_index], y_label[test_index], verbose=0)
        if max_acc<accuracy:
            max_acc = accuracy
        accuracies.append(accuracy)

    total_accuracy = np.mean(accuracies)
    print(f"SSB_data _ Age label total accuracy: {total_accuracy}",max_acc)

    max_acc = 0
    accuracies = []

    data_norm = np.array(selected_AP_data)
    for train_index, test_index in kf.split(data_norm):

        # 4. 使用训练集特征向量训练模型
        model.fit(data_norm[train_index], y_label[train_index])

        y_pred = model.predict(data_norm[test_index])
        y_pred = np.array(y_pred,dtype = np.int64)

        accuracy = accuracy_score(y_label[test_index], y_pred)
        # loss, accuracy = rf.evaluate(data_norm[test_index], y_label[test_index], verbose=0)
        if max_acc<accuracy:
            max_acc = accuracy
        accuracies.append(accuracy)

    total_accuracy = np.mean(accuracies)
    print(f"AP_data _ Age label total accuracy: {total_accuracy}",max_acc)

    accuracies = []
    max_acc = 0
    data_norm = np.array(selected_RNA_seq_data)
    for train_index, test_index in kf.split(data_norm):

        # 4. 使用训练集特征向量训练模型
        model.fit(data_norm[train_index], y_label[train_index])

        y_pred = model.predict(data_norm[test_index])
        y_pred = np.array(y_pred,dtype = np.int64)

        accuracy = accuracy_score(y_label[test_index], y_pred)
        # loss, accuracy = rf.evaluate(data_norm[test_index], y_label[test_index], verbose=0)
        if max_acc<accuracy:
            max_acc = accuracy
        accuracies.append(accuracy)

    total_accuracy = np.mean(accuracies)
    print(f"RNA_seq_data _ Age label total accuracy: {total_accuracy}",max_acc)

#merge the three type data 
def gen_merge_data(selected_SSB_data,selected_AP_data,selected_RNA_seq_data):
    # AP_SSB_merge_data = selected_SSB_data[selected_AP_data.columns.intersection(selected_SSB_data.columns)]
    # SSB_RNA_seq_merge_data = selected_SSB_data[selected_RNA_seq_data.columns.intersection(selected_SSB_data.columns)]
    # AP_RNA_seq_merge_data = selected_AP_data[selected_RNA_seq_data.columns.intersection(selected_AP_data.columns)]
    # ALL_merge_data = AP_SSB_merge_data[selected_RNA_seq_data.columns.intersection(AP_SSB_merge_data.columns)]
    AP_SSB_merge_data = np.add(selected_SSB_data,selected_AP_data)
    SSB_RNA_seq_merge_data = np.add(selected_SSB_data,selected_RNA_seq_data)
    AP_RNA_seq_merge_data = np.add(selected_AP_data,selected_RNA_seq_data)
    ALL_merge_data = np.add(AP_RNA_seq_merge_data,selected_SSB_data)
    # AP_SSB_merge_data = (AP_SSB_merge_data - AP_SSB_merge_data.min(axis=0))/(AP_SSB_merge_data.max(axis=0)-AP_SSB_merge_data.min(axis=0))
    # SSB_RNA_seq_merge_data = (SSB_RNA_seq_merge_data - SSB_RNA_seq_merge_data.min(axis=0))/(SSB_RNA_seq_merge_data.max(axis=0)-SSB_RNA_seq_merge_data.min(axis=0))
    # AP_RNA_seq_merge_data = (AP_RNA_seq_merge_data - AP_RNA_seq_merge_data.min(axis=0))/(AP_RNA_seq_merge_data.max(axis=0)-AP_RNA_seq_merge_data.min(axis=0))
    # ALL_merge_data = (ALL_merge_data - ALL_merge_data.min(axis=0))/(ALL_merge_data.max(axis=0)-ALL_merge_data.min(axis=0))
    # AP_SSB_merge_data = (AP_SSB_merge_data - AP_SSB_merge_data.mean(axis=0))/AP_SSB_merge_data.std(axis=0)
    # SSB_RNA_seq_merge_data =  (SSB_RNA_seq_merge_data - SSB_RNA_seq_merge_data.mean(axis=0))/SSB_RNA_seq_merge_data.std(axis=0)
    # AP_RNA_seq_merge_data =  (AP_RNA_seq_merge_data - AP_RNA_seq_merge_data.mean(axis=0))/AP_RNA_seq_merge_data.std(axis=0)
    # ALL_merge_data =  (ALL_merge_data - ALL_merge_data.mean(axis=0))/ALL_merge_data.std(axis=0)
    # data_norm = (data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
    # data_norm = (data_norm - data_norm.mean(axis=0))/data_norm.std(axis=0)
    print(AP_SSB_merge_data.shape,SSB_RNA_seq_merge_data.shape,AP_RNA_seq_merge_data.shape,ALL_merge_data.shape)
    # print(AP_SSB_merge_data.head,SSB_RNA_seq_merge_data.head,AP_RNA_seq_merge_data.head,ALL_merge_data.head)
    return AP_RNA_seq_merge_data,SSB_RNA_seq_merge_data,AP_SSB_merge_data,ALL_merge_data

# svc = LinearDiscriminantAnalysis()
# accuracies = []
# data_norm = np.array(AP_SSB_merge_data)
# data_norm = (data_norm - data_norm.mean(axis=0))/data_norm.std(axis=0)

# max_acc = 0
# for train_index, test_index in kf.split(data_norm):

#     # 4. 使用训练集特征向量训练模型
#     svc.fit(data_norm[train_index], y_label[train_index])

#     y_pred = svc.predict(data_norm[test_index])

#     accuracy = accuracy_score(y_age[test_index], y_pred)
#     # loss, accuracy = rf.evaluate(data_norm[test_index], y_label[test_index], verbose=0)
#     if max_acc<accuracy:
#         max_acc = accuracy
#     accuracies.append(accuracy)

# total_accuracy = np.mean(accuracies)
# print("SSB_data&AP_data age total accuracy:",total_accuracy,max_acc)

# data_norm = np.array(SSB_RNA_seq_merge_data)
# data_norm = (data_norm - data_norm.min(axis=0))/(data_norm.max(axis=0)-data_norm.min(axis=0))

# data_norm = (data_norm - data_norm.mean(axis=0))/data_norm.std(axis=0)

# max_acc = 0
# accuracies = []
# for train_index, test_index in kf.split(data_norm):

#     # 4. 使用训练集特征向量训练模型
#     svc.fit(data_norm[train_index], y_age[train_index])

#     y_pred = svc.predict(data_norm[test_index])

#     accuracy = accuracy_score(y_age[test_index], y_pred)
#     # loss, accuracy = rf.evaluate(data_norm[test_index], y_label[test_index], verbose=0)
#     if max_acc<accuracy:
#         max_acc = accuracy
#     accuracies.append(accuracy)

# total_accuracy = np.mean(accuracies)
# print("SSB_data&RNA_data age total accuracy:",total_accuracy,max_acc)

# data_norm = np.array(AP_RNA_seq_merge_data)
# data_norm = (data_norm - data_norm.min(axis=0))/(data_norm.max(axis=0)-data_norm.min(axis=0))

# data_norm = (data_norm - data_norm.mean(axis=0))/data_norm.std(axis=0)

# max_acc = 0
# accuracies = []
# for train_index, test_index in kf.split(data_norm):

#     # 4. 使用训练集特征向量训练模型
#     svc.fit(data_norm[train_index], y_age[train_index])

#     y_pred = svc.predict(data_norm[test_index])

#     accuracy = accuracy_score(y_age[test_index], y_pred)
#     # loss, accuracy = rf.evaluate(data_norm[test_index], y_label[test_index], verbose=0)
#     if max_acc<accuracy:
#         max_acc = accuracy
#     accuracies.append(accuracy)

# total_accuracy = np.mean(accuracies)
# print("RNA_data&AP_data age total accuracy:",total_accuracy,max_acc)


# data_norm = np.array(ALL_merge_data)
# data_norm = (data_norm - data_norm.min(axis=0))/(data_norm.max(axis=0)-data_norm.min(axis=0))

# data_norm = (data_norm - data_norm.mean(axis=0))/data_norm.std(axis=0)

# max_acc = 0
# accuracies = []
# for train_index, test_index in kf.split(data_norm):

#     # 4. 使用训练集特征向量训练模型
#     svc.fit(data_norm[train_index], y_age[train_index])

#     y_pred = svc.predict(data_norm[test_index])

#     accuracy = accuracy_score(y_age[test_index], y_pred)
#     # loss, accuracy = rf.evaluate(data_norm[test_index], y_label[test_index], verbose=0)
#     if max_acc<accuracy:
#         max_acc = accuracy
#     accuracies.append(accuracy)

# total_accuracy = np.mean(accuracies)
# print("RNA_data&AP_data&SSB_data age total accuracy:",total_accuracy,max_acc)

def  print_analize_result_(svc,selected_SSB_data,selected_AP_data,selected_RNA_seq_data,y_label,selected_gene_num,AP_RNA_seq_merge_data=None,SSB_RNA_seq_merge_data=None,AP_SSB_merge_data=None,ALL_merge_data=None):
    #ROC
    n_classes = ['0', '1', '2', '3', '4']
    # n_classes = ['0', '1']

    age_info = ['3','12','19','22','24']
    tissue_info = ['B-','H-','L-','M-','P-','S-']
    colors = ['blue','yellow','green','orange','black']
    is_age_label = None
    if np.max(y_label) > 4 :
        label_info = tissue_info
        is_age_label = 0
    else :
        is_age_label = 1
        label_info = age_info

    def softmax(x):
        
        max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum 
        return f_x
    linear_model = 0
    decisionTree_model = 0
    if svc == 'Lasso' or svc == 'ElasticNet' or svc == 'Ridge' or svc == 'LinearRegression' or svc == 'LogisticRegression' or svc == 'DecisionTreeRegressor' or svc == 'DecisionTreeClassifier' or svc == 'LGBMRegressor':
        linear_model = 1
        if svc == 'LinearRegression':
            model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=-1)
        elif svc == 'Ridge':
            model = Ridge(random_state = 42)
        elif svc == 'Lasso':
            model = Lasso(random_state = 42)
        elif svc == 'ElasticNet':
            model = ElasticNet(random_state = 42)
        elif svc == 'LogisticRegression':
            model = LogisticRegression(multi_class='multinomial',verbose=0,random_state=42)
        elif svc == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=42)
        elif svc == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(criterion='gini', splitter='best',min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=42,min_impurity_decrease=0.0)
        elif svc == 'LGBMRegressor':
            model = DecisionTreeClassifier(criterion='gini', splitter='best',min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=42,min_impurity_decrease=0.0)

    else:
        decisionTree_model = 1
        if svc == 'RandomForestClassifier':
            model = RandomForestClassifier(n_estimators=1000,max_depth=2, random_state=0)
        elif svc == 'XGBClassifier':
            model = XGBClassifier( booster='gbtree',objective='multi:softmax',random_state=42,learning_rate=0.1,n_jobs=-1)
        elif svc == 'MultinomialNB':
            model = GaussianNB()
        elif svc == 'LGBMClassifier':
            model = LGBMClassifier(n_estimators=1000,objective="multiclass")
        elif svc == 'AdaBoostClassifier':
            model = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1, algorithm='SAMME.R', random_state=42)
    to_save_results =[]
    to_save_metrics =[]
    x_train,x_test,y_train,y_test = train_test_split(np.array(selected_SSB_data),y_label,test_size=0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    plt.subplots_adjust(top=0.85)
    fig = plt.figure(figsize=(20, 20)) 
    plt.subplot(3, 3, 1)    
    if linear_model == 1 :
        pred_label = model.predict(x_test)

        if is_age_label :
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
        else:
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
        pred_label = np.array(pred_label,dtype=np.int64)
        pred = to_categorical(pred_label)
        to_plot_multi_SSB_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
        to_plot_multi_SSB_f1_score = f1_score(y_test, pred_label, average='weighted')
        to_save_result = 'SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(linear_auc)+'\n'
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'SSB' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
        to_save_metrics.append(to_save_metric)
    else:
        pred = model.predict_proba(x_test)
        pred_label = model.predict(x_test)
        pred_roc_label = to_categorical(pred_label)
        y_roc_test = to_categorical(y_test)
        to_plot_multi_SSB_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
        to_plot_multi_SSB_f1_score = f1_score(y_test, pred_label, average='weighted')
        to_save_result = 'SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(multi_auc)+'\n'
        pred = softmax(pred)
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'SSB' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
        to_save_metrics.append(to_save_metric)
    # print(pred)

    to_save_results.append(to_save_result)
    # print(to_save_result)

    # pred = to_categorical(pred)
    y_test = to_categorical(y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_fpr = []
    mean_tpr = []   

    for i in range(len(n_classes)):
        # print(y_test[:, i],pred[:, i])
        fpr[i],  tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
        print(fpr[i], tpr[i],thresholds)
        mean_fpr.append(fpr[i])
        mean_tpr.append(tpr[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                label='SSB curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]),alpha = 0.2)
    if linear_model == 0 or decisionTree_model == 1:
        mean_fpr = np.array(mean_fpr).mean(axis=0)
        mean_tpr = np.array(mean_tpr).mean(axis=0)
        mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
        plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='SSB curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))
    else:    
        plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
        # print(mean_fpr)
        
    plt.title('SSB validation ROC')
    plt.plot([0,1],[0,1],'k--',label = 'base')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel ('True Positive Rate')
    plt.xlabel ('False Positive Rate')
    plt.legend()

    plt.subplot(3, 3, 2)
    x_train,x_test,y_train,y_test = train_test_split(np.array(selected_AP_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    if linear_model == 1 :
        pred_label = model.predict(x_test)
        if is_age_label :
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
        else:
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
        pred_label = np.array(pred_label,dtype=np.int64)
        pred = to_categorical(pred_label)
        to_plot_multi_AP_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
        to_plot_multi_AP_f1_score = f1_score(y_test, pred_label, average='weighted')
        to_save_result = 'AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(linear_auc)+'\n'
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
        to_save_metrics.append(to_save_metric)
    else:
        pred  = model.predict_proba(x_test)
        pred_label = model.predict(x_test)
        pred = softmax(pred)
        pred_roc_label = to_categorical(pred_label)
        y_roc_test = to_categorical(y_test)
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
        to_save_metrics.append(to_save_metric)
        to_plot_multi_AP_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
        to_plot_multi_AP_f1_score = f1_score(y_test, pred_label, average='weighted')
        to_save_result = 'AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info)+'multi_AUC_result:'+str(multi_auc)+'\n'
    to_save_results.append(to_save_result)
    # print(to_save_result)
    # print(pred)
    
    # pred = to_categorical(pred)
    y_test = to_categorical(y_test)
    # fpr,tpr,threshold = metrics.roc_curve(y_test,pred)
    # roc_curce = metrics.auc(fpr,tpr)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_fpr = []
    mean_tpr = []
    for i in range(len(n_classes)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
        # print(fpr[i], tpr[i], thresholds)
        mean_fpr.append(fpr[i])
        mean_tpr.append(tpr[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                label='AP curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]),alpha = 0.2)
    # print(mean_fpr)
    if linear_model == 0 or decisionTree_model == 1:
        mean_fpr = np.array(mean_fpr).mean(axis=0)
        mean_tpr = np.array(mean_tpr).mean(axis=0)
        mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
        plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='AP curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))
    else:    
        plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
    plt.title('AP validation ROC')
    plt.plot([0,1],[0,1],'k--',label = 'base')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel ('True Positive Rate')
    plt.xlabel ('False Positive Rate')
    plt.legend()

    plt.subplot(3, 3, 3)
    x_train,x_test,y_train,y_test = train_test_split(np.array(selected_RNA_seq_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    if linear_model == 1 :
        pred_label = model.predict(x_test)
        if is_age_label :
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
        else:
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
        pred_label = np.array(pred_label,dtype=np.int64)
        pred = to_categorical(pred_label)
        to_plot_multi_RNA_seq_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
        to_plot_multi_RNA_seq_f1_score = f1_score(y_test, pred_label, average='weighted')

        to_save_result = 'RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(linear_auc)+'\n'
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
        to_save_metrics.append(to_save_metric)
    else:
        pred  = model.predict_proba(x_test)
        pred_label = model.predict(x_test)
        pred = softmax(pred)
        pred_roc_label = to_categorical(pred_label)
        y_roc_test = to_categorical(y_test)
        to_plot_multi_RNA_seq_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
        to_plot_multi_RNA_seq_f1_score = f1_score(y_test, pred_label, average='weighted')       
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
        to_save_metrics.append(to_save_metric)
        to_save_result = 'RNA-seq :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(multi_auc)+'\n'
    to_save_results.append(to_save_result)
    print(to_save_result)
    
    # pred = to_categorical(pred)
    y_test = to_categorical(y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_fpr = []
    mean_tpr = []
    for i in range(len(n_classes)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
        mean_fpr.append(fpr[i])
        mean_tpr.append(tpr[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                label='RNA-seq curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]),alpha = 0.2)
    if linear_model == 0 or decisionTree_model == 1:
        mean_fpr = np.array(mean_fpr).mean(axis=0)
        mean_tpr = np.array(mean_tpr).mean(axis=0)
        mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
        plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='RNA-seq curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))
    else:    
        plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
    plt.title('RNA-seq validation ROC')
    plt.plot([0,1],[0,1],'k--',label = 'base')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel ('True Positive Rate')
    plt.xlabel ('False Positive Rate')
    plt.legend()

    plt.subplot(3, 3, 4)
    x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_RNA_seq_merge_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    if linear_model == 1 :
        pred_label = model.predict(x_test)
        print(pred_label)
        if is_age_label :
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
        else:
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
        pred_label = np.array(pred_label,dtype=np.int64)
        pred = to_categorical(pred_label)
        print(pred,y_test)
        to_plot_multi_SSB_RNA_seq_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
        to_save_result = 'RNA-seq&SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(linear_auc)+'\n'
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&SSB' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
        to_save_metrics.append(to_save_metric)
    else:
        pred  = model.predict_proba(x_test)
        pred_label = model.predict(x_test)
        pred = softmax(pred)
        pred_roc_label = to_categorical(pred_label)
        y_roc_test = to_categorical(y_test)
        to_plot_multi_SSB_RNA_seq_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&SSB' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
        to_save_metrics.append(to_save_metric)
        to_save_result ='RNA-seq&SSB:\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+ str(multi_auc)+'\n'
    to_save_results.append(to_save_result)
    # print(to_save_result)
    
    # pred = to_categorical(pred)
    y_test = to_categorical(y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_fpr = []
    mean_tpr = []
    for i in range(len(n_classes)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
        print(fpr[i], tpr[i], thresholds)
        mean_fpr.append(fpr[i])
        mean_tpr.append(tpr[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                label='RNA-seq&SSB curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]),alpha = 0.2)
    if linear_model == 0 or decisionTree_model == 1:
        mean_fpr = np.array(mean_fpr).mean(axis=0)
        mean_tpr = np.array(mean_tpr).mean(axis=0)
        mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
        plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='RNA-seq&SSB curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))
    else:    
        plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))

    plt.title('RNA-seq&SSB validation ROC')
    plt.plot([0,1],[0,1],'k--',label = 'base')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel ('True Positive Rate')
    plt.xlabel ('False Positive Rate')
    plt.legend()

    plt.subplot(3, 3, 5)
    x_train,x_test,y_train,y_test = train_test_split(np.array(AP_RNA_seq_merge_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    if linear_model == 1 :
        pred_label = model.predict(x_test)
        if is_age_label :
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
        else:
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
        pred_label = np.array(pred_label,dtype=np.int64)
        pred = to_categorical(pred_label)
        to_plot_multi_AP_RNA_seq_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
        to_save_result = 'RNA-seq&AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(linear_auc)+'\n'
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
        to_save_metrics.append(to_save_metric)
    else:
        pred  = model.predict_proba(x_test)
        pred_label = model.predict(x_test)
        pred = softmax(pred)
        pred_roc_label = to_categorical(pred_label)
        y_roc_test = to_categorical(y_test)
        to_plot_multi_AP_RNA_seq_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
        to_save_metrics.append(to_save_metric)
        to_save_result ='RNA-seq&AP:\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(multi_auc)+'\n'
    to_save_results.append(to_save_result)
    # print(to_save_result)
    print("asdas",pred,y_test)    
    # pred = to_categorical(pred)
    y_test = to_categorical(y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_fpr = []
    mean_tpr = []
    for i in range(len(n_classes)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
        print(fpr[i],tpr[i],thresholds)
        mean_fpr.append(fpr[i])
        mean_tpr.append(tpr[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                label='RNA-seq&AP curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]),alpha = 0.2)
    if linear_model == 0 or decisionTree_model == 1:
        mean_fpr = np.array(mean_fpr).mean(axis=0)
        mean_tpr = np.array(mean_tpr).mean(axis=0)
        mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
        plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='RNA-seq&SSB curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))
    else:    
        plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
    plt.title('RNA-seq&AP validation ROC')
    plt.plot([0,1],[0,1],'k--',label = 'base')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel ('True Positive Rate')
    plt.xlabel ('False Positive Rate')
    plt.legend()

    plt.subplot(3, 3, 6)
    x_train,x_test,y_train,y_test = train_test_split(np.array(AP_SSB_merge_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    if linear_model == 1 :
        pred_label = model.predict(x_test)
        if is_age_label :
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
        else:
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
        pred_label = np.array(pred_label,dtype=np.int64)
        pred = to_categorical(pred_label)
        to_plot_multi_SSB_AP_seq_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
        to_save_result = 'SSB&AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(linear_auc)+'\n'
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'SSB&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
        to_save_metrics.append(to_save_metric)
    else:
        pred  = model.predict_proba(x_test)
        pred_label = model.predict(x_test)
        pred = softmax(pred)
        pred_roc_label = to_categorical(pred_label)
        y_roc_test = to_categorical(y_test)
        to_plot_multi_SSB_AP_seq_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'SSB&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
        to_save_metrics.append(to_save_metric)
        to_save_result ='SSB&AP:\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(multi_auc)+'\n'
    to_save_results.append(to_save_result)
    # print(to_save_result)
    
    # pred = to_categorical(pred)
    y_test = to_categorical(y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_fpr = []
    mean_tpr = []
    for i in range(len(n_classes)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
        mean_fpr.append(fpr[i])
        mean_tpr.append(tpr[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                label='AP&SSB curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]),alpha = 0.2)
    if linear_model == 0 or decisionTree_model == 1:
        mean_fpr = np.array(mean_fpr).mean(axis=0)
        mean_tpr = np.array(mean_tpr).mean(axis=0)
        mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
        plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='AP&SSB curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))
    else:    
        plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
    plt.title('AP&SSB validation ROC')
    plt.plot([0,1],[0,1],'k--',label = 'base')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel ('True Positive Rate')
    plt.xlabel ('False Positive Rate')
    plt.legend()

    plt.subplot(3, 3, 7)
    x_train,x_test,y_train,y_test = train_test_split(np.array(ALL_merge_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
    model.fit(x_train,y_train)
    if linear_model == 1 :
        pred_label = model.predict(x_test)
        if is_age_label :
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
        else:
            pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
        pred_label = np.array(pred_label,dtype=np.int64)
        pred = to_categorical(pred_label)
        to_plot_multi_all_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
        to_save_result = 'RNA-seq&SSB&AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(linear_auc)+'\n'
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&SSB&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
        to_save_metrics.append(to_save_metric)
    else:
        pred  = model.predict_proba(x_test)
        pred_label = model.predict(x_test)    
        pred = softmax(pred)
        pred_roc_label = to_categorical(pred_label)
        y_roc_test = to_categorical(y_test)
        to_plot_multi_all_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
        to_save_result ='RNA-seq&SSB&AP:\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(multi_auc)+'\n'
        to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&SSB&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
        to_save_metrics.append(to_save_metric)
    to_save_results.append(to_save_result)
    # print(to_save_result)
    if is_age_label == 0:
        output_result_file_path = './result/'+'classifier_tissue/selected_gene_'+str(selected_gene_num)+'/'
    else :       
        output_result_file_path = './result/'+'classifier_age/selected_gene_'+str(selected_gene_num)+'/'
    if not os.path.exists(output_result_file_path):
        os.makedirs(output_result_file_path)
    output_result_file = open(output_result_file_path+svc+'.txt','w')
    for res_ in to_save_results:
        output_result_file.write(res_)

    if is_age_label == 0:
        output_result_file_path = './result/'+'classifier_metrics_tissue/selected_gene_'+str(selected_gene_num)+'/'
    else :       
        output_result_file_path = './result/'+'classifier_metrics_age/selected_gene_'+str(selected_gene_num)+'/'
    if not os.path.exists(output_result_file_path):
        os.makedirs(output_result_file_path)
    output_result_file = open(output_result_file_path+svc+'.txt','w')
    for res_ in to_save_metrics:
        output_result_file.write(res_)

    # pred = to_categorical(pred)
    y_test = to_categorical(y_test)

    mean_fpr = []
    mean_tpr = []
    for i in range(len(n_classes)):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)

        mean_fpr.append(fpr[i])
        mean_tpr.append(tpr[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                label='RNA-seq&SSB&AP curve of class {0} (AUC = {1:0.2f})'
                ''.format(i, roc_auc[i]),alpha = 0.2)
    if linear_model == 0 or decisionTree_model == 1:
        mean_fpr = np.array(mean_fpr).mean(axis=0)
        mean_tpr = np.array(mean_tpr).mean(axis=0)
        mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
        plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='RNA-seq&SSB&AP curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))

    else:    
        plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
    plt.title('ALL validation ROC')
    plt.plot([0,1],[0,1],'k--',label = 'base')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel ('True Positive Rate')
    plt.xlabel ('False Positive Rate')
    plt.legend()
    if is_age_label == 0:
        output_result_picture_file_path = './result_pictures/selected_gene_'+str(selected_gene_num)+'/tissue/'
    else :
        output_result_picture_file_path = './result_pictures/selected_gene_'+str(selected_gene_num)+'/age/'
    if not os.path.exists(output_result_picture_file_path):
        os.makedirs(output_result_picture_file_path)
    plt.savefig(output_result_picture_file_path+svc+'.svg',dpi = 1080,format = 'svg')
    # plt.show()
    return to_plot_multi_SSB_auc,to_plot_multi_AP_auc,to_plot_multi_RNA_seq_auc,to_plot_multi_SSB_AP_seq_auc,to_plot_multi_SSB_RNA_seq_auc,to_plot_multi_AP_RNA_seq_auc,to_plot_multi_all_auc,to_plot_multi_SSB_f1_score,to_plot_multi_AP_f1_score,to_plot_multi_RNA_seq_f1_score


#pipeline
def ana_pipeline():
    SSB_data,AP_data,RNA_seq_data,y_age,y_tissue = read_and_preprocess_data()
    # label = SSB_data.columns.tolist()
    # SSB_data = SSB_data.T[:-12].T
    # AP_data = AP_data.T[:-12].T
    # RNA_seq_data = RNA_seq_data.T[:-12].T
    for  y_label in [y_age]:
        # y_label = y_label[:-12]
        selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas = select_topN_genes('svc',SSB_data,AP_data,RNA_seq_data,y_label,1,"all")
    #select#ed_SSB_datas,selected_AP_datas,selected_RNA_seq_datas = select_topN_genes('svc',SSB_data,AP_data,RNA_seq_data,y_age,1,500)
    # caculte_corrs(SSB_data,AP_data,RNA_seq_data,y_age)
    # for selected_gene_for_heatmap in [50,100,200]:
    #     selected_SSB_data = selected_SSB_datas.T[:selected_gene_for_heatmap].T
    #     selected_AP_data = selected_AP_datas.T[:selected_gene_for_heatmap].T
    #     selected_RNA_seq_data = selected_RNA_seq_datas.T[:selected_gene_for_heatmap].T
    #     caculte_topN_genes_corrs(selected_SSB_data,selected_AP_data,selected_RNA_seq_data,selected_gene_for_heatmap,label)
    # clus_methods = ['PCA','TSNE','SpectralEmbedding','LDA','KMeans','KNeighborsClassifier','DBSCAN']
    # colors = ['blue','yellow','green','orange','black','pink','m']
    # colors = ['#f89588','#f74d4d','#B883D4','#76da91','#7898e1','#BEB8DC','#A1A9D0','#eddd86','#8ECFC9','#63b2ee','#943c39']

    # linestyles = ['1','2','3','4','h','x','D']    
    # # # clus_methods = ['KNeighborsClassifier']
    # # # cluster_compare(clus_methods,SSB_data,AP_data,RNA_seq_data,y_age,y_tissue)
    # svcs = ['ElasticNet','RandomForestClassifier','DecisionTreeRegressor','DecisionTreeClassifier','AdaBoostClassifier','LogisticRegression','LinearRegression','LGBMClassifier','XGBClassifier','Lasso']
    # # # svcs = ['DecisionTreeRegressor','DecisionTreeClassifier','AdaBoostClassifier','LogisticRegression','LinearRegression','LGBMClassifier','XGBClassifier','Lasso']
    # # # svcs = ['LogisticRegression','LinearRegression','Lasso']
    # # # svcs = ['XGBClassifier','Lasso']
    # # # # svcs = ['ElasticNet','RandomForestClassifier']
    # # svcs = ['DecisionTreeClassifier','AdaBoostClassifier','XGBClassifier']
    # SSB_data = SSB_data.T[:-12].T
    # AP_data = AP_data.T[:-12].T
    # RNA_seq_data = RNA_seq_data.T[:-12].T
    # select_topN_gene_nums = [50,100,150,200,250,300,400,500,1000,'all']
    # for svc in svcs:
    #     print("processing method:",svc)
    #     for y_label in [y_age,y_tissue]:
    #         to_plot_multi_SSB_aucs = []
    #         to_plot_multi_AP_aucs = []
    #         to_plot_multi_RNA_seq_aucs = []
    #         to_plot_multi_SSB_AP_seq_aucs = []        
    #         to_plot_multi_SSB_RNA_seq_aucs = []
    #         to_plot_multi_AP_RNA_seq_aucs = []
    #         to_plot_multi_all_aucs = []    
    #         to_plot_multi_SSB_f1_scores = []
    #         to_plot_multi_AP_f1_scores = []
    #         to_plot_multi_RNA_seq_f1_scores = []            
    #         # selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas = select_topN_genes(svc,SSB_data,AP_data,RNA_seq_data,y_label,500)
    #         # selected_SSB_datas = (selected_SSB_datas - selected_SSB_datas.mean(axis=0))/selected_SSB_datas.std(axis=0)*2
    #         # selected_AP_datas =  (selected_AP_datas - selected_AP_datas.mean(axis=0))/selected_AP_datas.std(axis=0)*2
    #         # selected_RNA_seq_datas =  (selected_RNA_seq_datas - selected_RNA_seq_datas.mean(axis=0))/selected_RNA_seq_datas.std(axis=0)*10
    #         for selected_gene in select_topN_gene_nums:
    #             selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas = select_topN_genes(svc,SSB_data,AP_data,RNA_seq_data,y_label,saveFlag=0, feature_numbers=selected_gene)
    #             # selected_SSB_datas = (selected_SSB_datas - selected_SSB_datas.mean(axis=0))/selected_SSB_datas.std(axis=0)*5
    #             # selected_AP_datas = (selected_AP_datas - selected_AP_datas.mean(axis=0))/selected_AP_datas.std(axis=0)*5
    #             # selected_RNA_seq_datas =  (selected_RNA_seq_datas - selected_RNA_seq_datas.mean(axis=0))/selected_RNA_seq_datas.std(axis=0)*5

    #             # print(selected_SSB_datas.head)
    #             # selected_AP_datas =  (selected_AP_datas - selected_AP_datas.mean(axis=0))/selected_AP_datas.std(axis=0)*10
    #             # selected_RNA_seq_datas =  (selected_RNA_seq_datas - selected_RNA_seq_datas.mean(axis=0))/selected_RNA_seq_datas.std(axis=0)*2
    #             if isinstance(selected_gene,int):
    #                 selected_SSB_data = selected_SSB_datas.T[:selected_gene].T
    #                 print(selected_SSB_data.shape)
    #                 selected_AP_data = selected_AP_datas.T[:selected_gene].T
    #                 selected_RNA_seq_data = selected_RNA_seq_datas.T[:selected_gene].T
    #                 #test_the_topN_features(svc,selected_SSB_data,selected_AP_data,selected_RNA_seq_data,y_label)
    #                 AP_RNA_seq_merge_data,SSB_RNA_seq_merge_data,AP_SSB_merge_data,ALL_merge_data = gen_merge_data(selected_SSB_data,selected_AP_data,selected_RNA_seq_data)

    #             else :
    #                 selected_SSB_data,selected_AP_data,selected_RNA_seq_data = SSB_data.T,AP_data.T,RNA_seq_data.T
    #                 AP_RNA_seq_merge_data,SSB_RNA_seq_merge_data,AP_SSB_merge_data,ALL_merge_data = gen_merge_data(SSB_data.T,AP_data.T,RNA_seq_data.T)
    #             to_plot_multi_SSB_auc,to_plot_multi_AP_auc,to_plot_multi_RNA_seq_auc,to_plot_multi_SSB_AP_seq_auc,to_plot_multi_SSB_RNA_seq_auc,to_plot_multi_AP_RNA_seq_auc,to_plot_multi_all_auc,to_plot_multi_SSB_f1_score,to_plot_multi_AP_f1_score,to_plot_multi_RNA_seq_f1_score\
    #                   = print_analize_result_(svc,selected_SSB_data,selected_AP_data,selected_RNA_seq_data,y_label,selected_gene,AP_RNA_seq_merge_data,SSB_RNA_seq_merge_data,AP_SSB_merge_data,ALL_merge_data)
    #             to_plot_multi_SSB_aucs.append(to_plot_multi_SSB_auc)
    #             to_plot_multi_AP_aucs.append(to_plot_multi_AP_auc)
    #             to_plot_multi_RNA_seq_aucs.append(to_plot_multi_RNA_seq_auc)
    #             to_plot_multi_SSB_AP_seq_aucs.append(to_plot_multi_SSB_AP_seq_auc)  
    #             to_plot_multi_SSB_RNA_seq_aucs.append(to_plot_multi_SSB_RNA_seq_auc)
    #             to_plot_multi_AP_RNA_seq_aucs.append(to_plot_multi_AP_RNA_seq_auc)
    #             to_plot_multi_all_aucs.append(to_plot_multi_all_auc)
    #             to_plot_multi_SSB_f1_scores.append(to_plot_multi_SSB_f1_score)
    #             to_plot_multi_AP_f1_scores.append(to_plot_multi_AP_f1_score)
    #             to_plot_multi_RNA_seq_f1_scores.append(to_plot_multi_RNA_seq_f1_score)
    #         plt.clf()
    #         plt.figure(figsize=(6, 6))
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_SSB_aucs,color=colors[int(0)],marker = linestyles[int(0)],lw=2,label='SSB multi_auc curve ')
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_AP_aucs,color=colors[int(1)],marker = linestyles[int(1)],lw=2,label='AP multi_auc curve ')
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_RNA_seq_aucs,color=colors[int(2)],marker = linestyles[int(2)],lw=2,label='RNA-seq multi_auc curve ')
    #         # plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_SSB_RNA_seq_aucs,color=colors[int(3)],marker = linestyles[int(3)],lw=2,label='RNA-seq&SSB multi_auc curve ')
    #         # plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_AP_RNA_seq_aucs,color=colors[int(4)],marker = linestyles[int(4)],lw=2,label='RNA-seq&AP multi_auc curve ')
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_SSB_AP_seq_aucs,color=colors[int(5)],marker = linestyles[int(5)],lw=2,label='SSB&AP multi_auc curve ')
    #         # plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_all_aucs,color=colors[int(6)],marker = linestyles[int(6)],lw=2,label='RNA-seq&SSB&AP multi_auc curve ')
    #         plt.xlabel('selected_gene_num')
    #         plt.ylabel('multi_AUC_score')
    #         plt.ylim(-0.05,1.05)
    #         plt.xticks(np.arange(len(select_topN_gene_nums)),select_topN_gene_nums)
    #         plt.title(svc)
    #         plt.legend()
    #         if np.max(y_label)<=4 :
    #             output_result_picture_file_path = './result_metrics_pictures_Top1000_71_0.4/'+'age/'
    #         else :
    #             output_result_picture_file_path = './result_metrics_pictures_Top1000_71_0.4/'+'tissue/'
    #         if not os.path.exists(output_result_picture_file_path):
    #             os.makedirs(output_result_picture_file_path)
    #         plt.savefig(output_result_picture_file_path+svc+'all.jpg',dpi = 400,format = 'jpg')
    #         plt.clf()
    #         plt.figure(figsize=(6,6))
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_SSB_aucs,color=colors[int(0)],marker = linestyles[int(0)],lw=2,label='SSB multi_auc curve ')
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_RNA_seq_aucs,color=colors[int(2)],marker = linestyles[int(2)],lw=2,label='RNA-seq multi_auc curve ')
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_SSB_RNA_seq_aucs,color=colors[int(3)],marker = linestyles[int(3)],lw=2,label='RNA-seq&SSB multi_auc curve ')
    #         plt.xlabel('selected_gene_num')
    #         plt.ylabel('multi_AUC_score')
    #         plt.xticks(np.arange(len(select_topN_gene_nums)),select_topN_gene_nums)
    #         plt.title(svc)
    #         plt.legend()
    #         if np.max(y_label)<=4 :
    #             output_result_picture_file_path = './result_metrics_pictures_Top1000_71_0.4/'+'age/'
    #         else :
    #             output_result_picture_file_path = './result_metrics_pictures_Top1000_71_0.4/'+'tissue/'
    #         if not os.path.exists(output_result_picture_file_path):
    #             os.makedirs(output_result_picture_file_path)
    #         plt.savefig(output_result_picture_file_path+svc+'_RNA&SSB.png',dpi = 400,format = 'png')
    #         plt.clf()
    #         plt.figure(figsize=(6,6))
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_AP_aucs,color=colors[int(1)],marker = linestyles[int(1)],lw=2,label='AP multi_auc curve ')
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_RNA_seq_aucs,color=colors[int(2)],marker = linestyles[int(2)],lw=2,label='RNA-seq multi_auc curve ')
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_AP_RNA_seq_aucs,color=colors[int(4)],marker = linestyles[int(4)],lw=2,label='RNA-seq&AP multi_auc curve ')
    #         plt.xlabel('selected_gene_num')
    #         plt.ylabel('multi_AUC_score')
    #         plt.ylim(-0.05,1.05)
    #         plt.xticks(np.arange(len(select_topN_gene_nums)),select_topN_gene_nums)
    #         plt.title(svc)
    #         plt.legend()
    #         if np.max(y_label)<=4 :
    #             output_result_picture_file_path = './result_metrics_pictures_Top1000_71_0.4/'+'age/'
    #         else :
    #             output_result_picture_file_path = './result_metrics_pictures_Top1000_71_0.4/'+'tissue/'
    #         if not os.path.exists(output_result_picture_file_path):
    #             os.makedirs(output_result_picture_file_path)
    #         plt.savefig(output_result_picture_file_path+svc+'_RNA&AP.png',dpi = 400,format = 'png')

    #         plt.clf()
    #         plt.figure(figsize=(6,6))
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_AP_f1_scores,color=colors[int(-2)],marker = linestyles[int(1)],lw=2,label='AP multi_f1_score curve ')
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_RNA_seq_f1_scores,color=colors[int(-3)],marker = linestyles[int(2)],lw=2,label='RNA-seq multi_f1_score curve ')
    #         plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_SSB_f1_scores,color=colors[int(-1)],marker = linestyles[int(4)],lw=2,label='SSB multi_f1_score curve ')
    #         plt.xlabel('selected_gene_num')
    #         plt.ylabel('multi_f1_score')
    #         plt.xticks(np.arange(len(select_topN_gene_nums)),select_topN_gene_nums)
    #         plt.title(svc)
    #         plt.legend()
    #         if np.max(y_label)<=4 :
    #             output_result_picture_file_path = './result_metrics_pictures_Top1000_71_0.4/'+'age/'
    #         else :
    #             output_result_picture_file_path = './result_metrics_pictures_Top1000_71_0.4/'+'tissue/'
    #         if not os.path.exists(output_result_picture_file_path):
    #             os.makedirs(output_result_picture_file_path)
    #         plt.savefig(output_result_picture_file_path+svc+'_f1_score.png',dpi = 400,format = 'png')

if __name__ == "__main__":
    ana_pipeline()