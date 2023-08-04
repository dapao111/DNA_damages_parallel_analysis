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
from sklearn import preprocessing
warnings.filterwarnings("ignore")

# 读取文件并合并
def read_and_preprocess_data():
    rna_seq_file_path = r'./RNA_data_sorted_tpm.csv'
    RNA_seq_data = pd.read_csv(rna_seq_file_path,header=0,index_col=0).T
    # RNA_seq_data = RNA_seq_data.iloc[0:,1:]
    # RNA_seq_data = RNA_seq_data.T.apply(lambda row: row[row < row.max()] if row.max() < 100 else row, axis=1).dropna()


    SSB_file_path = r'./New_SSB_data_sorted_tpm.csv'
    SSB_data = pd.read_csv(SSB_file_path,header=0,index_col=0).T
    # SSB_data = SSB_data.T.apply(lambda row: row[row < row.max()] if row.max() < 100 else row, axis=1).dropna()
    ssb_columns = SSB_data.columns.tolist()
    # print(ssb_columns)
    # AP_file_path = r'./fasdgasdgasd.csv'
    AP_file_path = r'./New_AP_data_sorted_tpm.csv'

    AP_data = pd.read_csv(AP_file_path,header=0,index_col=0).T
    print(SSB_data.shape,AP_data.shape,RNA_seq_data.shape)

    AP_columns = AP_data.columns.tolist()


    # AP_data_mean = AP_data.values.mean()
    # AP_data_std = AP_data.values.std()
    # SSB_data_mean = SSB_data.values.mean()
    # SSB_data_std = SSB_data.values.std()
    # RNA_seq_data_mean = RNA_seq_data.values.mean()
    # RNA_seq_data_std = RNA_seq_data.values.std()
    RNA_seq_data = RNA_seq_data[RNA_seq_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
    AP_data = AP_data[AP_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
    SSB_data = SSB_data[SSB_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
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
    new_AP_columns = []
    for column in AP_columns:
        column = column.split('.')[0]
        new_AP_columns.append(column)
    print(new_AP_columns)
    RNA_seq_data.columns = new_AP_columns
    AP_data.columns =new_AP_columns
    SSB_data.columns = new_AP_columns

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
    # #筛选相同的基因
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


          
# 训练 LightGBM 模型
def select_topN_genes(svc,AP_data, SSB_data, RNA_seq_data, y_label,feature_numbers=500):
  
    from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif,chi2,f_regression,r_regression,mutual_info_regression

    AP_data = SelectKBest(f_classif, k=feature_numbers).fit_transform(AP_data.T, y_label)
    SSB_data = SelectKBest(f_classif, k=feature_numbers).fit_transform(SSB_data.T , y_label)
    RNA_seq_data = SelectKBest(f_classif, k=feature_numbers).fit_transform(RNA_seq_data.T, y_label)
    return AP_data,SSB_data,RNA_seq_data





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

def  print_analize_result_(svcs,selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas,y_label,selected_gene_nums,AP_RNA_seq_merge_data=None,SSB_RNA_seq_merge_data=None,AP_SSB_merge_data=None,ALL_merge_data=None):
    #ROC
    n_classes = ['0', '1', '2', '3', '4']
    if y_label.shape[0] == 83:
        age_info = ['3','12','19','22','24']
    else:
        age_info = ['3','12','19','22']
        n_classes = ['0', '1', '2', '3']

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
    fig = plt.figure(figsize=(40, 10)) 
    plt.subplots_adjust(top=0.9)
    len_select_genes = len(selected_gene_nums)
    len_svc = len(svcs)       
    idx_select_gene_num = 0
    plt.title('Ten algorithms of ROC about different Top genes ')
    for svc in svcs:
        for selected_gene_num in selected_gene_nums:
            idx_select_gene_num = idx_select_gene_num+1
            selected_SSB_data = selected_SSB_datas.T[:selected_gene_num].T
            selected_AP_data = selected_AP_datas.T[:selected_gene_num].T
            selected_RNA_seq_data = selected_RNA_seq_datas.T[:selected_gene_num].T
            # print(selected_AP_data.shape)
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
                elif svc == 'LGBMRegressor':
                    model = DecisionTreeClassifier(criterion='gini', splitter='best',min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=42,min_impurity_decrease=0.0)
                elif svc == 'DecisionTreeClassifier':
                    model = DecisionTreeClassifier(criterion='gini', splitter='best',min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=42,min_impurity_decrease=0.0)
            else:
                decisionTree_model = 1
                if svc == 'RandomForestClassifier':
                    model = RandomForestClassifier(n_estimators=200,max_depth=2, random_state=0)
                elif svc == 'XGBClassifier':
                    model = XGBClassifier( booster='gbtree',objective='multi:softmax',random_state=42,learning_rate=0.1,n_jobs=-1)
                elif svc == 'MultinomialNB':
                    model = GaussianNB()
                elif svc == 'LGBMClassifier':
                    model = LGBMClassifier(n_estimators=1000,objective="multiclass")
                elif svc == 'AdaBoostClassifier':
                    model = AdaBoostClassifier(n_estimators=50, learning_rate=0.1, algorithm='SAMME.R', random_state=42)
                elif svc == 'DecisionTreeClassifier':
                    model = DecisionTreeClassifier(criterion='gini', splitter='best',min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=42,min_impurity_decrease=0.0)
            to_save_results =[]
            to_save_metrics =[]
            x_train,x_test,y_train,y_test = train_test_split(np.array(selected_SSB_data),y_label,test_size=0.2,random_state=44,stratify=y_label)
            model.fit(x_train,y_train)
            plt.subplot(2, 5, idx_select_gene_num)    
            if linear_model == 1 :
                pred_label = model.predict(x_test)
                if is_age_label :
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) *5
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
                # print(fpr[i], tpr[i],thresholds)
                mean_fpr.append(fpr[i])
                mean_tpr.append(tpr[i])
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])
                # plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                #         label='SSB curve of class {0} (AUC = {1:0.2f})'
                #         ''.format(i, roc_auc[i]),alpha = 0.2)
            if linear_model == 1:
                fpr_length = max(len(sublst) for sublst in mean_fpr)
                new_mean_fpr = []
                new_mean_tpr = []
                for sublst in mean_fpr:
                    x = np.linspace(sublst[0], sublst[-1], len(sublst))
                    y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
                    new_mean_fpr.append(y)
                tpr_length = max(len(sublst) for sublst in mean_tpr)
                for sublst in mean_tpr:
                    x = np.linspace(sublst[0], sublst[-1], len(sublst))
                    y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], tpr_length)]
                    new_mean_tpr.append(y)
                mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                plt.plot(mean_fpr,mean_tpr, color='#f74d4d', lw=2,label='SSB (AUC = %0.3f)'''%(mean_roc_auc))
            else:    
                try:
                    mean_fpr = np.array(mean_fpr).mean(axis=0)
                except:
                    fpr_length = max(len(sublst) for sublst in mean_fpr)
                    new_mean_fpr = []
                    for sublst in mean_fpr:
                        x = np.linspace(sublst[0], sublst[-1], len(sublst))
                        y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
                        new_mean_fpr.append(y)
                    mean_fpr = new_mean_fpr
                try:    
                    mean_tpr = np.array(mean_tpr).mean(axis=0)
                except:
                    fpr_length = max(len(sublst) for sublst in mean_fpr)
                    new_mean_tpr = []
                    for sublst in mean_tpr:
                        x = np.linspace(sublst[0], sublst[-1], len(sublst))
                        y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
                        new_mean_tpr.append(y)
                    mean_tpr = new_mean_tpr
                mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                plt.plot(mean_fpr,mean_tpr, color='#f74d4d', lw=2,label='SSB (AUC = %0.3f)'''%(mean_roc_auc))
                # plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
                # print(mean_fpr)
                
            # plt.title('{}: Top{} selected genes validation ROC'.format(svc,selected_gene_num))
            # plt.plot([0,1],[0,1],'k--',label = 'base')
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.ylabel ('True Positive Rate')
            # plt.xlabel ('False Positive Rate')
            # plt.legend()

            x_train,x_test,y_train,y_test = train_test_split(np.array(selected_AP_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
            model.fit(x_train,y_train)
            if linear_model == 1 :
                pred_label = model.predict(x_test)
                if is_age_label :
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) *5
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
                # plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                #         label='AP curve of class {0} (AUC = {1:0.2f})'
                #         ''.format(i, roc_auc[i]),alpha = 0.2)
            # print(mean_fpr)
            if linear_model == 1:
                fpr_length = max(len(sublst) for sublst in mean_fpr)
                new_mean_fpr = []
                new_mean_tpr = []
                for sublst in mean_fpr:
                    x = np.linspace(sublst[0], sublst[-1], len(sublst))
                    y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
                    new_mean_fpr.append(y)
                tpr_length = max(len(sublst) for sublst in mean_tpr)

                for sublst in mean_tpr:
                    x = np.linspace(sublst[0], sublst[-1], len(sublst))
                    y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], tpr_length)]
                    new_mean_tpr.append(y)
                mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                plt.plot(mean_fpr,mean_tpr, color='#82B0D2', lw=2,label='AP (AUC = %0.3f)'''%(mean_roc_auc))
            else:    
                try:
                    mean_fpr = np.array(mean_fpr).mean(axis=0)
                except:
                    fpr_length = max(len(sublst) for sublst in mean_fpr)
                    new_mean_fpr = []
                    for sublst in mean_fpr:
                        x = np.linspace(sublst[0], sublst[-1], len(sublst))
                        y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
                        new_mean_fpr.append(y)
                    mean_fpr = new_mean_fpr
                try:    
                    mean_tpr = np.array(mean_tpr).mean(axis=0)
                except:
                    fpr_length = max(len(sublst) for sublst in mean_fpr)
                    new_mean_tpr = []
                    for sublst in mean_tpr:
                        x = np.linspace(sublst[0], sublst[-1], len(sublst))
                        y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
                        new_mean_tpr.append(y)
                    mean_tpr = new_mean_tpr
                mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                plt.plot(mean_fpr,mean_tpr, color='#82B0D2', lw=2,label='AP (AUC = %0.3f)'''%(mean_roc_auc))
                # plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
            # plt.title('AP validation ROC')
            # plt.plot([0,1],[0,1],'k--',label = 'base')
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.ylabel ('True Positive Rate')
            # plt.xlabel ('False Positive Rate')
            # plt.legend()

            x_train,x_test,y_train,y_test = train_test_split(np.array(selected_RNA_seq_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
            model.fit(x_train,y_train)
            if linear_model == 1 :
                pred_label = model.predict(x_test)
                if is_age_label :
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
                else:
                    pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) *5
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
                # plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
                #         label='RNA-seq curve of class {0} (AUC = {1:0.2f})'
                #         ''.format(i, roc_auc[i]),alpha = 0.2)
            if linear_model == 1:
                fpr_length = max(len(sublst) for sublst in mean_fpr)
                new_mean_fpr = []
                new_mean_tpr = []
                for sublst in mean_fpr:
                    x = np.linspace(sublst[0], sublst[-1], len(sublst))
                    y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
                    new_mean_fpr.append(y)
                tpr_length = max(len(sublst) for sublst in mean_tpr)
                for sublst in mean_tpr:
                    x = np.linspace(sublst[0], sublst[-1], len(sublst))
                    y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], tpr_length)]
                    new_mean_tpr.append(y)
                mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                plt.plot(mean_fpr,mean_tpr, color='#C76DA2', lw=2,label='RNA-seq (AUC = %0.3f)'''%(mean_roc_auc))
            else:    
                try:
                    mean_fpr = np.array(mean_fpr).mean(axis=0)
                except:
                    fpr_length = max(len(sublst) for sublst in mean_fpr)
                    new_mean_fpr = []
                    for sublst in mean_fpr:
                        x = np.linspace(sublst[0], sublst[-1], len(sublst))
                        y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
                        new_mean_fpr.append(y)
                    mean_fpr = new_mean_fpr
                try:    
                    mean_tpr = np.array(mean_tpr).mean(axis=0)
                except:
                    fpr_length = max(len(sublst) for sublst in mean_fpr)
                    new_mean_tpr = []
                    for sublst in mean_tpr:
                        x = np.linspace(sublst[0], sublst[-1], len(sublst))
                        y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
                        new_mean_tpr.append(y)
                    mean_tpr = new_mean_tpr
                mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                plt.plot(mean_fpr,mean_tpr, color='#C76DA2', lw=2,label='RNA-seq (AUC = %0.3f)'''%(mean_roc_auc))
                # plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
            plt.title('{}:Top{} selected genes validation ROC'.format(svc,selected_gene_num))
            plt.plot([0,1],[0,1],'k--',label = 'base')
            plt.xlim([-0.05,1.05])
            plt.ylim([-0.05,1.05])
            plt.ylabel ('True Positive Rate')
            plt.xlabel ('False Positive Rate')
        
            plt.legend()
    if is_age_label == 0:
        output_result_picture_file_path = './result_ROC_pictures/tissue'
    else :
        output_result_picture_file_path = './result_ROC_pictures/age'
    if not os.path.exists(output_result_picture_file_path):
        os.makedirs(output_result_picture_file_path)
    # plt.savefig(output_result_picture_file_path+'/results_Top500_71_'+str(selected_gene_nums)+'genes.png',format = 'png')
    plt.savefig(output_result_picture_file_path+'/results_Top500_83_'+str(selected_gene_nums)+'genes.svg',format = 'svg')
            # plt.title('RNA-seq validation ROC')
            # plt.plot([0,1],[0,1],'k--',label = 'base')
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.ylabel ('True Positive Rate')
            # plt.xlabel ('False Positive Rate')
            # plt.legend()

            # plt.subplot(3, 3, 4)
            # x_train,x_test,y_train,y_test = train_test_split(np.array(SSB_RNA_seq_merge_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
            # model.fit(x_train,y_train)
            # if linear_model == 1 :
            #     pred_label = model.predict(x_test)
            #     print(pred_label)
            #     if is_age_label :
            #         pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
            #     else:
            #         pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
            #     pred_label = np.array(pred_label,dtype=np.int64)
            #     pred = to_categorical(pred_label)
            #     print(pred,y_test)
            #     to_plot_multi_SSB_RNA_seq_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
            #     to_save_result = 'RNA-seq&SSB :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(linear_auc)+'\n'
            #     to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&SSB' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
            #     to_save_metrics.append(to_save_metric)
            # else:
            #     pred  = model.predict_proba(x_test)
            #     pred_label = model.predict(x_test)
            #     pred = softmax(pred)
            #     pred_roc_label = to_categorical(pred_label)
            #     y_roc_test = to_categorical(y_test)
            #     to_plot_multi_SSB_RNA_seq_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
            #     to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&SSB' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
            #     to_save_metrics.append(to_save_metric)
            #     to_save_result ='RNA-seq&SSB:\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+ str(multi_auc)+'\n'
            # to_save_results.append(to_save_result)
            # # print(to_save_result)
            
            # # pred = to_categorical(pred)
            # y_test = to_categorical(y_test)
            # fpr = dict()
            # tpr = dict()
            # roc_auc = dict()
            # mean_fpr = []
            # mean_tpr = []
            # for i in range(len(n_classes)):
            #     fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
            #     print(fpr[i], tpr[i], thresholds)
            #     mean_fpr.append(fpr[i])
            #     mean_tpr.append(tpr[i])
            #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            #     plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
            #             label='RNA-seq&SSB curve of class {0} (AUC = {1:0.2f})'
            #             ''.format(i, roc_auc[i]),alpha = 0.2)
            # if linear_model == 1:
            #     fpr_length = max(len(sublst) for sublst in mean_fpr)
            #     new_mean_fpr = []
            #     new_mean_tpr = []
            #     for sublst in mean_fpr:
            #         x = np.linspace(sublst[0], sublst[-1], len(sublst))
            #         y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
            #         new_mean_fpr.append(y)
            #     tpr_length = max(len(sublst) for sublst in mean_tpr)
            #     for sublst in mean_tpr:
            #         x = np.linspace(sublst[0], sublst[-1], len(sublst))
            #         y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], tpr_length)]
            #         new_mean_tpr.append(y)
            #     mean_fpr = np.array(new_mean_fpr).mean(axis=0)
            #     mean_tpr = np.array(new_mean_tpr).mean(axis=0)
            #     mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
            #     plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='RNA-seq&SSB (AUC = %0.3f)'''%(mean_roc_auc))
            # else:                 
            # mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                # mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                # mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                # plt.plot(mean_fpr,mean_tpr, color='#f74d4d', lw=2,label='SSB (AUC = %0.3f)'''%(mean_roc_auc))
                # plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))

            # plt.title('RNA-seq&SSB validation ROC')
            # plt.plot([0,1],[0,1],'k--',label = 'base')
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.ylabel ('True Positive Rate')
            # plt.xlabel ('False Positive Rate')
            # plt.legend()

            # plt.subplot(3, 3, 5)
            # x_train,x_test,y_train,y_test = train_test_split(np.array(AP_RNA_seq_merge_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
            # model.fit(x_train,y_train)
            # if linear_model == 1 :
            #     pred_label = model.predict(x_test)
            #     if is_age_label :
            #         pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
            #     else:
            #         pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
            #     pred_label = np.array(pred_label,dtype=np.int64)
            #     pred = to_categorical(pred_label)
            #     to_plot_multi_AP_RNA_seq_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
            #     to_save_result = 'RNA-seq&AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(linear_auc)+'\n'
            #     to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
            #     to_save_metrics.append(to_save_metric)
            # else:
            #     pred  = model.predict_proba(x_test)
            #     pred_label = model.predict(x_test)
            #     pred = softmax(pred)
            #     pred_roc_label = to_categorical(pred_label)
            #     y_roc_test = to_categorical(y_test)
            #     to_plot_multi_AP_RNA_seq_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
            #     to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
            #     to_save_metrics.append(to_save_metric)
            #     to_save_result ='RNA-seq&AP:\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(multi_auc)+'\n'
            # to_save_results.append(to_save_result)
            # # print(to_save_result)
            # print("asdas",pred,y_test)    
            # # pred = to_categorical(pred)
            # y_test = to_categorical(y_test)
            # fpr = dict()
            # tpr = dict()
            # roc_auc = dict()
            # mean_fpr = []
            # mean_tpr = []
            # for i in range(len(n_classes)):
            #     fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
            #     print(fpr[i],tpr[i],thresholds)
            #     mean_fpr.append(fpr[i])
            #     mean_tpr.append(tpr[i])
            #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            #     plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
            #             label='RNA-seq&AP curve of class {0} (AUC = {1:0.2f})'
            #             ''.format(i, roc_auc[i]),alpha = 0.2)
            # if linear_model == 1:
            #     fpr_length = max(len(sublst) for sublst in mean_fpr)
            #     new_mean_fpr = []
            #     new_mean_tpr = []
            #     for sublst in mean_fpr:
            #         x = np.linspace(sublst[0], sublst[-1], len(sublst))
            #         y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
            #         new_mean_fpr.append(y)
            #     tpr_length = max(len(sublst) for sublst in mean_tpr)
            #     for sublst in mean_tpr:
            #         x = np.linspace(sublst[0], sublst[-1], len(sublst))
            #         y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], tpr_length)]
            #         new_mean_tpr.append(y)
            #     mean_fpr = np.array(new_mean_fpr).mean(axis=0)
            #     mean_tpr = np.array(new_mean_tpr).mean(axis=0)
            #     mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
            #     plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='RNA-seq&SSB (AUC = %0.3f)'''%(mean_roc_auc))
            # else:                 mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                # mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                # mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                # plt.plot(mean_fpr,mean_tpr, color='#f74d4d', lw=2,label='SSB (AUC = %0.3f)'''%(mean_roc_auc))
                # plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
            # plt.title('RNA-seq&AP validation ROC')
            # plt.plot([0,1],[0,1],'k--',label = 'base')
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.ylabel ('True Positive Rate')
            # plt.xlabel ('False Positive Rate')
            # plt.legend()

            # plt.subplot(3, 3, 6)
            # x_train,x_test,y_train,y_test = train_test_split(np.array(AP_SSB_merge_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
            # model.fit(x_train,y_train)
            # if linear_model == 1 :
            #     pred_label = model.predict(x_test)
            #     if is_age_label :
            #         pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
            #     else:
            #         pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
            #     pred_label = np.array(pred_label,dtype=np.int64)
            #     pred = to_categorical(pred_label)
            #     to_plot_multi_SSB_AP_seq_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
            #     to_save_result = 'SSB&AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(linear_auc)+'\n'
            #     to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'SSB&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
            #     to_save_metrics.append(to_save_metric)
            # else:
            #     pred  = model.predict_proba(x_test)
            #     pred_label = model.predict(x_test)
            #     pred = softmax(pred)
            #     pred_roc_label = to_categorical(pred_label)
            #     y_roc_test = to_categorical(y_test)
            #     to_plot_multi_SSB_AP_seq_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
            #     to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'SSB&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
            #     to_save_metrics.append(to_save_metric)
            #     to_save_result ='SSB&AP:\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(multi_auc)+'\n'
            # to_save_results.append(to_save_result)
            # # print(to_save_result)
            
            # # pred = to_categorical(pred)
            # y_test = to_categorical(y_test)
            # fpr = dict()
            # tpr = dict()
            # roc_auc = dict()
            # mean_fpr = []
            # mean_tpr = []
            # for i in range(len(n_classes)):
            #     fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
            #     mean_fpr.append(fpr[i])
            #     mean_tpr.append(tpr[i])
            #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            #     plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
            #             label='AP&SSB curve of class {0} (AUC = {1:0.2f})'
            #             ''.format(i, roc_auc[i]),alpha = 0.2)
            # if linear_model == 1:
            #     fpr_length = max(len(sublst) for sublst in mean_fpr)
            #     new_mean_fpr = []
            #     new_mean_tpr = []
            #     for sublst in mean_fpr:
            #         x = np.linspace(sublst[0], sublst[-1], len(sublst))
            #         y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
            #         new_mean_fpr.append(y)
            #     tpr_length = max(len(sublst) for sublst in mean_tpr)
            #     for sublst in mean_tpr:
            #         x = np.linspace(sublst[0], sublst[-1], len(sublst))
            #         y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], tpr_length)]
            #         new_mean_tpr.append(y)
            #     mean_fpr = np.array(new_mean_fpr).mean(axis=0)
            #     mean_tpr = np.array(new_mean_tpr).mean(axis=0)
            #     mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
            #     plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='AP&SSB (AUC = %0.3f)'''%(mean_roc_auc))
            # else:                 mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                # mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                # mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                # plt.plot(mean_fpr,mean_tpr, color='#f74d4d', lw=2,label='SSB (AUC = %0.3f)'''%(mean_roc_auc))
                # plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
            # plt.title('AP&SSB validation ROC')
            # plt.plot([0,1],[0,1],'k--',label = 'base')
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.ylabel ('True Positive Rate')
            # plt.xlabel ('False Positive Rate')
            # plt.legend()

            # plt.subplot(3, 3, 7)
            # x_train,x_test,y_train,y_test = train_test_split(np.array(ALL_merge_data),y_label,test_size =0.2,random_state=44,stratify=y_label)
            # model.fit(x_train,y_train)
            # if linear_model == 1 :
            #     pred_label = model.predict(x_test)
            #     if is_age_label :
            #         pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label)- np.min(pred_label)) *4
            #     else:
            #         pred_label = (pred_label - np.min(pred_label))/(np.max(pred_label) - np.min(pred_label)) * 5
            #     pred_label = np.array(pred_label,dtype=np.int64)
            #     pred = to_categorical(pred_label)
            #     to_plot_multi_all_auc = linear_auc = roc_auc_score(y_test, pred, multi_class='ovo')
            #     to_save_result = 'RNA-seq&SSB&AP :\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(linear_auc)+'\n'
            #     to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&SSB&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(linear_auc) + '\n'
            #     to_save_metrics.append(to_save_metric)
            # else:
            #     pred  = model.predict_proba(x_test)
            #     pred_label = model.predict(x_test)    
            #     pred = softmax(pred)
            #     pred_roc_label = to_categorical(pred_label)
            #     y_roc_test = to_categorical(y_test)
            #     to_plot_multi_all_auc = multi_auc = roc_auc_score(y_roc_test, pred_roc_label, multi_class='ovo')
            #     to_save_result ='RNA-seq&SSB&AP:\n'+classification_report(y_true=y_test,y_pred=pred_label,target_names=label_info) +'multi_AUC_result:'+str(multi_auc)+'\n'
            #     to_save_metric = svc + '\t' + str(selected_gene_num) + '\t' + 'RNA-seq&SSB&AP' + '\t' + str(accuracy_score(y_test, pred_label)) + '\t' + str(precision_score(y_test, pred_label, average='weighted')) + '\t' + str(recall_score(y_test, pred_label, average='weighted')) + '\t' + str(f1_score(y_test, pred_label, average='weighted')) + '\t' + str(multi_auc) + '\n'
            #     to_save_metrics.append(to_save_metric)
            # to_save_results.append(to_save_result)
            # # print(to_save_result)
            # if is_age_label == 0:
            #     output_result_file_path = './result/'+'classifier_tissue/selected_gene_'+str(selected_gene_num)+'/'
            # else :       
            #     output_result_file_path = './result/'+'classifier_age/selected_gene_'+str(selected_gene_num)+'/'
            # if not os.path.exists(output_result_file_path):
            #     os.makedirs(output_result_file_path)
            # output_result_file = open(output_result_file_path+svc+'.txt','w')
            # for res_ in to_save_results:
            #     output_result_file.write(res_)

            # if is_age_label == 0:
            #     output_result_file_path = './result/'+'classifier_metrics_tissue/selected_gene_'+str(selected_gene_num)+'/'
            # else :       
            #     output_result_file_path = './result/'+'classifier_metrics_age/selected_gene_'+str(selected_gene_num)+'/'
            # if not os.path.exists(output_result_file_path):
            #     os.makedirs(output_result_file_path)
            # output_result_file = open(output_result_file_path+svc+'.txt','w')
            # for res_ in to_save_metrics:
            #     output_result_file.write(res_)

            # # pred = to_categorical(pred)
            # y_test = to_categorical(y_test)

            # mean_fpr = []
            # mean_tpr = []
            # for i in range(len(n_classes)):
            #     fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)

            #     mean_fpr.append(fpr[i])
            #     mean_tpr.append(tpr[i])
            #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            #     plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
            #             label='RNA-seq&SSB&AP curve of class {0} (AUC = {1:0.2f})'
            #             ''.format(i, roc_auc[i]),alpha = 0.2)
            # if linear_model == 1:
            #     fpr_length = max(len(sublst) for sublst in mean_fpr)
            #     new_mean_fpr = []
            #     new_mean_tpr = []
            #     for sublst in mean_fpr:
            #         x = np.linspace(sublst[0], sublst[-1], len(sublst))
            #         y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], fpr_length)]
            #         new_mean_fpr.append(y)
            #     tpr_length = max(len(sublst) for sublst in mean_tpr)
            #     for sublst in mean_tpr:
            #         x = np.linspace(sublst[0], sublst[-1], len(sublst))
            #         y = [np.interp(i, x, sublst) for i in np.linspace(sublst[0], sublst[-1], tpr_length)]
            #         new_mean_tpr.append(y)
            #     mean_fpr = np.array(new_mean_fpr).mean(axis=0)
            #     mean_tpr = np.array(new_mean_tpr).mean(axis=0)
            #     mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
            #     plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='RNA-seq&SSB&AP (AUC = %0.3f)                '''%(mean_roc_auc))

            # else:    mean_fpr = np.array(new_mean_fpr).mean(axis=0)
                # mean_tpr = np.array(new_mean_tpr).mean(axis=0)
                # mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
                # plt.plot(mean_fpr,mean_tpr, color='#f74d4d', lw=2,label='SSB (AUC = %0.3f)'''%(mean_roc_auc))
                # plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
            # plt.title('ALL validation ROC')
            # plt.plot([0,1],[0,1],'k--',label = 'base')
            # plt.xlim([-0.05,1.05])
            # plt.ylim([-0.05,1.05])
            # plt.ylabel ('True Positive Rate')
            # plt.xlabel ('False Positive Rate')
            # plt.legend()
            
            # # plt.show()
            # return to_plot_multi_SSB_auc,to_plot_multi_AP_auc,to_plot_multi_RNA_seq_auc,to_plot_multi_SSB_AP_seq_auc,to_plot_multi_SSB_RNA_seq_auc,to_plot_multi_AP_RNA_seq_auc,to_plot_multi_all_auc,to_plot_multi_SSB_f1_score,to_plot_multi_AP_f1_score,to_plot_multi_RNA_seq_f1_score


#pipeline
def ana_pipeline():
    SSB_data,AP_data,RNA_seq_data,y_age,y_tissue = read_and_preprocess_data()
    # caculte_corrs(SSB_data,AP_data,RNA_seq_data),'DBSCAN'
    # clus_methods = ['PCA','TSNE','SpectralEmbedding','LDA','KMeans','KNeighborsClassifier']
    # colors = ['blue','yellow','green','orange','black','pink','m']
    colors = ['#f89588','#f74d4d','#B883D4','#76da91','#7898e1','#BEB8DC','#A1A9D0','#eddd86','#8ECFC9','#63b2ee','#943c39']

    linestyles = ['1','2','3','4','h','x','D']    
    svcs = ['ElasticNet','RandomForestClassifier','DecisionTreeRegressor','DecisionTreeClassifier','AdaBoostClassifier','LogisticRegression','LinearRegression','LGBMClassifier','XGBClassifier','Lasso']
    svcs = ['LogisticRegression','MultinomialNB','RandomForestClassifier','LGBMClassifier','XGBClassifier','ElasticNet','AdaBoostClassifier','DecisionTreeRegressor','Lasso','DecisionTreeClassifier']
    svcs = ['LogisticRegression']
    # svcs = ['DecisionTreeRegressor','DecisionTreeClassifier','AdaBoostClassifier','LGBMClassifier','XGBClassifier','Lasso']
    # svcs = ['Lasso','LogisticRegression','LinearRegression']
    # svcs = ['XGBClassifier','Lasso']
    # svcs = ['ElasticNet','RandomForestClassifier']
    # svcs = ['LGBMClassifier','AdaBoostClassifier']

    select_topN_gene_nums = [5,10,25,50,75,100,150,200,250,300,400,500]
    select_topN_gene_nums = [50,200,300,400,500]
    select_topN_gene_nums = [500]
    # SSB_data = SSB_data.T[:-12].T
    # AP_data = AP_data.T[:-12].T
    # RNA_seq_data = RNA_seq_data.T[:-12].T
    for y_label in [y_age,y_tissue]:
        # y_label = y_label[:-12]
        selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas = select_topN_genes('svc',SSB_data,AP_data,RNA_seq_data,y_label,500)
        # selected_SSB_datas = (selected_SSB_datas - selected_SSB_datas.mean(axis=0))/selected_SSB_datas.std(axis=0)
        # print(selected_SSB_datas.head)
        # selected_AP_datas =  (selected_AP_datas - selected_AP_datas.mean(axis=0))/selected_AP_datas.std(axis=0)*10
        AP_RNA_seq_merge_data,SSB_RNA_seq_merge_data,AP_SSB_merge_data,ALL_merge_data = gen_merge_data(selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas)
        print_analize_result_(svcs,selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas,y_label,select_topN_gene_nums,AP_RNA_seq_merge_data,SSB_RNA_seq_merge_data,AP_SSB_merge_data,ALL_merge_data)

    #ROC
if __name__ == "__main__":
    ana_pipeline()