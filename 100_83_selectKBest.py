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
from scipy import stats
import argparse

# args = argparse.
warnings.filterwarnings("ignore")

# 读取文件并合并
def read_and_preprocess_data():
    rna_seq_file_path = r'./RNA_data_sorted_tpm.csv'
    RNA_seq_data = pd.read_csv(rna_seq_file_path,header=0,index_col=0).T
    print(RNA_seq_data.shape)

    SSB_file_path = r'./New_SSB_data_sorted_tpm.csv'
    SSB_data = pd.read_csv(SSB_file_path,header=0,index_col=0).T
    ssb_columns = SSB_data.columns.tolist()
    AP_file_path = r'./New_AP_data_sorted_tpm.csv'

    AP_data = pd.read_csv(AP_file_path,header=0,index_col=0).T
    AP_columns = AP_data.columns.tolist()

    RNA_seq_data = RNA_seq_data[RNA_seq_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
    AP_data = AP_data[AP_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
    SSB_data = SSB_data[SSB_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()

    new_AP_columns = []
    for column in AP_columns:
        column = column.split('.')[0]
        new_AP_columns.append(column)
    print(new_AP_columns)
    RNA_seq_data.columns = new_AP_columns
    AP_data.columns =new_AP_columns
    SSB_data.columns = new_AP_columns



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
 

def select_topN_genes( y_label):
  
    if np.max(y_label) <=4:
        AP_data = pd.read_csv("./data_topN_features/selected_AP_data_filter_all.csv",header=0,index_col=0)
        SSB_data = pd.read_csv("./data_topN_features/selected_SSB_data_filter_all.csv",header=0,index_col=0)
        RNA_seq_data = pd.read_csv("./data_topN_features/selected_RNA_seq_data_filter_all.csv",header=0,index_col=0)
    else:
        AP_data = pd.read_csv("./data_topN_features/selected_AP_data_tissue_filter_all.csv",header=0,index_col=0)
        SSB_data = pd.read_csv("./data_topN_features/selected_SSB_data_tissue_filter_all.csv",header=0,index_col=0)
        RNA_seq_data = pd.read_csv("./data_topN_features/selected_RNA_seq_data_tissue_filter_all.csv",header=0,index_col=0)
 
    return AP_data,SSB_data,RNA_seq_data

font_dict=dict(fontsize=12,
              color='black',
              family='Arial',
              weight='light',
              )

def boxplot_roc_results():
    select_topN_gene_nums = [15,25,50,75,100,150,200,250,300,400,500,'all']
    svcs = ['ElasticNet','RandomForestClassifier','DecisionTreeRegressor','DecisionTreeClassifier','AdaBoostClassifier','LogisticRegression','GaussianNB','LGBMClassifier','XGBClassifier','Lasso']
    select_topN_gene_nums = [15,25,50,75,100,150,200,250,300,400,500,'all']
    import palettable
    labels = ["LPKM-SSB", "LPKM-AP", "TPM"]
    colors = [(202/255.,96/255.,17/255.), (255/255.,217/255.,102/255.), (137/255.,128/255.,68/255.)]
    for y_label in ["age",'tissue']:

        for svc in svcs:
            if y_label == "age":
                colors = palettable.tableau.Tableau_10.mpl_colors[0:3]
                output_models_results_path = './result_metrics_pictures_mssb_100_0.4_83/'+'age_model_results/'+str(svc)+'/'
                output_result_roc_boxplot_file_path = "./result_metrics_pictures_mssb_100_0.4_83/roc_box_age_plot_results/"
            else :
                colors = palettable.tableau.Tableau_10.mpl_colors[3:6]
                output_models_results_path = './result_metrics_pictures_mssb_100_0.4_83/'+'tissue_model_results/'+str(svc)+'/'
                output_result_roc_boxplot_file_path = "./result_metrics_pictures_mssb_100_0.4_83/roc_box_tissue_plot_results/"
            data_aucs_to_plot = []        
            for selected_gene in select_topN_gene_nums:
                data_rocs_path = output_models_results_path+"selectedTop_"+str(selected_gene)+"_100_model_results.csv"
                data_rocs = pd.read_csv(data_rocs_path,header=0,index_col=0)
                data_one_iter = []
                data_one_iter.append(data_rocs["ssb_auc"])
                data_one_iter.append(data_rocs["AP_auc"])
                data_one_iter.append(data_rocs["RNA_seq_auc"])
                data_aucs_to_plot.append(data_one_iter)

            #绘制箱型图
            plt.figure(figsize=(6,6))
            for idx in range(len(select_topN_gene_nums)):
                pos = idx+idx*0.5
                bplot = plt.boxplot(data_aucs_to_plot[idx], patch_artist=True,labels=labels,positions=(pos,pos+0.4,pos+0.8),widths=0.3) 
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)
            plt.xticks([i+0.4+i*0.5 for i in range(len(select_topN_gene_nums))],select_topN_gene_nums,fontdict=font_dict)
            plt.ylabel('AUC score',fontdict=font_dict)
            plt.xlabel("Number of genes",fontdict=font_dict)
            plt.title(str(svc),fontdict=font_dict)
            plt.grid(linestyle="--", alpha=0.5)  #绘制图中虚线 透明度0.3

            plt.legend(bplot['boxes'],labels,loc='upper right',fontsize=12,prop={'family':'Arial'})  #绘制表示框，右下角绘制
            if not os.path.exists(output_result_roc_boxplot_file_path):
                os.makedirs(output_result_roc_boxplot_file_path)
            plt.savefig(output_result_roc_boxplot_file_path+str(svc)+"rocs_box_plot.svg",dpi = 1080)  
            # plt.show()
            
def calculate_confidencial_interval():
    svcs = ['ElasticNet','RandomForestClassifier','DecisionTreeRegressor','DecisionTreeClassifier','AdaBoostClassifier','LogisticRegression','GaussianNB','LGBMClassifier','XGBClassifier','Lasso']
    select_topN_gene_nums = [15,25,50,75,100,150,200,250,300,400,500,'all']
    # select_topN_gene_nums = [15,25]

    lower_percentile, upper_percentile = [(100-95)/2, 100-(100-95)/2]
    with pd.ExcelWriter('exon_83_'+'age_tissue'+'confidentcial_results.xlsx') as writer:
        multi_index = pd.MultiIndex.from_product([["tissue",'age'],svcs,select_topN_gene_nums, ['SSB', 'AP', 'RNA_seq']])
        confidencial_interval_result = pd.DataFrame(columns=['CI lowwer','CI upper'], index=multi_index)
        for svc in svcs:
            for y_label in ["tissue",'age']:
                if y_label == "age":
                    output_result_confidencial_file_path = "./result_metrics_pictures_mssb_100_0.4_83/age_confidencial_interval_results/"
                else:
                    output_result_confidencial_file_path = "./result_metrics_pictures_mssb_100_0.4_83/tissue_confidencial_interval_results/"
                if not os.path.exists(output_result_confidencial_file_path):
                    os.makedirs(output_result_confidencial_file_path)
                if y_label == "age":
                    output_models_results_path = './result_metrics_pictures_mssb_100_0.4_83/'+'age_model_results/'+str(svc)+'/'
                else :
                    output_models_results_path = './result_metrics_pictures_mssb_100_0.4_83/'+'tissue_model_results/'+str(svc)+'/'
                for selected_gene in select_topN_gene_nums:
                    data_rocs_path = output_models_results_path+"selectedTop_"+str(selected_gene)+"_100_model_results.csv"
                    data_rocs = pd.read_csv(data_rocs_path,header=0,index_col=0)
                    ssb_rocs = data_rocs['ssb_auc']
                    ap_rocs = data_rocs['AP_auc']
                    rna_rocs = data_rocs['RNA_seq_auc']

                    ssb_sorted_rocs = np.sort(np.array(ssb_rocs))
                    ap_sorted_rocs = np.sort(np.array(ap_rocs))
                    rna_sorted_rocs = np.sort(np.array(rna_rocs))

                    ssb_lower_value = ssb_sorted_rocs[int(lower_percentile/100*len(ssb_sorted_rocs))]
                    ssb_upper_value = ssb_sorted_rocs[int(upper_percentile/100*len(ssb_sorted_rocs))]
                
                    ap_lower_value = ap_sorted_rocs[int(lower_percentile/100*len(ap_sorted_rocs))]
                    ap_upper_value = ap_sorted_rocs[int(upper_percentile/100*len(ap_sorted_rocs))]
                                    
                    rna_lower_value = rna_sorted_rocs[int(lower_percentile/100*len(rna_sorted_rocs))]
                    rna_upper_value = rna_sorted_rocs[int(upper_percentile/100*len(rna_sorted_rocs))]
                    # print(selected_gene,ssb_lower_value,ssb_upper_value)
                    confidencial_interval_result.loc[(y_label,str(svc),selected_gene,'SSB'), 'CI lowwer'] = ssb_lower_value
                    confidencial_interval_result.loc[(y_label,str(svc),selected_gene,'SSB'), 'CI upper'] = ssb_upper_value
                    confidencial_interval_result.loc[(y_label,str(svc),selected_gene,'SSB'), 'mean auc'] = ssb_rocs.mean()
                    confidencial_interval_result.loc[(y_label,str(svc),selected_gene,'AP'), 'CI lowwer'] = ap_lower_value
                    confidencial_interval_result.loc[(y_label,str(svc),selected_gene,'AP'), 'CI upper'] = ap_upper_value
                    confidencial_interval_result.loc[(y_label,str(svc),selected_gene,'AP'), 'mean auc'] = ap_rocs.mean()
                    confidencial_interval_result.loc[(y_label,str(svc),selected_gene,'RNA_seq'), 'CI lowwer'] = rna_lower_value
                    confidencial_interval_result.loc[(y_label,str(svc),selected_gene,'RNA_seq'), 'CI upper'] = rna_upper_value
                    confidencial_interval_result.loc[(y_label,str(svc),selected_gene,'RNA_seq'), 'mean auc'] = rna_rocs.mean()
            confidencial_interval_result.to_excel(writer)

def print_analize_result_(svc,selected_SSB_data,selected_AP_data,selected_RNA_seq_data,y_label,selected_gene_num,AP_RNA_seq_merge_data=None,SSB_RNA_seq_merge_data=None,AP_SSB_merge_data=None,ALL_merge_data=None,randomstate = None):
    #ROC
    n_classes = ['0', '1', '2', '3','4']
    # n_classes = ['0', '1']

    age_info = ['3','12','19','22','24']
    # age_info = ['3','12','19','22']
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
            model = Ridge(random_state = 42,max_iter=1000)
        elif svc == 'Lasso':
            model = Lasso(random_state = 42,max_iter=1000)
        elif svc == 'ElasticNet':
            model = ElasticNet(random_state = 42,max_iter=1000)
        elif svc == 'LogisticRegression':
            model = LogisticRegression(multi_class='multinomial',verbose=0,random_state=42)
        elif svc == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=42,min_samples_split=2,max_features="auto")
        elif svc == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(criterion='gini', splitter='best',min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=42,min_impurity_decrease=0.0)
        # elif svc == 'LGBMRegressor':
        #     model = DecisionTreeClassifier(criterion='gini', splitter='best',min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=42,min_impurity_decrease=0.0)

    else:
        decisionTree_model = 1
        if svc == 'RandomForestClassifier':
            model = RandomForestClassifier(n_estimators=1000,max_depth=2, random_state=0)
        elif svc == 'XGBClassifier':
            model = XGBClassifier( booster='gbtree',objective='multi:softmax',random_state=42,learning_rate=0.1,n_jobs=-1,n_estimators=1000)
        elif svc == 'GaussianNB':
            model = GaussianNB(var_smoothing=1e-8)
        elif svc == 'LGBMClassifier':
            model = LGBMClassifier(n_estimators=1000,objective="multiclass")
        elif svc == 'AdaBoostClassifier':
            model = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1, algorithm='SAMME.R', random_state=42)
    to_save_results =[]
    to_save_metrics =[]
    x_train,x_test,y_train,y_test = train_test_split(np.array(selected_SSB_data),y_label,test_size=0.2,random_state=randomstate,stratify=y_label)
    model.fit(x_train,y_train)
    # plt.subplots_adjust(top=0.85)
    # fig = plt.figure(figsize=(20, 20)) 
    # plt.subplot(3, 3, 1)    
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

    # for i in range(len(n_classes)):
    #     # print(y_test[:, i],pred[:, i])
    #     fpr[i],  tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
    #     # print(fpr[i], tpr[i],thresholds)
    #     mean_fpr.append(fpr[i])
    #     mean_tpr.append(tpr[i])
    #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    #     # plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
    #     #         label='SSB curve of class {0} (AUC = {1:0.2f})'
    #     #         ''.format(i, roc_auc[i]),alpha = 0.2)
    # if linear_model == 0 or decisionTree_model == 1:
    #     mean_fpr = np.array(mean_fpr).mean(axis=0)
    #     mean_tpr = np.array(mean_tpr).mean(axis=0)
    #     mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
    #     plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='SSB curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))
    # else:    
    #     plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
        # print(mean_fpr)
        
    # plt.title('SSB validation ROC')
    # plt.plot([0,1],[0,1],'k--',label = 'base')
    # plt.xlim([-0.05,1.05])
    # plt.ylim([-0.05,1.05])
    # plt.ylabel ('True Positive Rate')
    # plt.xlabel ('False Positive Rate')
    # plt.legend()

    # plt.subplot(3, 3, 2)
    x_train,x_test,y_train,y_test = train_test_split(np.array(selected_AP_data),y_label,test_size =0.2,random_state=randomstate,stratify=y_label)
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
    # for i in range(len(n_classes)):
    #     fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
    #     # print(fpr[i], tpr[i], thresholds)
    #     mean_fpr.append(fpr[i])
    #     mean_tpr.append(tpr[i])
    #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        # plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
        #         label='AP curve of class {0} (AUC = {1:0.2f})'
        #         ''.format(i, roc_auc[i]),alpha = 0.2)
    # print(mean_fpr)
    # if linear_model == 0 or decisionTree_model == 1:
    #     mean_fpr = np.array(mean_fpr).mean(axis=0)
    #     mean_tpr = np.array(mean_tpr).mean(axis=0)
    #     mean_roc_auc = metrics.auc(mean_fpr,mean_tpr)
    #     plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='AP curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))
    # else:    
    #     plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
    # plt.title('AP validation ROC')
    # plt.plot([0,1],[0,1],'k--',label = 'base')
    # plt.xlim([-0.05,1.05])
    # plt.ylim([-0.05,1.05])
    # plt.ylabel ('True Positive Rate')
    # plt.xlabel ('False Positive Rate')
    # plt.legend()

    # plt.subplot(3, 3, 3)
    x_train,x_test,y_train,y_test = train_test_split(np.array(selected_RNA_seq_data),y_label,test_size =0.2,random_state=randomstate,stratify=y_label)
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
    # print(to_save_result)
    
    # pred = to_categorical(pred)
    y_test = to_categorical(y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_fpr = []
    mean_tpr = []
    # for i in range(len(n_classes)):
    #     fpr[i], tpr[i], thresholds = metrics.roc_curve(y_test[:, i],pred[:, i],drop_intermediate=False)
    #     mean_fpr.append(fpr[i])
    #     mean_tpr.append(tpr[i])
    #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    #     # plt.plot(fpr[i], tpr[i], color=colors[int(i)], lw=2,
    #     #         label='RNA-seq curve of class {0} (AUC = {1:0.2f})'
    #     #         ''.format(i, roc_auc[i]),alpha = 0.2)
    # if linear_model == 0 or decisionTree_model == 1:
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
    #     plt.plot(mean_fpr,mean_tpr, color='pink', lw=2,label='RNA-seq curve of Mean_classes (AUC = {%0.6f})'''%(mean_roc_auc))
    # else:    
    #     plt.plot(np.linspace(0,1,20),[linear_auc]*20,'c-',label = 'mean_AUC= {%0.6f}'%(linear_auc))
    # plt.title('RNA-seq validation ROC')
    # plt.plot([0,1],[0,1],'k--',label = 'base')
    # plt.xlim([-0.05,1.05])
    # plt.ylim([-0.05,1.05])
    # plt.ylabel ('True Positive Rate')
    # plt.xlabel ('False Positive Rate')
    # plt.legend()

    return to_plot_multi_SSB_auc,to_plot_multi_AP_auc,to_plot_multi_RNA_seq_auc,to_plot_multi_SSB_f1_score,to_plot_multi_AP_f1_score,to_plot_multi_RNA_seq_f1_score


#pipeline
def ana_pipeline():
    SSB_data,AP_data,RNA_seq_data,y_age,y_tissue = read_and_preprocess_data()
    colors = ['#f89588','#f74d4d','#B883D4','#76da91','#7898e1','#BEB8DC','#A1A9D0','#eddd86','#8ECFC9','#63b2ee','#943c39']

    linestyles = ['1','2','3','4','h','x','D']    
    svcs = ['ElasticNet','RandomForestClassifier','DecisionTreeRegressor','DecisionTreeClassifier','AdaBoostClassifier','LogisticRegression','GaussianNB','LGBMClassifier','XGBClassifier','Lasso']
    # svcs = ['ElasticNet','RandomForestClassifier','DecisionTreeRegressor','DecisionTreeClassifier','AdaBoostClassifier','LogisticRegression','GaussianNB','LGBMClassifier']
    svcs = ['XGBClassifier']
    select_topN_gene_nums = [15,25,50,75,100,150,200,250,300,400,500,'all']
    for y_label in [y_age,y_tissue]:
        # y_label = y_label[:-12]            
        for svc in svcs:
            if np.max(y_label)<=4 :
                output_models_results_path = './result_metrics_pictures_mssb_100_0.4_83_xgb1000/'+'age_model_results/'+str(svc)+'/'
            else :
                output_models_results_path = './result_metrics_pictures_mssb_100_0.4_83_xgb1000/'+'tissue_model_results/'+str(svc)+'/'
            if not os.path.exists(output_models_results_path):
                os.makedirs(output_models_results_path)
            print("processing method:",svc)

            selected_AP_datas,selected_SSB_datas,selected_RNA_seq_datas = select_topN_genes(y_label)
            to_plot_multi_mean_SSB_aucs = []
            to_plot_multi_mean_AP_aucs = []
            to_plot_multi_mean_RNA_seq_aucs = [] 
            to_plot_multi_mean_SSB_f1_scores = []
            to_plot_multi_mean_AP_f1_scores = []
            to_plot_multi_mean_RNA_seq_f1_scores = []  
            for selected_gene in select_topN_gene_nums:
                iters_result =pd.DataFrame()
                seeds = []                   
                to_plot_multi_SSB_aucs = []
                to_plot_multi_AP_aucs = []
                to_plot_multi_RNA_seq_aucs = []
                to_plot_multi_SSB_f1_scores = []
                to_plot_multi_AP_f1_scores = []
                to_plot_multi_RNA_seq_f1_scores = []    
                for iter_i in range(100):
                    import time 
                    seed = int(time.time())+100*iter_i+(3*iter_i)**2
                    seeds.append(seed)
                    if isinstance(selected_gene,int):
                        selected_SSB_data = selected_SSB_datas.T[:selected_gene].T
                        print(selected_SSB_data.shape)
                        selected_AP_data = selected_AP_datas.T[:selected_gene].T
                        selected_RNA_seq_data = selected_RNA_seq_datas.T[:selected_gene].T
                    else :
                        selected_SSB_data,selected_AP_data,selected_RNA_seq_data = selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas
                    to_plot_multi_SSB_auc,to_plot_multi_AP_auc,to_plot_multi_RNA_seq_auc,to_plot_multi_SSB_f1_score,to_plot_multi_AP_f1_score,to_plot_multi_RNA_seq_f1_score\
                        = print_analize_result_(svc,selected_SSB_data,selected_AP_data,selected_RNA_seq_data,y_label,selected_gene,randomstate=seed)
                    to_plot_multi_SSB_aucs.append(to_plot_multi_SSB_auc)
                    to_plot_multi_AP_aucs.append(to_plot_multi_AP_auc)
                    to_plot_multi_RNA_seq_aucs.append(to_plot_multi_RNA_seq_auc)

                    to_plot_multi_SSB_f1_scores.append(to_plot_multi_SSB_f1_score)
                    to_plot_multi_AP_f1_scores.append(to_plot_multi_AP_f1_score)
                    to_plot_multi_RNA_seq_f1_scores.append(to_plot_multi_RNA_seq_f1_score)
                    iters_result = iters_result.append({"seed":seed,"ssb_auc":to_plot_multi_SSB_auc,"AP_auc":to_plot_multi_AP_auc,"RNA_seq_auc":to_plot_multi_RNA_seq_auc,"ssb_f1_score":to_plot_multi_SSB_f1_score,"AP_f1_score":to_plot_multi_AP_f1_score,"RNA_seq_f1_score":to_plot_multi_RNA_seq_f1_score},ignore_index=True)
                iters_result.to_csv(output_models_results_path+"selectedTop_"+str(selected_gene)+"_100_model_results.csv")
                to_plot_multi_mean_SSB_aucs.append(np.array(to_plot_multi_SSB_aucs).mean())
                to_plot_multi_mean_AP_aucs.append(np.array(to_plot_multi_AP_aucs).mean())
                to_plot_multi_mean_RNA_seq_aucs.append(np.array(to_plot_multi_RNA_seq_aucs).mean())
                to_plot_multi_mean_SSB_f1_scores.append(np.array(to_plot_multi_SSB_f1_scores).mean())
                to_plot_multi_mean_AP_f1_scores.append(np.array(to_plot_multi_AP_f1_scores).mean())
                to_plot_multi_mean_RNA_seq_f1_scores.append(np.array(to_plot_multi_RNA_seq_f1_scores).mean())
            plt.clf()
            plt.figure(figsize=(6, 6))
            plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_mean_SSB_aucs,color=colors[int(0)],marker = linestyles[int(0)],lw=2,label='SSB multi_auc curve ')
            plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_mean_AP_aucs,color=colors[int(1)],marker = linestyles[int(1)],lw=2,label='AP multi_auc curve ')
            plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_mean_RNA_seq_aucs,color=colors[int(2)],marker = linestyles[int(2)],lw=2,label='RNA-seq multi_auc curve ')
            plt.xlabel('selected_gene_num')
            plt.ylabel('multi_AUC_score')
            plt.ylim(-0.05,1.05)
            plt.xticks(np.arange(len(select_topN_gene_nums)),select_topN_gene_nums)
            plt.title(svc)
            plt.legend()
            if np.max(y_label)<=4 :
                output_result_picture_file_path = './result_metrics_pictures_mssb_100_0.4_83_xgb1000/'+'age/'
            else :
                output_result_picture_file_path = './result_metrics_pictures_mssb_100_0.4_83_xgb1000/'+'tissue/'
            if not os.path.exists(output_result_picture_file_path):
                os.makedirs(output_result_picture_file_path)
            plt.savefig(output_result_picture_file_path+svc+'.jpg',dpi = 400,format = 'jpg')

            plt.clf()
            plt.figure(figsize=(6,6))
            plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_mean_AP_f1_scores,color=colors[int(-2)],marker = linestyles[int(1)],lw=2,label='AP multi_f1_score curve ')
            plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_mean_RNA_seq_f1_scores,color=colors[int(-3)],marker = linestyles[int(2)],lw=2,label='RNA-seq multi_f1_score curve ')
            plt.plot(np.arange(len(select_topN_gene_nums)),to_plot_multi_mean_SSB_f1_scores,color=colors[int(-1)],marker = linestyles[int(4)],lw=2,label='SSB multi_f1_score curve ')
            plt.xlabel('selected_gene_num')
            plt.ylabel('multi_f1_score')
            plt.xticks(np.arange(len(select_topN_gene_nums)),select_topN_gene_nums)
            plt.title(svc)
            plt.legend()
            if np.max(y_label)<=4 :
                output_result_picture_file_path = './result_metrics_pictures_mssb_100_0.4_83_xgb1000/'+'age/'
            else :
                output_result_picture_file_path = './result_metrics_pictures_mssb_100_0.4_83_xgb1000/'+'tissue/'
            if not os.path.exists(output_result_picture_file_path):
                os.makedirs(output_result_picture_file_path)
            plt.savefig(output_result_picture_file_path+svc+'_f1_score.png',dpi = 400,format = 'png')

if __name__ == "__main__":
    # ana_pipeline()
    boxplot_roc_results()
    # calculate_confidencial_interval()