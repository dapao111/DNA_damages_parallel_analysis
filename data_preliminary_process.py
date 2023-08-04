import pandas as pd 
import numpy as np
import os 

def read_and_preprocess_data(RNA_seq_file_path,SSB_file_path,AP_file_path):
    # rna_seq_file_path = r'./RNA_data_sorted_tpm.csv'
    RNA_seq_data = pd.read_csv(RNA_seq_file_path,header=0,index_col=0).T
    print(RNA_seq_data.shape)

    # SSB_file_path = r'./New_SSB_data_sorted_tpm.csv'
    SSB_data = pd.read_csv(SSB_file_path,header=0,index_col=0).T
    SSB_columns = SSB_data.columns.tolist()
    # AP_file_path = r'./New_AP_data_sorted_tpm.csv'

    AP_data = pd.read_csv(AP_file_path,header=0,index_col=0).T
    AP_columns = AP_data.columns.tolist()

    RNA_seq_data = RNA_seq_data[RNA_seq_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
    AP_data = AP_data[AP_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()
    SSB_data = SSB_data[SSB_data.apply(lambda x: (x.max()>1) & ((x == 0).sum()/len(x) <= 0.4), axis=1)].dropna()

    new_ssb_columns = []
    for column in AP_columns:
        column = column.split('.')[0]
        new_ssb_columns.append(column)
    RNA_seq_data.columns = new_ssb_columns
    AP_data.columns =new_ssb_columns
    SSB_data.columns = new_ssb_columns


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

    # print(y_age,y_tissue,SSB_data.shape,AP_data.shape,RNA_seq_data.shape)
    return SSB_data,AP_data,RNA_seq_data,y_age,y_tissue

# ===========f-selection(variance analysis) -select TopN========================
def select_topN_genes(svc,SSB_data, AP_data,RNA_seq_data, y_label,output_file,saveDataFlag = 0,feature_numbers=500):
  
    from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif,chi2,f_regression,r_regression,mutual_info_regression
    
    #adopt f_classify parameters to fit a variance analysis model for multi-age-classification
    model = SelectKBest(f_classif, k=feature_numbers)

    # AP part Top gene selection
    selected_AP_data = model.fit_transform(AP_data.T, y_label)

    AP_scores = model.scores_
    AP_p_values = model.pvalues_
    indices = np.argsort(AP_scores)[::-1]
    AP_names = set(model.get_feature_names_out(AP_data.T.columns.tolist()))

    if feature_numbers == 'all':
        selected_AP_data = AP_data.T.iloc[:,indices]
    else:
        selected_AP_data = AP_data.T.iloc[:,indices[:feature_numbers]]

    # SSB part Top gene selection
    selected_SSB_data = model.fit_transform(SSB_data.T , y_label)
    SSB_scores = model.scores_

    SSB_p_values = model.pvalues_
    indices = np.argsort(SSB_scores)[::-1]
    # k_best_features = list(SSB_data.T.columns.values[indices])
    SSB_names = set(model.get_feature_names_out(SSB_data.T.columns.tolist()))
    # k_best_features = set(k_best_features)
    if feature_numbers == 'all':
        selected_SSB_data = SSB_data.T.iloc[:,indices]
    else:
        selected_SSB_data = SSB_data.T.iloc[:,indices[:feature_numbers]]
    model = SelectKBest(f_classif, k=feature_numbers)

    #RNA_seq part Top gene selection
    selected_RNA_seq_data = model.fit_transform(RNA_seq_data.T, y_label)

    RNA_scores = model.scores_
    RNA_p_values = model.pvalues_
    indices = np.argsort(RNA_p_values)
    indices = np.argsort(RNA_scores)[::-1]
    # k_best_features = list(RNA_seq_data.T.columns.values[indices])
    RNA_names = set(model.get_feature_names_out(RNA_seq_data.T.columns.tolist()))
    # k_best_features = set(k_best_features)


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

    # topN_save_path = './data_topN_features/genes_metrics'
    if saveDataFlag:
        topN_save_path_genes_metrics = output_file+'/genes_metrics'

        if not os.path.exists(topN_save_path_genes_metrics):
            os.makedirs(topN_save_path_genes_metrics)
        if np.max(y_label) >4:
            AP_feature_importance_df.to_csv(topN_save_path_genes_metrics+'/selected_AP_data_tissue_metrics'+str(feature_numbers)+'.csv')
            SSB_feature_importance_df.to_csv(topN_save_path_genes_metrics+'/selected_SSB_data_tissue_metrics'+str(feature_numbers)+'.csv')
            RNA_feature_importance_df.to_csv(topN_save_path_genes_metrics+'/selected_RNA_seq_data_tissue_metrics'+str(feature_numbers)+'.csv')
        else :
            AP_feature_importance_df.to_csv(topN_save_path_genes_metrics+'/selected_AP_data_metrics'+str(feature_numbers)+'.csv')
            SSB_feature_importance_df.to_csv(topN_save_path_genes_metrics+'/selected_SSB_data_metrics'+str(feature_numbers)+'.csv')
            RNA_feature_importance_df.to_csv(topN_save_path_genes_metrics+'/selected_RNA_seq_data_metrics'+str(feature_numbers)+'.csv')
    
        topN_save_path_genes_data = output_file+'/genes_TopN_data/'
        if not os.path.exists(topN_save_path_genes_data):
            os.makedirs(topN_save_path_genes_data)
        if np.max(y_label) >4:
            if selected_AP_data.shape[0]==83:
                selected_AP_data.to_csv(topN_save_path_genes_data+'selected_AP_data_tissue_filter_'+str(feature_numbers)+'.csv')
                selected_SSB_data.to_csv(topN_save_path_genes_data+'selected_SSB_data_tissue_filter_'+str(feature_numbers)+'.csv')
                selected_RNA_seq_data.to_csv(topN_save_path_genes_data+'selected_RNA_seq_data_tissue_filter_'+str(feature_numbers)+'.csv')
            else:
                selected_AP_data.to_csv(topN_save_path_genes_data+'selected_AP_data_tissue_filter_71_'+str(feature_numbers)+'.csv')
                selected_SSB_data.to_csv(topN_save_path_genes_data+'selected_SSB_data_tissue_filter_71_'+str(feature_numbers)+'.csv')
                selected_RNA_seq_data.to_csv(topN_save_path_genes_data+'selected_RNA_seq_data_tissue_filter_71_'+str(feature_numbers)+'.csv')
        else :
            if selected_AP_data.shape[0]==83:
                selected_AP_data.to_csv(topN_save_path_genes_data+'selected_AP_data_filter_'+str(feature_numbers)+'.csv')
                selected_SSB_data.to_csv(topN_save_path_genes_data+'selected_SSB_data_filter_'+str(feature_numbers)+'.csv')
                selected_RNA_seq_data.to_csv(topN_save_path_genes_data+'selected_RNA_seq_data_filter_'+str(feature_numbers)+'.csv')
            else:
                selected_AP_data.to_csv(topN_save_path_genes_data+'selected_AP_data_filter_71_'+str(feature_numbers)+'.csv')
                selected_SSB_data.to_csv(topN_save_path_genes_data+'selected_SSB_data_filter_71_'+str(feature_numbers)+'.csv')
                selected_RNA_seq_data.to_csv(topN_save_path_genes_data+'selected_RNA_seq_data_filter_71_'+str(feature_numbers)+'.csv')

    return selected_AP_data,selected_SSB_data,selected_RNA_seq_data

def read_topN_genes(dataPath,ageFlag):

    if ageFlag:
        AP_data = pd.read_csv(dataPath+"/selected_AP_data_filter_all.csv",header=0,index_col=0)
        SSB_data = pd.read_csv(dataPath+"/selected_SSB_data_filter_all.csv",header=0,index_col=0)
        RNA_seq_data = pd.read_csv(dataPath+"/selected_RNA_seq_data_filter_all.csv",header=0,index_col=0)
    else:
        AP_data = pd.read_csv(dataPath+"/selected_AP_data_tissue_filter_all.csv",header=0,index_col=0)
        SSB_data = pd.read_csv(dataPath+"/selected_SSB_data_tissue_filter_all.csv",header=0,index_col=0)
        RNA_seq_data = pd.read_csv(dataPath+"/selected_RNA_seq_data_tissue_filter_all.csv",header=0,index_col=0)
 
    return AP_data,SSB_data,RNA_seq_data


# def pre_process(SSB_file_path,RNA_seq_file_path,AP_file_path):
    SSB_file_path = r'./New_ssb_data_tpm.csv'
    SSB_data = pd.read_csv(SSB_file_path,header=0,index_col=0)
    SSB_data.columns = SSB_data.columns.str.replace("bone_","")
    # SSB_data = SSB_data.apply(lambda row: row[row < row.max()] if row.max() < 50 else row, axis=1).dropna()
    ssb_columns = SSB_data.columns.tolist()

    AP_file_path = r'./New_AP_data_tpm.csv'
    AP_data = pd.read_csv(AP_file_path,header=0,index_col=0)
    # AP_data = AP_data.apply(lambda row: row[row < row.max()] if row.max() < 50 else row, axis=1).dropna()
    AP_columns = AP_data.columns.tolist()

    RNA_seq_file_path = r'./RNA_seq_data.xlsx'
    RNA_seq_data = pd.read_excel(RNA_seq_file_path,header=0,index_col=0)
    RNA_seq_columns = RNA_seq_data.columns.tolist()
    RNA_seq_data = RNA_seq_data.iloc[0:,1:]

    #filter by label
    import re
    # pattern_SSB_num = re.compile(r'X\d+\.\w\.')
    pattern_num = re.compile(r'\d+\-')
    pattern_alph = re.compile(r'\d+\-\w\-')
    pattern_SSB_num = re.compile(r'\.\d+\w\.\w')
    pattern_SSB_alpha = re.compile(r'SSB\.\w+\.')
    RNA_seq_sorted_columns = sorted([c for c in RNA_seq_columns if pattern_alph.search(c)], key=lambda x: (int(pattern_alph.search(x).group().split('-')[0]),pattern_alph.search(x).group().split('-')[1]))
    ssb_sorted_columns = sorted([c for c in ssb_columns if pattern_SSB_num.search(c)], key=lambda x: (int(pattern_SSB_num.search(x).group().split('.')[1][:-1]),pattern_SSB_alpha.search(x).group().split('.')[1][0].upper()))
    AP_sorted_columns = sorted([c for c in AP_columns if pattern_alph.search(c)], key=lambda x: (int(pattern_alph.search(x).group().split('-')[0]),pattern_alph.search(x).group().split('-')[1][0]))

    # print(ssb_sorted_columns)
    with open("labels.txt","w") as f:
        for i,j in enumerate(ssb_sorted_columns):
            if i <83:
                f.write(j+'\t'+RNA_seq_sorted_columns[i]+'\t'+AP_sorted_columns[i]+'\n') 
    # print(RNA_seq_sorted_columns[-1],RNA_seq_sorted_columns[-2])
    SSB_data = SSB_data.sort_index(axis=1, ascending=True)
    AP_data = AP_data.sort_index(axis=1,ascending=True)
    RNA_seq_data = RNA_seq_data.sort_index(axis=1,ascending=True)

    #re-arrange the samples
    SSB_data = SSB_data[ssb_sorted_columns].T
    AP_data = AP_data[AP_sorted_columns].T
    RNA_seq_data = RNA_seq_data[RNA_seq_sorted_columns].T

    #delete the sample not common having 
    SSB_data = SSB_data.drop(SSB_data.index[54],axis=0)
    RNA_seq_data= RNA_seq_data.drop(RNA_seq_data.index[54],axis=0)

    AP_data.to_csv("New_AP_data_sorted_tpm.csv")
    SSB_data.to_csv("New_SSB_data_sorted_tpm.csv")
    RNA_seq_data.to_csv("RNA_data_sorted_tpm.csv")


