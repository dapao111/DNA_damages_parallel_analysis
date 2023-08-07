import argparse
from argparse import RawTextHelpFormatter
import warnings
from data_preliminary_process import read_and_preprocess_data,select_topN_genes
from AnalysisBases import TopN_genes_age_tissue_correlation_analysis,TopN_genes_correlation_analysis,parallel_N_iters_results,print_ROC_result,calculate_confidencial_interval,boxplot_roc_results

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-sf", "--SSB_file", help = "SSB expression file(csv) ", required=False,default='./data/New_SSB_data_sorted_tpm.csv')
parser.add_argument("-pf", "--AP_file", help = "AP expression file(csv) ", required=False,default='./data/New_AP_data_sorted_tpm.csv')
parser.add_argument("-rf", "--RNA_file", help = "RNA expression file(csv)", required=False,default='./data/RNA_data_sorted_tpm.csv')
parser.add_argument("-o", "--output_file", help = "output_file_path", required=False,default='./DataTopnGenesFeatures')
parser.add_argument("-fn", "--feature_num", help = "selected_Top_num_genes", required=False,default=500)
parser.add_argument("-in", "--iter_num", help = "iter_numbers", required=False,default=100)
parser.add_argument("-caa", "--caa_flag", help = "correlation_analysis_with_age", required=False,default=1)
parser.add_argument("-cat", "--cat_flag", help = "correlation_analysis_with_age_tissue", required=False,default=1)
parser.add_argument("-rpf", "--rocplot_flag", help = "correlation_analysis_with_age_tissue", required=False,default=0)
parser.add_argument("-ccf", "--confidencial_cal_flag", help = "calculate_confidencial_interval", required=False,default=0)

parser.add_argument("-s", "--seed", help = "random_seed", required=False)

warnings.filterwarnings("ignore")

args = parser.parse_args()



#pipeline
def ana_pipeline():

    RNA_seq_file_path = args.RNA_file
    SSB_file_path = args.SSB_file
    AP_file_path = args.AP_file
    out_file = args.output_file
    iters_number = int(args.iter_num)
    features_num = args.feature_num 

    SSB_data,AP_data,RNA_seq_data,y_age,y_tissue = read_and_preprocess_data(RNA_seq_file_path,SSB_file_path,AP_file_path)
    if isinstance(features_num,str):
        if features_num[0] >'0' and features_num[0]<='9':
            features_num = int(features_num)
        else:
            print("please check your -n feature_num parameter,make sure it is integer or \"all\"")
    for y_label in [y_age,y_tissue]:
        save_selected_data_metrics_flag = 1
        selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas = select_topN_genes('svc',SSB_data,AP_data,RNA_seq_data,y_label,out_file,save_selected_data_metrics_flag,features_num)
        print(selected_SSB_datas.shape,selected_RNA_seq_datas.shape)

        if args.cat_flag:
            TopN_genes_age_tissue_correlation_analysis(y_label,selected_AP_datas,selected_SSB_datas,selected_RNA_seq_datas,out_file)
        if args.caa_flag:
            TopN_genes_correlation_analysis(y_label,selected_AP_datas,selected_SSB_datas,selected_RNA_seq_datas,out_file)

    parallel_N_iters_results(y_age,y_tissue,out_file,iters_number,features_num)

    if args.rocplot_flag:
        print_ROC_result(y_age,y_tissue,features_num,out_file)

    if args.confidencial_cal_flag:
        calculate_confidencial_interval(out_file)

if __name__ == "__main__":
    ana_pipeline()