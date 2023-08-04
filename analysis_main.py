import argparse
from argparse import RawTextHelpFormatter
import warnings
from data_preliminary_process import read_and_preprocess_data
from AnalysisBases import select_topN_genes,TopN_genes_age_tissue_correlation_analysis,TopN_genes_correlation_analysis,parallel_100_iters_results,print_ROC_result,calculate_confidencial_interval

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-sf", "--SSB_file", help = "SSB expression file(csv) ", required=False,default='./data/New_SSB_data_sorted_tpm.csv')
parser.add_argument("-pr", "--AP_file", help = "AP expression file(csv) ", required=False,default='./data/New_AP_data_sorted_tpm.csv')
parser.add_argument("-rf", "--RNA_file", help = "RNA expression file(csv)", required=False,default='./data/RNA_data_sorted_tpm.csv')
parser.add_argument("-o", "--output_file", help = "output_file_path", required=False,default='./DataTopnGenesFeatures')
parser.add_argument("-caa", "--caa_flag", help = "correlation_analysis_with_age", required=False,default=1)
parser.add_argument("-cat", "--cat_flag", help = "correlation_analysis_with_age_tissue", required=False,default=1)
parser.add_argument("-rp", "--rocplot_flag", help = "correlation_analysis_with_age_tissue", required=False,default=0)
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

    SSB_data,AP_data,RNA_seq_data,y_age,y_tissue = read_and_preprocess_data(RNA_seq_file_path,SSB_file_path,AP_file_path)

    for y_label in [y_age,y_tissue]:
        save_selected_data_metrics_flag = 1 
        selected_SSB_datas,selected_AP_datas,selected_RNA_seq_datas = select_topN_genes('svc',SSB_data,AP_data,RNA_seq_data,y_label,out_file,save_selected_data_metrics_flag ,"all")
        if args.cat:
            TopN_genes_age_tissue_correlation_analysis(y_label,selected_AP_datas,selected_SSB_datas,selected_RNA_seq_datas,out_file)
        if args.caa:
            TopN_genes_correlation_analysis(y_label,selected_AP_datas,selected_SSB_datas,selected_RNA_seq_datas,out_file)

        parallel_100_iters_results(y_age,y_tissue,out_file)

        if args.rocplot_flag:
            print_ROC_result(y_age,y_tissue,out_file)

        if args.confidencial_cal_flag:
            calculate_confidencial_interval(out_file)


if __name__ == "__main__":
    ana_pipeline()