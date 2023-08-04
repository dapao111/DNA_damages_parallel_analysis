import pandas as pd
#导入文件
SSB_file_path = r'./New_ssb_data_tpm.csv'
SSB_data = pd.read_csv(SSB_file_path,header=0,index_col=0)
SSB_data.columns = SSB_data.columns.str.replace("bone_","")
# SSB_data = SSB_data.apply(lambda row: row[row < row.max()] if row.max() < 50 else row, axis=1).dropna()
ssb_columns = SSB_data.columns.tolist()

AP_file_path = r'./New_AP_data_tpm.csv'
AP_data = pd.read_csv(AP_file_path,header=0,index_col=0)
# AP_data = AP_data.apply(lambda row: row[row < row.max()] if row.max() < 50 else row, axis=1).dropna()
AP_columns = AP_data.columns.tolist()
print(AP_data.head(5))

RNA_seq_file_path = r'./RNA_seq_data.xlsx'
RNA_seq_data = pd.read_excel(RNA_seq_file_path,header=0,index_col=0)
RNA_seq_columns = RNA_seq_data.columns.tolist()
RNA_seq_data = RNA_seq_data.iloc[0:,1:]
print(RNA_seq_data.shape,AP_data.shape,SSB_data.shape)

#筛选标签
import re
# pattern_SSB_num = re.compile(r'X\d+\.\w\.')
pattern_num = re.compile(r'\d+\-')
pattern_alph = re.compile(r'\d+\-\w\-')
pattern_SSB_num = re.compile(r'\.\d+\w\.\w')
pattern_SSB_alpha = re.compile(r'SSB\.\w+\.')
print(ssb_columns)
RNA_seq_sorted_columns = sorted([c for c in RNA_seq_columns if pattern_alph.search(c)], key=lambda x: (int(pattern_alph.search(x).group().split('-')[0]),pattern_alph.search(x).group().split('-')[1]))
ssb_sorted_columns = sorted([c for c in ssb_columns if pattern_SSB_num.search(c)], key=lambda x: (int(pattern_SSB_num.search(x).group().split('.')[1][:-1]),pattern_SSB_alpha.search(x).group().split('.')[1][0].upper()))
AP_sorted_columns = sorted([c for c in AP_columns if pattern_alph.search(c)], key=lambda x: (int(pattern_alph.search(x).group().split('-')[0]),pattern_alph.search(x).group().split('-')[1][0]))

print(ssb_sorted_columns)
with open("labels.txt","w") as f:
    for i,j in enumerate(ssb_sorted_columns):
        if i <83:
            f.write(j+'\t'+RNA_seq_sorted_columns[i]+'\t'+AP_sorted_columns[i]+'\n') 
print(RNA_seq_sorted_columns[-1],RNA_seq_sorted_columns[-2])
SSB_data = SSB_data.sort_index(axis=1, ascending=True)
AP_data = AP_data.sort_index(axis=1,ascending=True)
RNA_seq_data = RNA_seq_data.sort_index(axis=1,ascending=True)

#再把样本排顺序
SSB_data = SSB_data[ssb_sorted_columns].T
AP_data = AP_data[AP_sorted_columns].T
RNA_seq_data = RNA_seq_data[RNA_seq_sorted_columns].T
#删除没有的共同样本"22-B-1"

SSB_data = SSB_data.drop(SSB_data.index[54],axis=0)
RNA_seq_data= RNA_seq_data.drop(RNA_seq_data.index[54],axis=0)

AP_data.to_csv("New_AP_data_sorted_tpm.csv")
SSB_data.to_csv("New_SSB_data_sorted_tpm.csv")
RNA_seq_data.to_csv("RNA_data_sorted_tpm.csv")