import pandas as pd
file = open('mm10.exon.bed','r')
# output_file = open("mm10.gene.bed",'w')
import numpy as np
# gene_length = {}
# gene_ids = set()
# gene_info = {}
# for line in file :
#     content = line.split()
#     gene_id = content[4]
#     left_pos = int(content[1]) 
#     right_pos = int(content[2])
#     if gene_id in gene_ids:
#         gene_length[gene_id].append(list([left_pos,right_pos]))
#     else:
#         gene_ids.add(gene_id)
#         gene_length[gene_id] = []
#         gene_length[gene_id].append(list([left_pos,right_pos]))
#         gene_info[gene_id] = [content[0],content[5]]
# gene_exon_length = {}   
# with open("mm10.exon.merge.bed","w") as f:
#     for gene_id in gene_length:
#         gene_exons = list(gene_length[gene_id])
#         gene_exons = sorted(gene_exons,key=lambda x: x[0])
#         merge_exon = []
#         for sub_exon in gene_exons:

#             if not merge_exon or merge_exon[-1][1] < sub_exon[0]:
#                     merge_exon.append(sub_exon)
#             else:
#                 # 否则的话，我们就可以与上一区间进行合并
#                 merge_exon[-1][1] = max(merge_exon[-1][1], sub_exon[1])
#         gene_exon_length[gene_id] = merge_exon
#         for sub_dis in gene_exon_length[gene_id]:
#             f.write(gene_info[gene_id][0]+'\t'+str(sub_dis[0])+'\t'+str(sub_dis[1])+'\t'+gene_id+'\t'+gene_info[gene_id][1]+'\n')
#     exon_valid_length = 0
#     for i in merge_exon:
#          exon_valid_length += i[1] - i[0]
#     gene_exon_length[gene_id] = exon_valid_length
    
file = open('mm10.exon.merge.bed','r')
gene_length = {}
gene_ids = set()
for line in file :
    content = line.split()
    gene_id = content[3]
    left_pos = int(content[1]) 
    right_pos = int(content[2])
    if gene_id in gene_ids:
        gene_length[gene_id] = gene_length[gene_id] + right_pos - left_pos
    else:
        gene_ids.add(gene_id)
        gene_length[gene_id] =right_pos - left_pos
print(gene_length["0610007P14Rik"])

file = open('count.AP.pos.txt','r')
# output_file = open("mm10.gene.bed",'w')
AP_length = {}
for line in file :
    content = line.split()
    AP_length[content[0]] = int(content[1])

file = open('count.SSB.pos.txt','r')
SSB_length = {}
for line in file :
    content = line.split()
    SSB_length[content[0]] = int(content[1])

SSB_file = r'./New_SSB_position_exon_output.xlsx'
AP_file = r'./New_AP_position_exon_output.xlsx'

ssb_data = pd.read_excel(SSB_file,header=0,index_col=0)
AP_data = pd.read_excel(AP_file,header=0,index_col=0)
print(ssb_data.sum(axis=0))

ssb_data_normlize = ssb_data.apply(lambda x:x.div(SSB_length[x.name[:-4]],axis=0)).mul(10**10)
AP_data_normlize = AP_data.apply(lambda x:x.div(AP_length[x.name[:-4]],axis=0)).mul(10**10)
print(AP_data_normlize.head)

ssb_data_normlize = ssb_data_normlize.apply(lambda x:x.div(gene_length[x.name],axis=0),axis=1)
AP_data_normlize = AP_data_normlize.apply(lambda x:x.div(gene_length[x.name],axis=0),axis=1)
print(AP_data_normlize.head)

AP_data_normlize.to_csv("New_AP_data_TPM.csv")

ssb_data_normlize.to_csv("New_SSB_data_TPM.csv")

