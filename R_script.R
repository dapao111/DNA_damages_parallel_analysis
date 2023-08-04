
library(dplyr)
library(stringr)
library(pheatmap)
library(clusterProfiler)

#heatmap_color=RColorBrewer::brewer.pal(name = "Blues",n=5)

heatmap_color=RColorBrewer::brewer.pal(11,"RdBu")[3:8]
ann_colors=RColorBrewer::brewer.pal(11,"Set3")[1:11]
pal=rev(colorRampPalette(heatmap_color)(100))

pheatmap(width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(ssb))+1),annotation_col = colanno, filename = 'ssb_tissue500.pdf',annotation_colors = list(age = c("24"=ann_colors[4],"22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))
pheatmap(width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(ap))+1),annotation_col = colanno, filename = 'ap_tissue500.pdf',annotation_colors = list(age = c("24"=ann_colors[4],"22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))
pheatmap(width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(rnaseq))+1),annotation_col = colanno, filename = 'rna_tissue500.pdf',annotation_colors = list(age = c("24"=ann_colors[4],"22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))

data_path = "./DataTopnGenesFeatures/genes_TopN_data/"
#tissue
ap <- read.csv(data_path+"selected_AP_data_tissue_filter_500.csv",row.names = 1)
ap_gene <- colnames(ap)

ssb <- read.csv(data_path+"selected_SSB_data_tissue_filter_500.csv",row.names = 1)
ssb_gene <- colnames(ssb)

rnaseq <- read.csv(data_path+"selected_RNA_seq_data_tissue_filter_500.csv",row.names = 1)
rnaseq_gene <- colnames(rnaseq)

#age
ap <- read.csv(data_path+"selected_AP_data_filter_500.csv",row.names = 1)
ap_gene <- colnames(ap)

ssb <- read.csv(data_path+"selected_SSB_data_filter_500.csv",row.names = 1)
ssb_gene <- colnames(ssb)

rnaseq <- read.csv(data_path+"selected_RNA_seq_data_filter_500.csv",row.names = 1)
rnaseq_gene <- colnames(rnaseq)

colanno <- data.frame(row.names = rownames(rnaseq),
                      age = rownames(rnaseq) %>% str_remove_all("-.*") %>% as.character(),
                      tissue = rownames(rnaseq)  %>% str_extract_all(pattern = "B|P|S|M|L|H") %>% unlist())

#Three data pheatmap analysis
ssb_p = pheatmap(main = "83 ssb samples",width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(ssb))+1),annotation_col = colanno,annotation_colors = list(age = c("24"=ann_colors[4],"22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))
ap_p = pheatmap(main = "83 ssb samples",width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(ap))+1),annotation_col = colanno, annotation_colors = list(age = c("24"=ann_colors[4],"22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))
rna_p = pheatmap(main = "83 ssb samples",width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(rnaseq))+1),annotation_col = colanno, annotation_colors = list(age = c("24"=ann_colors[4],"22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))

# ssb_p_71 = pheatmap(main = "71 ssb samples",width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(ssb))+1),annotation_col = colanno,annotation_colors = list(age = c("22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))
# ap_p_71 = pheatmap(main = "71 ap samples",width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(ap))+1),annotation_col = colanno, annotation_colors = list(age = c("22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))
# rna_p_71 = pheatmap(main = "71 rna samples",width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(rnaseq))+1),annotation_col = colanno, annotation_colors = list(age = c("22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))

# library(ggplotify)
# library(patchwork)
# (as.ggplot(ssb_p) + as.ggplot(ssb_p_71)) / (as.ggplot(ap_p)+ as.ggplot(ap_p_71)) / (as.ggplot(rna_p)+ as.ggplot(rna_p_71))

# library(ggplot2)
# ggsave(filename = "./exon_heatmapTop500tissue_71&83.png",height = 20,width =20)

# pheatmap(width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(ssb))+1),annotation_col = colanno, filename = 'ssb_age500.pdf',annotation_colors = list(age = c("24"=ann_colors[4],"22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))
# pheatmap(width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(ap))+1),annotation_col = colanno, filename = 'ap_age500.pdf',annotation_colors = list(age = c("24"=ann_colors[4],"22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))
# pheatmap(width = 12,height = 7,cluster_rows = 0,clcolor = pal,show_rownames = 0,log(as.data.frame(t(rnaseq))+1),annotation_col = colanno, filename = 'rna_age500.pdf',annotation_colors = list(age = c("24"=ann_colors[4],"22"=ann_colors[2],"19"=ann_colors[3],"12"=ann_colors[11],"3"=ann_colors[5])))





#==========================================

source("./get_venn_content.R")

genelist <- list(ap_gene = ap_gene[1:500],ssb_gene = ssb_gene[1:500],rnaseq_gene = rnaseq_gene[1:500])
venn <- get_venn_content(gene_list = genelist)
genelist <-venn

library('VennDiagram')
venn.plot <- venn.diagram(
  list(A=ssb_gene[1:500],B=ap_gene[1:500],C=rnaseq_gene[1:500]),
  filename = NULL,
  main = paste0("data_set_tissue_Top500"),
  category.names = c("SSB","AP","RNA-seq"), 
  col = "transparent",
  fill = c("#f89588", "#eddd86", "#B883D4"),
  #alpha = 0.50,
  #cex = 1.5,
  #fontfamily = "serif",
  #fontface = "bold",
  #rotation.degree = 270,
  #margin = 0.2
);
pdf("venn500_tissue.pdf")
grid.draw(venn.plot)
dev.off()

#colors = ['#f89588','#f74d4d','#B883D4','#76da91','#7898e1','#BEB8DC','#A1A9D0','#eddd86','#8ECFC9','#63b2ee','#943c39']
for (i in c(50,100,200,300,400,500)){
  venn.plot <- venn.diagram(
    list(A=ssb_gene[1:i],B=ap_gene[1:i],C=rnaseq_gene[1:i]),
    paste0("Venn_3data_set_age_",i,".jpeg"),
    main = paste0("data_set_age_Top",i),
    category.names = c("SSB","AP","RNA-seq"), 
    col = "transparent",
    fill = c("#f89588", "#eddd86", "#B883D4"),
    #alpha = 0.50,
    #cex = 1.5,
    #fontfamily = "serif",
    #fontface = "bold",
    #rotation.degree = 270,
    #margin = 0.2
  );
}
for (i in c(50,100,200,300,400,500)){
  venn.plot <- venn.diagram(
    list(A=ssb_gene[1:i],B=ap_gene[1:i],C=rnaseq_gene[1:i]),
    paste0("Venn_3data_set_tissue_",i,".jpeg"),
    main = paste0("data_set_tissue_Top",i),
    category.names = c("SSB","AP","RNA-seq"), 
    col = "transparent",
    fill = c("#f89588", "#eddd86", "#B883D4"),
    #alpha = 0.50,
    #cex = 1.5,
    #fontfamily = "serif",
    #fontface = "bold",
    #rotation.degree = 270,
    #margin = 0.2
  );
}


library("dplyr")
source("./run_go.R")
library("org.Mm.eg.db")
run_go(genelist = venn,use_go_RData = F,plot_type = "dot",facet = F,outputpath = paste0("./go_age_83_inters/Top",500,"_results"),keyType = "SYMBOL",OrgDb = "org.Mm.eg.db")

run_go(genelist = genelist,use_go_RData = T,plot_type = "dot",facet = F,outputpath = paste0("./go_age_83/Top",500,"_results"),keyType = "SYMBOL",OrgDb = "org.Mm.eg.db")
run_go(genelist = genelist,use_go_RData = T,plot_type = "dot",facet = F,outputpath = paste0("./go_tissue_83/Top",500,"_results"),keyType = "SYMBOL",OrgDb = "org.Mm.eg.db")

library(patchwork)
ap_pictures_list = list()
ssb_pictures_list = list()
rna_seq_pictures_list = list()
for (i in c(50,100,200,300,400,500)) {
  genelist <- list(ap_gene = ap_gene[1:i],ssb_gene = ssb_gene[1:i],rnaseq_gene = rnaseq_gene[1:i])
  
  bp_pictures = run_go(genelist = genelist,use_go_RData = F,plot_type = "dot",facet = F,outputpath = paste0("./go_age_83/Top",i,"_results"),keyType = "SYMBOL",OrgDb = "org.Mm.eg.db")
  # browser()
  ap_pictures_list <- append(ap_pictures_list,bp_pictures[1])
  ssb_pictures_list<- append(ssb_pictures_list,bp_pictures[2])
  rna_seq_pictures_list <-append(rna_seq_pictures_list,bp_pictures[3])
}

final = ap_pictures_list[[1]]
for (i in c(2:6))
{
  tmpp <- ap_pictures_list[[i]]
  final<-final + tmpp
}
ggsave(filename = "./go_age_83/joined_all_apgenes_go_analysis_age_result_83.pdf",width = 20,height = 15)

final = ssb_pictures_list[[1]]
for (i in c(2:6))
{
  tmpp <- ssb_pictures_list[[i]]
  final<-final + tmpp
}
ggsave(filename = "./go_age_83/joined_all_ssbgenes_go_analysis_age_result_83.pdf",width = 20,height = 15)

final = rna_seq_pictures_list[[1]]
  for (i in c(2:6))
{
  tmpp <- rna_seq_pictures_list[[i]]
  final<-final + tmpp
}
ggsave(filename = "./go_age_83/joined_all_rna_seq_genes_go_analysis_age_result_83.pdf",width = 20,height = 15)




###############################################################################
########################## Disease Ontology (DO)  &  Plot  ####################
###############################################################################
library(DOSE)
library("AnnotationDbi")
library("org.Mm.eg.db")
library(org.Hs.eg.db)

library(ggplot2)
library(clusterProfiler)
outputpath = "./DOSE_83"

if (!dir.exists(paste0(outputpath,"/dotplot/"))) {
  dir.create(paste0(outputpath,"/dotplot/"),recursive = T)
}
if (!dir.exists(paste0(outputpath,"/res/"))) {
  dir.create(paste0(outputpath,"/res/"),recursive = T)
}

mm2hg = read.table(file = './mm2hgSymbolId.txt',header = T)
library(stringr)
keys(org.Mm.eg.db,keytype="SYMBOL")
for (topn in c(5,15,25,50,100,200,300,400,500)) {
  genelist <- list(ap_gene = ap_gene[1:i],ssb_gene = ssb_gene[1:i],rnaseq_gene = rnaseq_gene[1:i])
  for (name in names(genelist)) {
    name <- as.character(name)
    select_mm2hg_gene <- mm2hg[mm2hg$MM %in% genelist[[name]],1]
    
    class(select_mm2hg_gene)
    
    
    select_mm2hg_gene <- str_to_upper(select_mm2hg_gene)[1:500]
    
    #gname_entrezid<-mapIds(x=org.Hs.eg.db,
     #                    keys = select_mm2hg_gene,
      #                   keytype = "SYMBOL",
       #                  column = "ENTREZID")
    entrez <- bitr(geneID = select_mm2hg_gene,fromType = "SYMBOL",toType = "ENTREZID",OrgDb = org.Hs.eg.db)
    
    entrez <- entrez$ENTREZID
  
    do <- enrichDO(gene  = entrez,  ##此处选择待分析的基因，通常可以是Different expresssion gene
                   ont  = "DO",   ##选择疾病本体论这一方式进行分析
                   pvalueCutoff  = 1,##选择p值小于0.05的进行保留
                   pAdjustMethod = "BH",
                   qvalueCutoff  = 1)
    do = setReadable(do,OrgDb = "org.Hs.eg.db",keyType = "ENTREZID")
    if(length(do)!=0){
     dotplot(do)
    
    ggsave(paste0(outputpath,"/dotplot/",name,"_age.pdf"),width = 9,height = 4)
    #pdf(file =  paste0(outputpath,"/dotplot/",name,"_tissue.pdf"),width = 9,height = 5)
    
    do_res <- as.data.frame(do)
    for (i in c(1:nrow(do_res))){
      do_res[i,"geneID"] <- paste(collapse = '/',mm2hg[mm2hg$hg %in% unlist(str_split(do_res[i,"geneID"],pattern = "\\/")),1])
    }
    
    do_res$geneID
    write.csv(do_res,file = paste0(outputpath,"/res/",name,"_",topn,"_age.csv"))}
  }
}

# deResult <- go(ap_gene, species="mouse", ont="BP", pvalueCutoff=0.05, qvalueCutoff=1)

# library(msigdbr)
# m_df = msigdbr(species = "Mus musculus")
# genelist <- list(ap_gene = ap_gene,ssb_gene = ssb_gene,rnaseq_gene = rnaseq_gene)
# genelist <- genelist[[2]]
# msigdbr_t2g = m_df %>% dplyr::select(gs_name, gene_symbol) %>% as.data.frame()
# locol<- enricher(gene = ap_gene[1:10], TERM2GENE = msigdbr_t2g, pvalueCutoff = 0.05,qvalueCutoff = 1)
# write.csv(locol,"msigbdr_ap.csv")

# #Sys.setenv("http_proxy"="https://127.0.0.1:10809")
# library(ReactomePA)
# library("reactome.db")

# entrez <- bitr(geneID = ap_gene[1:15],fromType = "SYMBOL",toType = "ENTREZID",OrgDb = "org.Mm.eg.db")
# entrez <- bitr(geneID = ssb_gene[1:15],fromType = "SYMBOL",toType = "ENTREZID",OrgDb = "org.Mm.eg.db")
# entrez <- bitr(geneID = rnaseq_gene[1:15],fromType = "SYMBOL",toType = "ENTREZID",OrgDb = "org.Mm.eg.db")

# entreid <- entrez$ENTREZID

# enrich_reactome = enrichPathway(gene = entreid,organism = "mouse",pvalueCutoff = 1,qvalueCutoff = 1)


# aa = setReadable(enrich_reactome,OrgDb = "org.Mm.eg.db",keyType = "ENTREZID")
# if (!dir.exists(paste0("Reactome","/res/"))) {
#   dir.create(paste0("Reactome","/res/"),recursive = T)
# }
# write.csv(aa@result,file = paste0("Reactome","/res/","ap_83_age_genes",".csv"))
# write.csv(aa@result,file = paste0("Reactome","/res/","ssb_83_age_genes",".csv"))
# write.csv(aa@result,file = paste0("Reactome","/res/","rna_83_age_seq_genes",".csv"))

# write.csv(aa@result,file = paste0("Reactome","/res/","ap_71_age_genes",".csv"))
# write.csv(aa@result,file = paste0("Reactome","/res/","ssb_71_age_genes",".csv"))
# write.csv(aa@result,file = paste0("Reactome","/res/","rna_71_age_seq_genes",".csv"))


# library("figpatch")
# p1 <- fig("./Venn_3data_set_age_50.jpeg")
# p2 <- fig("./Venn_3data_set_age_100.jpeg")
# p3 <- fig("./Venn_3data_set_age_200.jpeg")
# p4 <- fig("./Venn_3data_set_age_300.jpeg")
# p5 <- fig("./Venn_3data_set_age_400.jpeg")
# p6 <- fig("./Venn_3data_set_age_500.jpeg")
# p1+p2+p3+p4+p5+p6

# p1 <- fig("./Venn_3data_set_tissue_50.jpeg")
# p2 <- fig("./Venn_3data_set_tissue_100.jpeg")
# p3 <- fig("./Venn_3data_set_tissue_200.jpeg")
# p4 <- fig("./Venn_3data_set_tissue_300.jpeg")
# p5 <- fig("./Venn_3data_set_tissue_400.jpeg")
# p6 <- fig("./Venn_3data_set_tissue_500.jpeg")
# p1+p2+p3+p4+p5+p6


# age_related_genes = read.table(file = './mm_age_related_tops.txt',header = FALSE)
# mm2hg = read.table(file = './mm2hgSymbolId.txt',header = T)
# age_related_mm_genes = age_related_genes[,1]
# age_related_mm_genes[str_to_upper(age_related_mm_genes) %in% str_to_upper(ap_gene)]
# age_related_mm_genes[str_to_upper(age_related_mm_genes) %in% str_to_upper(ssb_gene)]
# age_related_mm_genes[str_to_upper(age_related_mm_genes) %in% str_to_upper(rnaseq_gene)]




#==================caculate the gene distance==============================================
library(org.Mm.eg.db)
library(GOSemSim)
ranseqID <- bitr(geneID = rnaseq_gene,fromType = "SYMBOL",toType = "ENTREZID",OrgDb = org.Mm.eg.db)
apID <- bitr(geneID = ap_gene,fromType = "SYMBOL",toType = "ENTREZID",OrgDb = org.Mm.eg.db)
ssbID <- bitr(geneID = ssb_gene,fromType = "SYMBOL",toType = "ENTREZID",OrgDb = org.Mm.eg.db)

mmbp_database <- godata('org.Mm.eg.db', ont="BP", computeIC=FALSE)

geneSim(apID$ENTREZID, ssbID$ENTREZID, semData=mmbp_database, measure="Wang")
geneSim(rnaseqID$ENTREZID, ssbID$ENTREZID, semData=mmbp_database, measure="Wang")
geneSim(apID$ENTREZID, ranseqID$ENTREZID, semData=mmbp_database, measure="Wang")
