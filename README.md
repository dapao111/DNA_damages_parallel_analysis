# DNA_damages_parallel_analysis
Parallel Analysis for AP,SSB,RNA-seq expressions in aging biomarker

# Usage
1. Clone repository.
```
git clone https://github.com/dapao111/DNA_damages_parallel_analysis.git
```
2. Install the requirments.
```
pip install -r requirements.txt
```

Input:
    -sf : SSB expression file(csv) PATH
    -pf : AP expression file(csv) PATH
    -rf : RNA expression file(csv) PATH
    -o : output_file_path
    -fn : selected_Top_num_genes of whole genes, using "all" means all genes would be as inputs.
    -in : selected the numbers of iteration to test the dataset, 100 default.
    -caa : correlation_analysis_with_age or_not 
    -cat : correlation_analysis_with_age or_not 
    -rp : roc_results_plot or_not 
    -ccf : calculate_confidencial_interval or_not

    Note: Please make sure your AP,SSB Matrix expression data are normalized with the formula we gave in Paper as well as the mm gtf data is Compatible

Output:
    you will get the parallel analysis results between the three datasets by adpoting ten classical machine learning algorithms

## Reproducing our experiments results
   

Run experiments with the default parameters.

    python analysis_main.py 

If you want to run your data:

    python analysis_main.py -sf (your ssb expression file ) -pf (your ap expression file ) -rf (your rna expression file ) -o (output_file path) -caa 1 or 0 -cat 1 or 0 -rp 1 or 0 -ccf 1 or 0
    
