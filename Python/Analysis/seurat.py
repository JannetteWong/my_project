import scanpy as sc
import numpy as np
import scvi
from scvi.model import SCVI
import pandas as pd
import matplotlib.pyplot as plt


def run_Seurat(annData,resolution):
    sc.pp.normalize_total(annData, target_sum=1e4)
    sc.pp.log1p(annData)
    sc.pp.highly_variable_genes(annData, min_mean=0.0125, max_mean=3, min_disp=0.5)
    annData = annData[:, annData.var.highly_variable]
    sc.pp.scale(annData, max_value=10)
    sc.tl.pca(annData, svd_solver='arpack')
    sc.pp.neighbors(annData, n_neighbors=10, n_pcs=40)
    sc.tl.umap(annData)

    sc.tl.leiden(annData, resolution)
    annData.obs['Seurat'] = annData.obs['leiden'].astype('category')
    sc.pl.umap(annData, color=['Seurat'])
    plt.show()

    return annData.obs[['cell_ontology_class', 'Seurat']].copy()

