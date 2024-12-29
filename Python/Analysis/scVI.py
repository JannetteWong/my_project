import scanpy as sc
import numpy as np
import scvi
from scvi.model import SCVI
import pandas as pd
import matplotlib.pyplot as plt

def run_scVI(adata,resolution,batch_key="donor", output_dir="D:/GTM/output/scvi",
                    csv_dir="D:/GTM/result/scVI_Liver_results.csv"):

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)

    scvi_model = SCVI(adata)
    scvi_model.train()

    scvi_model.save(output_dir)

    loaded_scvi_model = SCVI.load(output_dir, adata)
    latent = loaded_scvi_model.get_latent_representation()
    adata.obsm["X_scVI"] = latent

    sc.pp.neighbors(adata, use_rep='X_scVI')
    sc.tl.leiden(adata, resolution, key_added='scVI')

    sc.pl.umap(adata, color=['scVI'])

    clustering_results = adata.obs['scVI'].to_frame(name='Cluster')
    clustering_results.to_csv(csv_dir, index_label='cell_id')

    return adata.obs[['cell_ontology_class', 'scVI']].copy()
