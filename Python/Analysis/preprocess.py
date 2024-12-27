adata = ad.read_h5ad(f"Data_Class/{dataset}.h5ad")
# 添加assigned_cluster和batch_indices列
adata.obs['assigned_cluster'] = adata.obs['cell_ontology_class']  # 细胞类型
adata.obs['cell_types'] = adata.obs['cell_ontology_class']  # 细胞类型
adata.obs['batch_indices'] = 0  # 所有细胞归为同一批次
adata.write(f'adjusted_data/scETM_{dataset}.h5ad')
