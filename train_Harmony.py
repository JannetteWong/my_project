import anndata
import random
import scanpy as sc
from pathlib import Path
import scanpy as sc
import numpy as np
from time import strftime, time
import psutil
import os
import logging
import matplotlib
import harmonypy as hm
import argparse
from scETM import evaluate, initialize_logger
from arg_parser import add_plotting_arguments


logger = logging.getLogger(__name__)
initialize_logger(logger=logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-h5ad', type=str, required=True, help='path to input h5ad file')
    parser.add_argument('--checkpoint-dir', type=str, help='directory for saving checkpoints',
                        default=os.path.join('..', 'results'))
    parser.add_argument('--skip-be', action='store_true', help='skip batch mixing entropy calculation')
    parser.add_argument('--skip-eval', action='store_true', help='quit immediately after training')
    parser.add_argument('--feature-reduction', type=int, default=50, help='reduce raw data to this many features before integration')
    parser.add_argument('--random-seed', type=int, default=-1, help='set seed')
    add_plotting_arguments(parser)
    params = parser.parse_args()

    if params.random_seed >= 0:
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)

    matplotlib.use('Agg')
    sc.settings.set_figure_params(
        dpi=params.dpi_show, dpi_save=params.dpi_save, facecolor='white', fontsize=params.fontsize, figsize=params.figsize)

    # load dataset
    dataset = anndata.read_h5ad(params.input_h5ad)
    dataset_name = Path(params.input_h5ad).stem
    dataset.obs_names_make_unique()
    checkpoint_path = os.path.join(params.checkpoint_dir, f'{dataset_name}_Harmony_seed{params.random_seed}_{strftime("%m_%d-%H_%M_%S")}')
    os.makedirs(checkpoint_path)

    start_time = time()
    initial_memory = psutil.Process().memory_info().rss
    logger.info(f'Before model initialization and training: {psutil.Process().memory_info()}')

    # preprocess
    sc.pp.normalize_total(dataset, target_sum=1e4)
    sc.pp.log1p(dataset)
    if params.feature_reduction:
        sc.pp.highly_variable_genes(dataset, flavor='seurat', n_top_genes=3000)
        sc.pp.pca(dataset, n_comps=params.feature_reduction, use_highly_variable=True)
        data_matrix = dataset.obsm['X_pca']
    else:
        sc.pp.scale(dataset)
        data_matrix = np.array(dataset.X)

    harmony_output = hm.run_harmony(data_matrix, meta_data=dataset.obs, vars_use=['single_dataset'], max_iter_harmony=100)

    time_taken = time() - start_time
    memory_used = psutil.Process().memory_info().rss - initial_memory
    logger.info(f'Duration: {time_taken:.1f} s ({time_taken / 60:.1f} min)')
    logger.info(f'After model initialization and training: {psutil.Process().memory_info()}')

    embeddings = anndata.AnnData(X=harmony_output.result().T, obs=dataset.obs)
    embeddings.write_h5ad(os.path.join(checkpoint_path, f"{dataset_name}_Harmony_seed{params.random_seed}.h5ad"))

    if not params.skip_eval:
        evaluation_result = evaluate(embeddings, embedding_key="X", resolutions=params.resolutions, plot_dir=checkpoint_path, plot_fname=f"{dataset_name}_Harmony_seed{params.random_seed}_eval")
        with open(os.path.join(params.checkpoint_dir, 'table1.tsv'), 'a+') as file:
            # dataset, model, seed, ari, nmi, ebm, k_bet
            file.write(f'{dataset_name}\tHarmony\t{params.random_seed}\t{evaluation_result["ari"]}\t{evaluation_result["nmi"]}\t{evaluation_result["asw"]}\t{evaluation_result["ebm"]}\t{evaluation_result["k_bet"]}\t{time_taken}\t{memory_used/1024}\n')