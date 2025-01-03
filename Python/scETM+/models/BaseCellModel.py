from typing import Any, Callable, Mapping, Sequence, Tuple, Union, Iterable
import anndata
import logging
import numpy as np
import torch
from torch import nn, optim


_logger = logging.getLogger(__name__)


class BaseCellModel(nn.Module):

    clustering_input: str
    emb_names: Sequence[str]

    def __init__(self,
        n_trainable_genes: int,
        n_batches: int,
        n_fixed_genes: int = 0,
        need_batch: bool = False,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:

        super().__init__()
        self.device: torch.device = device
        self.n_trainable_genes: int = n_trainable_genes
        self.n_fixed_genes: int = n_fixed_genes
        self.n_batches: int = n_batches
        self.need_batch: bool = need_batch

    @property
    def n_genes(self):
        return self.n_trainable_genes + self.n_fixed_genes
    
    def train_step(self,
        optimizer: optim.Optimizer,
        data_dict: Mapping[str, torch.Tensor],
        hyper_param_dict: Mapping[str, Any],
        loss_update_callback: Union[None, Callable] = None
    ) -> Mapping[str, torch.Tensor]:

        self.train()
        optimizer.zero_grad()
        loss, fwd_dict, new_record = self(data_dict, hyper_param_dict)
        if loss_update_callback is not None:
            loss, new_record = loss_update_callback(loss, fwd_dict, new_record)
        loss.backward()
        norms = torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
        new_record['max_norm'] = norms.cpu().numpy()
        optimizer.step()
        return new_record

    def _apply_to(self,
        adata: anndata.AnnData,
        batch_col: str = 'batch_indices',
        batch_size: int = 2000,
        hyper_param_dict: Union[dict, None] = None,
        callback: Union[Callable, None] = None
    ) -> None:
        """Docstring (TODO)
        """

        sampler = CellSampler(adata, batch_size=batch_size, sample_batch_id=self.need_batch, n_epochs=1, batch_col=batch_col, shuffle=False)
        self.eval()
        for data_dict in sampler:
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
            fwd_dict = self(data_dict, hyper_param_dict=hyper_param_dict)
            if callback is not None:
                callback(data_dict, fwd_dict)

    def get_cell_embeddings_and_nll(self,
        adata: anndata.AnnData,
        batch_size: int = 2000,
        emb_names: Union[str, Iterable[str], None] = None,
        batch_col: str = 'batch_indices',
        inplace: bool = True
    ) -> Union[Union[None, float], Tuple[Mapping[str, np.ndarray], Union[None, float]]]:

        assert adata.n_vars == self.n_fixed_genes + self.n_trainable_genes
        nlls = []
        if self.need_batch and adata.obs[batch_col].nunique() != self.n_batches:
            _logger.warning(
                f'adata.obs[{batch_col}] contains {adata.obs[batch_col].nunique()} batches, '
                f'while self.n_batches == {self.n_batches}.'
            )
            if self.need_batch:
                _logger.warning('Disable decoding. You will not get reconstructed cell-gene matrix or nll.')
                nlls = None
        if emb_names is None:
            emb_names = self.emb_names
        self.eval()
        if isinstance(emb_names, str):
            emb_names = [emb_names]

        embs = {name: [] for name in emb_names}
        hyper_param_dict = dict(decode=nlls is not None)

        def store_emb_and_nll(data_dict, fwd_dict):
            for name in emb_names:
                embs[name].append(fwd_dict[name].detach().cpu())
            if nlls is not None:
                nlls.append(fwd_dict['nll'].detach().item())

        self._apply_to(adata, batch_col, batch_size, hyper_param_dict, callback=store_emb_and_nll)

        embs = {name: torch.cat(embs[name], dim=0).numpy() for name in emb_names}
        if nlls is not None:
            nll = sum(nlls) / adata.n_obs
        else:
            nll = None

        if inplace:
            adata.obsm.update(embs)
            return nll
        else:
            return embs, nll

    def _get_batch_indices_oh(self, data_dict: Mapping[str, torch.Tensor]):

        if 'batch_indices_oh' in data_dict:
            w_batch_id = data_dict['batch_indices_oh']
        else:
            batch_indices = data_dict['batch_indices'].unsqueeze(1)
            w_batch_id = torch.zeros((batch_indices.shape[0], self.n_batches), dtype=torch.float32, device=self.device)
            w_batch_id.scatter_(1, batch_indices, 1.)
            w_batch_id = w_batch_id[:, :self.n_batches - 1]
            data_dict['batch_indices_oh'] = w_batch_id
        return w_batch_id
