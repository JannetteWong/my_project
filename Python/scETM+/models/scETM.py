from typing import Any, Iterable, Mapping, Sequence, Tuple, Union
import anndata
import numpy as np
import logging
from scipy.sparse import spmatrix
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal, Independent


from scETM.logging_utils import log_arguments
from .BaseCellModel import BaseCellModel
from .model_utils import (
    InputPartlyTrainableLinear,
    PartlyTrainableParameter2D,
    get_fully_connected_layers,
    get_kl
)

_logger = logging.getLogger(__name__)


class scETM(BaseCellModel):

    clustering_input: str = "delta"
    emb_names: Sequence[str] = ['delta', 'theta']
    max_logsigma = 10
    min_logsigma = -10

    @log_arguments
    def __init__(self,
                 n_trainable_genes: int,
                 n_batches: int,
                 n_fixed_genes: int = 0,
                 n_topics: int = 50,
                 trainable_gene_emb_dim: int = 400,
                 hidden_sizes: Sequence[int] = (128,),
                 bn: bool = True,
                 dropout_prob: float = 0.1,
                 normalize_beta: bool = False,
                 normed_loss: bool = True,
                 norm_cells: bool = True,
                 input_batch_id: bool = False,
                 enable_batch_bias: bool = True,
                 enable_global_bias: bool = False,
                 rho_fixed_emb: Union[None, np.ndarray, spmatrix, torch.Tensor] = None,
                 rho_fixed_gene: Union[None, np.ndarray, spmatrix, torch.Tensor] = None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                 ):


        super().__init__(n_trainable_genes, n_batches, n_fixed_genes,
                         need_batch=n_batches > 1 and (input_batch_id or enable_batch_bias), device=device)

        self.n_topics: int = n_topics
        self.trainable_gene_emb_dim: int = trainable_gene_emb_dim
        self.hidden_sizes: Sequence[int] = hidden_sizes
        self.bn: bool = bn
        self.dropout_prob: float = dropout_prob
        self.normalize_beta: bool = normalize_beta
        self.normed_loss: bool = normed_loss
        self.norm_cells: bool = norm_cells
        self.input_batch_id: bool = input_batch_id
        self.enable_batch_bias: bool = enable_batch_bias
        self.enable_global_bias: bool = enable_global_bias

        if self.n_batches <= 1:
            _logger.warning(f'n_batches == {self.n_batches}, disabling batch bias')
            self.enable_batch_bias = False
            self.input_batch_id = False

        self.attn_heads = 8
        self.attn_dim = self.hidden_sizes[0]
        self.multihead_attention = MultiheadAttention(embed_dim=self.attn_dim, num_heads=self.attn_heads,
                                                      batch_first=True)

        self.q_delta: nn.Sequential = get_fully_connected_layers(
            n_trainable_input=self.n_trainable_genes + ((self.n_batches - 1) if self.input_batch_id else 0),
            hidden_sizes=self.hidden_sizes,
            bn=self.bn,
            dropout_prob=self.dropout_prob,
            n_fixed_input=self.n_fixed_genes
        )

        hidden_dim = self.hidden_sizes[-1]
        self.mu_q_delta: nn.Linear = nn.Linear(hidden_dim, self.n_topics, bias=True)
        self.logsigma_q_delta: nn.Linear = nn.Linear(hidden_dim, self.n_topics, bias=True)

        self.rho_fixed_emb: Union[None, torch.Tensor] = None
        self.rho_trainable_emb: Union[None, PartlyTrainableParameter2D] = None
        self._init_rho_trainable_emb()

        if self.trainable_gene_emb_dim > 0 and self.n_fixed_genes > 0 and rho_fixed_gene is not None:
            assert rho_fixed_gene.shape == (self.trainable_gene_emb_dim, self.n_fixed_genes)
            self.rho_trainable_emb.fixed = torch.FloatTensor(rho_fixed_gene)
        if rho_fixed_emb is not None:
            assert rho_fixed_emb.shape[1] == self.n_fixed_genes + self.n_trainable_genes
            self.rho_fixed_emb = torch.FloatTensor(rho_fixed_emb).to(device)

        self.alpha: nn.Parameter = nn.Parameter(torch.randn(self.n_topics, self.trainable_gene_emb_dim + (
            self.rho_fixed_emb.shape[0] if self.rho_fixed_emb is not None else 0)))
        self._init_batch_and_global_biases()

        self.to(device)

    @property
    def rho(self) -> torch.Tensor:
        """The fixed and trainable combined gene embedding rho.

        This is a read-only property. To modify the gene embeddings, please
        change self.rho_fixed_emb and self.rho_trainable_emb.
        """

        rho = [param for param in (self.rho_fixed_emb, self.rho_trainable_emb.get_param()) if param is not None]
        rho = torch.cat(rho, dim=0) if len(rho) > 1 else rho[0]
        return rho



    def _init_encoder_first_layer(self) -> None:
        """Initializes the first layer of the encoder given the constant
        attributes.
        """

        trainable_dim = self.n_trainable_genes + ((self.n_batches - 1) if self.input_batch_id else 0)
        if self.n_fixed_genes > 0:
            self.q_delta[0] = InputPartlyTrainableLinear(self.n_fixed_genes, self.hidden_sizes[0], trainable_dim)
        else:
            self.q_delta[0] = nn.Linear(trainable_dim, self.hidden_sizes[0])

    def _init_rho_trainable_emb(self) -> None:

        if self.trainable_gene_emb_dim > 0:
            self.rho_trainable_emb = PartlyTrainableParameter2D(self.trainable_gene_emb_dim, self.n_fixed_genes, self.n_trainable_genes)

    def _init_batch_and_global_biases(self) -> None:

        if self.enable_batch_bias:
            self.batch_bias: nn.Parameter = nn.Parameter(torch.randn(self.n_batches, self.n_fixed_genes + self.n_trainable_genes))
        
        self.global_bias: nn.Parameter = nn.Parameter(torch.randn(1, self.n_fixed_genes + self.n_trainable_genes)) if self.enable_global_bias else None

    def decode(self,
        theta: torch.Tensor,
        batch_indices: Union[None, torch.Tensor]
    ) -> torch.Tensor:

        beta = self.alpha @ self.rho

        if self.normalize_beta:
            recon = torch.mm(theta, F.softmax(beta, dim=-1))
            recon_log = (recon + 1e-30).log()
        else:
            recon_logit = torch.mm(theta, beta)  # [batch_size, n_genes]
            if self.enable_global_bias:
                recon_logit += self.global_bias
            if self.enable_batch_bias:
                recon_logit += self.batch_bias[batch_indices]
            recon_log = F.log_softmax(recon_logit, dim=-1)
        return recon_log

    def forward(self,
                data_dict: Mapping[str, torch.Tensor],
                hyper_param_dict: Mapping[str, Any] = dict()
                ) -> Mapping[str, Any]:

        cells, library_size = data_dict['cells'], data_dict['library_size']
        normed_cells = cells / library_size
        input_cells = normed_cells if self.norm_cells else cells

        if self.input_batch_id:
            input_cells = torch.cat((input_cells, self._get_batch_indices_oh(data_dict)), dim=1)

        input_cells_for_attention = input_cells.unsqueeze(1)

        attn_output, _ = self.multihead_attention(input_cells_for_attention, input_cells_for_attention,
                                                  input_cells_for_attention)

        attn_output = attn_output.squeeze(1)

        q_delta = self.q_delta(attn_output)

        mu_q_delta = self.mu_q_delta(q_delta)
        logsigma_q_delta = self.logsigma_q_delta(q_delta).clamp(self.min_logsigma, self.max_logsigma)

        q_delta = Independent(Normal(
            loc=mu_q_delta,
            scale=logsigma_q_delta.exp()
        ), 1)

        delta = q_delta.rsample()
        theta = F.softmax(delta, dim=-1)

        if not self.training:
            theta = F.softmax(mu_q_delta, dim=-1)
            fwd_dict = dict(theta=theta, delta=mu_q_delta)
            if 'decode' in hyper_param_dict and hyper_param_dict['decode']:
                recon_log = self.decode(theta, data_dict.get('batch_indices', None))
                fwd_dict['recon_log'] = recon_log
                fwd_dict['nll'] = (-recon_log * (normed_cells if self.normed_loss else cells)).sum()
            return fwd_dict

        recon_log = self.decode(theta, data_dict.get('batch_indices', None))
        nll = (-recon_log * normed_cells if self.normed_loss else cells).sum(-1).mean()
        kl_delta = get_kl(mu_q_delta, logsigma_q_delta).mean()
        loss = nll + hyper_param_dict['kl_weight'] * kl_delta
        record = dict(loss=loss, nll=nll, kl_delta=kl_delta)
        record = {k: v.detach().item() for k, v in record.items()}

        fwd_dict = dict(
            theta=theta,
            delta=delta,
            recon_log=recon_log
        )

        return loss, fwd_dict, record

    def get_all_embeddings_and_nll(self,
        adata: anndata.AnnData,
        batch_size: int = 2000,
        emb_names: Union[str, Iterable[str], None] = None,
        batch_col: str = 'batch_indices',
        inplace: bool = True,
        writer: Union[None, SummaryWriter] = None
    ) -> Union[Union[None, float], Tuple[Mapping[str, np.ndarray], Union[None, float]]]:


        result = super().get_cell_embeddings_and_nll(adata, batch_size=batch_size, emb_names=emb_names, batch_col=batch_col, inplace=inplace)
        if writer is not None:
            self.write_topic_gene_embeddings_to_tensorboard(writer, adata.var_names)
        if inplace:
            adata.varm['rho'] = self.rho.T.detach().cpu().numpy()
            adata.uns['alpha'] = self.alpha.detach().cpu().numpy()
            return result
        else:
            result_dict, nll = result
            result_dict['rho'] = self.rho.T.detach().cpu().numpy()
            result_dict['alpha'] = self.alpha.detach().cpu().numpy()
            return result_dict, nll

    def write_topic_gene_embeddings_to_tensorboard(self,
        writer: Union[None, SummaryWriter] = None,
        gene_names: Union[None, Sequence[str]] = None,
        tag: Union[None, str] = None
    ) -> None:


        if writer is None:
            return

        _logger.info('Writing topic and gene embeddings to tensorboard...')

        if gene_names is None:
            gene_names = [f'gene{i}' for i in range(self.n_trainable_genes + self.n_fixed_genes)]
        else:
            gene_names = list(gene_names)
        topic_names = [f'topic{i}' for i in range(self.n_topics)]
        names = gene_names + topic_names
        is_topic = [False] * (self.n_trainable_genes + self.n_fixed_genes) + [True] * self.n_topics
        metadata = np.array([names, is_topic]).T.tolist()

        gene_emb = self.rho.T.detach().cpu().numpy()
        topic_emb = self.alpha.detach().cpu().numpy()
        embs = np.concatenate([gene_emb, topic_emb], axis=0)

        writer.add_embedding(embs, metadata, tag=tag, metadata_header=['name', 'is_topic'])
