from typing import Sequence, Union
from torch import nn
import torch
import math

class InputPartlyTrainableLinear(nn.Module):

    def __init__(self, n_fixed_input: int, n_output: int, n_trainable_input: int = 0, bias: bool = True) -> None:

        super().__init__()
        self.fixed: nn.Linear = nn.Linear(n_fixed_input, n_output, bias=False)
        self.fixed.requires_grad_(False)
        self.trainable_bias: Union[None, nn.Parameter] = None
        if n_trainable_input > 0:
            self.trainable: nn.Linear = nn.Linear(n_trainable_input, n_output, bias=bias)
        elif bias:
            self.trainable_bias = nn.Parameter(torch.Tensor(n_output))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fixed.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.trainable_bias, -bound, bound)
        self.n_fixed_input: int = n_fixed_input
        self.n_trainable_input: int = n_trainable_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.n_trainable_input > 0:
            x_fixed, x_trainable = x[:, :self.n_fixed_input], x[:, self.n_fixed_input:]
            with torch.no_grad():
                out = self.fixed(x_fixed)
            return out + self.trainable(x_trainable)
        elif self.trainable_bias is not None:
            return self.fixed(x) + self.trainable_bias
        else:
            return self.fixed(x)

    @property
    def weight(self) -> torch.Tensor:

        if self.n_trainable_input > 0:
            return torch.cat([self.fixed.weight, self.trainable.weight], dim=1)
        else:
            return self.fixed.weight
    
    @property
    def bias(self) -> Union[torch.Tensor, None]:


        if self.n_trainable_input > 0:
            return self.trainable.bias
        else:
            return self.trainable_bias

class OutputPartlyTrainableLinear(nn.Module):

    def __init__(self, n_input: int, n_fixed_output: int, n_trainable_output: int = 0, bias: bool = True) -> None:


        super().__init__(self)
        self.fixed: nn.Linear = nn.Linear(n_input, n_fixed_output, bias=False)
        self.fixed.requires_grad_(False)
        self.trainable_bias: Union[None, nn.Parameter] = None
        if bias:
            self.trainable_bias = nn.Parameter(torch.Tensor(n_fixed_output))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fixed.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.trainable_bias, -bound, bound)
        if n_trainable_output > 0:
            self.trainable: nn.Linear = nn.Linear(n_input, n_trainable_output, bias=bias)
        self.n_fixed_output: int = n_fixed_output
        self.n_trainable_output: int = n_trainable_output
        self.enable_bias: bool = bias

    def forward(self, x: torch.Tensor):


        # calculate fixed output
        with torch.no_grad():
            fixed_output = self.fixed(x)
        if self.trainable_bias is not None:
            fixed_output = fixed_output + self.trainable_bias
        # return full output
        if self.n_trainable_output > 0:
            return torch.cat([fixed_output, self.trainable(x)], dim=-1) 
        else:
            return fixed_output

    @property
    def weight(self):


        if self.n_trainable_output > 0:
            return torch.cat([self.fixed.weight, self.trainable.weight], dim=0)
        else:
            return self.fixed.weight

    @property
    def bias(self):
        if not self.enable_bias:
            return None
        if self.n_trainable_output > 0:
            return torch.cat([self.trainable_bias, self.trainable.bias], dim=0)
        else:
            return self.trainable_bias

class PartlyTrainableParameter2D(nn.Module):

    def __init__(self, height: int, n_fixed_width: int, n_trainable_width: int) -> None:
        super().__init__()
        self.height: int = height
        self.n_fixed_width: int = n_fixed_width
        self.n_trainable_width: int = n_trainable_width
        self.fixed: Union[None, torch.Tensor] = None
        self.trainable: Union[None, nn.Parameter] = None
        if n_fixed_width > 0:
            self.fixed = torch.randn(height, n_fixed_width)
        if n_trainable_width > 0:
            self.trainable = nn.Parameter(torch.randn(height, n_trainable_width))
    
    def get_param(self) -> Union[None, torch.Tensor]:

        params = [param for param in (self.fixed, self.trainable) if param is not None]
        if len(params) == 2:
            return torch.cat(params, dim=1)
        elif len(params) == 1:
            return params[0]
        else:
            return None

    def __repr__(self):
        return f'{self.__class__.__name__}(height={self.height}, fixed={self.n_fixed_width}, trainable={self.n_trainable_width})'


def get_fully_connected_layers(
    n_trainable_input: int,
    hidden_sizes: Union[int, Sequence[int]],
    n_trainable_output: Union[None, int] = None,
    bn: bool = True,
    bn_track_running_stats: bool = True,
    dropout_prob: float = 0.,
    n_fixed_input: int = 0,
    n_fixed_output: int = 0
) -> nn.Sequential:

    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    layers = []
    for i, size in enumerate(hidden_sizes):
        if i == 0 and n_fixed_input > 0:
            layers.append(InputPartlyTrainableLinear(n_fixed_input, size, n_trainable_input))
        else:
            layers.append(nn.Linear(n_trainable_input, size))
        layers.append(nn.ReLU())
        if bn:
            layers.append(nn.BatchNorm1d(size, track_running_stats=bn_track_running_stats))
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))
        n_trainable_input = size
    if n_trainable_output is not None:
        if n_fixed_output > 0:
            layers.append(OutputPartlyTrainableLinear(n_trainable_input, n_fixed_output, n_trainable_output))
        else:
            layers.append(nn.Linear(n_trainable_input, n_trainable_output))
    return nn.Sequential(*layers)


def get_kl(mu: torch.Tensor, logsigma: torch.Tensor):

    logsigma = 2 * logsigma
    return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)