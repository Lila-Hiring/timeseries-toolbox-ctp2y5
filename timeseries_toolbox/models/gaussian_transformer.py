import copy
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer

from timeseries_toolbox.models.base_timeseries import BaseTimeSeriesModel


class causal_self_attention(nn.Module):
    """
    Causal self-attention layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_seq_len: int,
    ):
        """
        Constructor.

        Args:
            embedding_dim: The dimension of the embedding.
            max_seq_len: The maximum sequence length for the causal mask.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        # Instantiate projections for Q, K, V
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim,
                                  bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim,
                                  bias=False)
        # Create causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1,
                                                                  max_seq_len,
                                                                  max_seq_len),
        )

    def forward(
        self,
        net_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the causal attention.

        Args:
            net_in: Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            Output of shape (batch_size, seq_len, embedding_dim).
        """
        # B = batch_size, L = seq_len, C = embedding_dim
        B, L, C = net_in.size()
        # Get Q, K, V, each has shape (B, L, C)
        query, key, value = self.qkv_proj(net_in).chunk(3, dim=-1)
        # Compute attention scores
        attn = (query @ key.transpose(-2, -1))  # Attn resulting shape (B, L, C)
        attn /= math.sqrt(C)
        attn = attn.masked_fill(self.causal_mask[:, :L, :L] == 0,
                                float('-inf'))
        attn = F.softmax(attn, dim=-1)
        # Apply attention to values
        net_out = attn @ value  # net_out shape (B, L, C)
        return self.out_proj(net_out)  # Final output shape (B, L, C)


class transformer_block(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        max_seq_len: int,
    ):
        """Constructor.

        Args:
            embedding_dim: The dimension of the embedding.
            num_heads: The number of attention heads.
            max_seq_len: The maximum sequence length.
        """
        super().__init__()
        self.attention = causal_self_attention(embedding_dim,
                                               max_seq_len)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(
        self,
        net_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Args:
            net_in: Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            Output of shape (batch_size, seq_len, embedding_dim).
        """
        # B = batch_size, L = seq_len, C = embedding_dim
        # B, L, C = net_in.size()
        net_out = self.attention(self.norm1(net_in))  # net_out shape (B, L, C)
        net_out = self.norm2(self.mlp(net_out))  # net_out shape (B, L, C)
        return net_out


class GaussianTransformer(BaseTimeSeriesModel):
    """
    Time series model which predicts a gaussian distribution for each
    dimension of the time series model.
    """

    def __init__(
        self,
        data_dim: int,
        embedding_dim: int,
        max_seq_len: int,
        num_blocks: int,
        in_mean: Tensor | None,
        in_std: Tensor | None,
        out_mean: Tensor | None,
        out_std: Tensor | None,
        learning_rate: float = 3e-4,
    ):
        """
        Constructor

        Args:
            data_dim (int): The dimension of the input and output data.
            embedding_dim (int): The dimension of the embedding.
            max_seq_len (int): The maximum sequence length.
            num_blocks (int): The number of transformer blocks.
            in_mean (Tensor | None): The mean for input normalization.
            in_std (Tensor | None): The standard deviation for input normalization.
            out_mean (Tensor | None): The mean for output normalization.
            out_std (Tensor | None): The standard deviation for output normalization.
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__(
            data_dim=data_dim,
            in_mean=in_mean,
            in_std=in_std,
            out_mean=out_mean,
            out_std=out_std,
        )
        self.learning_rate = learning_rate
        self.blocks = nn.ModuleList([
            transformer_block(embedding_dim, max_seq_len)
            for _ in range(num_blocks)
        ])
        self.embedding_projection = nn.Linear(data_dim, embedding_dim)
        self.mean_out_projection = nn.Linear(embedding_dim, data_dim)
        self.var_out_projection = nn.Linear(embedding_dim, data_dim)

    def forward(
        self,
        net_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer.

        Args:
            net_in: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Output of shape (batch_size, seq_len, output_dim) for the
            mean and variance.
        """
        # B = batch_size, L = seq_len, D = input_dim, C = embedding_dim
        # B, L, D = net_in.size()
        # Embed the input
        net_out = self.embedding_projection(net_in)  # net_out shape (B, L, C)
        # Pass through transformer blocks
        for block in self.blocks:
            net_out = block(net_out)  # net_out shape (B, L, C)
        # Project to mean and variance
        mean_out = self.mean_out_projection(net_out)  # mean_out shape (B, L, D)
        var_out = self.var_out_projection(net_out)  # var_out shape (B, L, D)
        return mean_out, var_out

    @property
    def optimizer(self) -> Optimizer:
        """
        The optimizer to use for training the model.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_loss(
        self,
        net_in: Tensor,
        labels: Tensor,
        masks: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute the training loss for the model.

        Args:
            net_in (batch_size, seq_len, num_features): The sequence of time
                series features that are fed into the model and used for the
                next time step prediction.
            labels (batch_size, seq_len, num_features): The labels for the
                corresponding time series features.
            masks (batch_size, seq_len): The masks which are used to indicate
                which time steps are real.

        Returns:
            * loss (Tensor): The training loss.
            * metrics (dict): A dictionary containing the training metrics.
                These metrics should be somewhat cheap to compute.
        """
        # B = batch_size, L = seq_len, D = num_features
        # B, L, D = net_in.size()
        self.train()
        masks = masks.unsqueeze(-1)  # Mask shape -> (1, B, L)
        pred_mean, pred_var = self.forward(net_in)  # Preds shape (B, L, D)
        pred_var = torch.relu(pred_var) + 1e-6
        sq_err = (labels - pred_mean).square()
        loss = 0.5 * (
            # torch.log(torch.tensor(2 * torch.pi))
            + torch.log(pred_var)
            + sq_err / pred_var
        ).mean()
        return loss, {
            "loss": loss.item(),
            "nll": loss.item(),
            "mse": sq_err.mean().item(),
        }

    def evaluation_metrics(
        self,
        net_in: Tensor,
        labels: Tensor,
        masks: Tensor,
    ) -> dict[str, float]:
        """
        Compute the evaluations loss for the model. These metrics can be
        expensive to compute. We want every metric indicative of performance
        to be returned from this method.

        Args:
            net_in (batch_size, seq_len, num_features): The sequence of time
                series features that are fed into the model and used for the
                next time step prediction.
            labels (batch_size, seq_len, num_features): The labels for the
                corresponding time series features.
            masks (batch_size, seq_len): The masks which are used to indicate
                which time steps are real.

        Returns: Evaluation metrics.
        """
        self.eval()
        masks = masks.unsqueeze(-1)
        with torch.no_grad():
            pred_mean, pred_var = self.forward(net_in)
        pred_var = torch.relu(pred_var) + 1e-6
        sq_err = (labels - pred_mean).square()
        loss = 0.5 * (
            torch.log(torch.tensor(2 * torch.pi))
            + torch.log(pred_var)
            + sq_err / pred_var
        ).mean()
        return {
            "loss": loss.item(),
            "nll": loss.item(),
            "mse": sq_err.mean().item(),
        }

    def forecast(
        self,
        observations: Tensor,
        n_steps: int,
        deterministic: bool = True,
    ) -> Tensor:
        """
        Do a forecast into the future.

        Args:
            observations (seq_len, num_features): The observations seen
                thusfar for the time series we would like to forecast. The
                observations here should be not yet be normalized.
            n_steps (int): The number of steps to forecast into the future.
            deterministic (bool): Whether to do a stochastic sample for the
                future forecast or not.

        Returns: Tensor of shape (seq_len + n_steps, num_features) containing
            the original observations plus the forecasted values.
        """
        self.eval()
        inputs = copy.deepcopy(observations)
        for _ in range(n_steps):
            with torch.no_grad():
                mean_pred, var_pred = self.forward(
                    self.normalize_input(inputs.unsqueeze(0))
                )
            mean_pred = mean_pred.squeeze(0)[[-1]]
            var_pred = var_pred.squeeze(0)[[-1]]
            if deterministic:
                pred = mean_pred
            else:
                pred = mean_pred + torch.exp(var_pred / 2) * torch.randn_like(
                    mean_pred
                )
            observations = torch.cat((observations, pred), dim=0)
            inputs = torch.cat((inputs, mean_pred), dim=0)
        return observations
