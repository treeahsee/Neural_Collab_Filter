import warnings
from typing import Union

import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.retrieval import RetrievalNormalizedDCG


def get_optimizer_by_type(model, optimizer_type, learning_rate, weight_decay):
    if optimizer_type == 'adam':
        return torch.optim.Adam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
    else:
        return torch.optim.SGD(model.parameters(), lr=learning_rate)


def get_loss_by_type(input, target, loss_type):
    # result and input are n x m ratings
    if loss_type == "mse":
        return F.mse_loss(F.sigmoid(input), target, reduction="none")
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(input, target, reduction="none")
    if loss_type == "kld":
        return F.kl_div(F.sigmoid(input), target, reduction="none")


# Basic structure referenced from https://avandekleut.github.io/vae/
class VariationalEncoder(nn.Module):
    def __init__(self, item_dim, embedding_dim, latent_dim, hidden_dim):
        super(VariationalEncoder, self).__init__()
        self.item_embedding = nn.Embedding(item_dim, embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.mlp_input = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mlp_mean = nn.Linear(hidden_dim, latent_dim)
        self.mlp_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.sparse_coo):
        e = torch.sparse.mm(x, self.item_embedding.weight) / torch.bincount(x.indices()[0], minlength=x.shape[0]).unsqueeze(1)
        if self.training:
            e = self.dropout(e)
        h = self.mlp_input(e)
        mu = self.mlp_mean(h)
        log_sigma = self.mlp_var(h)
        return mu, log_sigma


class VariationalDecoder(nn.Module):
    def __init__(self, item_dim, latent_dim, hidden_dim):
        super(VariationalDecoder, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.mlp_output = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, item_dim),
        )

    def forward(self, x):
        if self.training:
            x = self.dropout(x)
        return self.mlp_output(x)


class VariationalAutoencoder(L.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VariationalAutoencoder")
        parser.add_argument("--embedding_dim", type=int, default=128)
        parser.add_argument("--latent_dim", type=int, default=200)
        parser.add_argument("--hidden_dim", type=int, default=600)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--loss_type", type=str, default="mse")
        parser.add_argument("--total_anneal_steps", type=int, default=20000)
        parser.add_argument("--anneal_cap", type=float, default=0.2)
        return parent_parser

    def __init__(self, item_dim, embedding_dim=128, latent_dim=200, hidden_dim=600, learning_rate=1e-3, weight_decay=1e-4, loss_type="mse", total_anneal_steps=20000, anneal_cap=0.2, **kwargs):
        super(VariationalAutoencoder, self).__init__()
        self.item_dim = item_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap
        self.update_count = 0
        self.encoder = VariationalEncoder(item_dim, embedding_dim, latent_dim, hidden_dim)
        self.decoder = VariationalDecoder(item_dim, latent_dim, hidden_dim)

    def random_masking(self, x: torch.sparse_coo):
        warnings.filterwarnings(action='ignore', category=UserWarning)
        # x[0] of user ids are sorted
        indices = x.indices()
        user_ids, num_movie_ratings_per_user = torch.unique_consecutive(indices[0], return_counts=True)
        movie_ratings_offset_per_user = torch.cumsum(num_movie_ratings_per_user, dim=0)
        k_ratings_to_omit_per_user = torch.randint(self.item_dim, size=(10, num_movie_ratings_per_user.size()[0]), device=self.device) % num_movie_ratings_per_user
        masked_movie_ratings_offset = movie_ratings_offset_per_user - k_ratings_to_omit_per_user - 1
        aggregated_masked_movie_ratings_indices = masked_movie_ratings_offset.unsqueeze(0).unique()
        keep_movie_ratings_indices = torch.ones(indices.shape[1], dtype=torch.bool)
        keep_movie_ratings_indices[aggregated_masked_movie_ratings_indices] = False
        keep_mask = torch.sparse_coo_tensor(
            indices[:, keep_movie_ratings_indices],
            torch.ones_like(torch.where(keep_movie_ratings_indices)[0], dtype=torch.bool),
            size=(x.size()[0], self.item_dim),
            device=self.device
        )
        mask_mask = torch.sparse_coo_tensor(
            indices[:, aggregated_masked_movie_ratings_indices],
            torch.ones_like(aggregated_masked_movie_ratings_indices, dtype=torch.bool),
            size=(x.size()[0], self.item_dim),
            device=self.device
        )
        keep_x = torch.masked.masked_tensor(x, keep_mask)
        mask_x = torch.masked.masked_tensor(x, mask_mask)
        return keep_x, mask_x

    def training_step(self, batch, batch_idx):
        user_ids, train_ratings = batch
        keep_ratings, discard_ratings = self.random_masking(train_ratings)
        mu, log_sigma = self.encoder.forward(keep_ratings.get_data())
        sigma = torch.exp(0.5 * log_sigma)
        z = mu + sigma * torch.randn_like(sigma)
        raw_output = self.decoder(z)
        kld_loss = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=1)
        anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
        self.update_count += 1
        loss = self._train_loss_fn(raw_output, train_ratings) + anneal * kld_loss.mean()
        self.log("train_loss", loss, batch_size=len(batch[0]))
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, train_ratings, test_ratings = batch
        mu, log_sigma = self.encoder.forward(train_ratings)
        raw_output = self.decoder(mu)
        loss = self._train_loss_fn(raw_output, train_ratings)
        self.log("val_loss", loss, batch_size=len(batch[0]))
        return loss

    def test_step(self, batch, batch_idx):
        user_ids, train_ratings, test_ratings = batch
        mu, log_sigma = self.encoder.forward(train_ratings)
        raw_output = self.decoder(mu)
        loss, ndcg = self._test_loss_fn(raw_output, test_ratings)
        metrics = {"test_ndcg@100": ndcg, "test_loss": loss}
        self.log_dict(metrics, batch_size=len(batch[0]))
        return metrics

    def configure_optimizers(self):
        optimizer = get_optimizer_by_type(self, "adam", self.learning_rate, self.weight_decay)
        return optimizer


    def _train_loss_fn(self, raw_output, target_ratings):
        input_loss = self._target_loss_sparse(raw_output, target_ratings)
        return input_loss.mean()

    def _test_loss_fn(self, raw_output, output_ratings):
        loss = self._target_loss_sparse(raw_output, output_ratings)
        pred = F.sigmoid(raw_output).sparse_mask(output_ratings).values()
        target = (output_ratings.values() * 5).long()
        ndcg = RetrievalNormalizedDCG(top_k=100)(pred, target, output_ratings.indices()[1])
        return loss.mean(), ndcg

    def _target_loss_sparse(self, input, target):
        values = get_loss_by_type(input.sparse_mask(target).values(), target.values(), self.loss_type)
        formatted_loss = torch.sparse_coo_tensor(target.indices(), values, size=target.size())
        return torch.sparse.sum(formatted_loss, dim=1).to_dense() / torch.bincount(target.indices()[0], minlength=target.shape[0])