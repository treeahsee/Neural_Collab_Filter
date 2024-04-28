from ray.air import RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from data_loader import load_data, train_test_split_shuffle, MovielensAllMovieRatingsPerUserDataset, MovielensConcatDataset
import torch
import lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from model import VariationalAutoencoder
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def load_datasets(batch_size):
    dataset = '20m'
    data, num_users, num_items = load_data(dataset, rescale_data=True)
    all_train_data, test_data = train_test_split_shuffle(data, 0.4)
    train_data, validation_data = train_test_split_shuffle(all_train_data, 0.2)

    train = MovielensAllMovieRatingsPerUserDataset(data=train_data, num_users=num_users, num_items=num_items)
    validation = MovielensConcatDataset(dataset1=train_data, dataset2=validation_data, num_users=num_users, num_items=num_items)
    all_train = MovielensAllMovieRatingsPerUserDataset(data=all_train_data, num_users=num_users, num_items=num_items)
    test = MovielensConcatDataset(dataset1=all_train_data, dataset2=test_data, num_users=num_users, num_items=num_items)

    train_dataloader = DataLoader(train, batch_size=batch_size, num_workers=4, collate_fn=MovielensAllMovieRatingsPerUserDataset.sparse_collate)
    validation_dataloader = DataLoader(validation, batch_size=batch_size, num_workers=4, collate_fn=MovielensConcatDataset.sparse_collate)
    all_train_dataloader = DataLoader(all_train, batch_size=batch_size, num_workers=4, collate_fn=MovielensAllMovieRatingsPerUserDataset.sparse_collate)
    test_dataloader = DataLoader(test, batch_size=batch_size, num_workers=4, collate_fn=MovielensConcatDataset.sparse_collate)

    return num_users, num_items, train_dataloader, validation_dataloader, all_train_dataloader, test_dataloader


def train(config):
    model = VariationalAutoencoder(
        item_dim=num_items,
        **config,
    )
    trainer = L.Trainer(accelerator="gpu", devices="auto", max_epochs=config["max_epochs"],
                        logger=TensorBoardLogger(save_dir="logs/"))
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
    trainer.test(ckpt_path='best')


def train_ray(config):
    model = VariationalAutoencoder(
        item_dim=num_items,
        **config,
    )
    trainer = L.Trainer(accelerator="gpu", devices="auto", max_epochs=config["max_epochs"],
                    strategy=RayDDPStrategy(),
                    callbacks=[RayTrainReportCallback()],
                    plugins=[RayLightningEnvironment()],
                    logger=TensorBoardLogger(save_dir="logs/"))
    trainer = prepare_trainer(trainer)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)


def tune_ray(num_samples=4):
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "latent_dim": tune.choice([100, 200, 300]),
        "hidden_dim": tune.choice([400, 600, 800]),
        "batch_size": 1000,
        "loss_type": "mse",
        "total_anneal_steps": tune.lograndint(10000, 300000),
        "anneal_cap": tune.uniform(0.1, 0.3),
        "embedding_dim": tune.choice([64, 128, 256])
    }
    scheduler = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )
    ray_trainer = TorchTrainer(
        train_ray,
        run_config=run_config,
    )
    tuner = tune.Tuner(
        ray_trainer,
        param_space={"config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    parser = ArgumentParser()
    parser.add_argument("--tune", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser = VariationalAutoencoder.add_model_specific_args(parser)
    args = parser.parse_args()

    config = vars(args)

    num_users, num_items, train_dataloader, validation_dataloader, all_train_dataloader, test_dataloader = load_datasets(config["batch_size"])

    if config["tune"]:
        results = tune_ray()
        print(results)
    else:
        train(config)