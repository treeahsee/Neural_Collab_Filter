from data_loader import load_data, train_test_split, MovielensDataset
from ncf_mlp import NCF_MLP
import numpy as np
import torch
from torch import nn 
from torch.optim import Adam
import argparse
import yaml
from torch.utils.data import DataLoader

### TODO:  enable device cuda
###        pytorch dataset class
###        new models + param tuning
###        post training analysis on users/item learning
###        trianing on 20m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="configs.yml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()
    for batch, (user, item, y) in enumerate(dataloader):
        user, item, y = user.to(device), item.to(device), y.to(device)
        pred = model(user, item)
        loss = loss_fn(pred.squeeze(dim = 1), y.to(torch.float32))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * config['batch_size'] + len(user)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for user, item, y in dataloader:
            user, item, y = user.to(device), item.to(device), y.to(device)
            pred = model(user, item)
            test_loss += loss_fn(pred.squeeze(dim = 1), y).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    data, num_users, num_items = load_data(config['movielens_data'])
   
    train, test = train_test_split(data)

    train = MovielensDataset(users=train['user_id'], movies=train['movie_id'], ratings = train['rating'])
    test = MovielensDataset(users=test['user_id'], movies=test['movie_id'], ratings = test['rating'])

    train_dataloader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test, batch_size=config['batch_size'], shuffle=True, num_workers= 4)

    model = NCF_MLP(num_users= num_users, num_items=num_items, latent_dims=32).to(device)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])

    for t in range(config['epochs']):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")