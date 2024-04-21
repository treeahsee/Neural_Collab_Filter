
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from sklearn.metrics import mean_squared_error

from gmf import GMF
from ncf_mlp import NCF_MLP
from neural_mf import NEURAL_MF

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    model.train()
    for batch, (user, item, y) in enumerate(dataloader):
        user, item, y = user.to(device), item.to(device), y.to(device)
        pred = model(user, item)
        loss = loss_fn(pred.squeeze(dim = 1), y.to(torch.float32))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # If we're using a smaller dataset, print more frequently
        if size < 1_000_000:
            batch_num = 100
        else:
            batch_num = 1000

        if batch % batch_num == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(user)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    y_list = list()
    pred_list = list()
    with torch.no_grad():
        for user, item, y in dataloader:
            user, item, y = user.to(device), item.to(device), y.to(device)
            pred = model(user, item)
            # test_loss += loss_fn(pred.squeeze(dim = 1), y).item()
            y_list.extend(y.tolist())
            pred_list.extend(pred.tolist())

    # TODO: Fix this to work with Cross Entropy Loss
    # test_loss /= num_batches
    # print(f"Avg loss: {test_loss:>8f} \n")

    test_mse = mean_squared_error(y_list, pred_list)
    test_rmse = mean_squared_error(y_list, pred_list, squared=False)
    print(f"Test MSE", test_mse)
    print(f"Test RMSE", test_rmse)
    return test_rmse

def get_optimizer_by_type(model, optimizer_type, learning_rate, weight_decay):
    if optimizer_type == 'adam':
        return torch.optim.Adam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
    else:
        return torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_gmf(train_loader,
              test_loader,
              num_users,
              num_items,
              epochs,
              latent_dims,
              learning_rate,
              optimizer_type,
              criterion,
              device,
              weight_decay=None):
    model = GMF(num_users=num_users,
                num_items=num_items,
                embed_dim=latent_dims,
                output_top=True).to(device)

    optimizer = get_optimizer_by_type(model, optimizer_type, learning_rate, weight_decay)

    for i in range(epochs):
        print('Epoch', i+1)
        print('------------------------')

        train_loop(train_loader, model, criterion, optimizer, device)
        test_loop(test_loader, model, criterion, device)
    
    return model

def train_mlp(train_loader,
              test_loader,
              num_users,
              num_items,
              epochs,
              latent_dims,
              learning_rate,
              optimizer_type,
              criterion,
              device,
              weight_decay=None):
    model = NCF_MLP(num_users=num_users,
                    num_items=num_items,
                    latent_dims=latent_dims,
                    output_top=True).to(device)

    optimizer = get_optimizer_by_type(model, optimizer_type, learning_rate, weight_decay)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, criterion, optimizer, device)
        test_loop(test_loader, model, criterion, device)

    return model

def train_joint_nerual_mf(train_loader,
                          test_loader,
                          num_users,
                          num_items,
                          epochs,
                          latent_dims,
                          learning_rate,
                          optimizer_type,
                          criterion,
                          device,
                          weight_decay=None):
    """A naive implementation of GMF + MLP fusion for Neural MF.
    
    In the paper, the GMF and MLP models are pretrained so that the embeddings start
    in a good place, and there's a hyperparameter weighing the relative importance
    of each model's contriubitions at the start of training the joint model. This
    implementation, instead, simply trains everything jointly.
    """
    model = NEURAL_MF(num_users=num_users,
                      num_items=num_items,
                      latent_dims=latent_dims).to(device)

    optimizer = get_optimizer_by_type(model, optimizer_type, learning_rate, weight_decay)

    for i in range(epochs):
        print('Epoch', i+1)
        print('------------------------')

        train_loop(train_loader, model, criterion, optimizer, device)
        test_loop(test_loader, model, criterion, device)
    
    return model

# Note: specifying None for gmf/mlp is equivalent to using train_joint_neural_mf above
def train_neural_mf(train_loader,
                    test_loader,
                    gmf_pretrained,
                    mlp_pretrained,
                    num_users,
                    num_items,
                    epochs,
                    latent_dims,
                    learning_rate,
                    optimizer_type,
                    criterion,
                    device,
                    alpha,
                    weight_decay=None):

    model = NEURAL_MF(num_users=num_users,
                      num_items=num_items,
                      latent_dims=latent_dims,
                      gmf=gmf_pretrained,
                      mlp=mlp_pretrained,
                      alpha=alpha).to(device)
    
    optimizer = get_optimizer_by_type(model, optimizer_type, learning_rate, weight_decay)

    for i in range(epochs):
        print('Epoch', i+1)
        print('------------------------')

        train_loop(train_loader, model, criterion, optimizer, device)
        test_loop(test_loader, model, criterion, device)
    
    return model
