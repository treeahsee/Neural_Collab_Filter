from data_loader import load_data, train_test_split, MovielensDataset
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import argparse
import yaml
from torch.utils.data import DataLoader
import tqdm
from sklearn.metrics import mean_squared_error

from gmf import GMF
from ncf_mlp import NCF_MLP
from neural_mf import NEURAL_MF

from train import train_gmf, train_joint_nerual_mf, train_mlp, train_neural_mf

### TODO:  new models + param tuning
###        post training analysis on users/item learning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="configs.yml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Ensure we don't error out if optional params aren't set
def get_optional_config(key, default=None):
    return config[key] if key in config else default

if __name__ == '__main__':
    data, num_users, num_items = load_data(config['movielens_data'])
    train, test = train_test_split(data)

    train = MovielensDataset(users=train['user_id'], movies=train['movie_id'], ratings = train['rating'])
    test = MovielensDataset(users=test['user_id'], movies=test['movie_id'], ratings = test['rating'])

    train_dataloader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test, batch_size=config['batch_size'], shuffle=True, num_workers= 4)

    epochs = config['epochs']
    latent_dims = config['latent_dims']
    learning_rate = config['learning_rate']
    optimizer_type = config['optimizer']

    weight_decay = get_optional_config('weight_decay')
    alpha = get_optional_config('alpha', 0.5)

    if config['model'] == 'gmf':
        model = train_gmf(train_dataloader,
                          test_dataloader,
                          num_users,
                          num_items,
                          epochs,
                          latent_dims,
                          learning_rate,
                          optimizer_type,
                          criterion=nn.CrossEntropyLoss(),
                          device=device,
                          weight_decay=weight_decay
                          )

    elif config['model'] == 'mlp':
        model = train_mlp(train_dataloader,
                          test_dataloader,
                          num_users,
                          num_items,
                          epochs,
                          latent_dims,
                          learning_rate,
                          optimizer_type,
                          criterion=nn.CrossEntropyLoss(),
                          device=device,
                          weight_decay=weight_decay
                          )

    elif config['model'] == 'nmf':
        if 'gmf_weights_file' in config:
            print('Loading pretrained GMF')
            gmf = GMF(num_users, num_items, latent_dims, output_top=False)
            gmf.load_state_dict(torch.load(config['gmf_weights_file']), strict=False)
            gmf = gmf.to(device)
        else:
            print('Training GMF')
            print('----------------------')
            gmf = train_gmf(train_dataloader,
                            test_dataloader,
                            num_users,
                            num_items,
                            epochs,
                            latent_dims,
                            learning_rate,
                            optimizer_type,
                            criterion=nn.CrossEntropyLoss(),
                            device=device,
                            weight_decay=weight_decay
                            )

        if 'mlp_weights_file' in config:
            print('Loading pretrained MLP')
            mlp = NCF_MLP(num_users, num_items, latent_dims, output_top=False)
            mlp.load_state_dict(torch.load(config['mlp_weights_file']), strict=False)
            mlp = mlp.to(device)
        else:
            print('Training MLP')
            print('----------------------')
            mlp = train_mlp(train_dataloader,
                            test_dataloader,
                            num_users,
                            num_items,
                            epochs,
                            latent_dims,
                            learning_rate,
                            optimizer_type,
                            criterion=nn.CrossEntropyLoss(),
                            device=device,
                            weight_decay=weight_decay
                            )
        
        gmf.output_top = False
        mlp.output_top = False

        print('Training NEURAL MF WITH PRETRAINED GMF, MLP')
        print('----------------------')
        model = train_neural_mf(train_dataloader,
                               test_dataloader,
                               gmf,
                               mlp,
                               num_users,
                               num_items,
                               epochs,
                               latent_dims,
                               learning_rate,
                               optimizer_type,
                               criterion=nn.CrossEntropyLoss(),
                               device=device,
                               alpha=alpha,
                               weight_decay=weight_decay
                               )

    else:
        model = train_joint_nerual_mf(train_dataloader,
                                      test_dataloader,
                                      num_users,
                                      num_items,
                                      epochs,
                                      latent_dims,
                                      learning_rate,
                                      optimizer_type,
                                      criterion=nn.CrossEntropyLoss(),
                                      device=device,
                                      weight_decay=weight_decay
                                      )

    # Save the trained model to the specified file path
    if 'weight_file' in config:
        torch.save(model.state_dict(), config['weight_file'])

    print("Done!")
