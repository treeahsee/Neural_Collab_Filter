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
    if config['criterion'] == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
        top_depth = 1 # Don't use sigmoid activation with classification task
        rescale_data = False
    else: # MSE By Default, can support others with an elif later if necessary
        criterion = nn.MSELoss()
        top_depth = get_optional_config('top_depth', 1)
        rescale_data = True

    print("Using Loss", criterion)
    print("Rescaling Data", rescale_data)
    data, num_users, num_items = load_data(config['movielens_data'], rescale_data=rescale_data)
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
    top_depth = get_optional_config('top_depth', 1)

    if config['model'] == 'gmf':
        model = train_gmf(train_dataloader,
                          test_dataloader,
                          num_users,
                          num_items,
                          epochs,
                          latent_dims,
                          learning_rate,
                          optimizer_type,
                          criterion=criterion,
                          device=device,
                          weight_decay=weight_decay,
                          rescale_data=rescale_data
                          )[0]

    elif config['model'] == 'mlp':
        model = train_mlp(train_dataloader,
                          test_dataloader,
                          num_users,
                          num_items,
                          epochs,
                          latent_dims,
                          learning_rate,
                          optimizer_type,
                          criterion=criterion,
                          device=device,
                          weight_decay=weight_decay,
                          rescale_data=rescale_data
                          )[0]

    elif config['model'] == 'nmf':
        if 'gmf_weights_file' in config:
            print('Loading pretrained GMF')
            gmf = GMF(num_users, num_items, latent_dims, top_depth=top_depth)
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
                            criterion=criterion,
                            device=device,
                            weight_decay=weight_decay,
                            top_depth=top_depth,
                            rescale_data=rescale_data
                            )[0]

        if 'mlp_weights_file' in config:
            print('Loading pretrained MLP')
            mlp = NCF_MLP(num_users, num_items, latent_dims, top_depth=top_depth)
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
                            criterion=criterion,
                            device=device,
                            weight_decay=weight_decay,
                            top_depth=top_depth,
                            rescale_data=rescale_data
                            )[0]
        
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
                               criterion=criterion,
                               device=device,
                               alpha=alpha,
                               weight_decay=weight_decay,
                               top_depth=top_depth,
                               rescale_data=rescale_data
                               )[0]

    else:
        model = train_joint_nerual_mf(train_dataloader,
                                      test_dataloader,
                                      num_users,
                                      num_items,
                                      epochs,
                                      latent_dims,
                                      learning_rate,
                                      optimizer_type,
                                      criterion=criterion,
                                      device=device,
                                      weight_decay=weight_decay,
                                      rescale_data=rescale_data
                                      )

    # Save the trained model to the specified file path
    if 'weight_file' in config:
        torch.save(model.state_dict(), config['weight_file'])

    print("Done!")
