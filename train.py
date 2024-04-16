from data_loader import load_data, train_test_split
from ncf_mlp import NCF_MLP
import numpy as np
import torch
from torch import nn 
from torch.optim import Adam
import argparse
import yaml

### TODO:  enable device cuda
###        pytorch dataset class
###        new models + param tuning
###        post training analysis on users/item learning
###        trianing on 20m

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="configs.yml")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':
    data, num_users, num_items = load_data(config['movielens_data'])
   
    train, test = train_test_split(data)

    model = NCF_MLP(num_users= num_users, num_items=num_items, latent_dims=32)

    batch = config['batch_size']
    
    avg = []
    model.train(True)

    criterion = nn.MSELoss()
    opt = Adam(model.parameters(), lr = config['learning_rate'])
    for epoch in range(config['epochs']):
        for b in range(len(train)//batch):
            df = train.sample(frac=batch/len(train))

            opt.zero_grad()
            logits = model.forward(torch.tensor(df.user_id.values), torch.tensor(df.movie_id.values))
            loss = criterion(logits.squeeze(), torch.FloatTensor(df.rating.values))
            loss.backward()
            opt.step()
            avg.append(loss.item())

        print(f"epoch loss {epoch+1}:", round(sum(avg)/len(avg),5))
        avg = []

    with torch.no_grad():
        model.train(False)
        users = torch.tensor(test.user_id.values)
        items = torch.tensor(test.movie_id.values)
        
        test['prediction'] = model(users,items)

    mae = round(abs(test['prediction'] - test['rating']).mean(), 4)
    print(f'Test Set Mean Absolute Error {mae}')
        