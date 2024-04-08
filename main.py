from data_loader import load_data, train_test_split
from ncf_mlp import NCF_MLP
import numpy as np
import torch
from torch import nn 
from torch.optim import SparseAdam,Adam,Adagrad,SGD

if __name__ == '__main__':
    data, num_users, num_items = load_data()
   
    train, test = train_test_split(data)

    model = NCF_MLP(num_users= num_users, num_items=num_items, latent_dims=32)

    batch = 1000
    
    avg = []
    mx = []
    states = {}
    model.train(True)

    criterion = nn.MSELoss()
    opt = Adam(model.parameters(), lr = 1e-3)
    for epoch in range(8):
        for b in range(len(train)//batch):
            df = train.sample(frac=batch/len(train))

            opt.zero_grad()
            logits = model.forward(torch.tensor(df.user_id.values), torch.tensor(df.movie_id.values))
            loss = criterion(logits, torch.FloatTensor(df.rating.values).reshape(-1, 1))
            loss.backward()
            opt.step()
            avg.append(loss.item())

        print(f"EPOCH {epoch+1}:", round(sum(avg)/len(avg),5))
        avg = []

    with torch.no_grad():
        model.train(False)
        users = test.user_id.values
        items = test.movie_id.values
        test['prediction'] = model(users,items).numpy()

    mae = round(abs(test['prediction'] - test['rating']).mean(), 4)
    print(f'Test Set Mean Absolute Error {mae}')
        