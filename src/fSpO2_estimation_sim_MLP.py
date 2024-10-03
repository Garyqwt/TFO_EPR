import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, torch
import torch.nn as nn

# custom api
from DataPreparer import DataPreparer
from model import FusionMLP
from train_val import train, inference

with open('../data/simulation/epr_740nm_dataset.pkl', 'rb') as file:
    ratio_740_dataset = pickle.load(file)
with open('../data/simulation/epr_850nm_dataset.pkl', 'rb') as file:
    ratio_850_dataset = pickle.load(file)

# fSaO2 dataset
with open('../data/simulation/fsao2_dataset.pkl', 'rb') as file:
    fSaO2_dataset = pickle.load(file)
with open('../data/simulation/fsao2_stats.pkl', 'rb') as file:
    fsao2_stats = pickle.load(file)

subject_all = list(fSaO2_dataset.keys())

config = {
    'method': 'random',
    'val_ratio': 0.2
}

my_dataset = DataPreparer(config, [ratio_740_dataset, ratio_850_dataset], fSaO2_dataset, verbose=True)
X_train_dict, X_test_dict, y_train_dict, y_test_dict = my_dataset.prepare_dataset_dict()
X_train, X_test, y_train, y_test = my_dataset.prepare_dataset()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

hidden_size = 64
initial_model = FusionMLP(input_size=15, hidden_size=hidden_size).to(device)
layer_arch = f'FusionMLP_{hidden_size}'

def weights_init_he(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
initial_model.apply(weights_init_he)

torch.save(initial_model, f'../runtime/model/{layer_arch}_initial.pth')
with open(f'../runtime/model/{layer_arch}_initial.txt', 'w') as file:
    print(initial_model, file=file)
    print("Number of parameters in the model:", sum(p.numel() for p in initial_model.parameters()))

loaded_model = torch.load(f'../runtime/model/{layer_arch}_initial.pth')
initial_model.load_state_dict(loaded_model.state_dict())

model, _, _, val_loss, val_mae, loss_history = train(initial_model, X_train, y_train, device, sample_cnt=None, batch_size=64, X_test=X_test, y_test=y_test, epochs=500, val_patience=20)

torch.save(model.state_dict(), f'../results/models/simulation/{layer_arch}_trained_model.pth')






# # Inference
# model = FusionMLP(input_size=10, hidden_size=hidden_size).to(device)
# path = f'../results/models/{layer_arch}_sim_trained_model.pth'
# model.load_state_dict(torch.load(path))

# mae_results = []
# y_true, y_pred = {}, {}
# all_y_true, all_y_pred = [], []
# for idx, subject in enumerate(subject_all):
#     X_test = torch.Tensor(X_test_dict[subject])
#     y_test = np.array(y_test_dict[subject][:,0]).reshape(-1)
#     y_pred_test = inference(model, X_test, device)

#     # Unscale the labels
#     mean, std = fsao2_stats['mean'], fsao2_stats['std']
#     y_test = y_test*std + mean
#     y_pred_test = y_pred_test*std + mean

#     mae_loss = nn.L1Loss()
#     mae = mae_loss(torch.tensor(y_pred_test), torch.tensor(y_test))
#     mae = mae.item()*100
#     print(f'Depth {subject} MAE: {mae:.2f}%')

#     mae_results.append({
#         'depth': subject,
#         'mae': f'{mae:.2f}'
#     })

#     all_y_true.extend(y_test)
#     all_y_pred.extend(y_pred_test)

#     if idx % 2 == 0:
#         y_true[subject] = y_test[::30]*100
#         y_pred[subject] = y_pred_test[::30]*100
    
# overall_mae = mae_loss(torch.tensor(all_y_pred), torch.tensor(all_y_true))
# overall_mae_percentage = overall_mae.item() * 100
# print(f'Overall MAE: {overall_mae_percentage:.2f}%')

# mae_results.append({
#     'depth': 'all',
#     'mae': f'{overall_mae_percentage:.2f}'
# })

# mae_df = pd.DataFrame(mae_results)
# mae_df.to_csv(f'../results/sim_results_{layer_arch}.csv', index=False)

