import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, torch
from scipy import stats
import torch.nn as nn
import os

# custom api
from DataPreparer import DataPreparer
from model import FusionMLP
from train_val import train, inference

os.environ["CUDA_VISIBLE_DEVICES"]="1"

with open('../data/experiment/epr_740nm_dataset.pkl', 'rb') as file:
    ratio_740_dataset = pickle.load(file)
with open('../data/experiment/epr_850nm_dataset.pkl', 'rb') as file:
    ratio_850_dataset = pickle.load(file)

# fSaO2 dataset
with open('../data/experiment/fsao2_dataset.pkl', 'rb') as file:
    fSaO2_dataset = pickle.load(file)
with open('../data/experiment/fsao2_stats.pkl', 'rb') as file:
    fsao2_stats = pickle.load(file)


shp_rd_all = ['S4_R1_sp2022',
                'S10_R1_sp2022',
                'S5_R2_su2020',
                'S5_R1_su2020',
                'S2_R2_sp2022',
                'S10_R2_sp2022',
                'S4_R2_sp2022',
                'S1_R3_sp2021',
                ]

total_samples = sum([len(ratio_740_dataset[subject]) for subject in shp_rd_all])
print(f"Total samples: {total_samples}", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

initial_model = FusionMLP(input_size=10, hidden_size=16).to(device)
layer_arch = 'FusionMLP_16'

def weights_init_he(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data)
initial_model.apply(weights_init_he)

with open(f'../runtime/model/{layer_arch}_initial.txt', 'w') as file:
    print(initial_model, file=file)
    print("Number of parameters in the model:", sum(p.numel() for p in initial_model.parameters()), flush=True)

loaded_model = torch.load(f'../runtime/model/{layer_arch}_initial.pth')

error_history, val_loss_history, val_mae_history = {}, {}, {}
val_st_percentage = [0, 0.2, 0.4, 0.6, 0.8]
for itr, st_ratio in enumerate(val_st_percentage):
    config = {
        'method': 'temporal',
        'val_ratio': 0.2,
        'val_st_per': st_ratio
    }
    my_dataset = DataPreparer(config, [ratio_740_dataset, ratio_850_dataset], fSaO2_dataset, verbose=True)
    X_train, X_test, y_train, y_test = my_dataset.prepare_dataset()
    X_train_dict, X_test_dict, y_train_dict, y_test_dict, sample_cnt = my_dataset.prepare_dataset_dict()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    initial_model = FusionMLP(input_size=10, hidden_size=16).to(device)
    initial_model.load_state_dict(loaded_model.state_dict())

    model, _, _, val_loss, val_mae, loss_history = train(initial_model, X_train, y_train, device, sample_cnt=sample_cnt, batch_size=32, X_test=X_test, y_test=y_test, epochs=500, val_patience=25,
                                                     lr=1e-4, weight_decay=1e-4, momentum=0.98, nesterov=True)
    error_history[st_ratio] = loss_history
    val_loss_history[st_ratio] = val_loss
    val_mae_history[st_ratio] = val_mae
    torch.save(model.state_dict(), f'../results/models/experiment/{layer_arch}_trained_model_itr{itr+1}.pth')

    del model, initial_model
    torch.cuda.empty_cache()