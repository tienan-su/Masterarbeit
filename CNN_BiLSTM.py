#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:24:38 2025

@author: forootan
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_features, cnn_filters=64, kernel_size=3,
                 lstm_hidden=64, lstm_layers=1, output_size=1, dropout=0.5):
        super(CNN_BiLSTM, self).__init__()
        
        # 1D CNN Layer
        self.conv1d = nn.Conv1d(in_channels=input_features,
                                out_channels=cnn_filters,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2)  # same padding
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # BiLSTM Layer
        self.bilstm = nn.LSTM(input_size=cnn_filters,
                              hidden_size=lstm_hidden,
                              num_layers=lstm_layers,
                              batch_first=True,  # batch size is the first dimension
                              dropout=dropout if lstm_layers > 1 else 0,  # dropout only if there are multiple layers (prevents overfitting)
                              bidirectional=True)  # output size is 2*lstm_hidden

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden, output_size)  # 2 for bidirectional # make output size 1 for regression

    def forward(self, x):
        # Input shape: (batch, time, features)
        x = x.permute(0, 2, 1)  # to (batch, features, time) for Conv1d

        x = self.conv1d(x)       # -> (batch, cnn_filters, time)
        x = self.relu(x)
        x = self.pool(x)         # -> (batch, cnn_filters, time/2)

        x = x.permute(0, 2, 1)   # back to (batch, time, features) for LSTM

        output, _ = self.bilstm(x)  # -> (batch, time, 2*lstm_hidden) 
                                    # output, (hn, cn) = self.bilstm(x) # hn is the hidden state of the last time step, cn is the cell state of the last time step
        x = output[:, -1, :]        # Take last time step

        x = self.dropout(x)
        out = self.fc(x)            # -> (batch, output_size)
        return out


# Dummy example
model = CNN_BiLSTM(input_features=10, output_size=1)
'''
x = torch.randn(32, 50, 10)  # (batch_size, time_steps, input_features)
y = model(x).detach()
# print(y.shape)  # -> torch.Size([32, 1])
# print(f'x: {x[0]}')
# print(f'x.shape: {x.shape}')
print(f'y: {y}')
print(f'y.shape: {y.shape}')
'''


# Simulate Realistic Energy Data
import numpy as np

def generate_energy_data(seq_len, batch_size, input_features):
    hours = torch.arange(seq_len).repeat(batch_size, 1) % 24
    days = torch.arange(seq_len).repeat(batch_size, 1) // 24 % 7

    # Temperature
    base_temp = 10 + 10 * torch.sin((hours / 24) * 2 * np.pi)
    temp_noise = torch.randn_like(base_temp) * 1.5
    temperature = base_temp + temp_noise

    # Derived usage
    load = (
        1.5 +
        0.03 * (temperature - 22).clamp(min=0) +
        0.04 * (10 - temperature).clamp(min=0) +
        0.1 * torch.sin((hours / 24) * 2 * np.pi) +
        0.05 * torch.randn_like(temperature)
    )

    # Additional synthetic features
    humidity = 50 + 10 * torch.randn_like(temperature)
    solar = (torch.sin((hours / 24) * np.pi) * 800).clamp(min=0)  # 0~800 W/m²
    pressure = 1013 + 5 * torch.randn_like(temperature)
    wind_speed = torch.abs(torch.randn_like(temperature) * 3)
    co2 = 400 + 20 * torch.randn_like(temperature)
    weekend_flag = ((days >= 5).float())  # Saturday/Sunday = 1

    # Normalize time-related features
    hour_norm = hours / 24.0
    day_norm = days / 7.0

    # Stack all features
    features = torch.stack([
        temperature,
        hour_norm,
        day_norm,
        humidity,
        solar,
        pressure,
        wind_speed,
        co2,
        weekend_flag,
        load  # you can also treat this as a lagged feature
    ], dim=-1)  # (batch, time, 10 features)

    # Target: predict electricity at final time step
    x = features[:, :-1, :]          # (batch, 49, 10)
    y = load[:, -1].unsqueeze(-1)    # (batch, 1)

    return x.float(), y.float()

def generate_energy_dataset(n_samples=1024, seq_len=51, batch_size=32):
    all_x = []
    all_y = []
    for _ in range(n_samples // batch_size):
        x, y = generate_energy_data(seq_len=seq_len, batch_size=batch_size, input_features=10)
        all_x.append(x)
        all_y.append(y)
    return torch.cat(all_x), torch.cat(all_y)


x, y = generate_energy_dataset(n_samples=1024, seq_len=51, batch_size=32)

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# train the model
import torch.optim as optim

def train_model(model, dataloader, epochs=100, lr=0.001, criterion=nn.MSELoss()):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    model.train()  # Put model in training mode

    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        loss_value = loss.item()  # Convert to float (detach from graph)
        loss_history.append(loss_value)
        print(f'Epoch {epoch+1}, Loss: {loss_value:.4f}')
        
        
    return loss_history, loss_value

# y = torch.randn(32, 1)
loss_history, loss_value = train_model(model, loader, epochs=100, lr=0.001)


# plot the loss
def plot_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.savefig("CNN_BiLSTM_loss_curve.png")


plot_loss(loss_history)

import os
os.system("open CNN_BiLSTM_loss_curve.png")

