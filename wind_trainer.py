#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:52:55 2024

@author: forootan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:35:53 2023
@author: forootani
"""

import numpy as np
import sys
import os


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir


root_dir = setting_directory(0)

from pathlib import Path
import torch
from scipy import linalg

import torch.nn as nn
import torch.nn.init as init

from siren_modules import Siren



from wind_loop_process import WindLoopProcessor

from wind_loss import wind_loss_func

from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time


warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod

##################################

class WindTrain(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def train_func(self):
        pass


###################################################
###################################################


class LSTMTrainer(WindTrain):
    def __init__(
        self,
        model,
        optim_adam,
        scheduler,
        num_epochs=1500,
        learning_rate=1e-5
    ):
        super().__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim_adam
        self.scheduler = scheduler
        self.loss_total = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_func(self, train_loader, test_loader):
        loop = tqdm(range(self.num_epochs), leave=False)

        for epoch in loop:
            self.model.train()
            loss_data_total = 0
            start_time = time.time()

            for batch_idx, (input_data, output_data) in enumerate(train_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()
                u_pred = self.model(input_data)
                
                # Check if u_pred is a tuple and extract the tensor if necessary
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]  # Extract the output tensor from the tuple

                # Ensure u_pred and output_data have the same shape
                if u_pred.shape != output_data.shape:
                    print(f"Shape mismatch: u_pred {u_pred.shape}, output_data {output_data.shape}")

                loss = self.loss_function(output_data, u_pred)
                loss.backward()
                self.optimizer.step()

                loss_data_total += loss.item()

            # Validation phase
            val_loss = self.validate(test_loader)

            # Step the scheduler with the validation loss
            self.scheduler.step(val_loss)

            avg_loss = loss_data_total / len(train_loader)
            self.loss_total.append(avg_loss)
            loop.set_postfix(training_loss=avg_loss, validation_loss=val_loss)

        self.loss_total = np.array(self.loss_total)
        return self.loss_total

    def validate(self, test_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (input_data, output_data) in enumerate(test_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                u_pred = self.model(input_data)
                
                # Check if u_pred is a tuple and extract the tensor if necessary
                if isinstance(u_pred, tuple):
                    u_pred = u_pred[0]  # Extract the output tensor from the tuple

                loss = self.loss_function(output_data, u_pred)
                val_loss += loss.item()

        return val_loss / len(test_loader)

    def loss_function(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


####################################################
####################################################


class CNNBiLSTMTrainer(WindTrain):
    def __init__(
        self,
        model,
        optim_adam,
        scheduler,
        num_epochs=1500,
        learning_rate=1e-5,
        grad_clip: float | None = None,   # set e.g. 1.0 to enable clipping
    ):
        super().__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optim_adam
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.loss_total = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_func(self, train_loader, test_loader):
        loop = tqdm(range(self.num_epochs), leave=False)

        for epoch in loop:
            self.model.train()
            loss_data_total = 0.0

            for batch_idx, (input_data, output_data) in enumerate(train_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()

                # Forward
                y_pred = self.model(input_data)
                if isinstance(y_pred, tuple):  # handle (out, hidden/aux) returns
                    y_pred = y_pred[0]

                # Shape sanity check
                if y_pred.shape != output_data.shape:
                    print(f"[Epoch {epoch} | Batch {batch_idx}] Shape mismatch: "
                          f"y_pred {tuple(y_pred.shape)} vs target {tuple(output_data.shape)}")

                loss = self.loss_function(output_data, y_pred)

                # Backward
                loss.backward()

                # Optional gradient clipping (useful for RNNs/LSTMs)
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

                loss_data_total += loss.item()

                # Validation + scheduler
                val_loss = self.validate(test_loader)
                self.scheduler.step(val_loss)

                avg_loss = loss_data_total / max(1, len(train_loader))
                self.loss_total.append(avg_loss)
                loop.set_postfix(training_loss=avg_loss, validation_loss=val_loss)

            self.loss_total = np.array(self.loss_total)
            return self.loss_total

    def validate(self, test_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (input_data, output_data) in enumerate(test_loader):
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                y_pred = self.model(input_data)
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                loss = self.loss_function(output_data, y_pred)
                val_loss += loss.item()

        return val_loss / max(1, len(test_loader))

    def loss_function(self, y_true, y_pred):
        # MSE (same as LSTMTrainer)
        return torch.mean((y_true - y_pred) ** 2)
