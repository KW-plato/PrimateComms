""""
Module: early_stopping.py
Author: Saurabh Biswas
Institution: University of Osnabrueck
Created on: 15.03.2021

Early stops the training if validation loss doesn't improve after a given patience.
Ref: "Early Stopping -- but when?" by Lutz Prechelt. https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf

"""
import torch

class EarlyStopping:

    def __init__(self, patience=5, criteria="UP", delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            criteria (str): Critertia type for determining stop point. Allowed values : 'GL/'PQ'/'UP'
                            Default: 'UP'
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0. Always zero for criteria 'UP'
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'

        """
        self.patience = patience
        self.counter = 0
        self.criteria = criteria
        self.early_stop = False
        self.val_error_min = None
        self.train_errors = []
        self.delta = delta
        self.path = path

    def __call__(self, val_error=0, train_error=0, model=None):
        """
        Args:
            val_error (float): validation error
                            Default: 5
            train_error (float): training error. Criteria 'UP' doesnt use training error.
            model (torch.nn.models) : the models state for the iteration
        """
        if self.criteria == "GL":
            if self.val_error_min is None:
                self.val_error_min = val_error
                self.save_checkpoint(model)
            elif (val_error / self.val_error_min - 1) >= self.delta:
                self.early_stop = True
            elif val_error < self.val_error_min:
                self.val_error_min = val_error
                self.save_checkpoint(model)
        if self.criteria == "UP":
            if self.val_error_min is None:
                self.val_error_min = val_error
                self.save_checkpoint(model)
            elif val_error > self.val_error_min:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.val_error_min = val_error
                self.save_checkpoint(model)
                self.counter = 0
        if self.criteria == "PQ":
            if self.val_error_min is None:
                self.val_error_min = val_error
                self.save_checkpoint(model)
                self.train_errors.append(train_error)
                self.counter += 1
            else:
                self.train_errors.append(train_error)
                self.counter += 1
                if val_error < self.val_error_min:
                    self.val_error_min = val_error
                    self.save_checkpoint(model)
                if self.counter >= self.patience:
                    PK = sum(self.train_errors) / (self.counter * min(self.train_errors)) - 1
                    GL = val_error / self.val_error_min - 1
                    if GL / PK > self.delta:
                        self.early_stop = True
                    else:
                        self.counter = 0
                        self.train_errors = []


    def save_checkpoint(self, model):
        """
        Saves models on any iteration with validation loss lower than previous threshold
        """
        torch.save(model.state_dict(), self.path)

