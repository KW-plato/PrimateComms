#!/usr/bin/env python3
import torch.nn as nn
from torch.utils import data as D
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Defines the model
1. Input: Max pool
2. Conv layers: 2 X 4, Activation: ReLU, Batchnormalization: Yes, Maxpool: Yes
3. Linear Layer: Flatten, Fully Connected, Batchnormalization, ReLU, Dropout, Fully connected
"""
class ChimpModel(nn.Module):
    def __init__(self, num_labels=4):
        super().__init__()
        self.num_labels = num_labels
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels= 1, out_channels= 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64*8*7, out_features=1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=num_labels, bias=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return(x)

"""
The class to handle the dataset for training and validation:
1. Not created directly, but by calling the function that reads the main input file and spilts it into train and test
2. stores locations of the spectrograms and corresponding labels 
3. Reads the spectrogram from disk, applies transforms if any and returns a dict with the spectrogram and label
"""

class AudioSpectDataset(D.Dataset):

    def __init__(self, spectrograms, labels, transforms=None): # calling program must check len of spect and label lists are same
        self.spectrograms = spectrograms
        self.labels = labels
        self.transforms = transforms
        self.length = len(labels)

    def __getitem__(self, index):
        with open(self.spectrograms[index], 'rb') as fread:
            spect = torch.from_numpy(np.load(fread)).float().unsqueeze(dim=0)
            if self.transforms:
                spect = self.transforms(spect)
        return ({
            'spectrogram': spect,
            'label': self.labels[index]
        })

    def __len__(self):
        return self.length

    def get_datasets(inputfilelocation, train_ratio=0.90, transforms=None):  # training ratio controls fraction of data to use for training
        data = pd.read_csv(inputfilelocation)
        file_locations = data.iloc[:, 0].tolist()
        labels = data.iloc[:, 1].tolist()
        train_spects, test_spects, train_labels, test_labels = \
            train_test_split(file_locations, labels, train_size=train_ratio, random_state=42, shuffle=True)

        return AudioSpectDataset(train_spects, train_labels, transforms), \
               AudioSpectDataset(test_spects, test_labels, transforms)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, criteria="UP", delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'

        """
        self.patience = patience
        self.counter = 0
        self.criteria = criteria # GL,PQ or UP
        self.early_stop = False
        self.val_error_min = None
        self.train_errors = []
        self.delta = delta
        self.path = path

    def __call__(self, val_error, train_error, model):
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
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)


"""
First train the model and then validate its performance
"""
# Supply the source datafile and set up hyper-parameters
#input = "~/Techspace/Chimp/ml-meets-animal-communication/Helpers/TrainerModuleInput.csv"
input = "/net/projects/scratch/winter/valid_until_31_July_2021/sbiswas/Data/UniqueInput_2types.csv"
num_labels = 2
learning_rate = 0.0003
epsilon = 0.001
batch_size = 32
num_epochs = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Build the Data supply pipeline for the traning and validation
1. Fetch the training and eval dataset using AudioSpectDataset class
2. Build the dataloaders for training and validation runs
"""

data_train, data_test = AudioSpectDataset.get_datasets(inputfilelocation=input)

train_loader = D.DataLoader(data_train, batch_size=batch_size, num_workers=1, shuffle=True)
eval_loader = D.DataLoader(data_test, batch_size=batch_size, num_workers=1, shuffle=True)

"""
Getting ready for training
1. Set up the model instance
2. Set up the Loss function
3. Set up the Optimizer
4. Set up the early stopping module
"""
classifier = ChimpModel(num_labels=num_labels).float().to(device=device)

loss_func = nn.CrossEntropyLoss()

optimizer_func = torch.optim.Adam(classifier.parameters(), lr=learning_rate, eps=epsilon,weight_decay=)

early_stopping = EarlyStopping(patience=5, delta=0.05, criteria="UP", path="checkpoint.pt")

"""
Train the model
"""

train_loss = []
eval_loss = []
train_accuracy = []
eval_accuracy = []
constraintHigh = 1
constraintLow = 0

for epoch in range(num_epochs):

    iter_loss = 0.0
    correct = 0
    iterations = 0

    for i, samples in enumerate(train_loader):

        input_spects = samples['spectrogram'].to(device)
        classes = samples['label'].to(device)

        classifier.train()  # Put the network into training mode

        optimizer_func.zero_grad()  # Clear off the gradients from any past operation
        outputs = classifier(input_spects)  # Do the forward pass
        loss = loss_func(outputs, classes)  # Calculate the loss
        iter_loss += loss.item()  # accumulate the loss

        loss.backward()  # Calculate the gradients with help of back propagation
        optimizer_func.step()  # Ask the optimizer to adjust the parameters based on the gradients
        #weight normalization
        classifier.layer1.weight = torch.nn.Parameter(
            constraintLow +
            (constraintHigh - constraintLow) * (
                (classifier.layer1.weight - torch.min(classifier.layer1.weight))/(
                    torch.max(classifier.layer1.weight) - torch.min(classifier.layer1.weight)
                )
            )
        )

        # Record the correct predictions for training data
        _, predicted = torch.max(outputs, 1)  # the indexes are the predicted classes. Need only that from torch.max
        # set up metrics calculations
        correct += (predicted == classes).sum()
        iterations += 1

# Record the training loss and accuracy
    train_loss.append(iter_loss/iterations)
    train_accuracy.append(correct / len(train_loader.dataset))

############################
# Validate - How did we do on the unseen dataset?
############################

    loss = 0.0
    correct = 0
    iterations = 0
    labels_input = []
    labels_preds = []
    confusion_matrix_list = []

    classifier.eval()  # Put the network into evaluation mode

    with torch.no_grad():
        for i, samples in enumerate(eval_loader):
            input_spects = samples['spectrogram'].to(device)
            classes = samples['label'].to(device)
            outputs = classifier(input_spects)  # run inputs through the model and get output
            loss += loss_func(outputs, classes).item()  # calculate the loss
            _, predicted = torch.max(outputs, 1)  # the indexes are the predicted classes. Need only that from torch.max
            iterations += 1
            # for current iteration- set up metrics calculations and update confusion matrix
            correct += (predicted == classes).sum()
            labels_preds.append(predicted)
            labels_input.append(classes)

    # Record the validation loss
    eval_loss.append(loss / iterations)
    # Record the validation accuracy
    eval_accuracy.append(correct / len(eval_loader.dataset))

    # Code for bookkeeping metrices at end of every epoch
    all_input_labels = torch.cat(labels_input)
    all_preds_labels = torch.cat(labels_preds)
    precision = precision_score(all_input_labels, all_preds_labels, average='weighted')
    recall = recall_score(all_input_labels, all_preds_labels, average='weighted')
    f1 = f1_score(all_input_labels, all_preds_labels, average='weighted')
    cfm = confusion_matrix(all_input_labels, all_preds_labels,labels=range(num_labels))

    # build a list of dictionary with the metrics for saving at end of program
    confusion_matrix_list.append({"cfm": cfm, "accuracy": eval_accuracy[-1].item(), "precision": precision,
                                  "recall": recall, "f1": f1})
    # print out key metrics for current epoch
    print('Epoch %d/%d, Training ( Loss: %.4f, Acc: %.4f ), '
          'Validation (Loss: %.4f, Acc: %.2f , precision: %.2f, recall: %.2f, f1: %.2f) '
            % (epoch + 1, num_epochs, train_loss[-1], train_accuracy[-1] * 100,
                eval_loss[-1], eval_accuracy[-1] * 100, precision * 100, recall * 100, f1 * 100))

    early_stopping(eval_loss[-1], train_loss[-1], classifier)
    if early_stopping.early_stop:
        classifier.load_state_dict(torch.load("checkpoint.pt"))
        print("Early stopping. Last Checkpoint saved as model state")
        break

"""
save model to disk
"""
torch.save(classifier.state_dict(), "Model4_150321.pth")

"""
save metrices for every epoch to disk
"""
conf_df = pd.DataFrame(confusion_matrix_list, columns=['cfm', 'accuracy', 'precision', 'recall', 'f1'])
conf_df.to_csv("/net/projects/scratch/winter/valid_until_31_July_2021/sbiswas/Data/ConfusionMatrices.csv", header=True, index=False)

"""
Plot loss versus iterations
"""

f = plt.figure(figsize=(10, 8))
plt.plot(train_loss, label='training loss')
plt.plot(eval_loss, label='validation loss')
plt.legend()
plt.savefig("Loss.png")


"""
Plot accuracy versus iterations
"""

f = plt.figure(figsize=(10, 8))
plt.plot(train_accuracy, label='training accuracy')
plt.plot(eval_accuracy, label='validation accuracy')
plt.legend()
plt.savefig("Accuracy.png")


