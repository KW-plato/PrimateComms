"""
train_spectrogram.py
Author: Saurabh Biswas
Institution: University of Osnabrueck

Train a classifier from the spectrograms.

Inputs:
1. Data: Path to the CSV file with location and call-type label of each spectrogram to be used in training
2. (Optional) Output directory. To restart from a checkpoint the exact output directory needs to be supplied.
The exact option to restart is printed to the console during every run for easy reference.

Hyperparameters need to be changed in the code

Output:
1. The trained model (the best performing epoch)
2. CSV with metrics for all epochs

How to run? Examples are provided below:
train_spectrogram.py mFolder/training_samples.csv
train_spectrogram.py myFolder/training_samples.csv -o myOutputFolder
"""
import argparse
from datetime import datetime
from handlers.datahandler import AudioSpectDataset
from models.chimp_model import ChimpCallClassifier
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils import data
import warnings
warnings.filterwarnings('once')

def create_folders(dname):
    try:
        os.makedirs(dname)
        print("Folder created!")
    except Exception as e:
        print("Cant create folder {} {}".format(dname, e))
        exit()


def parse_cli():
    """
    parses command line arguments
    """
    now = datetime.now()
    tmpstamp = now.strftime("%d%m%-y-%H%M%S")
    model_name = 'CNN' + tmpstamp

    cmd_parse = argparse.ArgumentParser(
        usage= "%(prog)s [-o OUTPUT] Samples.csv",
        description="Training a Custom CNN \n e.g. python %(prog)s myfolder/training_data.csv")
    cmd_parse.add_argument("Samples",
                           help="Location of the CSV file with location and label of each spectrogram to be used in training")

    cmd_parse.add_argument("-o", "--output",
                           help="Directory where outputs are placed. Folder created in current directory if not provided. Mandatory to restart from a checkpoint")

    args = cmd_parse.parse_args()
    if not os.path.isfile(args.Samples):
        print("Training data file not found. Exiting...")
        exit()

    if args.output is None:
        curr_dir = os.getcwd()
        output_dir = os.path.join(curr_dir,"output",model_name)
        print("Outputs will be in folder {}".format(output_dir))
        create_folders(output_dir)
    else:
        if not os.path.isdir(args.output):
            print("Provided Output location either doesn't exist or isn't a directory.")
            exit()
        else:
            output_dir = args.output
            print("Outputs will be in folder {}".format(output_dir))

    return (model_name, args.Samples, output_dir)

def get_labels(data_file):
    try:
        df = pd.read_csv(data_file)
        labels = df['label'].drop_duplicates().sort_values().to_list()
        return len(labels), {labels[i]: i for i in range(len(labels))}
    except Exception as e:
        print("Couldn't get call-type labels from input samples.")
        print(e)
        print("Exiting...")
        exit()

def train_model(training_data, output_dir, params):

    """
    Create the output file paths
    """
    checkpoint_file = os.path.join(output_dir,"chkpt.pth")
    trained_model = os.path.join(output_dir,"model.pth")
    metrics_file = os.path.join(output_dir,'metrics.csv')
    best_model_file = os.path.join(output_dir,'model_early_stop.pth')

    """
    Build the Data supply pipeline for the training and validation
    """
    input_data = {}
    input_data['train'], input_data['eval'] = AudioSpectDataset.get_datasets(
        inputfile=training_data,
        train_ratio=0.9,
        label_dict=params["label_dict"],
    )

    dataset_sizes = {x: len(input_data[x]) for x in ['train', 'eval']}

    """
    Handle class imbalance by oversampling. Define the sampler
    """
    all_labels = input_data['train'].get_all_labels()
    class_freq = np.bincount(all_labels)
    class_weights = 100.0 / class_freq
    each_samples_weight = class_weights[all_labels]
    sampler = data.WeightedRandomSampler(each_samples_weight, len(input_data['train']))

    dataloaders = {}
    for x in ['train', 'eval']:
        if x == 'train':
            dataloaders[x] = data.DataLoader(input_data[x],
                                             batch_size=params["batch_size"],
                                             sampler=sampler,
                                             num_workers=1)
        else:
            dataloaders[x] = data.DataLoader(input_data[x],
                                             batch_size=params["batch_size"],
                                             num_workers=1)
    """
    Getting ready for training
    1. Set up the model instance. 
    2. If checkpoint exists from an interrupted previous run, load from the checkpoint
    3. Set up the Loss function
    4. Set up the Optimizer
    """
    classifier = ChimpCallClassifier(
        num_labels=params["num_labels"],
        spectrogram_shape=params["spectrogram_shape"],
        dropout=params["dropout"]
    ).float()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            classifier = nn.DataParallel(classifier)
    classifier.to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer_func = torch.optim.Adam(
        classifier.parameters(),
        lr=params["learning_rate"],
        eps=params["epsilon"],
        weight_decay=params["weight_decay"]
    )

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_func, step_size=params["scheduler_step_size"], gamma=0.1)

    # if checkpoint from an interrupted run exists, load the model, optimizer, early stopper and metrics
    if os.path.exists(checkpoint_file):
        try:
            checkpoint_data = torch.load(checkpoint_file)
            classifier.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer_func.load_state_dict(checkpoint_data['optim_state_dict'])
            start_epoch = checkpoint_data['epoch'] + 1
            metrics_list = checkpoint_data['metrics']
            lowest_loss = checkpoint_data['lowest_loss']
            best_epoch = checkpoint_data['best_epoch']
            print("Checkpoint found. Loaded!")
            print("Re-starting training with epoch# {}".format(start_epoch + 1)) # epochs shown on screen start from 1
        except Exception as e:
            print("Error in reinstating interrupted run. Error: {}".format(e))
            exit()
    else:
        start_epoch = 0
        metrics_list = []
        lowest_loss = 1e14
        best_epoch = 0
    """
    Training and validation
    """
    print("*" * 70)
    print("Training Starting")
    print("To restart, when launching use the option: -o {}".format(output_dir))
    print("*" * 70)
    for epoch in range(start_epoch, params["num_epochs"]):
        epoch_loss = {'train': 0.0, 'eval': 0.0}
        epoch_accuracy = {'train': 0.0, 'eval': 0.0}
        labels_input = []
        labels_preds = []
        precision = recall = f1 = 0

        for phase in ['train', 'eval']:
            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                classifier.train()
            else:
                classifier.eval()

            for samples in dataloaders[phase]:
                input_spects = samples['spectrogram'].to(device)
                input_labels = samples['label'].to(device)
                optimizer_func.zero_grad()  # Clear off the gradients from any past operation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = classifier(input_spects)  # Do the forward pass
                    loss = loss_func(outputs, input_labels)  # Calculate the loss
                    if phase == 'train':
                        loss.backward()  # Calculate the gradients with help of back propagation
                        optimizer_func.step()  # Ask the optimizer to adjust the parameters based on the gradients
                # Record the predictions
                _, predicted = torch.max(outputs, 1)  # the indexes are the predicted classes. Need only that from torch.max
                # set up metrics
                running_loss += loss.item() * input_labels.size(0)  # accumulate the loss
                running_corrects += (predicted == input_labels).sum()
                if phase == 'eval':
                    labels_preds.append(predicted)
                    labels_input.append(input_labels)

            if phase == 'train':
                exp_lr_scheduler.step()

            """
            Calculate performance metrics for the train and eval runs of this epoch
            """
            epoch_loss[phase] = running_loss / dataset_sizes[phase]
            epoch_accuracy[phase] = running_corrects.item() / dataset_sizes[phase]
            if phase == 'eval':
                all_input_labels = torch.cat(labels_input).cpu()
                all_preds_labels = torch.cat(labels_preds).cpu()
                precision = precision_score(all_input_labels, all_preds_labels, average='weighted')
                recall = recall_score(all_input_labels, all_preds_labels, average='weighted')
                f1 = f1_score(all_input_labels, all_preds_labels, average='weighted')
                cfm = confusion_matrix(all_input_labels, all_preds_labels, labels=range(params["num_labels"]))

        # Finish up the Epoch: Save model & optimizer state, metrics and earlystop. Print performance. Check early stopping
        metrics_list.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss['train'],
                "train_acc": epoch_accuracy['train'],
                "eval_loss": epoch_loss['eval'],
                "accuracy": epoch_accuracy['eval'],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cfm": cfm
            }
            )
        print('Epoch %2d/%d, Training (Loss: %.4f, Acc: %.2f ), '
                      'Validation (Loss: %.4f, Acc: %.2f , precision: %.2f, recall: %.2f, f1: %.2f) '
                      % (epoch + 1, params["num_epochs"], epoch_loss['train'], epoch_accuracy['train'] * 100,
                         epoch_loss['eval'], epoch_accuracy['eval'] * 100, precision * 100, recall * 100, f1 * 100))

        if epoch_loss['eval'] < lowest_loss:
            best_epoch = epoch
            lowest_loss = epoch_loss['eval']
            torch.save(classifier.state_dict(), best_model_file)


        torch.save(
            {
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optim_state_dict': optimizer_func.state_dict(),
                    'best_epoch': best_epoch,
                    'lowest_loss': lowest_loss,
                    'metrics': metrics_list
            }, checkpoint_file)

    """
    Save to disk: final model and all epoch metrics as csv file
    Final Model: the best performing model is always saved at the location 'best_model_file' because of the Early Stop code
    """
    classifier.load_state_dict(torch.load(best_model_file))
    torch.save(classifier.state_dict(), trained_model)
    pd.DataFrame(
        metrics_list,
        columns=["epoch","train_loss","train_acc","eval_loss","accuracy","precision","recall","f1","cfm"]
        ).to_csv(metrics_file, index=False, header=True)

    #  clean up, remove the temporary files used to store runtime state
    if os.path.exists(best_model_file):
        os.remove(best_model_file)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print("Training Complete! Epoch #{:2d} saved".format(best_epoch + 1))
    print("Model at: {}".format(trained_model))
    print("Metrics at: {}".format(metrics_file))


if __name__ == "__main__":

    # parse command line inputs, get the input csv, the call labels and output folder
    model_name, training_data_file, output_directory = parse_cli()

    # Set up the parameters
    params = {}
    params["model_name"] = model_name
    params["spectrogram_shape"] = (257, 254)
    params["num_labels"], params["label_dict"] = get_labels(training_data_file)
    print("Call type labels for training: {}".format(list(params["label_dict"].keys())))
    params["learning_rate"] = 0.0003
    params["epsilon"] = 0.001
    params["weight_decay"] = 0.05
    params["batch_size"] = 32
    params["num_epochs"] = 30
    params["dropout"] = 0.4
    params["scheduler_step_size"] = 7

    # call the trainer function
    train_model(training_data_file, output_directory, params)




