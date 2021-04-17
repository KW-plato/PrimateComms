import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from models.chimp_model import ChimpCallClassifier
from handlers.datahandler import AudioSpectDataset
from handlers.early_stopping import EarlyStopping


# Supply the source datafile and set up hyper-parameters
input = "~/Techspace/Chimp/ml-meets-animal-communication/Helpers/TrainerModuleInput.csv"
model_name = 'Model_1'
num_labels = 4
spectrogram_shape = (257, 254)
label_dict = {'ph': 0, 'sm': 1, 'phsm': 2, 'phtbsm': 3}
learning_rate = 0.0003
epsilon = 0.001
weight_decay = 0.1
batch_size = 36
num_epochs = 60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Build the Data supply pipeline for the training and validation
1. Fetch the training and eval dataset using AudioSpectDataset class
"""

dataset = {}
dataset['train'], dataset['eval'] = AudioSpectDataset.get_datasets(inputfilelocation=input, label_dict=label_dict)

"""
Getting ready for training
1. Set up the model instance
2. Set up the Loss function
3. Set up the Optimizer
4. Set up the early stopping module
"""
classifier = ChimpCallClassifier(
    num_labels=num_labels,
    spectrogram_shape=spectrogram_shape
    ).float().to(device=device)

loss_func = torch.nn.CrossEntropyLoss()

optimizer_func = torch.optim.Adam(
    classifier.parameters(),
    lr=learning_rate,
    eps=epsilon,
    weight_decay=weight_decay
    )

earlystop = EarlyStopping(
    patience=5,
    criteria="UP",
    path="checkpoint.pt"
    )

"""
create the storage for metrics, this is saved as an output along with the trained model
"""
confusion_matrix_list = []

"""
Training and validation
"""
for epoch in range(num_epochs):
    epoch_loss = {'train': 0.0, 'eval': 0.0}
    epoch_accuracy = {'train': 0.0, 'eval': 0.0}
    labels_input = []
    labels_preds = []
    precision = recall = f1 = 0
    for run in ['train', 'eval']:
        iter_loss = 0.0
        correct = 0
        data_loader = torch.utils.data.DataLoader(dataset[run], batch_size=batch_size, shuffle=True, num_workers=1)
        dataset_len = len(data_loader.dataset)
        if run == 'train':
            classifier.train()
        else:
            classifier.eval()
        for samples in data_loader:
            input_spects = samples['spectrogram'].to(device)
            input_labels = samples['label'].to(device)
            optimizer_func.zero_grad()  # Clear off the gradients from any past operation
            with torch.set_grad_enabled(run == 'train'):
                outputs = classifier(input_spects)  # Do the forward pass
                loss = loss_func(outputs, input_labels)  # Calculate the loss
                if run == 'train':
                    loss.backward()  # Calculate the gradients with help of back propagation
                    optimizer_func.step()  # Ask the optimizer to adjust the parameters based on the gradients
            # Record the correct predictions
            _, predicted = torch.max(outputs, 1)  # the indexes are the predicted classes. Need only that from torch.max
            # set up metrics
            iter_loss += loss.item() * input_labels.size(0)  # accumulate the loss
            correct += (predicted == input_labels).sum()
            if run == 'eval':
                labels_preds.append(predicted)
                labels_input.append(input_labels)
        """
        Calculate and store performance metrics for the train and eval runs of this epoch
        """
        epoch_loss[run] = iter_loss / dataset_len
        epoch_accuracy[run] = correct.item() / dataset_len
        if run == 'eval':
            all_input_labels = torch.cat(labels_input)
            all_preds_labels = torch.cat(labels_preds)
            precision = precision_score(all_input_labels, all_preds_labels, average='weighted')
            recall = recall_score(all_input_labels, all_preds_labels, average='weighted')
            f1 = f1_score(all_input_labels, all_preds_labels, average='weighted')
            cfm = confusion_matrix(all_input_labels, all_preds_labels, labels=range(num_labels))
            confusion_matrix_list.append({"cfm": cfm, "accuracy": epoch_accuracy[run], "precision": precision,
                                          "recall": recall, "f1": f1})

    print('Epoch %d/%d, Training (Loss: %.4f, Acc: %.2f ), '
                  'Validation (Loss: %.4f, Acc: %.2f , precision: %.2f, recall: %.2f, f1: %.2f) '
                  % (epoch + 1, num_epochs, epoch_loss['train'], epoch_accuracy['train'] * 100,
                     epoch_loss['eval'], epoch_accuracy['eval'] * 100, precision * 100, recall * 100, f1 * 100))

    earlystop(epoch_loss['eval'], epoch_loss['train'], classifier)
    if earlystop.early_stop:
        classifier.load_state_dict(torch.load("checkpoint.pt"))
        print("Early stopping. Last Checkpoint saved as model state")
        break

"""
save model to disk
"""
torch.save(classifier.state_dict(), "{}.pth".format(model_name))

"""
save metrices for every epoch to disk
"""
conf_df = pd.DataFrame(confusion_matrix_list, columns=['cfm', 'accuracy', 'precision', 'recall', 'f1'])
conf_df.to_csv("Metrics_{}.csv".format(model_name), header=True, index=False)


