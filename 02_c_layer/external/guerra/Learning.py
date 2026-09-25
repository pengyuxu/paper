##### MARCO GUERRA 2025

# Run the learning algorithm for classification of SHREC 2025
# This file is meant to be run either directly or via nohup

import sys, os
from pathlib import Path
import numpy as np

import numpy as np
from src.data_reader import DataSource

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary as torch_summary

from src.learning.nn import SimpleNN, train, predict, Autoencoder
import csv


if sys.version_info < (3, 10):
    raise RuntimeError("This project requires Python 3.10 or higher.")

print('** Training a classification model **', flush=True)

# READ THE GROUND TRUTH

Ground_Truth = {}
with open('./data/data/test_set_ground_truth.csv', 'r') as csvfile:
    truths = csv.reader(csvfile)

    next(truths, None) # skip first row, it's a header

    for t in truths:
        Ground_Truth[t[0]] = int(t[1])

ground_truths = []
test_set = DataSource( './data/data/test_set/', base_path='./data/data/test_set')

for f in test_set:

    filename = os.path.basename(f)
    filename, _ = os.path.splitext(filename)
    
    ground_truths.append(Ground_Truth[filename])

ground_truths = np.array(ground_truths).reshape((len(ground_truths),))


# READ THE DESCRIPTORS DATA

dataAlphaTrain = np.load('./data/savedAlpha/AlphaDescriptors_train_aug.npy')
dataSectorsTrain = np.load('./data/savedSectors/train_set_aug.npy')

dataAlphaTest = np.load('./data/savedAlpha/AlphaDescriptors_test_aug.npy')
dataSectorsTest = np.load('./data/savedSectors/test_set_aug.npy')

labels = np.load('./data/labels/labels_train_aug.npy')

# IF USING A DUMMY DATASET, 
# MAKE LABELS AND GROUND TRUTH DUMMY TOO
# TO AVOID PROBLEMS WITH CROSS-ENTROPY

#labels = np.random.randint(2, size=labels.shape, dtype=int)
#ground_truths = np.random.randint(2, size=ground_truths.shape, dtype=int)

data = np.concatenate( [dataSectorsTrain , dataAlphaTrain] , axis = 1 )
test = np.concatenate( [dataSectorsTest , dataAlphaTest] , axis = 1 )

print('Shape of training data')
print(data.shape, flush=True)
print('Shape of test data')
print(test.shape, flush=True)

# SET UP LEARNING MODEL

StrShSp = StratifiedShuffleSplit(n_splits=15, train_size=0.8, random_state=None)

# get indices of split
train_idx, val_idx = next(StrShSp.split(data, labels))

train_data = data[train_idx,:]
train_labels = labels[train_idx]

val_data = data[val_idx]
val_labels = labels[val_idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device = ',device)

X_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)

X_val = torch.tensor(val_data, dtype=torch.float32)
y_val = torch.tensor(val_labels, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128)

# TRAINING

m = train_data.shape[1]
c = np.unique(labels).shape[0]

print('Input size = ', m, ' Output size = ',c, flush=True)

# Model save path
os.makedirs('./data/trained_models/', exist_ok=True)
trained_model_path = './data/trained_models/best_model.pth'

model = SimpleNN(m, c)

# Uncomment to print a summary of the model
#summary = torch_summary(model,None)
#print(summary, flush=True)

# UNCOMMENT IF YOU WANT TO RELOAD A TRAINING RUN
try: 
    pass
    #model.load_state_dict(torch.load(trained_model_path))
except:
    pass

criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)  # Adam optimizer

epochs = 5000
# Early Stopping Parameters
patience = int(epochs/10)  # Stop if no improvement for 'patience' epochs

print('Start training', flush=True)

train(epochs, model, criterion, optimizer, train_loader, val_loader, trained_model_path, patience)

print('Finished training \n', flush=True)

print('Model saved at ./data/trained_models/best_model.pth \n')

# PREDICT ON TEST SET

print('Prediction on test set \n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = test.shape[1]
c = np.unique(labels).shape[0]
trained_model_path = './data/trained_models/best_model.pth'

model = SimpleNN(m, c)
model.load_state_dict(torch.load(trained_model_path))
model.to(device)
model.eval()

X_predict = torch.tensor(test, dtype=torch.float32).to(device)
output = model( X_predict )

y_pred = torch.argmax(output, dim=1)
y_pred = y_pred.cpu().numpy()

# FINAL CLASSIFICATION

print('Majority voting and class frequencies')

# y_pred has 8 predictions for each original protein. We now compute the majority vote
# and break ties by a-priori class frequency

# a-priori class frequency
uni_classes , class_counts = np.unique(labels, return_counts=True)
a_priori_freq = {}
for i,c in enumerate(uni_classes):
    a_priori_freq[c] = class_counts[i]/60

# majority vote
bl_pred = y_pred.reshape(-1,8) # reshape in blocks of 8, each is one protein
uni_counts = [np.unique(block, return_counts=True) for block in bl_pred]
sorted_classes = []

for protein in uni_counts:
    classes = protein[0]
    counts = protein[1]

    order = np.argsort(counts)[::-1]

    classes = classes[order]
    counts = counts[order]

    sorted_classes.append( [ classes, counts ] )

sorted_classes = [ [x[0] , [y for y in x[1]]] for x in sorted_classes ]
sorted_classes = [ [ x[0].tolist(), x[1]] for x in sorted_classes]

# put the two together
majority_voting = []

for classes, counts in sorted_classes:

    max_freq = np.max(counts) # the most frequent classes have this number of instances

    tied_classes = [cls for cls, cts in zip(classes, counts) if cts == max_freq] # find which have that frequency

    if len(tied_classes)==1: # if there are no ties
        majority_voting.append(tied_classes[0]) # it' simply that class

    else: # if there is more than one class with the same prediction confidence
        top_class = max( tied_classes, key = lambda n : a_priori_freq[n] ) # choose based on the a-priori frequency
        majority_voting.append(top_class)

majority_voting = np.array(majority_voting).reshape((len(majority_voting)),)
np.save('./data/prediction.npy', majority_voting)

print('Final accuracy score:')
print(accuracy_score(majority_voting, ground_truths))

print('\nFinal prediction saved at ./data/prediction.npy')

# WRITE PREDICTION



