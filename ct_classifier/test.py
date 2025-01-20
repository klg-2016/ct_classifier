import os
import argparse
import yaml
import glob
from tqdm import trange
import wandb
import torch # this imports pytorch
import torch.nn as nn # this contains our loss function
from torch.utils.data import DataLoader # the pytorch dataloader class will take care of all kind of parallelization during training
from torch.optim import SGD # this imports the optimizer
# let's import our own classes and functions!
from util import init_seed
from dataset import CTDataset
from model import CustomResNet18
from datetime import datetime
import pandas as pd
import torch.nn.functional as F

# the goal is to test the model on images and see which ones are problematic
# for that, we need filename, prediction, ground truth and confidence. Preferably in a dataframe with those columns.

def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )
    return dataLoader

# Open the model
cfg = yaml.safe_load(open('../configs/exp_resnet18.yaml', 'r'))
model = CustomResNet18(cfg['num_classes'])
state = torch.load(open(f'/home/Kathryn/code/ct_classifier/model_states-2025-01-17_21-39-55/best.pt', 'rb'), map_location='cpu')
model.load_state_dict(state['model'])
device = cfg['device']
model.to(device)
model.eval() #evaluation mode, you freeze all the parameters

#initialize empty lists for the three things we're interested in
preds = []
trues = []
confs = []

# predict on val images
#this is running the model on the val data
dataLoader = create_dataloader(cfg, split='val')
with torch.no_grad():
    for idx, (data, ground_truths) in enumerate(dataLoader):
        # put data and labels on device, device is cuda (defined in config file)
        data, labels = data.to(device), ground_truths.to(device)
        # forward pass
        prediction = model(data) #for every image, returns a list of all categories
        pred_labels = torch.argmax(prediction, dim=1) #returns only the highest class
        confidence_scores = F.softmax(prediction, dim=1).max(dim=1)[0].tolist() #softmax turns the numbers into numbers from 0 to 1
        pred_labels = pred_labels.tolist()  # Use pred_labels instead of labels
        ground_truths = ground_truths.tolist()
        len_batch = len(pred_labels)
        for idx in range(len_batch): #idx changes every run, goes from 0 to batch size
            pred = pred_labels[idx]
            preds.append(pred) #add the prediction value to the list 

            true = ground_truths[idx]
            trues.append(true)

            conf = confidence_scores[idx]
            confs.append(conf)

            print(f"pred : {pred}")
            print(f"true : {true}")
            print(f"conf : {conf}")
            print("")

#extracts file names from data loader
dataset = dataLoader.dataset
filenames = [entry[0] for entry in dataset.data]
print(filenames)

#filenames is a list already
#need to get predictions into a list

#create df with file name, prediction, ground truth and confidence values
#a row for every image, and a column for each of those (pred, truth, conf)

# Combine them into a dictionary where keys are column names
data = {
    'file': filenames,
    'pred': preds,
    'true': trues,
    'conf': confs
}

# Create the DataFrame
df = pd.DataFrame(data)

print(df.head())

#note: DataFrame, list are both classes with functions that work on them