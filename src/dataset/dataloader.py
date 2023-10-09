from re import T
import torch
from torch.utils.data import Dataset, DataLoader
from .dataset import SSLDataset
import pandas as pd
import os
import random
root_folder = r'./data/content'
random.seed(0)

def convert_labels_to_tokens(labels):
    list_set = set(labels)
    tokens = (list(list_set))
    word_to_idx = {word: i for i, word in enumerate(tokens)}
    return word_to_idx


def get_mutated_dataloader():
    train_names = sorted(os.listdir(root_folder + '/Compiled_train'))
    names_train = random.sample(train_names, len(train_names))
    labels_train = [x.split('_')[0] for x in names_train]
    tokens=convert_labels_to_tokens(labels_train)
    training_dataset_mutated = SSLDataset(root_folder+'/Compiled_train',names_train, labels_train,tokens, train=True,mutate=True)
    dataloader_training_dl = DataLoader(training_dataset_mutated, batch_size=256, shuffle=True, num_workers=10)
    return dataloader_training_dl
    
def get_linear_dataloader():
    train_names = sorted(os.listdir(root_folder + '/Compiled_train'))
    names_train_10_percent = random.sample(train_names, len(train_names) // 10)
    labels_train_10_percent = [x.split('_')[0] for x in names_train_10_percent]
    tokens=convert_labels_to_tokens(labels_train_10_percent)
    linear_dataset = SSLDataset(root_folder+'/Compiled_train', names_train_10_percent, labels_train_10_percent,tokens, train=True,mutate=False)
    dataloader_linear_dl = DataLoader(linear_dataset, batch_size=125, shuffle=True, num_workers=2)
    return dataloader_linear_dl

def get_test_dataloader():
    label_val=pd.read_csv ('./data/val_annotations.txt', sep='\t',header=None)[[0,1]]
    label_val.columns=['img','label']
    label_val=label_val.set_index('img').T.to_dict('list')
    label_names=label_val.values()
    label_names = [x[0].split('_')[0] for x in label_names]
    test_names = sorted(os.listdir(root_folder + '/Compiled_val'))
    names_test = random.sample(test_names, len(test_names))
    tokens=convert_labels_to_tokens(label_names)
    testing_dataset = SSLDataset(root_folder + '/Compiled_val', names_test, label_val,tokens, train=False,mutate=False)
    dataloader_testing_dl = DataLoader(testing_dataset, batch_size=125, shuffle=True, num_workers=2)
    return dataloader_testing_dl




