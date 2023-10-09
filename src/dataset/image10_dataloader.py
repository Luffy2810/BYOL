from re import T
import torch
from torch.utils.data import Dataset, DataLoader
from .image10_dataset import SSLDataset
import pandas as pd
import os
import glob
import random
root_folder = r'../imagenette2/'
random.seed(0)

def convert_labels_to_tokens(labels):
    list_set = set(labels)
    tokens = (list(list_set))
    word_to_idx = {word: i for i, word in enumerate(tokens)}
    return word_to_idx


def get_mutated_dataloader():

    train_names = sorted(glob.glob('../imagenette2/train/*/*.JPEG',recursive=True))
    names_train = random.sample(train_names, len(train_names))
    labels_train = [x.split('/')[3] for x in names_train]
    tokens=convert_labels_to_tokens(labels_train)
    training_dataset_mutated = SSLDataset('',names_train, labels_train,tokens,mutate=True)
    dataloader_training_dl = DataLoader(training_dataset_mutated, batch_size=256, shuffle=True, num_workers=20)
    return dataloader_training_dl
    
def get_linear_dataloader():
    train_names = sorted(glob.glob('../imagenette2/train/*/*.JPEG',recursive=True))
    names_train_10_percent = random.sample(train_names, len(train_names) // 10)
    labels_train_10_percent = [x.split('/')[3] for x in names_train_10_percent]
    tokens=convert_labels_to_tokens(labels_train_10_percent)
    linear_dataset = SSLDataset('', names_train_10_percent, labels_train_10_percent,tokens,mutate=False)
    dataloader_linear_dl = DataLoader(linear_dataset, batch_size=256, shuffle=True, num_workers=2)
    return dataloader_linear_dl

def get_test_dataloader():
    train_names = sorted(glob.glob('../imagenette2/train/*/*.JPEG',recursive=True))
    names_train_10_percent = random.sample(train_names, len(train_names) // 10)
    labels_train_10_percent = [x.split('/')[3] for x in names_train_10_percent]
    tokens=convert_labels_to_tokens(labels_train_10_percent)
    test_names = sorted(glob.glob('../imagenette2/val/*/*.JPEG',recursive=True))
    names_test = random.sample(test_names, len(test_names))
    labels_test = [x.split('/')[3] for x in names_test]
    testing_dataset = SSLDataset('', names_test, labels_test,tokens,mutate=False)
    dataloader_testing_dl = DataLoader(testing_dataset, batch_size=256, shuffle=True, num_workers=2)
    return dataloader_testing_dl


