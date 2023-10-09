from src.dataset.dataloader import get_linear_dataloader,get_test_dataloader
from src.model.Resnet import make_model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from tqdm import tqdm

class LinearNet(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(25, 200)

    def forward(self, x):
        x = self.fc1(x)
        return(x)
    



def get_mean_of_list(L):
    return sum(L) / len(L)

def Linear():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet=make_model().to(device)
    if(os.path.isfile("results/modelq.pth")):
        resnet.load_state_dict(torch.load("results/modelq.pth"))
    else:
        print("Model Does not exist")


    dataloader_training_dataset = get_linear_dataloader()
    dataloader_testing_dataset = get_test_dataloader()

    if not os.path.exists('linear'):
        os.makedirs('linear')

    linear_classifier = LinearNet()

    linear_classifier.to(device)

    linear_optimizer = optim.SGD(linear_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)


    num_epochs_linear = 10

    LINEAR_TRAINING = True

    losses_train_linear = []
    acc_train_linear = []
    losses_test_linear = []
    acc_test_linear = []


    max_test_acc = 0

    if(os.path.isfile("linear/model.pth")):


        linear_classifier.load_state_dict(torch.load("linear/model.pth"))
        linear_optimizer.load_state_dict(torch.load("linear/optimizer.pth"))

        temp = np.load("linear/linear_losses_train_file.npz")
        losses_train_linear = list(temp['arr_0'])
        temp = np.load("linear/linear_losses_test_file.npz")
        losses_test_linear = list(temp['arr_0'])
        temp = np.load("linear/linear_acc_train_file.npz")
        acc_train_linear = list(temp['arr_0'])
        temp = np.load("linear/linear_acc_test_file.npz")
        acc_test_linear = list(temp['arr_0'])

    for epoch in range(num_epochs_linear):
        print (epoch)
        if LINEAR_TRAINING:

            linear_classifier.train()

            epoch_losses_train_linear = []
            epoch_acc_train_num_linear = 0.0
            epoch_acc_train_den_linear = 0.0

            for (_, sample_batched) in enumerate(tqdm(dataloader_training_dataset)):

                x = sample_batched['image']
                y_actual = sample_batched['label']
                x = x.to(device)
                y_actual  = y_actual.to(device)
                y_intermediate = resnet(x)


                linear_optimizer.zero_grad()

                y_predicted = linear_classifier(y_intermediate)

                loss = nn.CrossEntropyLoss()(y_predicted, y_actual)

                epoch_losses_train_linear.append(loss.data.item())

                loss.backward()


                linear_optimizer.step()

                pred = np.argmax(y_predicted.cpu().data, axis=1)
                actual = y_actual.cpu().data


                epoch_acc_train_num_linear += (actual == pred).sum().item()
                epoch_acc_train_den_linear += len(actual)

                x = None
                y_intermediate = None
                y_predicted = None
                sample_batched = None

            losses_train_linear.append(get_mean_of_list(epoch_losses_train_linear))
            acc_train_linear.append(epoch_acc_train_num_linear / epoch_acc_train_den_linear)

        linear_classifier.eval()

        epoch_losses_test_linear = []
        epoch_acc_test_num_linear = 0.0
        epoch_acc_test_den_linear = 0.0

        for (_, sample_batched) in enumerate(tqdm(dataloader_testing_dataset)):
            
            x = sample_batched['image']
            y_actual = sample_batched['label']
            y_actual = np.asarray(y_actual)
            y_actual = torch.from_numpy(y_actual.astype('long'))
            x = x.to(device)
            y_actual  = y_actual.to(device)

            y_intermediate = resnet(x)

            y_predicted = linear_classifier(y_intermediate)
            loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
            epoch_losses_test_linear.append(loss.data.item())

            pred = np.argmax(y_predicted.cpu().data, axis=1)
            actual = y_actual.cpu().data
            epoch_acc_test_num_linear += (actual == pred).sum().item()
            epoch_acc_test_den_linear += len(actual)

        test_acc = epoch_acc_test_num_linear / epoch_acc_test_den_linear
        print(test_acc)

        if LINEAR_TRAINING:
            losses_test_linear.append(get_mean_of_list(epoch_losses_test_linear))
            acc_test_linear.append(epoch_acc_test_num_linear / epoch_acc_test_den_linear)


            fig = plt.figure(figsize=(10, 10))
            sns.set_style('darkgrid')
            plt.plot(losses_train_linear)
            plt.plot(losses_test_linear)
            plt.legend(['Training Losses', 'Testing Losses'])
            plt.savefig('linear/losses.png')
            plt.close()

            fig = plt.figure(figsize=(10, 10))
            sns.set_style('darkgrid')
            plt.plot(acc_train_linear)
            plt.plot(acc_test_linear)
            plt.legend(['Training Accuracy', 'Testing Accuracy'])
            plt.savefig('linear/accuracy.png')
            plt.close()

            print("Epoch completed")

            if test_acc >= max_test_acc:


                max_test_acc = test_acc
                torch.save(linear_classifier.state_dict(), 'linear/model.pth')
                torch.save(linear_optimizer.state_dict(), 'linear/optimizer.pth')


        np.savez("linear/linear_losses_train_file", np.array(losses_train_linear))
        np.savez("linear/linear_losses_test_file", np.array(losses_test_linear))
        np.savez("linear/linear_acc_train_file", np.array(acc_train_linear))
        np.savez("linear/linear_acc_test_file", np.array(acc_test_linear))

if __name__=="__main__":
    Linear()
