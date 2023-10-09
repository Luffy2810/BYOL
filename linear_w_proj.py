from src.dataset.image10_dataloader import get_linear_dataloader,get_test_dataloader
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
from torch.optim.lr_scheduler import StepLR
torch.cuda.set_device(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
class LinearNet(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(25, 10)


    def forward(self, x):
        # print (x.size())
        x = self.fc1(x)
        return(x)
    
   


def get_mean_of_list(L):
    return sum(L) / len(L)

def Linear():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet=make_model()
    if(os.path.isfile("results_pretrained/modelq.pth")):
        resnet.load_state_dict(torch.load("results_pretrained/modelq.pth"))
        resnet.eval().to(device)
    else:
        print("Model Does not exist")

    # resnet.fc = nn.Sequential(*list(resnet.fc.children())[:-3])
    dataloader_training_dataset = get_linear_dataloader()
    dataloader_testing_dataset = get_test_dataloader()

    if not os.path.exists('linear_w_proj'):
        os.makedirs('linear_w_proj')

    linear_classifier = LinearNet().to(device)
    # linear_classifier.eval()
    # linear_classifier.to(device)
    # if(os.path.isfile("linear/model.pth")):
    #     linear_classifier.load_state_dict(torch.load("linear/model.pth"))
    #     linear_classifier.train()
    
    linear_optimizer = optim.SGD(linear_classifier.parameters(), lr=0.3, momentum=0.9, weight_decay=1e-5)
    # linear_scheduler = StepLR(linear_optimizer, step_size=7, gamma=0.1)
    num_epochs_linear = 20

    LINEAR_TRAINING = True

    losses_train_linear = []
    acc_train_linear = []
    losses_test_linear = []
    acc_test_linear = []


    max_test_acc = 0


    # # linear_classifier.eval()
    epoch_losses_test_linear = []
    epoch_acc_test_num_linear = 0.0
    epoch_acc_test_den_linear = 0.0
    # # for (_, sample_batched) in enumerate(tqdm(dataloader_testing_dataset)):
        
    # #     x = sample_batched['image']
    # #     y_actual = sample_batched['label']
    # #     y_actual = np.asarray(y_actual)
    # #     y_actual = torch.from_numpy(y_actual.astype('long'))
    # #     x = x.to(device)
    # #     y_actual  = y_actual.to(device)

    # #     y_intermediate = resnet(x)

    # #     y_predicted = linear_classifier(y_intermediate)
    # #     loss = nn.CrossEntropyLoss()(y_predicted, y_actual)
    # #     epoch_losses_test_linear.append(loss.data.item())

    # #     pred = np.argmax(y_predicted.cpu().data, axis=1)
    # #     actual = y_actual.cpu().data
    # #     epoch_acc_test_num_linear += (actual == pred).sum().item()
    # #     epoch_acc_test_den_linear += len(actual)

    # # test_acc = epoch_acc_test_num_linear / epoch_acc_test_den_linear
    # # print(test_acc)



    for epoch in range(num_epochs_linear):
        print (epoch)
        if LINEAR_TRAINING:

            linear_classifier.train()
            resnet.eval()

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

            # linear_scheduler.step()

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
            plt.savefig('linear_w_proj/losses.png')
            plt.close()
            writer.add_scalar("linear_w_proj/accuracy_train", epoch_acc_train_num_linear / epoch_acc_train_den_linear, epoch)
            writer.add_scalar("linear_w_proj/accuracy_test", test_acc, epoch)
            fig = plt.figure(figsize=(10, 10))
            sns.set_style('darkgrid')
            plt.plot(acc_train_linear)
            plt.plot(acc_test_linear)
            plt.legend(['Training Accuracy', 'Testing Accuracy'])
            plt.savefig('linear_w_proj/accuracy.png')
            plt.close()

            print("Epoch completed")

            if test_acc >= max_test_acc:


                max_test_acc = test_acc
                torch.save(linear_classifier.state_dict(), 'linear_w_proj/model.pth')
                torch.save(linear_optimizer.state_dict(), 'linear_w_proj/optimizer.pth')


        np.savez("linear_w_proj/linear_losses_train_file", np.array(losses_train_linear))
        np.savez("linear_w_proj/linear_losses_test_file", np.array(losses_test_linear))
        np.savez("linear_w_proj/linear_acc_train_file", np.array(acc_train_linear))
        np.savez("linear_w_proj/linear_acc_test_file", np.array(acc_test_linear))
    # torch.save(linear_classifier,'linear_w_prod")):
        linear_classifier.load_state_dict(torch.load("linear_w_proj/model.pth"))
    linear_classifier.eval()

    # linear_classifier=torch.load('linear.pt').to(device)
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
    
if __name__=="__main__":
    Linear()
