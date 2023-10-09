from src.dataset.image10_dataloader import get_mutated_dataloader
from src.model.Resnet import ResNet18,MLPHead
from src.model.loss import loss_function
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy
torch.cuda.set_device(1)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
writer = SummaryWriter()



def get_mean_of_list(L):
    return sum(L) / len(L)

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_training_dataset_mutated = get_mutated_dataloader()

    resnetq=ResNet18().to(device)
    resnetk = copy.deepcopy(resnetq).to(device)


    predictor = MLPHead(in_channels=resnetq.projection.net[-1].out_features,mlp_hidden_size=512, projection_size=128).to(device)

    optimizer = torch.optim.SGD(list(resnetq.parameters()) + list(predictor.parameters()),0.2, weight_decay=1.5e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader_training_dataset_mutated), eta_min=0,
                                                           last_epoch=-1)
    losses_train = []
    num_epochs = 1200
    momentum=0.99
    queue = None


    if not os.path.exists('results'):
        os.makedirs('results')

    if(os.path.isfile("results/modelq.pth")):
        resnetq.load_state_dict(torch.load("results/modelq.pth"))
        resnetk.load_state_dict(torch.load("results/modelk.pth"))
        predictor.load_state_dict(torch.load("results/predictor.pth"))
        optimizer.load_state_dict(torch.load("results/optimizer.pth"))


        temp = np.load("results/lossesfile.npz")
        losses_train = list(temp['arr_0'])

    resnetq.train()

    for epoch in range(num_epochs):

        print(epoch)

        epoch_losses_train = []

        for (_, sample_batched) in enumerate(tqdm(dataloader_training_dataset_mutated)):

            optimizer.zero_grad()

            i1 = sample_batched['image1']
            i2 = sample_batched['image2']

            i1= i1.to(device)
            i2 = i2.to(device)

            predictions_from_view_1 = predictor(resnetq(i1))
            predictions_from_view_2 = predictor(resnetq(i2))
            with torch.no_grad():
                targets_to_view_2 = resnetk(i1)
                targets_to_view_1 = resnetk(i2)

            loss = loss_function(predictions_from_view_1, targets_to_view_1)
            loss += loss_function(predictions_from_view_2, targets_to_view_2)
            epoch_losses_train.append(loss.mean().cpu().data.item())
            loss.mean().backward()
            optimizer.step()
            

            for θ_k, θ_q in zip(resnetk.parameters(), resnetq.parameters()):
                θ_k.data.copy_(momentum*θ_k.data + θ_q.data*(1.0 - momentum))
        scheduler.step()
        losses_train.append(get_mean_of_list(epoch_losses_train))
        writer.add_scalar("Loss/train", get_mean_of_list(epoch_losses_train), epoch)
        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(losses_train)
        plt.legend(['Training Losses'])
        plt.savefig('losses.png')
        plt.close()

        torch.save(resnetq.state_dict(), 'results/modelq.pth')
        torch.save(resnetk.state_dict(), 'results/modelk.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')
        torch.save(predictor.state_dict(), 'results/predictor.pth')
        np.savez("results/lossesfile", np.array(losses_train))
        torch.save(queue, 'results/queue.pt')

if __name__=="__main__":
    train()