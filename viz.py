import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.model.Resnet import make_model
import seaborn as sns
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from src.dataset.image10_dataloader import get_linear_dataloader,get_test_dataloader
tsne = TSNE()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader_training_dataset=get_linear_dataloader()
dataloader_testing_dataset=get_test_dataloader()
torch.cuda.set_device(1)
resnet=make_model().to(device) 
if(os.path.isfile("results/modelq.pth")):
# torch.cuda.set_device(1)
    resnet.load_state_dict(torch.load("results/modelq.pth"))

def plot_vecs_n_labels(v,labels,fname):
    fig = plt.figure(figsize = (10, 10))
    plt.axis('off')
    sns.set_style("darkgrid")
    sns.scatterplot(x=v[:,0],y= v[:,1], hue=labels, legend='full', palette=sns.color_palette("bright", 10))
    
    plt.savefig(fname)
    plt.close()


TSNEVIS = True
if not os.path.exists('vis'):
    os.makedirs('vis')


if TSNEVIS:
    
    resnet.eval()

    
    for (_, sample_batched) in enumerate(tqdm(dataloader_training_dataset)):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'./vis/tsne_train_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

    
    for (_, sample_batched) in enumerate(tqdm(dataloader_testing_dataset)):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'./vis/tsne_test_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None


resnet.fc = nn.Sequential(*list(resnet.fc.children())[:-2])

if TSNEVIS:
    for (_, sample_batched) in enumerate(tqdm(dataloader_training_dataset)):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'./vis/tsne_train_second_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

    for (_, sample_batched) in enumerate(tqdm(dataloader_testing_dataset)):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'./vis/tsne_test_second_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None


resnet.fc = nn.Sequential(*list(resnet.fc.children())[:-1])

if TSNEVIS:
    for (_, sample_batched) in enumerate(tqdm(dataloader_training_dataset)):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'./vis/tsne_hidden_train.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

    for (_, sample_batched) in enumerate(tqdm(dataloader_testing_dataset)):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'./vis/tsne_hidden_test.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None