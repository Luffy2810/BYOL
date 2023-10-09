import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image



class SSLDataset(Dataset):
    def __init__(self, root_dir, filenames,labels,tokens ,train=True,mutate=True,img_size=64):
        self.root_dir = root_dir
        self.file_names = filenames
        self.labels = labels
        self.train = train
        self.mutate=mutate
        self.img_size=img_size
        self.token=tokens
        # print (tokens)
    def __len__(self):
        return len(self.file_names)

    def pil_loader(self,path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def get_color_distortion(self,s=1.0):
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
        rnd_gray = T.RandomGrayscale(p=0.2)
        color_distort = T.Compose([rnd_color_jitter, rnd_gray])

        return color_distort

    def tensorify(self, img):
        res = T.ToTensor()(img)
        res = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(res)
        return res

    def mutate_image(self, img):
        res = T.RandomResizedCrop(int(224*64/256))(img)
        res = self.get_color_distortion(1)(res)
        return res

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        img_name = os.path.join(self.root_dir, self.file_names[idx])
        image = self.pil_loader(img_name)
        if self.train:
          label = self.labels[idx]
          label=self.token[label]
        else:
          label=self.labels[self.file_names[idx]][0]
          label=self.token[label]
        image = T.Resize((self.img_size, self.img_size))(image)

        if self.mutate:
            image1 = self.mutate_image(image)
            image1 = self.tensorify(image1)
            image2 = self.mutate_image(image)
            image2 = self.tensorify(image2)
            sample = {'image1': image1, 'image2': image2, 'label': label}
        else:
            image = T.Resize((224, 224))(image)
            image = self.tensorify(image)
            sample = {'image': image, 'label': label}

        return sample