import os
import argparse
import pandas as pd
import numpy as np

import torchvision
import torch.nn as nn
import torch
import torchvision.transforms as transforms

from skimage import io, img_as_float32
from sklearn import metrics

from src.mammo_transforms import ToTensor3D
from dm_meta import DMMetaManager

import skimage.transform as sk_trans

import src.resnet50 as resnet_models 


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self):
        super(RegLog, self).__init__()
        self.bn = None
        self.model = nn.Linear(2048, 5)
        
    def forward(self, x):
        out = self.model(x)
        return out


def make_features_resnet(loader, resnet, device):
    """ Make features from images by passing them through simclr encoder
    """
    feats = []
    labels = []
    resnet.eval()
    for batch, (img, label) in enumerate(loader):
        img = img.to(device)
        label = label.to(device)
        
        with torch.no_grad():
            h_i = resnet(img)
        
        h_i = h_i.squeeze()
        
        feats.append(h_i)
        labels.append(label)
    
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)  
    
    return feats, labels


class ImageDataset(torch.utils.data.Dataset):
    """ Custom dataset for whole mammograms.
    """
    def __init__(self, image_list, labels=None, transforms=None):
        self.image_list = image_list
        self.labels = labels

        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        data = io.imread(self.image_list[idx])
        data = img_as_float32(data)

        if self.transforms:
            data = self.transforms(data)

        if self.labels is not None:
            return (data, self.labels[idx])
        else:
            return data
        
        
def get_last_exam_flatten_img_list(exam_gen):
    '''Get image-level last exam data lists. Adapted from dm_meta.DMManager.get_flatten_img_list.
            Returns a list of image dir names and their ground-truth labels.

    Args:
        meta ([bool]): whether to return meta info or not. Default is 
                False.
    '''
    img = []
    lab = []
    for subj_id, ex_idx, exam_dat in exam_gen:
        for idx, dat in exam_dat.iterrows():
            img_name = dat['filename']
            laterality = dat['laterality']
            try:
                cancer = dat['cancerL'] if laterality == 'L' else dat['cancerR']
                try:
                    cancer = int(cancer)
                except ValueError:
                    cancer = np.nan
            except KeyError:
                try:
                    cancer = int(dat['cancer'])
                except KeyError:
                    cancer = np.nan
            img.append(img_name)
            lab.append(cancer)

    return (img, lab)

        
def get_image_datasets():
    """ Returns a training and dev image dataset. Put in its own function to avoid 
        crowding the train function.
    """
    dm_man = DMMetaManager(img_tsv='../meta_info/images_crosswalk_2018-02-04.txt', 
                        exam_tsv='../meta_info/exams_metadata_2018-08-20.txt', 
                        img_folder='../mssm_mg_cohort_png', img_extension='png')
    
    train_subjs = pd.read_csv('../suggested_pat_split/pat_k5_train3_0padded.txt', 
                                header=None, dtype='str')
    dev_subjs = pd.read_csv('../suggested_pat_split/pat_k5_val3_0padded.txt', 
                                header=None, dtype='str')
    test_subjs = pd.read_csv('../suggested_pat_split/pat_k5_test3_0padded.txt', 
                                header=None, dtype='str')

    train_subjs = train_subjs[0].tolist()
    dev_subjs = dev_subjs[0].tolist()
    test_subjs = test_subjs[0].tolist()

    train_subj_gen = dm_man.last_exam_generator(subj_list=train_subjs)
    dev_subj_gen = dm_man.last_exam_generator(subj_list=dev_subjs)
    test_subj_gen = dm_man.last_exam_generator(subj_list=test_subjs)

    train_img_list, train_labs = get_last_exam_flatten_img_list(train_subj_gen)
    dev_img_list, dev_labs = get_last_exam_flatten_img_list(dev_subj_gen)
    test_img_list, test_labs = get_last_exam_flatten_img_list(test_subj_gen)
    
    train_img_set = ImageDataset(image_list=train_img_list, labels=train_labs,
                                    transforms=torchvision.transforms.Compose([Resize((576,448)), ToTensor3D()]))
    dev_img_set = ImageDataset(image_list=dev_img_list, labels=dev_labs,
                                    transforms=torchvision.transforms.Compose([Resize((576,448)), ToTensor3D()]))
    test_img_set = ImageDataset(image_list=test_img_list, labels=test_labs,
                                transforms=torchvision.transforms.Compose([Resize((576,448)), ToTensor3D()]))
    
    return train_img_set, dev_img_set, test_img_set

class Resize(object):
    """Resize the image in a sample.

    Args:
        img_size (2-tuple of int): Desired image size.
    
    Dependencies:
        skimage
    """

    def __init__(self, image_size):
        assert isinstance(image_size, tuple)
        assert len(image_size) == 2
        self.image_size = image_size

    def __call__(self, image):
        return sk_trans.resize(image, self.image_size, 
                               mode='reflect', 
                               anti_aliasing=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=500, type=int,
                        help='number of linear classifier training epochs')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        help='batch size')
    
    args=parser.parse_args()
    
    train(args)
   
    
def train(args):
    path = '.'
    device = torch.device('cuda')
    train_dataset = torchvision.datasets.ImageFolder(os.path.join('../DDSM_patches', "bps-train"))
    val_dataset = torchvision.datasets.ImageFolder(os.path.join('../DDSM_patches', "bps-val"))
    test_dataset = torchvision.datasets.ImageFolder(os.path.join('../DDSM_patches', "cbis-test"))
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    test_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        num_workers=6,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=50,
        num_workers=6,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=50,
        num_workers=6,
        pin_memory=True,
    )
    
    
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()
    resnet = nn.DataParallel(resnet, device_ids=[0, 1])    
    
    
    resnet = resnet.to(device)
        
    linear_model = nn.DataParallel(RegLog())
    linear_model = linear_model.to(device)
    linear_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-3)
    
    train_ftrs, train_labs = make_features_resnet(loader=train_loader, resnet=resnet, device=device)
    dev_ftrs, dev_labs = make_features_resnet(loader=val_loader, resnet=resnet, device=device)
    test_ftrs, test_labs = make_features_resnet(loader=test_loader, resnet=resnet, device=device)
    
    
    train_ftr_set = torch.utils.data.TensorDataset(train_ftrs, train_labs)
    dev_ftr_set = torch.utils.data.TensorDataset(dev_ftrs, dev_labs)
    test_ftr_set = torch.utils.data.TensorDataset(test_ftrs, test_labs)
    
    train_ftr_loader = torch.utils.data.DataLoader(train_ftr_set, batch_size=args.batch_size,
                                                   shuffle=True, drop_last=True)
    dev_ftr_loader = torch.utils.data.DataLoader(dev_ftr_set, batch_size=args.batch_size,)
    test_ftr_loader = torch.utils.data.DataLoader(test_ftr_set, batch_size=args.batch_size,)
    

    train_losses = []
    dev_epoch_losses = []
    train_AUCs = []
    dev_epoch_AUCs = []
    best_dev_AUC = 0
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_pos_prob = np.empty(0)
        train_labels = np.empty(0)
        for batch, (ftr, label) in enumerate(train_ftr_loader):
            optimizer.zero_grad()
            
            ftr = ftr.to(device)
            label = label.to(device)
            
            outputs = linear_model(ftr)
            
            loss = linear_criterion(outputs, label)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            labels = outputs.argmax(dim=1)
            epoch_pos_prob = np.append(epoch_pos_prob, labels.detach().cpu().numpy())
            train_labels = np.append(train_labels, label.cpu().numpy())
        
        train_epoch_AUC = metrics.accuracy_score(train_labels, epoch_pos_prob)
        train_epoch_loss = epoch_loss / len(train_ftr_loader)
        
        train_losses.append(train_epoch_loss)
        train_AUCs.append(train_epoch_AUC)
        
        epoch_loss = 0
        dev_labels = np.empty(0)
        epoch_pos_prob = np.empty(0)
        for batch, (ftr, label) in enumerate(dev_ftr_loader):
            optimizer.zero_grad()
            with torch.no_grad():
                ftr = ftr.to(device)
                label = label.to(device)
                
                outputs = linear_model(ftr)
                
                loss = linear_criterion(outputs, label)
                
            epoch_loss += loss.item()
            pos_prob = outputs.argmax(dim=1)
            epoch_pos_prob = np.append(epoch_pos_prob, pos_prob.detach().cpu().numpy())
            dev_labels = np.append(dev_labels, label.cpu().numpy())
            
        dev_epoch_AUC = metrics.accuracy_score(dev_labels, epoch_pos_prob)
        dev_epoch_loss = epoch_loss / len(dev_ftr_loader)
    
        dev_epoch_losses.append(dev_epoch_loss)
        dev_epoch_AUCs.append(dev_epoch_AUC)
        
        if dev_epoch_AUC > best_dev_AUC:
            torch.save(linear_model.state_dict(), os.path.join(path, "best_lin_model.pth"))

        if epoch % 50 == 0:
            print(f"Epoch: {epoch} \t Train loss: {train_epoch_loss} \t Train Acc: {train_epoch_AUC}" \
                f"Dev loss: {dev_epoch_loss} \t Dev Acc: {dev_epoch_AUC}")
        
    
    linear_model.load_state_dict(torch.load(os.path.join(path, "best_lin_model.pth")))
    test_loss = 0
    test_pos_prob = np.empty(0)
    epoch_pos_prob = np.empty(0)
    test_labels = np.empty(0)
    for batch, (ftr, label) in enumerate(test_ftr_loader):
        optimizer.zero_grad()
        with torch.no_grad():
            ftr = ftr.to(device)
            label = label.to(device)
            
            outputs = linear_model(ftr)
            
            loss = linear_criterion(outputs, label)
            
        test_loss += loss.item()
        pos_prob = outputs.argmax(dim=1)
        
        test_pos_prob = np.append(test_pos_prob, pos_prob.detach().cpu().numpy())
        test_labels = np.append(test_labels, label.cpu().numpy())
        
    test_epoch_AUC = metrics.accuracy_score(test_labels, test_pos_prob)
    test_epoch_loss = test_loss / len(test_ftr_loader)
    
    print(f"Test Dataset -- loss: {test_epoch_loss} -- Acc: {test_epoch_AUC}")
    
    train_losses = np.array(train_losses)
    train_AUCs = np.array(train_AUCs)
    dev_epoch_losses = np.array(dev_epoch_losses)
    dev_epoch_AUCs = np.array(dev_epoch_AUCs)
    testdata = np.array([test_epoch_loss, test_epoch_AUC])
    

    
    
if __name__ == '__main__':
    main()
    