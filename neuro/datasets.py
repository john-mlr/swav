import torch
import os
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import io

def get_patches_labels(start_path, data_path):
    """ Retrieves the paths to tiles images and gives them labels.
        I'm really disappointed how inelegant this is.
    
        args:
            start_path (str): path to top level folder (should contain background, nft, nft-ic)
            data_path (str): path to save the stratified train/val/test split indeces to
    """
    neg_path = os.path.join(start_path, 'background')
    pos_path1 = os.path.join(start_path, 'nft-ic')
    pos_path2 = os.path.join(start_path,'nft')
    
    paths = []
    labels = []
    for item in os.listdir(neg_path):
        temp_path = os.path.join(neg_path, item)
        paths.append(temp_path)
        labels.append(0)
        
    for item in os.listdir(pos_path1):
        temp_path = os.path.join(pos_path1, item)
        paths.append(temp_path)
        labels.append(1)
        
    for item in os.listdir(pos_path2):
        temp_path = os.path.join(pos_path2, item)
        paths.append(temp_path)
        labels.append(1)
        
    train_idx, eval_idx = train_test_split(np.arange(len(labels)),test_size=0.3,
                                            random_state=42, stratify=labels)

    test_idx, dev_idx = train_test_split(eval_idx, test_size=0.5, random_state=42,
                                        stratify=np.array(labels)[eval_idx])
    
    np.savez(os.path.join(data_path, 'indeces.npz'), train_idx=train_idx, dev_idx=dev_idx, test_idx=test_idx)
    
    train_paths = [paths[idx] for idx in train_idx]
    train_labs = [labels[idx] for idx in train_idx]
    
    dev_paths = [paths[idx] for idx in dev_idx]
    dev_labs = [labels[idx] for idx in dev_idx]
    
    test_paths = [paths[idx] for idx in test_idx]
    test_labs = [labels[idx] for idx in test_idx]
    
    return train_paths, train_labs, dev_paths, dev_labs, test_paths, test_labs
    
    
class PatchDataset(torch.utils.data.Dataset):
    """ Custom dataset for pretraining on patches."""
    def __init__(self, name_list, transforms=None):
        self.file_list = name_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        curr_path = self.file_list[idx]
        data = io.imread(curr_path)
        data = data[:,:,:-1]
        data = img_as_float32(data)
        
        if self.transforms is not None:
            img_1 = self.transforms(data)
            img_2 = self.transforms(data)
            
        return (img_1, img_2)