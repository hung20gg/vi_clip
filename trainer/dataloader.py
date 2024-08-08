from torch.utils.data import DataLoader, Dataset, BatchSampler
import torch
import os
from torchvision.io import read_image
import numpy as np
from PIL import Image

class ImageCaptionDataset(Dataset):
    def __init__(self, df, directory = ''):
        """_summary_

        Args:
            df (_type_): DataFrame of the dataset
            directory (_type_): _description_
        """
        super(ImageCaptionDataset, self).__init__()
        self.df = df
        self.images_id = df['image_id'].values
        self.imgs = df['image_path'].values
        self.descriptions = df['caption'].values
        
        self.directory = directory
        self.descriptions = []
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_dir = self.imgs[idx]
        img = Image.open(os.path.join(self.directory, img_dir))
        label = self.descriptions[idx]
        return img, label
    
class TensorCaptionDataset(Dataset):
    def __init__(self, df, directory = ''):
        """_summary_

        Args:
            df (_type_): DataFrame of the dataset
            directory (_type_): _description_
        """
        super(TensorCaptionDataset, self).__init__()
        self.df = df
        self.imgs = df['image_path'].values
        self.descriptions = df['caption'].values
        
        self.directory = directory
        self.descriptions = []
    
class CrossLingualDataset(Dataset):
    def __init__(self, df ):
        """_summary_

        Args:
            df (_type_): DataFrame of the dataset
            directory (_type_): _description_
        """
        super(CrossLingualDataset, self).__init__()
        self.original_text = df['en'].values
        self.translated_text = df['vi'].values

    def __len__(self):
        return len(self.original_text)
    
    def __getitem__(self, idx):
        return self.original_text[idx], self.translated_text[idx]
    
class mCLIPDataset(Dataset):
    def __init__(self, df , directory = ''):
        """_summary_

        Args:
            df (_type_): DataFrame of the dataset
            directory (_type_): _description_
        """
        super(mCLIPDataset, self).__init__()
        
        self.directory = directory
        self.images_id = df['image_id'].values
        self.imgs = df['image_path'].values
        self.original_text = df['text'].values
        self.translated_text = df['translated_text'].values
        assert len(self.original_text) == len(self.translated_text), "Original and translated text must have the same length"

    def __len__(self):
        return len(self.original_text)
    
    def __getitem__(self, idx):
        img_dir = self.imgs[idx]
        img = Image.open(os.path.join(self.directory, img_dir))
        return img, self.original_text[idx], self.translated_text[idx]
    
class CLIPSampler(BatchSampler):
    """_summary_

        Sampling strategy for CLIP training
        Only allow for one duplication, either text or image
        Must shuffle the dataset before using this sampler
    """
    def __init__(self, duplicate_id, batch_size):
        self.batch_size = batch_size
        self.duplicate_id = duplicate_id
        self.num_labels = len(len(duplicate_id.unique())) # Number of unique labels
        self.label_idx = dict()
        self.device = duplicate_id.device
        
        for i in range(self.num_labels):
            self.label_idx[i] = torch.where(duplicate_id == i)[0]
        
        self.extra = (batch_size - len(duplicate_id) % batch_size) % batch_size # Number of extra samples to add
        
    @staticmethod
    def random_mix(ts):
        order = torch.randperm(len(ts))
        return ts[order] 

        
    def __iter__(self):
        order = np.random.choice(self.num_labels, self.num_labels, replace=False)
        # Wrap into a matrix 
        idxs = torch.cat([self.label_idx[i] for i in order], dim=0).to(self.device) + 1
        idxs = torch.cat([idxs, torch.zeros(self.extra).long().to(self.device)], dim=0)
        idxs = torch.view(self.batch_size, -1).T
        
        for bs in range(idxs):
            get = torch.nonzero(bs)
            re = bs[get].squeeze() -1
            
            count = 0
            i = 0
            # print(self.random_mix(self.labels_idx[order[i]])[count])
            if re.shape[0] < self.bs:
                if count>len(self.labels_idx[order[i]]):
                    count = 0
                    i += 1
                lucky = torch.ones(1, device=self.device)*self.random_mix(self.labels_idx[order[i]])[count]
                re = torch.cat([re, lucky.long().to(self.device)])
                count += 1
            # print(re.tolist())
            # print(len(re))
            yield re