from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image

class CustomDataset(Dataset):
    
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None) -> None:
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 1])
        image = read_image(img_path).float()
        label = self.img_labels.iloc[index, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        
        