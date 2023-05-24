import os
import pandas as pd
from torchvision.io import read_image, write_jpeg
from tqdm import tqdm


image_save_path = "Data/images/image_"
label_save_path = "Data/labels/label_csv.csv"
load_path = "Rice_Image_Dataset/"

rice_types = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

labels = {
    "img_path": [], 
    "label": []
}

sample_idx = 0
# loop over the classes folders
for idx in tqdm(range(len(rice_types)), desc="folders: "):
    for image_path in tqdm(os.listdir(load_path + rice_types[idx]), desc="files: "):
        image = read_image(load_path + rice_types[idx] + "/" + image_path)
        write_jpeg(image, image_save_path+str(sample_idx)+".jpg")
        labels["label"].append(idx)
        labels["img_path"].append("image_"+str(sample_idx)+".jpg")
        sample_idx += 1

labels_df = pd.DataFrame(labels)
labels_df.to_csv(label_save_path)
print(labels_df)
        
