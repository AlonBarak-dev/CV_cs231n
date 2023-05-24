import torch
from torchvision.io import read_image
from model import RiceDetector
import sys

def load_model(path:str):
    model = RiceDetector()
    model.load_state_dict(torch.load(path))
    return model

def load_image(path:str):
    image = read_image(path).unsqueeze(0).float()
    return image

def class_2_label(class_num):
    l2c = {
        0 : "Arborio", 
        1 : "Basmati", 
        2 : "Ipsala", 
        3 : "Jasmine", 
        4 : "Karacadag"
    }
    return l2c[class_num]


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Expecting an image path!")
        exit()
    
    image_path = sys.argv[1]
    model = load_model("models/model_2.pt")
    image = load_image(image_path)
    
    pred = model(image)
    
    label = class_2_label(pred.argmax(1).item())
    print(label)