import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from FeatureEngineering import extract_features

class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []
        self.class_to_idx = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Neutral": 4, "Sad": 5, "Surprise": 6}
        
        # Load data
        for emotion in self.class_to_idx.keys():
            emotion_dir = os.path.join(self.root_dir, emotion)
            for img_name in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[emotion])
        
        # Set default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # For SVM and Random Forest, we need to extract features
        if self.mode in ['svm', 'rf']:
            features = extract_features(np.array(image))
            return torch.FloatTensor(features), label
        
        # For CNN, we return the image tensor directly
        return image, label

# Usage example:
# train_dataset = FacialExpressionDataset(root_dir='path/to/train/data', mode='train')
# svm_dataset = FacialExpressionDataset(root_dir='path/to/train/data', mode='svm')
# rf_dataset = FacialExpressionDataset(root_dir='path/to/train/data', mode='rf')