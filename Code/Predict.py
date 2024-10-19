import os
import shutil
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from datetime import datetime

# Define emotions list
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Set up device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
#print(f"Using device: {device}")

# Define model (FacialExpressionModel class remains the same)
class FacialExpressionModel(nn.Module):    
    def __init__(self, num_classes):
        super(FacialExpressionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
# Custom Dataset for facial expressions
class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for img_name in os.listdir(root_dir):
            img_path = os.path.join(root_dir, img_name)
            self.images.append(img_path)
            # We don't have labels for this dataset, so we'll use a dummy label
            self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Function to split dataset into train, test, and validation sets
def split_dataset(src_dir, train_dir, test_dir, val_dir, split=(0.7, 0.2, 0.1)):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_files = [f for f in os.listdir(src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_files)

    train_split = int(len(all_files) * split[0])
    test_split = int(len(all_files) * (split[0] + split[1]))

    for i, file in enumerate(all_files):
        src_path = os.path.join(src_dir, file)
        if i < train_split:
            dst_path = os.path.join(train_dir, file)
        elif i < test_split:
            dst_path = os.path.join(test_dir, file)
        else:
            dst_path = os.path.join(val_dir, file)
        shutil.copy(src_path, dst_path)

    print(f"Dataset split completed. Train: {train_split}, Test: {test_split - train_split}, Val: {len(all_files) - test_split}")


# Function to load the facial expression recognition model
def load_facial_expression_model(weights_path):
    model = FacialExpressionModel(num_classes=len(EMOTIONS_LIST))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    return model

# Function to fine-tune the model
def fine_tune_model(model, train_loader, val_loader, num_epochs=30, lr=1e-5):
    # Freeze early layers
    for param in model.features[:6].parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    train_losses,val_losses = [],[]
    epochs_no_improve = 0
    early_stop_threshold = 5
    

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_preds,all_labels = [],[]
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            #loss = criterion(outputs, torch.zeros(inputs.size(0), dtype=torch.long).to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds,val_labels = [],[]
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './Models/FineTunedModelForCeleb/best_fine_tuned_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stop_threshold:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Plot and save confusion matrix for each epoch
        cm = confusion_matrix(val_labels, val_preds)
        plot_confusion_matrix(cm, EMOTIONS_LIST, epoch)

    # Plot and save train/val loss
    plot_train_val_loss(train_losses, val_losses)
    
    print("Fine-tuning completed. Best model saved as 'best_fine_tuned_model.pth'")

def plot_confusion_matrix(cm, class_names, epoch):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Create filename with timestamp
    filename = f'./ModelsOutputCharts/confusion_matrix_epoch_{epoch+1}_{timestamp}.png'
    plt.savefig(filename)
    plt.close()


def plot_train_val_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Create filename with timestamp
    filename = f'./ModelsOutputCharts/train_val_loss_{timestamp}.png'
    plt.savefig(filename)
    plt.close()    

# Function to preprocess image for facial expression recognition
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(Image.fromarray(image)).to(device)


# Function to predict emotion
def predict_emotion(model, image):
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return EMOTIONS_LIST[predicted.item()], confidence.item()
        #_, predicted = torch.max(output, 1)
        #return EMOTIONS_LIST[predicted.item()]


# Function to detect faces and predict emotions
def detect_and_predict(yolo_model, emotion_model, image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image)
    
    emotions = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face = image[y1:y2, x1:x2]
            face_tensor = preprocess_image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            emotion = predict_emotion(emotion_model, face_tensor)
            emotions.append(emotion)
    
    return emotions


# Function to process the entire dataset
def process_dataset(yolo_model, emotion_model, data_dir):
    all_emotions = []
    total_images = len([f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for i, img_file in enumerate(tqdm(os.listdir(data_dir), total=total_images, desc="Processing images")):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(data_dir, img_file)
            emotions = detect_and_predict(yolo_model, emotion_model, img_path)
            all_emotions.extend(emotions)
        
        # Update progress
        progress = (i + 1) / total_images * 100
        print(f"\rProgress: {progress:.2f}% completed", end="")
    
    print("\nProcessing completed!")
    return all_emotions

def process_sample_image(yolo_model, emotion_model, image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Detect faces
    results = yolo_model(image)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract face
            face = image[y1:y2, x1:x2]
            
            # Predict emotion
            face_tensor = preprocess_image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            emotion = predict_emotion(emotion_model, face_tensor)
            
            # Draw rectangle around face
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put text of emotion above the rectangle
            cv2.putText(original_image, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the output image
    cv2.imwrite(output_path, original_image)
    print(f"Processed image saved to {output_path}")

def get_group_emotion(emotions):
    emotion_counts = Counter(emotions)
    return emotion_counts.most_common(1)[0][0]    

def process_and_save_image(yolo_model, emotion_model, image_path, output_folder):
    # Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Detect faces
    results = yolo_model(image)

    all_boxes = []
    for r in results:
        all_boxes.extend(r.boxes)
    
    emotions = []
    #for r in results:
    #for face_id, r in enumerate(results, start=1):
    for face_id, box in enumerate(all_boxes, start=1):
        #boxes = r.boxes
        #for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
        # Extract face
        face = image[y1:y2, x1:x2]
            
        # Predict emotion
        face_tensor = preprocess_image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        emotion, confidence = predict_emotion(emotion_model, face_tensor)
        emotions.append(emotion)
            
        # Draw rectangle around face
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label with face ID, emotion, and confidence
        label = f"Face {face_id}: {emotion} ({confidence:.2f})"
            
        # Put text of emotion above the rectangle
        cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Get total face count and group emotion
    total_faces = len(all_boxes)
    group_emotion = get_group_emotion(emotions)

    # Put total face count on top left
    cv2.putText(original_image, f"Total Faces: {total_faces}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Put group emotion on top middle
    text_size = cv2.getTextSize(f"Group Emotion: {group_emotion}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (original_image.shape[1] - text_size[0]) // 2
    cv2.putText(original_image, f"Group Emotion: {group_emotion}", (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Create output filename
    base_name = os.path.basename(image_path)
    output_name = f"{os.path.splitext(base_name)[0]}_output{os.path.splitext(base_name)[1]}"
    output_path = os.path.join(output_folder, output_name)

    # Save the output image
    cv2.imwrite(output_path, original_image)
    print(f"Processed image saved to {output_path}")
    print(f"Total faces detected: {total_faces}")
    print(f"Group emotion: {group_emotion}")

# Configuration dictionary to control the execution of each step
config = {
    "split_dataset": False,
    "load_pretrained_model": True,
    "prepare_datasets": True,
    "fine_tune_model": True,
    "load_fine_tuned_model": True,  
    "load_yolo_model": True,
    "process_test_data": True,
    "analyze_results": True
}

# Main execution
if __name__ == "__main__":
    # Set paths
    src_dir = 'dataset/Scraped-Dataset-for-GroupEmotion'
    train_dir = 'dataset/trainCelb'
    test_dir = 'dataset/testCelb'
    val_dir = 'dataset/valCelb'
    facial_expression_weights = './Models/facial_expression_recognition_weights.pth'
    output_folder = 'output'
    train_loader = None
    val_loader = None

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Split dataset
    if config["split_dataset"]:
        print("Splitting dataset...")
        split_dataset(src_dir, train_dir, test_dir, val_dir)

    # Step 2: Load pre-trained facial expression model
    if config["load_pretrained_model"]:
        print("Loading facial expression model...")
        emotion_model = load_facial_expression_model(facial_expression_weights)
        print("Facial expression model loaded!")

    # Step 3: Prepare datasets for fine-tuning
    if config["prepare_datasets"]:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = FacialExpressionDataset(train_dir, transform=transform)
        val_dataset = FacialExpressionDataset(val_dir, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Step 4: Fine-tune the model
    if config["fine_tune_model"]:
        if train_loader is None or val_loader is None:
            raise ValueError("Datasets not prepared. Set config['prepare_datasets'] to True.")
        print("Starting fine-tuning...")
        fine_tune_model(emotion_model, train_loader, val_loader, num_epochs=30)

    # Step 5: Load the best fine-tuned model
    if config["load_fine_tuned_model"]:
        print("Loading the best fine-tuned model...")
        emotion_model = load_facial_expression_model(facial_expression_weights)
        emotion_model.load_state_dict(torch.load('./Models/FineTunedModelForCeleb/best_fine_tuned_model.pth'))
        emotion_model.eval()
        #emotion_model.load_state_dict(torch.load('best_fine_tuned_model.pth'))
        #emotion_model.eval()

    # Step 6: Load pre-trained YOLOv8 model for face detection
    if config["load_yolo_model"]:
        print("Loading YOLOv8 model...")
        yolo_model = YOLO('./YoloModels/yolov8n-face.pt')  # Make sure to download this model
        print("YOLOv8 model loaded!")

    # Step 7: Process a sample image
    # sample_image_path = os.path.join(test_dir, os.listdir(test_dir)[0])  # Get the first image from test_dir
    # output_image_path = os.path.join(output_folder, 'sample_output.jpg')
    # process_sample_image(yolo_model, emotion_model, sample_image_path, output_image_path)
    all_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if all_images:
        random_image = random.choice(all_images)
        random_image_path = os.path.join(test_dir, random_image)
        print(f"Processing random image: {random_image}")
        process_and_save_image(yolo_model, emotion_model, random_image_path, output_folder)
    else:
        print("No images found in the test directory.")    

    # Step 8: Process the test dataset with the fine-tuned model
    if config["process_test_data"]:
        print("Processing the test dataset with fine-tuned model...")
        all_emotions = process_dataset(yolo_model, emotion_model, test_dir)

    # Step 9: Analyze results
    if config["analyze_results"]:
        print("Analyzing results...")
        emotion_counts = {emotion: all_emotions.count(emotion) for emotion in EMOTIONS_LIST}
        total_faces = sum(emotion_counts.values())

        print("\nResults:")
        for emotion, count in emotion_counts.items():
            percentage = (count / total_faces) * 100 if total_faces > 0 else 0
            print(f"{emotion}: {count} ({percentage:.2f}%)")

        # Plot emotion distribution
        plt.figure(figsize=(10, 6))
        plt.bar(emotion_counts.keys(), emotion_counts.values())
        plt.title('Emotion Distribution in Dataset (After Fine-tuning)')
        plt.xlabel('Emotions')
        plt.ylabel('Number of Faces')
        # Create filename with datetime prefix
        timestamp = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
        filename = f'./ModelsOutputCharts/{timestamp}_emotion_distribution_fine_tuned.png'
        plt.savefig(filename)
        plt.close()

    print("Analysis completed. Check the 'emotion_distribution_fine_tuned.png' for visualization.")
