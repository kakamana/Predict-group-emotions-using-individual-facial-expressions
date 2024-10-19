import streamlit as st
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from collections import Counter
import matplotlib.pyplot as plt 
import io
## unsupervised learning
from Autoencoder import Autoencoder, visualize_autoencoder
from KMeansClustering import extract_features, apply_kmeans, visualize_clusters, analyze_clusters
from PCAVisualization import apply_pca, visualize_pca
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
print(f"Using device: {device}")

# Define model
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

@st.cache_resource
def load_models():
    # Load emotion recognition model
    emotion_model = FacialExpressionModel(num_classes=len(EMOTIONS_LIST))
    emotion_model.load_state_dict(torch.load('./Models/facial_expression_recognition_weights.pth', map_location=device))
    emotion_model.to(device)
    emotion_model.eval()

    # Load YOLO model
    yolo_model = YOLO('./YoloModels/yolov8n-face.pt')

    # Load Autoencoder model
    dont_do_autoencoder = False
    if not dont_do_autoencoder:
        autoencoder = Autoencoder().to(device)
        autoencoder.load_state_dict(torch.load('./Models/autoencoder_weights.pth', map_location=device))
        autoencoder.eval()

    return emotion_model, yolo_model, autoencoder
    #return emotion_model, yolo_model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(Image.fromarray(image)).to(device)

def updated_preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Ensure the image is in RGB format
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB, no change needed
            pass
        else:
            raise ValueError("Unsupported image format")
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError("Unsupported image type")
    
    return transform(image).to(device)

def predict_emotion(model, image):
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return EMOTIONS_LIST[predicted.item()], confidence.item()

def get_group_emotion(emotions):
    emotion_counts = Counter(emotions)
    return emotion_counts.most_common(1)[0][0]

def process_image(yolo_model, emotion_model, image):
    # Convert PIL Image to numpy array
    np_image = np.array(image)
    
    # Detect faces
    results = yolo_model(np_image)

    # Image with boxes and IDs only
    boxed_image = np_image.copy()

    # Detailed emotion information
    emotions = []
    emotion_details = []

    for face_id, box in enumerate(results[0].boxes, start=1):
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Extract face
        face = np_image[y1:y2, x1:x2]
        
        # Predict emotion
        face_tensor = preprocess_image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        emotion, confidence = predict_emotion(emotion_model, face_tensor)
        emotions.append(emotion)
        
        # Draw rectangle around face
        cv2.rectangle(boxed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(boxed_image, f"{face_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Store emotion details
        emotion_details.append(f"id:{face_id} - {emotion} ({confidence:.2f})")        

    # Get total face count and group emotion
    total_faces = len(results[0].boxes)
    group_emotion = get_group_emotion(emotions)
    
    return (Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)),
            total_faces, group_emotion, emotion_details)

def process_frame(yolo_model, emotion_model, frame):
    results = yolo_model(frame)
    
    emotions = []
    emotion_details = []
    
    for face_id, box in enumerate(results[0].boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        face = frame[y1:y2, x1:x2]
        face_tensor = preprocess_image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        emotion, confidence = predict_emotion(emotion_model, face_tensor)
        emotions.append(emotion)
        
        # Draw rectangle and emotion on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{face_id}: {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        emotion_details.append(f"id:{face_id} - {emotion} ({confidence:.2f})")
    
    total_faces = len(results[0].boxes)
    group_emotion = get_group_emotion(emotions) if emotions else "No faces detected"
    
    return frame, total_faces, group_emotion, emotion_details

def process_video(yolo_model, emotion_model, video_source):
    cap = cv2.VideoCapture(video_source)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, total_faces, group_emotion, emotion_details = process_frame(yolo_model, emotion_model, frame)
        
        stframe.image(processed_frame, channels="RGB", use_column_width=True)
        
        # Display real-time statistics
        st.sidebar.write(f"Total faces: {total_faces}")
        st.sidebar.write(f"Group Emotion: {group_emotion}")
        for detail in emotion_details:
            st.sidebar.write(detail)
    
    cap.release()

def calculate_inertia(features, max_clusters):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
    return inertias    

def main():
    st.title("Milestone II: Facial Emotion Recognition App")

    # Load models
    emotion_model, yolo_model, autoencoder  = load_models()
    #emotion_model, yolo_model  = load_models()

    st.header("Unsupervised Learning Visualizations")
    unsupervised_method = st.selectbox(
    "Choose an unsupervised learning method to visualize:",
    ("Autoencoder", "K-means Clustering", "PCA"))

    if unsupervised_method == "Autoencoder":
        uploaded_file = st.file_uploader("Choose an image for autoencoder...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
            ])
            preprocessed_image = transform(image).to(device)
            visualize_autoencoder(autoencoder, preprocessed_image)

    elif unsupervised_method in ["K-means Clustering", "PCA"]:
        uploaded_files = st.file_uploader("Choose images for feature extraction...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            if st.button("Process Unsupervised Learning"):
                #features = extract_features(emotion_model, uploaded_files, device)
                features, file_names = extract_features(emotion_model, uploaded_files, device)
                
                if unsupervised_method == "K-means Clustering":
                    max_clusters = 10
                    inertias = calculate_inertia(features, max_clusters)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(range(1, max_clusters + 1), inertias, marker='o')
                    ax.set_xlabel('Number of Clusters (k)')
                    ax.set_ylabel('Inertia')
                    ax.set_title('Elbow Method for Optimal k')
                    st.pyplot(fig)
                    
                    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=max_clusters, value=5)
                    kmeans, labels, centers, _ = apply_kmeans(features, max_clusters=n_clusters)

                    if kmeans is not None:
                        st.subheader("K-means Clustering Visualization")
                        visualize_clusters(features, labels)
                    
                        cluster_info = analyze_clusters(features, labels, uploaded_files, centers)

                        st.subheader("Cluster Analysis")
                        for i, info in cluster_info.items():
                            st.write(f"Cluster {i}:")
                            st.write(f"  Size: {info['size']} images")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(info['representative_image'], caption="Representative Image", use_column_width=True)
                            with col2:
                                st.image(info['diverse_image'], caption="Most Diverse Image", use_column_width=True)
                            
                            if st.checkbox(f"Show all images in Cluster {i}"):
                                st.write(", ".join(info['all_images']))

                            # Predict emotions for representative and diverse images
                            rep_image = Image.open(io.BytesIO(info['representative_image']))
                            div_image = Image.open(io.BytesIO(info['diverse_image']))
                            rep_emotion, rep_confidence = predict_emotion(emotion_model, preprocess_image(np.array(rep_image)))
                            div_emotion, div_confidence = predict_emotion(emotion_model, preprocess_image(np.array(div_image)))
                            st.write(f"  Representative image emotion: {rep_emotion} (confidence: {rep_confidence:.2f})")
                            st.write(f"  Diverse image emotion: {div_emotion} (confidence: {div_confidence:.2f})")
                    
                        

                        # Analyze cluster-emotion relationship
                        st.subheader("Cluster-Emotion Relationship")
                        cluster_emotions = {i: [] for i in range(n_clusters)}
                        for idx, label in enumerate(labels):
                            image = Image.open(uploaded_files[idx])
                            emotion, _ = predict_emotion(emotion_model, updated_preprocess_image(np.array(image)))
                            cluster_emotions[label].append(emotion)
                        
                        for cluster, emotions in cluster_emotions.items():
                            emotion_counts = Counter(emotions)
                            st.write(f"Cluster {cluster} emotions:")
                            for emotion, count in emotion_counts.most_common():
                                st.write(f"  {emotion}: {count} ({count/len(emotions)*100:.2f}%)")
                    else:
                        st.warning("Unable to perform K-means clustering. Please upload more images.")
                    
                    """ kmeans, labels = apply_kmeans(features)
                    if kmeans is not None and labels is not None:
                        visualize_clusters(features, labels)
                    else:
                        st.warning("Unable to perform K-means clustering. Please upload more images.") """                
                else:  # PCA
                    pca = apply_pca(features)
                    visualize_pca(pca, (96, 96))
        else:
            st.write("Please upload some images to process.")

    # Choose between image and video
    input_type = st.radio("Choose input type:", ("Image", "Video"))

    if input_type == "Image":
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            original_image, boxed_image, total_faces, group_emotion, emotion_details = process_image(yolo_model, emotion_model, image)

            # Display original and processed images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_column_width=True)

            with col2:
                st.subheader("Processed Image")
                st.image(boxed_image, use_column_width=True)

            # Results section
            st.subheader("Emotion Recognition Results")
            st.write(f"Total faces: {total_faces}")
            st.write(f"Group Emotion: {group_emotion}")
            for detail in emotion_details:
                st.write(detail)
    elif input_type == "Video":
        video_source = st.radio("Choose video source:", ("Upload Video", "Live Camera"))
        if video_source == "Upload Video":
            uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_video.read())
                st.video(tfile.name)
                if st.button("Process Video"):
                    process_video(yolo_model, emotion_model, tfile.name)
                    os.unlink(tfile.name)
            
        elif video_source == "Live Camera":
            if st.button("Start Live Camera"):
                process_video(yolo_model, emotion_model, 0)  # 0 is typically the default camera
        
        

if __name__ == "__main__":
    main()