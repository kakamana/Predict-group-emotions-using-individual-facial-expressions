import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
import io

def extract_features_old(model, uploaded_files, device):
    features = []
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    model.eval()
    with torch.no_grad():
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            feature = model.features(image_tensor)
            features.append(feature.view(feature.size(0), -1).cpu().numpy())
    
    #return np.concatenate(features)
    return np.vstack(features)

def extract_features(model, uploaded_files, device):
    features = []
    file_names = []
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    model.eval()
    with torch.no_grad():
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            feature = model.features(image_tensor)
            features.append(feature.view(feature.size(0), -1).cpu().numpy())
            file_names.append(uploaded_file.name)
    
    return np.vstack(features), file_names

def apply_kmeans_old(features, max_clusters=5):
    n_samples = features.shape[0]
    n_clusters = min(n_samples, max_clusters)
    
    if n_samples < 2:
        st.warning("K-means clustering requires at least 2 samples. Please upload more images.")
        return None, None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return kmeans, cluster_labels

def apply_kmeans(features, max_clusters=5):
    n_samples = features.shape[0]
    n_clusters = min(n_samples, max_clusters)
    
    if n_samples < 2:
        st.warning("K-means clustering requires at least 2 samples. Please upload more images.")
        return None, None, None, None
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return kmeans, cluster_labels, kmeans.cluster_centers_, kmeans.inertia_

def analyze_clusters(features, labels, uploaded_files, cluster_centers):
    n_clusters = len(cluster_centers)
    cluster_info = {}
    
    for i in range(n_clusters):
        cluster_features = features[labels == i]
        cluster_files = [uploaded_files[j] for j in range(len(labels)) if labels[j] == i]
        
        # Find the image closest to the cluster center
        distances = euclidean_distances(cluster_features, [cluster_centers[i]])
        representative_idx = np.argmin(distances)
        
        # Find the most different image in the cluster
        if len(cluster_features) > 1:
            intra_distances = euclidean_distances(cluster_features)
            np.fill_diagonal(intra_distances, np.inf)
            diverse_idx = np.unravel_index(np.argmax(intra_distances), intra_distances.shape)[0]
        else:
            diverse_idx = 0
        
        # Read image data
        representative_image = Image.open(cluster_files[representative_idx])
        diverse_image = Image.open(cluster_files[diverse_idx])
        
        # Convert images to bytes
        rep_img_byte_arr = io.BytesIO()
        representative_image.save(rep_img_byte_arr, format='PNG')
        rep_img_byte_arr = rep_img_byte_arr.getvalue()

        div_img_byte_arr = io.BytesIO()
        diverse_image.save(div_img_byte_arr, format='PNG')
        div_img_byte_arr = div_img_byte_arr.getvalue()

        cluster_info[i] = {
            'size': len(cluster_features),
            'representative_image': rep_img_byte_arr,
            'diverse_image': div_img_byte_arr,
            'all_images': [file.name for file in cluster_files]
        }
    
    return cluster_info

def visualize_clusters(features, labels):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
    ax.set_title('K-means Clustering of Facial Features')
    plt.colorbar(scatter)
    st.pyplot(fig)