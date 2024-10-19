import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st

def apply_pca(features, n_components=50):
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca

def visualize_pca(pca, image_shape):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cumulative explained variance ratio
    ax1.plot(np.cumsum(pca.explained_variance_ratio_))
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance Ratio')
    ax1.set_title('PCA Explained Variance Ratio')
    
    # First principal component
    ax2.imshow(pca.components_[0].reshape(image_shape), cmap='gray')
    ax2.set_title('First Principal Component')
    
    st.pyplot(fig)