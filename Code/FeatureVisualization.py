import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
import dlib
from FeatureEngineering import preprocess_image, extract_hog_features, extract_lbp_features, extract_facial_landmarks
from datetime import datetime
import os
import random

def visualize_hog(image):
    preprocessed_image = preprocess_image(image)
    hog_features, hog_image = hog(preprocessed_image, orientations=8, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(preprocessed_image, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.axis('off')
    
    plt.tight_layout()
    #ModelsOutputCharts
     # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Create filename with timestamp
    filename = f'./ModelsOutputCharts/hog_visualization_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

def visualize_lbp(image):
    preprocessed_image = preprocess_image(image)
    lbp = local_binary_pattern(preprocessed_image, P=8, R=1, method='uniform')
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(preprocessed_image, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(lbp, cmap='gray')
    plt.title('LBP Visualization')
    plt.axis('off')
    
    plt.tight_layout()
     # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # Create filename with timestamp
    filename = f'./ModelsOutputCharts/lbp_visualization_{timestamp}.png'
    plt.savefig(filename)
    plt.close()

def visualize_facial_landmarks(image):
    preprocessed_image = preprocess_image(image)
    landmarks = extract_facial_landmarks(preprocessed_image)
    
    if landmarks is not None:
        landmarks = landmarks.reshape(-1, 2)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        #plt.imshow(image, cmap='gray')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(preprocessed_image, cmap='gray')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=3)
        plt.title('Facial Landmarks')
        plt.axis('off')
        
        plt.tight_layout()
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Create filename with timestamp
        filename = f'./ModelsOutputCharts/facial_landmarks_visualization_{timestamp}.png'
        plt.savefig(filename)        
        plt.close()
    else:
        print("No face detected in the image.")

def main():
    image_path=None
    test_dir = 'dataset/testCelb'
    # Load a sample image
    all_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if all_images:
        random_image = random.choice(all_images)
        random_image_path = os.path.join(test_dir, random_image)
        image_path=random_image_path
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate and save visualizations
    visualize_hog(image)
    visualize_lbp(image)
    visualize_facial_landmarks(image)

if __name__ == "__main__":
    main()