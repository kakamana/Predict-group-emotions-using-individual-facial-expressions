import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import dlib
from torchvision import transforms
from torchvision.models import vgg16
import torch

def preprocess_image(image, target_size=(48, 48)):
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values
    image = image.astype(np.float32) / 255.0
    
    return image

def augment_image(image):
    # Random rotation
    angle = np.random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random brightness and contrast adjustment
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.uniform(-30, 30)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image

def extract_hog_features(image):
    features, _ = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    return hist

def extract_facial_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    faces = detector(image, 1)
    if len(faces) == 0:
        return None
    
    shape = predictor(image, faces[0])
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
    return landmarks.flatten()

def extract_deep_features(image):
    model = vgg16(pretrained=True)
    model = torch.nn.Sequential(*list(model.features.children())[:-1])
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().cpu().numpy()

def extract_features(image):
    preprocessed_image = preprocess_image(image)
    augmented_image = augment_image(preprocessed_image)
    
    hog_features = extract_hog_features(augmented_image)
    lbp_features = extract_lbp_features(augmented_image)
    landmarks = extract_facial_landmarks(augmented_image)
    deep_features = extract_deep_features(augmented_image)
    
    return np.concatenate([hog_features, lbp_features, landmarks, deep_features])