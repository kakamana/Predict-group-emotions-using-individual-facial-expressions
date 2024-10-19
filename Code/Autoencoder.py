import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    

def train_autoencoder(model, dataloader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = img.to(device)
            _, decoded = model(img)
            loss = criterion(decoded, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def visualize_autoencoder(model, image):
    model.eval()
    with torch.no_grad():
        encoded, decoded = model(image.unsqueeze(0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image.permute(1, 2, 0).cpu())
    ax1.set_title('Original Image')
    ax2.imshow(decoded.squeeze().permute(1, 2, 0).cpu())
    ax2.set_title('Reconstructed Image')
    st.pyplot(fig)
    """ model.eval()
    with torch.no_grad():
        encoded, decoded = model(image.unsqueeze(0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title('Original Image')
    ax2.imshow(decoded.squeeze().permute(1, 2, 0))
    ax2.set_title('Reconstructed Image')
    st.pyplot(fig)

    # Visualize encoded features using t-SNE
    encoded_flat = encoded.view(encoded.size(0), -1).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    encoded_tsne = tsne.fit_transform(encoded_flat)
    
    fig, ax = plt.subplots()
    ax.scatter(encoded_tsne[:, 0], encoded_tsne[:, 1])
    ax.set_title('t-SNE visualization of encoded features')
    st.pyplot(fig) """