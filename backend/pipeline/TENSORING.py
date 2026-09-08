import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# CONVERT ENCODED DATA TO TENSOR
# =========================================================

def make_tensor(encoded_data):
    tensor = torch.tensor(
        encoded_data,
        dtype=torch.float32
    )
    return tensor


# =========================================================
# AUTOENCODER
# =========================================================

class AutoEncoder(nn.Module):

    def __init__(self, input_feature):
        super().__init__()
        self.input_feature = input_feature

        self.encoder = nn.Sequential(
            nn.Linear(input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_feature)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# =========================================================
# LOAD TRAINED AUTOENCODER
# =========================================================

def load_autoencoder(model_path, input_feature):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = AutoEncoder(input_feature).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, device


# =========================================================
# GENERATE EMBEDDING
# =========================================================

def make_embedding(tensor_data, model, device):
    tensor_data = tensor_data.to(device)

    with torch.no_grad():
        embedding = model.encoder(tensor_data)

    return embedding.cpu()


# =========================================================
# GET OR TRAIN AUTOENCODER (dynamic feature-count support)
# =========================================================

_model_cache = {}   # feature_count -> trained model

def get_or_train_autoencoder(tensor_data, device, pretrained_model=None, pretrained_features=27, epochs=50):
    input_feature = tensor_data.shape[1]

    if input_feature == pretrained_features and pretrained_model is not None:
        return pretrained_model, device

    if input_feature in _model_cache:
        return _model_cache[input_feature], device

    model = AutoEncoder(input_feature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    x = tensor_data.to(device)
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(x)
        loss = criterion(reconstructed, x)
        loss.backward()
        optimizer.step()

    model.eval()
    _model_cache[input_feature] = model
    return model, device


# =========================================================
# TRAINING (unchanged — this only runs when you execute
# this file directly, e.g. `python TENSORING.py`)
# =========================================================

if __name__ == "__main__":
    # ... your existing training block stays exactly as-is here
    pass