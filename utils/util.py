import os
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Loader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_csv(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        return pd.read_csv(file_path)

    def load_excel(self, file_name, sheet_name=0):
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        return pd.read_excel(file_path, sheet_name=sheet_name)

    def load_json(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        return pd.read_json(file_path)

class NeuralNetwork(nn.Module):
    def __init__(self, d, input, output):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input, 128//d)
        self.layer2 = nn.Linear(128//d, 128*2//d)
        self.layer3 = nn.Linear(128*2//d, 128//d)
        # self.layer4 = nn.Linear(128, 128)
        # self.layer5 = nn.Linear(128, 128)
        self.output = nn.Linear(128//d, output)
        self.relu = nn.ReLU()
        self.normalization = nn.BatchNorm1d(128//d)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.normalization(self.layer1(x)))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        # x = self.relu(self.layer4(x))
        # x = self.relu(self.layer5(x))
        x = self.sigmoid(self.output(x))
        return x

class DataSet:
    def __init__(self, data):
        self.X = data.drop(columns=['diabetes'])
        self.y = data['diabetes']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train_tensor = torch.tensor(self.X_train.values, dtype=torch.float32, device=device)
        self.y_train_tensor = torch.tensor(self.y_train.values, dtype=torch.float32, device=device).unsqueeze(1)
        self.X_test_tensor = torch.tensor(self.X_test.values, dtype=torch.float32, device=device)
        self.y_test_tensor = torch.tensor(self.y_test.values, dtype=torch.float32, device=device).unsqueeze(1)
        # One-hot encode the target variable using PyTorch
        self.y_train_onehot = torch.nn.functional.one_hot(self.y_train_tensor.to(torch.int64), num_classes=2).squeeze(1)
        self.y_test_onehot = torch.nn.functional.one_hot(self.y_test_tensor.to(torch.int64), num_classes=2).squeeze(1)

        self.train_dataset = DiabetesDataset(self.X_train_tensor, self.y_train_onehot)
        self.test_dataset = DiabetesDataset(self.X_test_tensor, self.y_test_onehot)
    def get_dataloaders(self, batch_size=64):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    
    def get_datasets(self):
        return [self.train_dataset, self.test_dataset]

class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]