import os
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
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
    def __init__(self, d, input, output, drop):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input, d)
        self.layer2 = nn.Linear(d, d*2)
        self.layer3 = nn.Linear(d*2, d*4)
        self.layer4 = nn.Linear(d*4, d*2)
        self.layer5 = nn.Linear(d*2, d)
        self.output = nn.Linear(d, output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop)
        self.normalization1 = nn.BatchNorm1d(d)
        self.normalization0 = nn.BatchNorm1d(input)
        self.normalization2 = nn.BatchNorm1d(d*2)
        self.normalization3 = nn.BatchNorm1d(d*4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.normalization1(self.layer1(self.normalization0(x))))
        x = self.relu((self.layer2(x)))
        x = self.dropout(self.relu(self.normalization3(self.layer3(x))))
        x = self.relu((self.layer4(x)))
        x = self.relu(self.normalization1(self.layer5(x)))
        x = self.sigmoid(self.output(x))
        return x

class DataSet:
    def __init__(self, data, y_col):
        self.X = data.drop(columns=[y_col])
        self.y = data[y_col]
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
        # Compute class counts
        class_counts = torch.bincount(self.y_train_tensor.to(torch.int64).flatten())
        # class_counts[1] = class_counts[1] // 2  # Increase the number of samples for class 1
        total_samples = int(sum(class_counts)) #len(self.y_train_tensor.to(torch.int64).flatten())

        # Compute weights for each class (inverse of frequency)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[self.y_train_tensor.to(torch.int64).flatten()]  # Assign weight to each sample

        # Create Weighted Sampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
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