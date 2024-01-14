import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Define the simple differential equation y' = -2x
def differential_eqn(x, y):
    return -2 * x

# Generate the dataset by solving the differential equation
x_range = np.linspace(0, 1, 100)
initial_y = 0.5
y_solution = np.zeros_like(x_range)
y_solution[0] = initial_y
for i in range(1, len(x_range)):
    h = x_range[i] - x_range[i - 1]
    y_solution[i] = y_solution[i - 1] + h * differential_eqn(x_range[i - 1], y_solution[i - 1])

# Create input-output pairs for the dataset
input_data = x_range.reshape(-1, 1)  # Reshape to (num_samples, num_features) format
output_data = y_solution.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Define a custom dataset for PyTorch
class DiffEqDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        self.output_data = torch.tensor(output_data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

# Create data loaders for training and testing
train_dataset = DiffEqDataset(X_train, y_train)
test_dataset = DiffEqDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class DiffEqCNN(nn.Module):
    def __init__(self):
        super(DiffEqCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 94, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model and define loss function and optimizer
model = DiffEqCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))  # Add a channel dimension for Conv1D
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss/len(train_loader)}")

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs.unsqueeze(1))  # Add a channel dimension for Conv1D
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader)}")
