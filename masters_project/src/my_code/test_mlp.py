import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.mlp_model import FluidNet
from torch.utils.data import DataLoader, Dataset
# from trained_cnn import FluidDataset


BATCH_SIZE = 64
N_POINTS = 51
num_features = N_POINTS**2

if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
    print("CUDA is available")
else: 
    print("CUDA is not available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_input = np.load('datasets/mlp/testing_inputs.npy').astype(np.float32)  # (18120, 51, 51)
test_input = test_input.reshape(test_input.shape[0], -1)  # Flatten the last dimension (18120, 1, 2601)

test_min = np.min(test_input)
test_max = np.max(test_input)
test_input = (test_input - test_min) / (test_max - test_min)

test_output = np.load('datasets/mlp/testing_outputs.npy').astype(np.float32)  # (18120, 51, 51)
test_output = test_output.reshape(test_output.shape[0], -1)  # Flatten the last dimension (18120, 1, 2601)

test_min = np.min(test_output)
test_max = np.max(test_output)
test_output = (test_output - test_min) / (test_max - test_min)

class FluidTestDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = torch.tensor(input_data)
        self.target_data = torch.tensor(target_data)
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.target_data[idx]

        return x, y

test_dataset = FluidTestDataset(test_input, test_output)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# print("test loader shape: ", len(test_dataset))  # 18120
# print("test output shape: ", test_output.shape)  # (18120, 2601)


# Define a function to calculate the Mean Squared Error (MSE)
def calculate_mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

# Function to test the model
def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    test_loss /= len(test_loader)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = calculate_mse(all_predictions, all_targets)

    return test_loss, mse, all_predictions


# Load the trained model
model = FluidNet(features=num_features).to(device)
model.load_state_dict(torch.load('datasets/mlp/trained_model_mlp.pth'))
model.eval()

# Define the loss function
criterion = nn.MSELoss()

# Test the model
test_loss, mse, p_predictions = test_model(model, test_loader, device)
test_loss, mse, p_predictions = test_model(model, test_loader, device)

print(f"Test Loss: {test_loss}")
print(f"Mean Squared Error: {mse}")


x = 30  
y = 30

NEW_SHAPE = (len(test_dataset), N_POINTS, N_POINTS)
test_preds_reshaped = p_predictions.reshape(NEW_SHAPE)
# print("pressure predicted all: ", test_preds_reshaped.shape)
pressure_pred = test_preds_reshaped[0:200, y, x]
pressure_calculated = test_output.reshape(NEW_SHAPE)[0:200, y, x]


# print("pressure predicted: ", pressure_pred)
# print("pressure calculated: ", pressure_calculated.shape)

# should separate labels
time = np.arange(pressure_calculated.shape[0])
plt.plot(time, pressure_pred, 'b', label='Predicted Pressure')  
plt.plot(time, pressure_calculated, 'g', label='Calculated Pressure')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.title('Predicted Pressure at Point ({}, {}) - Neural Net (MLP)'.format(x, y))
plt.legend()
plt.grid(True)
plt.savefig('Figures/pressure_over_time_mlp.png')   # mlp model


# get predicted pressure
p_predictions = (p_predictions * (test_max - test_min)) + test_min  # unnormalized output 
p_predictions = p_predictions.reshape(NEW_SHAPE)
p_predictions = np.array(p_predictions) # this is normalized
np.save('datasets/mlp/predicted_output.npy', p_predictions)
print(np.load('datasets/mlp/predicted_output.npy').astype(np.float32).shape)