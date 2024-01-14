import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.mlp_model import FluidNet 
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

 
if torch.cuda.is_available():
    print("CUDA is available")
else: 
    print("CUDA is not available")
 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100 
BATCH_SIZE = 32
N_POINTS = 51
num_features = N_POINTS**2



# Load the input & target data from .npy files - took about 87.1 min to generate 16000 datasets with 51 datapoints each
input_train = np.load('datasets/mlp/training_inputs.npy').astype(np.float32) # (3000, 51, 51)
target_train = np.load('datasets/mlp/training_outputs.npy').astype(np.float32)  # (3000, 51, 51)


input_valid = np.load('datasets/mlp/validation_inputs.npy').astype(np.float32) # (3000, 51, 51)
target_valid = np.load('datasets/mlp/validation_outputs.npy').astype(np.float32)  # (3000, 51, 51)

# print("input_data: ", input_data.shape)
# print("target data: ", target_data.shape)


# Reshape to (num_samples, num_features) format
input_train = input_train.reshape(input_train.shape[0], num_features)  # Flatten the last dimension (3000, 1, 2601)
target_train = target_train.reshape(target_train.shape[0], num_features)  # Flatten the last dimension (3000, 1, 2601)

input_valid = input_valid.reshape(input_valid.shape[0], num_features)  # Flatten the last dimension (3000, 1, 2601)
target_valid = target_valid.reshape(target_valid.shape[0], num_features)  # Flatten the last dimension (3000, 1, 2601)

print("target data shape: ", target_train.shape)

#################################### Preprocessing #######################################

# Min-Max Scaling (Normalization) 
input_min = np.min(input_train) 
input_max = np.max(input_train)
X_train = (input_train - input_min) / (input_max - input_min)

target_min = np.min(target_train)
target_max = np.max(target_train)
Y_train = (target_train - target_min) / (target_max - target_min)


input_min = np.min(input_valid) 
input_max = np.max(input_valid)
X_valid = (input_valid - input_min) / (input_max - input_min)

target_min = np.min(target_valid)
target_max = np.max(target_valid)
Y_valid = (target_valid - target_min) / (target_max - target_min)

# print([input_data[i, 0, 1601] for i in range(1, 502, 100)])
# print([target_data[j, 0, 1601] for j in range(1, 502, 100)])


# split the data into training and testing sets
# X_train, X_valid, Y_train, Y_valid = \
#         train_test_split(input_data, target_data, test_size=0.2, random_state=42, shuffle=True)


# Augmentation can be applied to the input data before training to increase the diversity of training samples
# Define a custom dataset for Pytorch
class FluidDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = torch.tensor(input_data)
        self.target_data = torch.tensor(target_data)
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.target_data[idx]

        return x, y # .squeeze(0)


# create the augmented datasets
train_dataset = FluidDataset(X_train, Y_train)
valid_dataset = FluidDataset(X_valid, Y_valid)

# print("Train dataset input data: ", train_dataset.input_data.shape)
# print("train dataset target data: ", train_dataset.target_data.shape)

# print("Valid dataset input data: ", valid_dataset.input_data.shape)
# print("Valid dataset target data: ", valid_dataset.target_data.shape)

# print("Test dataset input data: ", test_dataset.input_data.shape)

# Create a data loader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# print("TRAIN LOADER SIZE", len(train_loader))  # gives out number of batches
# print("VALID LOADER SIZE", len(valid_loader))  # gives out number of batches
# print("TEST LOADER SIZE", len(test_loader))  # gives out number of batches


# test_loader_shape = next(iter(test_loader))[0].shape
# print("Test Loader Shape:", test_loader_shape)

#################################### Training #######################################

# create an instance of model
model = FluidNet(features=num_features).to(device) 

# Define loss function & optimizer
loss_function = nn.MSELoss() # useful for regression
# loss_function = nn.CrossEntropyLoss() # useful for classification
# loss_function = nn.L1Loss()  # MAE loss

optimizer = optim.SGD(model.parameters(), lr=0.01)  # better for regression
# optimizer = optim.Adam(model.parameters(), lr=1e-3)  # better for classification

average_train_loss = []
average_valid_loss = []

# comment later ***
train_loader.requires_grad = True
valid_loader.requires_grad = True

# Training loop
for epoch in range(num_epochs):
    model.train() # set the model to training mode
    train_loss = []

    for batch_inputs, batch_targets in train_loader:
       
        # Move inputs and targets to CUDA device - if not, CPU
        batch_inputs = batch_inputs.to(device).float()
        batch_targets = batch_targets.to(device).float()

        # print("batch input size is:" , batch_inputs.shape) # with btc=64 => 1200 - two rounds loop
        # print("batch target size is:" , batch_targets.shape)
        
        # zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(batch_inputs) 
        # print("batch input size is: ", batch_inputs.unsqueeze(1).shape)
        # print("predicted output size is:" , outputs.shape)
        # print("batch target size is: ", batch_targets.shape)

        # Compute the loss
        loss = loss_function(outputs, batch_targets)

        # Backpropagation
        loss.backward(retain_graph=True) #retain_graph=True
        
        # Update the parameters
        optimizer.step()

        # accumulate the training loss
        train_loss.append(loss.item())

    # early stopping?
    if epoch % 50  == 0:
        loss = 0.0
    if epoch % 1000 == 0:
        torch.save(model.state_dict(), 'datasets/mlp/trained_model_mlp.pth')      

    # Calculate average training loss for the epoch
    average_train_loss.append(np.mean(train_loss))
    

    # validation loop
    model.eval() # set the model to evaluation mode
    valid_loss = []

    with torch.no_grad():
        for batch_inputs, batch_targets in valid_loader:

            # Move inputs and targets to CUDA device -if not, CPU
            batch_inputs = batch_inputs.to(device).float()
            batch_targets = batch_targets.to(device).float()
            # print("batch_target size is: ", batch_targets.shape) # with btc=64 => only 20  - one round loop

            # forward pass
            outputs = model(batch_inputs)

            # compute validation loss
            loss = loss_function(outputs, batch_targets)
            # total_loss = loss + lambda_reg * regularization_loss

            # accumulate the validation loss
            valid_loss.append(loss.item())

    average_valid_loss.append(np.mean(valid_loss))

    print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {np.mean(train_loss):.8f} | Valid Loss: {np.mean(valid_loss):.8f}")





# plt.style.use("dark_background")
plt.figure()

epochs = range(1, num_epochs + 1)

plt.plot(epochs, average_train_loss, 'b', label='Training Loss')
plt.plot(epochs, average_valid_loss, 'r', label='Validation Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
# Save the plot to an image file 
plt.savefig('Figures/mlp_loss_plot.png', dpi=300, bbox_inches='tight')
    
# Save the trained model
torch.save(model.state_dict(), 'datasets/mlp/trained_model_mlp.pth')       
 



