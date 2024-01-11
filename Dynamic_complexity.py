import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
tracks = pd.read_csv("/home/fanzeyu/数据集/inD-dataset-v1.0/data/00_tracks.csv")

# Normalize the features
scaler = MinMaxScaler()
tracks_normalized = scaler.fit_transform(tracks[['xCenter', 'yCenter', 'heading', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']])

# Sort and group the data by frame
tracks_sorted = tracks.sort_values(by='frame')
tracks_grouped = tracks_sorted.groupby('frame')

# Initialize an empty list to store the sequences
sequences = []

# For each group, concatenate the rows into a sequence
for name, group in tracks_grouped:
    sequence = group[['xCenter', 'yCenter', 'heading', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']].values.flatten()  # Only use the normalized features
    sequences.append(sequence)

# Determine the maximum sequence length
max_length = max(len(sequence) for sequence in sequences)

# Pad the sequences to the maximum length
sequences_padded = [np.pad(sequence, (0, max_length - len(sequence))) for sequence in sequences]

# Convert the list of sequences into a tensor
sequences_tensor = torch.tensor(sequences_padded, dtype=torch.float32)

# Define the encoder
encoder = nn.Sequential(
    nn.Linear(sequences_tensor.shape[1], 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 1),  # The output of the encoder is the scene complexity
    nn.Sigmoid()  
)

# Define the decoder
decoder = nn.Sequential(
    nn.Linear(1, 50),
    nn.ReLU(),
    nn.Linear(50, 100),
    nn.ReLU(),
    nn.Linear(100, sequences_tensor.shape[1])
)

# Define the optimizer and loss function
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.002)
loss_fn = nn.MSELoss()

# Initialize a list to store the losses for each epoch
losses = []

# Train the model
for epoch in range(5000):
    # Forward pass
    hidden = encoder(sequences_tensor)
    output = decoder(hidden)
    
    # Compute loss
    loss = loss_fn(output, sequences_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record the loss for this epoch
    losses.append(loss.item())
    
    # Print loss for every epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 5000, loss.item()))

# After training, plot the loss for each epoch
plt.figure(figsize=(10,5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')

# Calculate scene complexity
scene_complexity = encoder(sequences_tensor)
print(scene_complexity)

# Assume that `complexity` is the output of your model
complexity = scene_complexity.detach().numpy()

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the complexity
ax.plot(complexity)

# Set the labels
ax.set_xlabel('Frame')
ax.set_ylabel('Complexity')

# Show the plot
plt.show()


