import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import PointNet2Classification
from dataset import PointCloudDataset

def train_model(data_dir, epochs=100, batch_size=16):
    data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    dataset = PointCloudDataset(data_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PointNet2Classification(num_classes=2)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = inputs.permute(0, 2, 1)  # [B, N, C] -> [B, C, N]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')

    torch.save(model.state_dict(), 'pointnet2_human_detection.pth')

if __name__ == '__main__':
    data_dir = '../data/processed'
    train_model(data_dir)
