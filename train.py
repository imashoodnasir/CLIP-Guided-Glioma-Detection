
import torch
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from models.ech_vit import ECHViT
import torch.nn as nn

def dice_loss(pred, target):
    smooth = 1e-5
    pred = torch.softmax(pred, dim=1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images, ["whole tumor","tumor core","enhancing tumor"])
        loss = dice_loss(outputs, masks) + ce(outputs, masks)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECHViT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataset = BraTSDataset([])  # Fill with case paths
    loader = DataLoader(dataset, batch_size=1)
    train(model, loader, optimizer, device)
