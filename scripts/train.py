
import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from src.data.dataset import ASVspoofDataset
from src.training.trainer import Trainer

# SimpleCNN for test
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=80, stride=4)
        self.pool = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(16 * 3995, 2)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Démarrage sur: {device}")
    
    train_dataset = ASVspoofDataset(config, split='train')
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # --- CALCUL DES POIDS (BALANCING) ---
    # On calcule les poids par rapport au nombre de samples réels
    num_bonafide = (train_dataset.metadata['label'] == 'bonafide').sum()
    num_spoof = (train_dataset.metadata['label'] == 'spoof').sum()
    total = num_bonafide + num_spoof
    
    w_bonafide = total / (2 * num_bonafide)
    w_spoof = total / (2 * num_spoof)
    
    class_weights = torch.tensor([w_bonafide, w_spoof], dtype=torch.float).to(device)
    print(f" Weights appliqués: Bonafide={w_bonafide:.2f}, Spoof={w_spoof:.2f}")

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # AJOUT DU PARAMETRE WEIGHT
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    trainer = Trainer(model, train_loader, None, criterion, optimizer, config, device)
    
    for epoch in range(2):
        avg_loss, epoch_acc = trainer.train_epoch(epoch)
        trainer.save_checkpoint(epoch, epoch_acc)

if __name__ == "__main__":
    main()
