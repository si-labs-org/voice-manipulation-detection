
import torch
import yaml
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from src.data.dataset import ASVspoofDataset
# Importation d'un modèle simple pour le test (doit correspondre à celui du train)
from scripts.train import SimpleCNN 

def evaluate():
    # 1. Config & Device
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data (Split 'dev' pour l'évaluation)
    print(" Chargement des données de validation (Dev split)...")
    val_dataset = ASVspoofDataset(config, split='dev')
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # 3. Charger le Meilleur Modèle
    model = SimpleCNN().to(device)
    model_path = os.path.join(config['paths']['checkpoints'], "best_model.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    print(" Évaluation en cours...")
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # 4. Rapport & Confusion Matrix
    print("\n RAPPORT D'ÉVALUATION :")
    print(classification_report(all_labels, all_preds, target_names=['Bonafide', 'Spoof']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bonafide', 'Spoof'], yticklabels=['Bonafide', 'Spoof'])
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.title('Matrice de Confusion')
    
    # Sauvegarde du graphe pour le rapport PFE
    save_path = os.path.join(config['paths']['experiments_root'], "figures", "confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\n Matrice de confusion sauvegardée dans: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate()
