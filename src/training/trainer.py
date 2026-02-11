
import torch
import time
from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.best_accuracy = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # --- Calcul Metrics ---
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Affichage dynamique (Accuracy)
            if batch_idx % 5 == 0:
                current_acc = 100 * correct / total
                print(f" Epoch {epoch} | [{batch_idx}/{len(self.train_loader)}] "
                      f"| Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%")
                
        avg_loss = total_loss / len(self.train_loader)
        final_acc = 100 * correct / total
        duration = time.time() - start_time
        
        print(f"\n RESUME EPOCH {epoch}:")
        print(f" Loss Moyenne: {avg_loss:.4f} | Accuracy: {final_acc:.2f}% | Temps: {duration:.1f}s")
        return avg_loss, final_acc

    def save_checkpoint(self, epoch, accuracy):
        checkpoint_dir = Path(self.config['paths']['checkpoints'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        is_best = accuracy > self.best_accuracy
        if is_best:
            self.best_accuracy = accuracy
            print(f"🌟 NEW RECORD! Accuracy: {accuracy:.2f}% | Sauvegarde du meilleur modèle...")
            torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pt")
        
        # Sauvegarde normale de l'époque
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
        }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
