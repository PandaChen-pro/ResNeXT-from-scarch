import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import ResNext50
from dataset_loader import load_dataset
import wandb
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm
import torchvision.models as models
import torch.optim as optim
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100):
    best_val_acc = 0
    best_model_path = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"  ):
            images = images.to(device)
            labels = labels.to(device)
            


            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step_update(epoch * len(train_loader) + i)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 30 == 0:
                batch_acc = 100 * correct / total
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "batch": epoch * len(train_loader) + i,
                    "train_loss": loss.item(),
                    "train_acc": batch_acc,
                    "learning_rate": current_lr
                })
                tqdm.write(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}, learning_rate: {current_lr:.6f}")

        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        if epoch % 2 == 0:
            print('starting validation......')
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            wandb.log({
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            wandb.log({
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            os.makedirs("./checkpoints", exist_ok=True)
            if val_acc > best_val_acc:
                if best_model_path is not None and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                        print(f'删除旧的最佳模型: {best_model_path}')
                    except Exception as e:
                        print(f'删除旧模型失败: {e}')
                best_val_acc = val_acc
                new_model_path = f"./checkpoints/Cancer_Val_Epoch{epoch+1}_Acc{val_acc:.2f}.pth"
                torch.save(model.state_dict(), new_model_path)
                print(f'save model to {new_model_path}')
                best_model_path = new_model_path
    wandb.finish()
    return best_val_acc

def evaluate_model(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    return val_loss, val_acc


