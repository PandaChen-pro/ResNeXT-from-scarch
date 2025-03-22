import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from timm.scheduler.cosine_lr import CosineLRScheduler
from dataset_loader import load_dataset
from trainer import train_model
import torchvision.models as models
import wandb
import os

os.environ['http_proxy'] = 'http://172.17.0.4:7532'
os.environ['https_proxy'] = 'http://172.17.0.4:7532'


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset_path = "/home/code/experiment/modal/cancer/images/train"
    val_dataset_path = "/home/code/experiment/modal/cancer/images/val"
    num_classes = 4
    num_epochs = 100
    warmup_ratio = 0.1
    wandb.login()
    wandb.init(project='ResNext50-using-pretrain-model-Cancer',name='ResNext50-using-pretrain-model-Cancer')

    train_loader, val_loader = load_dataset(train_dataset_path, val_dataset_path, batch_size=32, is_train_shuffle=True, is_val_shuffle=False)
    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # 加载上一个checkpoint继续训练，/home/code/experiment/modal/resnext/checkpoints/Cancer_Val_Epoch23_Acc79.69.pth
    # checkpoint_path = "/home/code/experiment/modal/resnext/checkpoints/Cancer_Val_Epoch23_Acc79.69.pth"
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # if 'optimizer_state_dict' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)  
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=total_steps,        
        lr_min=1e-5,                  
        warmup_t=warmup_steps,       
        warmup_lr_init=1e-6,       
        t_in_epochs=False,          
        cycle_limit=1,             
        warmup_prefix=True          
    )
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,num_epochs)