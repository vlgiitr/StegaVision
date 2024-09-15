import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_load import load_model, get_available_models
from dataset_load import train_loader, val_loader, test_loader, visual_loader
from torchvision.models import vgg16
from torch.nn.functional import mse_loss
from enum import Enum

num_epochs = 500
warmup_epochs = 75
beta = 1.0
checkpoint_path = "/working/"  # Update with your checkpoints path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def psnr(original, compressed):
    mse = nn.MSELoss()(original, compressed)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(path, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(path, 'model_best.pth.tar')
        torch.save(state, best_filepath)

def validate(loader, model, device):
    psnr_meter = AverageMeter('PSNR', ':6.2f')
    model.eval()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            input_S = images[images.shape[0] // 2:]
            input_C = images[:images.shape[0] // 2]

            output_C, output_S = model(input_S, input_C)

            psnr_val = psnr(input_C, output_C)
            psnr_meter.update(psnr_val.item(), images.size(0))

    print(f' * PSNR {psnr_meter.avg:.3f}')
    return psnr_meter.avg

def train_model(model_name):
    model = load_model(model_name)
    optimizer = optim.Adam(model.parameters())
    
    # Define schedulers
    warmup = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(epoch / warmup_epochs, 1.0))
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    best_psnr = 0
    S_mseloss = nn.MSELoss().to(device)
    C_mseloss = nn.MSELoss().to(device)

    epoch_total_losses = []
    epoch_cover_losses = []
    epoch_secret_losses = []
    learning_rates = []

    for epoch in range(num_epochs):
        model.train()
        loss_all, c_loss, s_loss = [], [], []
        
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            input_C = images[:images.shape[0] // 2]
            input_S = images[images.shape[0] // 2:]

            optimizer.zero_grad()
            output_Cprime, output_Sprime = model(input_S, input_C)

            ssLoss = S_mseloss(input_S, output_Sprime)
            ccLoss = C_mseloss(input_C, output_Cprime)
            loss = beta * ssLoss + ccLoss

            loss.backward()
            optimizer.step()

            loss_all.append(loss.item())
            c_loss.append(ccLoss.item())
            s_loss.append(ssLoss.item())

        scheduler.step()

        mean_total_loss = np.mean(loss_all)
        mean_cover_loss = np.mean(c_loss)
        mean_secret_loss = np.mean(s_loss)

        epoch_total_losses.append(mean_total_loss)
        epoch_cover_losses.append(mean_cover_loss)
        epoch_secret_losses.append(mean_secret_loss)
        learning_rates.append(scheduler.get_last_lr()[0])

        print(f"[epoch = {epoch+1}] loss: {mean_total_loss:.4f}, s_loss = {mean_secret_loss:.4f}, c_loss = {mean_cover_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            psnr_value = validate(val_loader, model, device)
            is_best = psnr_value > best_psnr
            best_psnr = max(psnr_value, best_psnr)

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, checkpoint_path, filename=f'{model_name}_checkpoint.pth.tar')

    return epoch_total_losses, epoch_cover_losses, epoch_secret_losses, learning_rates, best_psnr

def plot_training_results(model_name, total_losses, cover_losses, secret_losses, learning_rates):
    epochs = range(1, len(total_losses) + 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, total_losses, label='Total Loss')
    plt.plot(epochs, cover_losses, label='Cover Loss')
    plt.plot(epochs, secret_losses, label='Secret Loss')
    plt.title(f'Training Losses - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, learning_rates)
    plt.title(f'Learning Rate - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_results.png')
    plt.close()

def evaluate_model(model, test_loader, device):
    model.eval()
    mse_loss = nn.MSELoss()
    results = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            input_S = images[images.shape[0] // 2:]
            input_C = images[:images.shape[0] // 2]

            for j in range(input_S.shape[0]):
                output_C, output_S = model(input_S[j].unsqueeze(0), input_C[j].unsqueeze(0))

                secret_loss = mse_loss(output_S, input_S[j].unsqueeze(0)).item()
                cover_loss = mse_loss(output_C, input_C[j].unsqueeze(0)).item()

                psnr_secret = psnr(input_S[j].unsqueeze(0), output_S).item()
                psnr_cover = psnr(input_C[j].unsqueeze(0), output_C).item()

                results.append({
                    'Pair': i * input_S.shape[0] + j + 1,
                    'Secret Loss': secret_loss,
                    'Cover Loss': cover_loss,
                    'PSNR Secret': psnr_secret,
                    'PSNR Cover': psnr_cover,
                })

                if len(results) == 500:  
                    return results

    return results

def main():
    available_models = get_available_models()
    
    for model_name in available_models:
        print(f"Training model: {model_name}")
        total_losses, cover_losses, secret_losses, learning_rates, best_psnr = train_model(model_name)
        
        plot_training_results(model_name, total_losses, cover_losses, secret_losses, learning_rates)
        
        # Load the best model for evaluation
        best_model = load_model(model_name)
        best_model.load_state_dict(torch.load(os.path.join(checkpoint_path, f'{model_name}_checkpoint.pth.tar'))['state_dict'])
        
        print(f"Evaluating model: {model_name}")
        evaluation_results = evaluate_model(best_model, test_loader, device)
        
        # Save evaluation results
        np.save(f'{model_name}_evaluation_results.npy', evaluation_results)
        
        print(f"Best PSNR for {model_name}: {best_psnr:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
