import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.transforms import Resize, InterpolationMode
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

# ===============================
# Универсальный U-Net (базовая архитектура)
# ===============================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=5, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ===============================
# DATL: Differentiable Anatomical Topology Layer
# ===============================
class DATL(nn.Module):
    def __init__(self, num_classes=5, device='cuda'):
        super(DATL, self).__init__()
        self.num_classes = num_classes
        self.device = device
        
        anat_matrix = torch.tensor([
            [1.0, 0.5, 0.7, 0.8, 0.6],  # фон
            [0.5, 1.0, 0.9, 0.7, 0.1],  # кость
            [0.7, 0.9, 1.0, 0.8, 0.3],  # мышцы
            [0.8, 0.7, 0.8, 1.0, 0.4],  # жир
            [0.6, 0.1, 0.3, 0.4, 1.0],  # лёгкие
        ], dtype=torch.float32, device=device)

        row_sums = anat_matrix.sum(dim=1, keepdim=True)
        anat_matrix = anat_matrix / (row_sums + 1e-8)
        self.anat_matrix = nn.Parameter(anat_matrix, requires_grad=True)

    def forward(self, logits):
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(B, H * W, C)
        logits_diffused = torch.matmul(logits_flat, self.anat_matrix.t())
        logits_diffused = logits_diffused.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return logits_diffused

# ===============================
# Полная модель с DATL
# ===============================
class UNetWithDATL(nn.Module):
    def __init__(self, n_channels=1, n_classes=5, bilinear=True, device='cuda'):
        super(UNetWithDATL, self).__init__()
        self.unet = UNet(n_channels, n_classes, bilinear)
        self.datl = DATL(n_classes, device)
        
    def forward(self, x):
        logits = self.unet(x)
        logits = self.datl(logits)
        return logits

# ===============================
# Обертка для инференса
# ===============================
class DATSPredictor:
    def __init__(self, weights_path="/app/weights/dats_best_model.pth", image_size=512):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size
        
        # Инициализация и загрузка весов
        self.model = UNetWithDATL(n_channels=1, n_classes=5, device=self.device).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.resize_img = Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR)

    def predict(self, image):
        """
        Принимает изображение (numpy array). 
        Возвращает маску (numpy array) с классами 0-4 в оригинальном размере.
        """
        # Перевод в grayscale, если изображение цветное
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        orig_h, orig_w = image.shape
        
        # Нормализация в диапазон [0, 1]
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Перевод в тензор формата [1, 1, H, W]
        tensor_img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        tensor_img = self.resize_img(tensor_img)
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(tensor_img)
            preds = outputs.argmax(dim=1).cpu().numpy()  # [1, H, W]
            
        mask = preds[0]  # [H, W] (512x512)
        
        # Возвращаем маску к оригинальному размеру среза
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        return mask_resized