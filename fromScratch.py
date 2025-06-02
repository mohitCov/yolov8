 import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2
import numpy as np
import math
import os
import random
from typing import List, Tuple
from PIL import Image

# ---------------------- Attention Modules ----------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# ---------------------- Basic Building Blocks ----------------------
class ConvBnAct(nn.Module):
    """Conv + BatchNorm + Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard bottleneck block with residual connection"""
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBnAct(in_channels, hidden_channels, 1)
        self.conv2 = ConvBnAct(hidden_channels, out_channels, 3, padding=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))

class C2f(nn.Module):
    """C2f block with CBAM attention"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, expansion=0.5):
        super().__init__()
        self.cv1 = ConvBnAct(in_channels, int(out_channels * 0.5), 1)
        self.cv2 = ConvBnAct(in_channels, int(out_channels * 0.5), 1)
        self.m = nn.Sequential(*[Bottleneck(int(out_channels * 0.5), int(out_channels * 0.5), shortcut, expansion) for _ in range(n)])
        self.cbam = CBAM(out_channels)  # Adding CBAM attention
        self.cv3 = ConvBnAct(out_channels, out_channels, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.m(self.cv2(x))
        x = self.cv3(torch.cat((y1, y2), dim=1))
        return self.cbam(x)  # Apply CBAM attention

# ---------------------- BiFPN Implementation ----------------------
class BiFPN_Block(nn.Module):
    """Bidirectional Feature Pyramid Network block"""
    def __init__(self, channels_list=[256, 512, 1024], num_levels=3, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.num_levels = num_levels
        
        # Feature fusion weights
        self.w1 = nn.Parameter(torch.ones(2, num_levels-1))  # Changed size for top-down
        self.w2 = nn.Parameter(torch.ones(3, num_levels-1))  # Changed size for bottom-up
        
        # Channel adjustment layers
        self.adjust_p3 = ConvBnAct(channels_list[0], channels_list[0], 1)
        self.adjust_p4 = ConvBnAct(channels_list[1], channels_list[0], 1)
        self.adjust_p5 = ConvBnAct(channels_list[2], channels_list[0], 1)
        
        # Top-down and bottom-up paths
        self.top_down_blocks = nn.ModuleList([
            ConvBnAct(channels_list[0], channels_list[0], 3, padding=1) for _ in range(num_levels - 1)
        ])
        self.bottom_up_blocks = nn.ModuleList([
            ConvBnAct(channels_list[0], channels_list[0], 3, padding=1) for _ in range(num_levels - 1)
        ])

    def forward(self, features: List[torch.Tensor]):
        # Adjust channels
        p3, p4, p5 = features
        p3 = self.adjust_p3(p3)
        p4 = self.adjust_p4(p4)
        p5 = self.adjust_p5(p5)
        features = [p3, p4, p5]
        
        # Top-down path
        w1 = F.relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        
        # Bottom-up path
        w2 = F.relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        
        # Top-down pathway
        td_features = [features[-1]]  # Start with P5
        for i in range(len(features)-2, -1, -1):  # P4 to P3
            td_feature = F.interpolate(td_features[-1], size=features[i].shape[-2:], mode='nearest')
            td_feature = w1[0, len(features)-i-2] * features[i] + w1[1, len(features)-i-2] * td_feature
            td_feature = self.top_down_blocks[len(features)-i-2](td_feature)
            td_features.append(td_feature)
        td_features = td_features[::-1]  # Reverse to get [P3, P4, P5]
        
        # Bottom-up pathway
        bu_features = [td_features[0]]  # Start with P3
        for i in range(1, len(features)):  # P4 to P5
            bu_feature = F.max_pool2d(bu_features[-1], kernel_size=2, stride=2)
            bu_feature = w2[0, i-1] * td_features[i] + w2[1, i-1] * bu_feature
            if i < len(features) - 1:
                bu_feature = bu_feature + w2[2, i-1] * features[i]
            bu_feature = self.bottom_up_blocks[i-1](bu_feature)
            bu_features.append(bu_feature)
        
        return bu_features

# ---------------------- Enhanced Detection Head ----------------------
class CustomDetectionHead(nn.Module):
    """Enhanced detection head with separate branches for classification and regression"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared feature extraction
        self.stem = nn.Sequential(
            ConvBnAct(in_channels, in_channels, 3, padding=1),
            CBAM(in_channels),  # Adding CBAM to head
            ConvBnAct(in_channels, in_channels, 3, padding=1)
        )
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            ConvBnAct(in_channels, in_channels, 3, padding=1),
            ConvBnAct(in_channels, in_channels, 3, padding=1)
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        
        # Regression branch
        self.reg_conv = nn.Sequential(
            ConvBnAct(in_channels, in_channels, 3, padding=1),
            ConvBnAct(in_channels, in_channels, 3, padding=1)
        )
        self.reg_pred = nn.Conv2d(in_channels, 4, 1)  # x, y, w, h
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        
        # Classification branch
        cls_feat = self.cls_conv(x)
        cls_out = self.cls_pred(cls_feat)
        
        # Regression branch
        reg_feat = self.reg_conv(x)
        reg_out = self.reg_pred(reg_feat)
        
        # Concatenate regression and classification outputs
        # [batch_size, 4+num_classes, H, W]
        return torch.cat([reg_out, cls_out], dim=1)

# ---------------------- Complete YOLOv8 Model ----------------------
class YOLOv8_Custom(nn.Module):
    """Complete YOLOv8 model with CBAM, BiFPN, and custom heads"""
    def __init__(self, num_classes=80, channels=3, depths=[64, 128, 256, 512, 1024]):
        super().__init__()
        # ---------------------- Backbone ----------------------
        self.stem = ConvBnAct(channels, depths[0], 3, 2, 1)  # /2
        
        # Dark stages
        self.dark2 = nn.Sequential(
            ConvBnAct(depths[0], depths[1], 3, 2, 1),  # /4
            C2f(depths[1], depths[1], n=1)
        )
        self.dark3 = nn.Sequential(
            ConvBnAct(depths[1], depths[2], 3, 2, 1),  # /8
            C2f(depths[2], depths[2], n=2)
        )
        self.dark4 = nn.Sequential(
            ConvBnAct(depths[2], depths[3], 3, 2, 1),  # /16
            C2f(depths[3], depths[3], n=2)
        )
        self.dark5 = nn.Sequential(
            ConvBnAct(depths[3], depths[4], 3, 2, 1),  # /32
            C2f(depths[4], depths[4], n=1),
            nn.Sequential(
                ConvBnAct(depths[4], depths[4], 1),
                ConvBnAct(depths[4], depths[4], 3, padding=1),
                ConvBnAct(depths[4], depths[4], 1)
            )
        )
        
        # ---------------------- BiFPN Neck ----------------------
        self.bifpn = BiFPN_Block(channels_list=[depths[2], depths[3], depths[4]], num_levels=3)
        
        # ---------------------- Detection Heads ----------------------
        self.heads = nn.ModuleList([
            CustomDetectionHead(depths[2], num_classes),  # P3
            CustomDetectionHead(depths[2], num_classes),  # P4 (now using same channel size)
            CustomDetectionHead(depths[2], num_classes)   # P5 (now using same channel size)
        ])
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone
        x1 = self.stem(x)       # /2
        x2 = self.dark2(x1)     # /4
        x3 = self.dark3(x2)     # /8  (P3)
        x4 = self.dark4(x3)     # /16 (P4)
        x5 = self.dark5(x4)     # /32 (P5)
        
        # BiFPN
        features = self.bifpn([x3, x4, x5])
        p3, p4, p5 = features
        
        # Heads
        outputs = []
        for head, feature in zip(self.heads, [p3, p4, p5]):
            outputs.append(head(feature))
        
        return outputs

# ---------------------- Dataset & Training Utilities ----------------------
def custom_collate_fn(batch):
    """Custom collate function to handle variable number of objects per image"""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    # Stack images (they should all be the same size)
    images = torch.stack(images, 0)
    
    # For targets, we'll return a list of tensors
    # Each tensor can have a different first dimension (number of objects)
    # but they all have 5 columns (class_id, x, y, w, h)
    return images, targets

class YOLODataset(Dataset):
    """Custom dataset loader with advanced augmentation"""
    def __init__(self, img_dir, label_dir, img_size=640, augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        
        # Color jitter - brightness, contrast, saturation, hue
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load labels (class_id, x_center, y_center, width, height)
        labels = []
        if os.path.exists(label_path):
            try:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, xc, yc, w, h = map(float, parts)
                        labels.append([class_id, xc, yc, w, h])
            except Exception as e:
                print(f"Error reading label file {label_path}: {str(e)}")
                labels = []
        
        # Convert to numpy array right after loading
        labels = np.array(labels, dtype=np.float32)
        
        # Data augmentation (only if we have labels)
        if self.augment and len(labels) > 0:
            try:
            img, labels = self.augment_image(img, labels)
            except Exception as e:
                print(f"Error in augmentation for {img_path}: {str(e)}")
        
        # Resize and normalize
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0  # Normalize to 0-1
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW format
        
        # Convert labels to tensor
        if len(labels) > 0:
            labels = torch.from_numpy(labels).float()
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)
            
        return img, labels
    
    def augment_image(self, img, labels):
        """Advanced augmentation pipeline"""
        # Ensure labels is a numpy array
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, dtype=np.float32)
        
        if len(labels) == 0:
            return img, labels
            
        # Random horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            if len(labels):
                # Update x-coordinate (horizontal flip)
                labels = labels.copy()  # Create a copy to avoid modifying original
            labels[:, 1] = 1.0 - labels[:, 1]  # Adjust x_center
            
        # Random vertical flip
        if random.random() < 0.2:
            img = img[::-1, :, :]
            if len(labels):
                # Update y-coordinate (vertical flip)
                labels = labels.copy()  # Create a copy to avoid modifying original
            labels[:, 2] = 1.0 - labels[:, 2]  # Adjust y_center
            
        # Convert image to PIL for color jitter
        img_pil = Image.fromarray(img.astype('uint8'))
        img = np.array(self.color_jitter(img_pil))
        
        # Random rotation (-15 to +15 degrees)
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            # Note: Would need to adjust labels for rotation (omitted for brevity)
            
        return img, labels

def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.01):
    """Complete training loop with validation"""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = compute_yolo_loss(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                val_loss += compute_yolo_loss(outputs, targets).item()
        
        # Adjust learning rate
        scheduler.step()
        
        # Print stats
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
    
    return model

def compute_yolo_loss(outputs, targets):
    """Compute YOLO loss for classification and regression"""
    total_loss = 0
    batch_size = outputs[0].shape[0]
    
    # Process each scale output
    for output in outputs:
        # Split predictions
        pred_reg = output[:, :4]  # [batch, 4, H, W]
        pred_cls = output[:, 4:]  # [batch, num_classes, H, W]
        
        # Initialize batch loss
        batch_loss = 0
        
        # Process each image in the batch
        for idx in range(batch_size):
            # Get targets for this image
            img_targets = targets[idx]  # [num_objects, 5]
            
            if len(img_targets) == 0:
                # If no objects, only compute background classification loss
                cls_loss = F.binary_cross_entropy_with_logits(
                    pred_cls[idx],
                    torch.zeros_like(pred_cls[idx]),
                    reduction='sum'
                )
                batch_loss += cls_loss * 0.1  # Lower weight for background
                continue
            
            # Get prediction grid size
            grid_h, grid_w = pred_cls.shape[-2:]
            
            # Scale targets to grid size
            scaled_targets = img_targets.clone()
            scaled_targets[:, 1:5] *= torch.tensor([grid_w, grid_h, grid_w, grid_h],
                                                 device=img_targets.device)
            
            # Convert target coordinates to grid cell coordinates
            grid_x = scaled_targets[:, 1].long().clamp(0, grid_w-1)
            grid_y = scaled_targets[:, 2].long().clamp(0, grid_h-1)
            
            # Classification loss for positive samples
            target_cls = torch.zeros_like(pred_cls[idx])
            target_cls[img_targets[:, 0].long(), grid_y, grid_x] = 1
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_cls[idx],
                target_cls,
                reduction='sum'
            )
            
            # Regression loss for positive samples
            pred_box = pred_reg[idx, :, grid_y, grid_x].t()  # [num_objects, 4]
            target_box = scaled_targets[:, 1:5]
            
            # IoU loss
            iou_loss = (1.0 - box_iou_matrix(pred_box.sigmoid(), target_box)).mean()
            
            # Combine losses
            batch_loss += cls_loss + iou_loss
            
        # Average batch loss
        total_loss += batch_loss / batch_size
    
    return total_loss / len(outputs)

def box_iou_matrix(box1, box2):
    """Calculate IoU between all pairs of boxes"""
    # Convert to x1, y1, x2, y2 format
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # Intersection area
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area.unsqueeze(1) + b2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    return iou

# ---------------------- Example Usage ----------------------
if __name__ == "__main__":
    # Initialize model
    model = YOLOv8_Custom(num_classes=80)
    
    # Example dataset paths
    train_dataset = YOLODataset("data/train/images", "data/train/labels", augment=True)
    val_dataset = YOLODataset("data/val/images", "data/val/labels", augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(model, train_loader, val_loader, device, epochs=100)
    
    # Save final model
    torch.save(trained_model.state_dict(), "yolov8_custom_final.pth")