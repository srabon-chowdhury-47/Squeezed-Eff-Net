# -*- coding: utf-8 -*-
"""
Optimized Brain Tumor Classification with SqueezeNet+EfficientNet Hybrid
Enhanced with comprehensive visualization - IMPROVED VERSION
LOCAL MACHINE READY
"""

# =====================
# 1. Environment Setup
# =====================
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings('ignore')

# Set a random seed for reproducibility across all relevant libraries
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Set to False for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

# ====================
# 2. Configuration
# ====================
class Config:
    # MODIFIED: Use relative paths for local machine
    BASE_DIR = './brain_tumor_data'  # Change this to your actual data path
    TRAIN_DIR = os.path.join(BASE_DIR, 'Training')
    TEST_DIR = os.path.join(BASE_DIR, 'Testing')
    IMG_SIZE = (224, 224) # MODIFIED: Reduced for local machine compatibility
    BATCH_SIZE = 16 # MODIFIED: Reduced for local machine memory constraints
    NUM_CLASSES = 4
    RANDOM_STATE = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INITIAL_LR = 1e-4 # Slightly reduced initial LR
    WEIGHT_DECAY = 1e-4 # Adjusted weight decay
    NUM_EPOCHS = 20 # MODIFIED: Reduced epochs for faster local testing
    PATIENCE = 5 # MODIFIED: Reduced patience for local testing
    LABEL_MAP = {
        'notumor': 0,
        'glioma': 1,
        'meningioma': 2,
        'pituitary': 3
    }
    INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    # Optimizer settings
    EPSILON = 1e-8 # For AdamW
    BETA1 = 0.9
    BETA2 = 0.999


# =====================
# 3. Data Preparation
# =====================
def create_dataframe(directory):
    """Create dataframe from directory structure"""
    file_list = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist!")
        return pd.DataFrame(file_list, columns=['file_path', 'label'])
    
    for root, _, files in os.walk(directory):
        label = os.path.basename(root)
        if label not in Config.LABEL_MAP:
            continue
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_list.append((os.path.join(root, f), Config.LABEL_MAP[label]))
    
    if len(file_list) == 0:
        print(f"Warning: No images found in {directory}")
    
    return pd.DataFrame(file_list, columns=['file_path', 'label'])

class OptimizedBrainMRI(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.image_paths = dataframe['file_path'].values
        self.labels = dataframe['label'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)

            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a dummy image
            dummy_img = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, torch.tensor(0, dtype=torch.long)

# Improved Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize(Config.IMG_SIZE),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
    ], p=0.6), # Increased jitter strength and probability
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=(-20, 20)), # Increased rotation range
    transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)), # Added RandomResizedCrop
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # Added Gaussian Blur
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(Config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# For Test Time Augmentation (TTA)
tta_transforms = transforms.Compose([
    transforms.Resize(Config.IMG_SIZE),
    transforms.RandomHorizontalFlip(p=1.0), # Apply horizontal flip
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =====================
# 4. Visualization Functions
# =====================
def visualize_training_samples(loader, num_samples=12):
    try:
        batch = next(iter(loader))
        images, labels = batch
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()

        plt.figure(figsize=(15, 10))
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 4, i+1)
            image = images[i].transpose((1, 2, 0))
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
            label_name = Config.INV_LABEL_MAP[labels[i]]
            plt.imshow(image)
            plt.title(f"Label: {label_name}", fontsize=10)
            plt.axis('off')
        plt.tight_layout()
        plt.suptitle("Training Samples Visualization (Augmented)", y=1.02)
        plt.show()
    except Exception as e:
        print(f"Error in visualization: {e}")

# =====================
# 5. Hybrid Model Architecture (Simplified for local machine)
# =====================
class SqueezeFire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], dim=1)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_ratio, se_ratio=0.25):
        super().__init__()
        expanded_channels = in_channels * expansion_ratio
        self.use_residual = in_channels == out_channels and stride == 1

        # Expansion phase
        if expansion_ratio != 1:
            self.expand = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(expanded_channels)
            self.swish0 = nn.SiLU(inplace=True)

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride, padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.swish1 = nn.SiLU(inplace=True)

        # Squeeze-and-Excitation
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, squeeze_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_channels, expanded_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Output phase
        self.project = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        # Expansion
        if hasattr(self, 'expand'):
            x = self.swish0(self.bn0(self.expand(x)))

        # Depthwise convolution
        x = self.swish1(self.bn1(self.depthwise(x)))

        # SE
        x = x * self.se(x)

        # Project
        x = self.bn2(self.project(x))

        # Residual connection
        if self.use_residual:
            x = x + residual

        return x

class HybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # ====================
        # SqueezeNet Pathway (Simplified for local machine)
        # ====================
        self.squeeze_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # MODIFIED: Reduced channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            SqueezeFire(32, 8, 32, 32),  # MODIFIED: Reduced channels
            SqueezeFire(64, 8, 32, 32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            SqueezeFire(64, 16, 64, 64), # MODIFIED: Reduced channels
            SqueezeFire(128, 16, 64, 64),
        )

        # ======================
        # EfficientNet Pathway (Simplified for local machine)
        # ======================
        self.efficient_features = nn.Sequential(
            # Stem
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),  # MODIFIED: Reduced channels
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),

            # MBConv blocks (reduced for local machine)
            MBConvBlock(16, 16, 3, 1, expansion_ratio=1),
            MBConvBlock(16, 24, 3, 2, expansion_ratio=6),
            MBConvBlock(24, 24, 3, 1, expansion_ratio=6),
            MBConvBlock(24, 40, 5, 2, expansion_ratio=6),
        )

        # Feature pooling
        self.squeeze_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.efficient_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate feature dimensions dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])

            squeeze_out = self.squeeze_features(dummy)
            squeeze_out = self.squeeze_pool(squeeze_out)
            self.squeeze_dim = squeeze_out.view(-1).shape[0]

            efficient_out = self.efficient_features(dummy)
            efficient_out = self.efficient_pool(efficient_out)
            self.efficient_dim = efficient_out.view(-1).shape[0]

        total_features = self.squeeze_dim + self.efficient_dim

        # Simplified classifier for local machine
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), # MODIFIED: Reduced dropout
            nn.Linear(total_features, 512), # MODIFIED: Reduced size
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # SqueezeNet pathway
        x1 = self.squeeze_features(x)
        x1 = self.squeeze_pool(x1)
        x1 = x1.view(x1.size(0), -1)

        # EfficientNet pathway
        x2 = self.efficient_features(x)
        x2 = self.efficient_pool(x2)
        x2 = x2.view(x2.size(0), -1)

        # Feature fusion
        combined = torch.cat((x1, x2), dim=1)
        return self.classifier(combined)

# =====================
# 6. Training Engine (Modified for local machine compatibility)
# =====================
def train_model():
    config = Config()
    set_seed(config.RANDOM_STATE)

    # Check if data directory exists
    if not os.path.exists(config.BASE_DIR):
        print(f"Error: Data directory '{config.BASE_DIR}' not found!")
        print("Please ensure your data is in the correct directory structure:")
        print(f"{config.BASE_DIR}/")
        print(f"  └── Training/")
        print(f"      ├── glioma/")
        print(f"      ├── meningioma/")
        print(f"      ├── notumor/")
        print(f"      └── pituitary/")
        print(f"  └── Testing/")
        print(f"      ├── glioma/")
        print(f"      ├── meningioma/")
        print(f"      ├── notumor/")
        print(f"      └── pituitary/")
        return None, None

    # Model setup
    model = HybridModel(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.INITIAL_LR,
                            weight_decay=config.WEIGHT_DECAY,
                            betas=(config.BETA1, config.BETA2), eps=config.EPSILON)
    criterion = nn.CrossEntropyLoss()
    
    # MODIFIED: Use GradScaler only if CUDA is available
    scaler = torch.cuda.amp.GradScaler() if config.DEVICE == 'cuda' else None

    # Data loading
    train_df = create_dataframe(config.TRAIN_DIR)
    test_df = create_dataframe(config.TEST_DIR)
    
    if len(train_df) == 0:
        print("No training data found! Please check your data directory.")
        return None, None
        
    if len(test_df) == 0:
        print("No test data found! Using validation split only.")
        test_df = train_df.sample(frac=0.1, random_state=config.RANDOM_STATE)  # Fallback

    train_df, val_df = train_test_split(train_df, test_size=0.15,
                                        stratify=train_df['label'],
                                        random_state=config.RANDOM_STATE)

    # MODIFIED: Reduced num_workers for local machine compatibility
    train_loader = DataLoader(
        OptimizedBrainMRI(train_df, train_transform),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # MODIFIED: Reduced for local machine
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    # Visualize training samples
    visualize_training_samples(train_loader)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.INITIAL_LR * 5,
        steps_per_epoch=len(train_loader),
        epochs=config.NUM_EPOCHS,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    val_loader = DataLoader(
        OptimizedBrainMRI(val_df, test_transform),
        batch_size=config.BATCH_SIZE * 2,
        num_workers=2,  # MODIFIED: Reduced for local machine
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    # Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    patience_counter = 0

    print(f"Starting training on {config.DEVICE}")
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            # MODIFIED: Handle mixed precision only for CUDA
            if config.DEVICE == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 20 == 0:  # MODIFIED: Less frequent printing
                batch_acc = predicted.eq(labels).sum().item()/labels.size(0)
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {batch_acc:.2%}")

        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        epoch_train_acc = correct / total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                outputs = model(inputs)
                val_correct += outputs.argmax(1).eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {epoch_train_acc:.2%}")
        print(f"Val Acc: {val_acc:.2%}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}\n")

        # Early Stopping Logic
        if val_acc > best_acc:
            torch.save(model.state_dict(), 'best_model.pth')
            best_acc = val_acc
            patience_counter = 0
            print("---- Saved Best Model ----\n")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{config.PATIENCE}\n")
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs. Best Validation Accuracy: {best_acc:.2%}")
                break

    return model, history

# =====================
# 7. Enhanced Evaluation & Visualization (Modified for local machine)
# =====================
def evaluate_model(model, test_loader, use_tta=False, num_tta_augmentations=3):  # MODIFIED: Reduced TTA by default
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(Config.DEVICE)
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(inputs.cpu().numpy())

            if use_tta and Config.DEVICE == 'cuda':  # MODIFIED: TTA only on CUDA
                tta_batch_probs = []
                # Original prediction
                with torch.cuda.amp.autocast():
                    outputs_original = model(inputs)
                    probs_original = torch.nn.functional.softmax(outputs_original, dim=1)
                tta_batch_probs.append(probs_original.cpu().numpy())

                for _ in range(num_tta_augmentations - 1):
                    augmented_inputs_list = []
                    for img_tensor in inputs:
                        unnorm_img = img_tensor * torch.tensor([0.229, 0.224, 0.225]).to(Config.DEVICE).view(3, 1, 1) + \
                                     torch.tensor([0.485, 0.456, 0.406]).to(Config.DEVICE).view(3, 1, 1)
                        unnorm_img = transforms.ToPILImage()(unnorm_img.cpu())
                        augmented_inputs_list.append(tta_transforms(unnorm_img).to(Config.DEVICE))
                    augmented_inputs = torch.stack(augmented_inputs_list)

                    with torch.cuda.amp.autocast():
                        outputs_tta = model(augmented_inputs)
                        probs_tta = torch.nn.functional.softmax(outputs_tta, dim=1)
                    tta_batch_probs.append(probs_tta.cpu().numpy())

                # Average probabilities from TTA
                avg_probs = np.mean(tta_batch_probs, axis=0)
                all_probs.extend(avg_probs)
                all_preds.extend(np.argmax(avg_probs, axis=1))

            else:
                if Config.DEVICE == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                    
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(outputs.argmax(1).cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=Config.LABEL_MAP.keys()))

    return all_preds, all_labels, all_probs, all_images

def visualize_results(test_df, all_preds, all_labels, all_probs, all_images, history):
    if not all_preds or not all_labels:
        print("No predictions to visualize!")
        return
        
    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # Training History
    if history and history['train_loss']:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy', color='green')
        if history['val_acc']:
            plt.plot(history['val_acc'], label='Val Accuracy', color='orange')
        plt.title("Accuracy Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.LABEL_MAP.keys(),
                yticklabels=Config.LABEL_MAP.keys())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Sample Predictions
    if all_images:
        plt.figure(figsize=(15, 8))
        num_samples_to_show = min(8, len(all_images))
        sample_indices = np.random.choice(len(all_images), num_samples_to_show, replace=False)

        for i, idx in enumerate(sample_indices):
            plt.subplot(2, 4, i+1)
            image = all_images[idx].transpose((1, 2, 0))
            # Denormalize the image for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0, 1)

            true_label = Config.INV_LABEL_MAP[all_labels[idx]]
            pred_label = Config.INV_LABEL_MAP[all_preds[idx]]
            color = 'green' if true_label == pred_label else 'red'

            plt.imshow(image)
            plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.suptitle("Sample Predictions (Green=Correct, Red=Incorrect)", y=1.05)
        plt.show()

    # Print metrics
    print(f"\nOverall Test Accuracy: {test_accuracy:.2%}")
    for class_name, class_idx in Config.LABEL_MAP.items():
        mask = np.array(all_labels) == class_idx
        if sum(mask) > 0:
            class_acc = accuracy_score(np.array(all_labels)[mask], np.array(all_preds)[mask])
            print(f"{class_name:12} Accuracy: {class_acc:.2%}")
        else:
            print(f"{class_name:12} Accuracy: No samples found")

# =====================
# 8. Main Execution
# =====================
if __name__ == "__main__":
    config = Config()
    set_seed(config.RANDOM_STATE)

    print("=" * 60)
    print("Brain Tumor Classification - Local Machine Version")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Image Size: {config.IMG_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("=" * 60)

    # Initialize and train
    print("Initializing Training...")
    model, history = train_model()

    if model is None:
        print("Training failed! Please check the error messages above.")
        exit()

    # Load best weights if available
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded best model weights.")
    else:
        print("Using final model weights (best model not found).")

    # Prepare test data
    test_df = create_dataframe(config.TEST_DIR)
    if len(test_df) == 0:
        print("No test data found! Using validation data for evaluation.")
        # Create a validation dataframe for evaluation
        train_df = create_dataframe(config.TRAIN_DIR)
        if len(train_df) > 0:
            _, test_df = train_test_split(train_df, test_size=0.2, 
                                         stratify=train_df['label'],
                                         random_state=config.RANDOM_STATE)
        else:
            print("No data available for evaluation!")
            exit()

    test_loader = DataLoader(
        OptimizedBrainMRI(test_df, test_transform),
        batch_size=config.BATCH_SIZE * 2,
        num_workers=2  # MODIFIED: Reduced for local machine
    )

    # Evaluate without TTA (faster)
    print("\nRunning Final Evaluation (without TTA)...")
    all_preds_no_tta, all_labels_no_tta, all_probs_no_tta, all_images_no_tta = evaluate_model(
        model, test_loader, use_tta=False)
    visualize_results(test_df, all_preds_no_tta, all_labels_no_tta, all_probs_no_tta, all_images_no_tta, history)

    # Evaluate with TTA only if CUDA is available
    if config.DEVICE == 'cuda':
        print("\nRunning Final Evaluation (with TTA)...")
        all_preds_tta, all_labels_tta, all_probs_tta, all_images_tta = evaluate_model(
            model, test_loader, use_tta=True, num_tta_augmentations=3)  # MODIFIED: Reduced TTA
        print(f"\nTest Time Augmentation Results:")
        print(f"Overall Test Accuracy (with TTA): {np.mean(np.array(all_preds_tta) == np.array(all_labels_tta)):.2%}")
    else:
        print("\nSkipping TTA evaluation (CPU mode)")

    print("\n" + "=" * 60)
    print("Training and Evaluation Complete!")
    print("=" * 60)
