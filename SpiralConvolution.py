import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional, Union, Callable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import warnings
warnings.filterwarnings('ignore')

# ==================== SPIRAL COORDINATE SYSTEM ====================
class SpiralCoordinateSystem:
    """
    Implements spiral coordinate transformations.
    Converts Cartesian (x,y) coordinates to spiral (r,θ) coordinates.
    """
    
    def __init__(self, 
                 spiral_radius: float = 0.5,
                 spiral_turns: int = 3,
                 clockwise: bool = False):
        """
        Args:
            spiral_radius: Maximum radius of the spiral
            spiral_turns: Number of complete turns in the spiral
            clockwise: Direction of spiral rotation
        """
        self.spiral_radius = spiral_radius
        self.spiral_turns = spiral_turns
        self.clockwise = clockwise
        self.direction = -1 if clockwise else 1
    
    def cartesian_to_spiral(self, 
                           x: torch.Tensor, 
                           y: torch.Tensor,
                           center_x: float = 0.0,
                           center_y: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to spiral coordinates.
        
        Args:
            x, y: Cartesian coordinates
            center_x, center_y: Center of the spiral
            
        Returns:
            spiral_r: Radial coordinate in spiral space
            spiral_theta: Angular coordinate in spiral space
        """
        # Center coordinates
        x_centered = x - center_x
        y_centered = y - center_y
        
        # Convert to polar coordinates
        polar_r = torch.sqrt(x_centered ** 2 + y_centered ** 2)
        polar_theta = torch.atan2(y_centered, x_centered)
        
        # Transform to spiral coordinates
        # In a spiral, radius increases with angle: r = a * θ
        spiral_theta = polar_theta
        spiral_r = self.spiral_radius * (polar_r / self.spiral_radius) ** (1 / (1 + 0.1 * polar_theta.abs()))
        
        # Apply spiral transformation
        spiral_r = spiral_r * (1 + 0.1 * self.direction * spiral_theta)
        
        return spiral_r, spiral_theta
    
    def spiral_to_cartesian(self,
                           spiral_r: torch.Tensor,
                           spiral_theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert spiral coordinates back to Cartesian coordinates.
        """
        # Inverse transformation
        polar_r = spiral_r / (1 + 0.1 * self.direction * spiral_theta)
        polar_theta = spiral_theta
        
        # Convert polar to Cartesian
        x = polar_r * torch.cos(polar_theta)
        y = polar_r * torch.sin(polar_theta)
        
        return x, y
    
    def create_spiral_grid(self, 
                          size: int,
                          normalize: bool = True) -> torch.Tensor:
        """
        Create a grid of points following a spiral pattern.
        
        Args:
            size: Grid size (size x size)
            normalize: Whether to normalize coordinates to [-1, 1]
            
        Returns:
            Grid tensor of shape (size*size, 2)
        """
        # Create base grid
        if normalize:
            grid = torch.linspace(-1, 1, size)
        else:
            grid = torch.linspace(0, size-1, size)
        
        x, y = torch.meshgrid(grid, grid, indexing='ij')
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # Convert to spiral coordinates
        spiral_r, spiral_theta = self.cartesian_to_spiral(x_flat, y_flat)
        
        # Stack coordinates
        spiral_grid = torch.stack([spiral_r, spiral_theta], dim=1)
        
        return spiral_grid

# ==================== SPIRAL SAMPLING ====================
class SpiralSampler:
    """
    Samples points along spiral paths for convolution.
    """
    
    def __init__(self,
                 kernel_size: int = 3,
                 num_spirals: int = 8,
                 spiral_type: str = 'archimedean'):
        """
        Args:
            kernel_size: Size of convolution kernel
            num_spirals: Number of spiral arms
            spiral_type: Type of spiral ('archimedean', 'logarithmic', 'golden')
        """
        self.kernel_size = kernel_size
        self.num_spirals = num_spirals
        self.spiral_type = spiral_type
        
        # Generate sampling positions
        self.sampling_positions = self._generate_sampling_positions()
    
    def _archimedean_spiral(self, t: torch.Tensor, a: float = 1.0, b: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Archimedean spiral: r = a + bθ"""
        r = a + b * t
        return r * torch.cos(t), r * torch.sin(t)
    
    def _logarithmic_spiral(self, t: torch.Tensor, a: float = 0.1, b: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Logarithmic spiral: r = a * exp(bθ)"""
        r = a * torch.exp(b * t)
        return r * torch.cos(t), r * torch.sin(t)
    
    def _golden_spiral(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Golden spiral based on Fibonacci sequence"""
        phi = (1 + math.sqrt(5)) / 2
        r = phi ** (2 * t / math.pi)
        return r * torch.cos(t), r * torch.sin(t)
    
    def _generate_sampling_positions(self) -> torch.Tensor:
        """
        Generate sampling positions for spiral convolution.
        
        Returns:
            Sampling positions tensor of shape (kernel_size, 2)
        """
        positions = []
        
        # Generate angles for sampling
        angles = torch.linspace(0, 2 * math.pi * 2, self.kernel_size)
        
        for i, angle in enumerate(angles):
            # Choose spiral type
            if self.spiral_type == 'archimedean':
                x, y = self._archimedean_spiral(angle, a=0.1, b=0.2)
            elif self.spiral_type == 'logarithmic':
                x, y = self._logarithmic_spiral(angle, a=0.05, b=0.25)
            elif self.spiral_type == 'golden':
                x, y = self._golden_spiral(angle)
            else:
                raise ValueError(f"Unknown spiral type: {self.spiral_type}")
            
            positions.append([x.item(), y.item()])
        
        positions = torch.tensor(positions)
        
        # Normalize positions
        positions = positions / positions.abs().max()
        
        return positions
    
    def get_sampling_pattern(self) -> torch.Tensor:
        """
        Get the sampling pattern as integer indices.
        
        Returns:
            Indices tensor of shape (kernel_size, 2)
        """
        # Convert normalized positions to integer indices
        indices = (self.sampling_positions * (self.kernel_size - 1) / 2).round().long()
        
        # Center indices
        indices = indices + self.kernel_size // 2
        
        return indices
    
    def visualize_sampling_pattern(self, ax=None):
        """Visualize the spiral sampling pattern."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        positions = self.sampling_positions
        
        # Plot spiral points
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c=range(len(positions)), cmap='viridis', 
                  s=100, alpha=0.7)
        
        # Connect points in order
        ax.plot(positions[:, 0], positions[:, 1], 
               'b-', alpha=0.3, linewidth=1)
        
        # Add center point
        ax.scatter([0], [0], c='red', s=200, marker='x', linewidth=2)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(f'Spiral Sampling Pattern ({self.spiral_type})')
        ax.grid(True, alpha=0.3)
        
        return ax

# ==================== SPIRAL CONVOLUTION LAYER ====================
class SpiralConv2d(nn.Module):
    """
    Spiral Convolution layer for 3-channel images.
    Applies convolution along spiral paths rather than rectangular grids.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 num_spirals: int = 8,
                 spiral_type: str = 'archimedean',
                 learnable_spiral: bool = True):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            spiral_type: Type of spiral pattern
            learnable_spiral: Whether to make spiral pattern learnable
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_spirals = num_spirals
        self.spiral_type = spiral_type
        self.learnable_spiral = learnable_spiral
        
        # Initialize spiral sampler
        self.spiral_sampler = SpiralSampler(
            kernel_size=kernel_size,
            num_spirals=num_spirals,
            spiral_type=spiral_type
        )
        
        # Get spiral sampling positions
        self.spiral_positions = self._initialize_spiral_positions()
        
        # Create convolution weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, 
                       kernel_size, kernel_size) * 0.1
        )
        
        # Create spiral-specific weights
        self.spiral_weights = nn.Parameter(
            torch.ones(num_spirals, kernel_size) * 0.1
        ) if learnable_spiral else None
        
        # Bias term
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self._init_parameters()
    
    def _initialize_spiral_positions(self) -> torch.Tensor:
        """Initialize spiral sampling positions."""
        positions = self.spiral_sampler.get_sampling_pattern()
        
        if self.learnable_spiral:
            # Make positions learnable with small random perturbation
            positions = positions.float()
            positions = positions + torch.randn_like(positions) * 0.1
            return nn.Parameter(positions)
        else:
            # Register as buffer (non-learnable)
            self.register_buffer('spiral_positions_buffer', positions)
            return self.spiral_positions_buffer
    
    def _init_parameters(self):
        """Initialize layer parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _apply_spiral_mask(self, weight: torch.Tensor) -> torch.Tensor:
        """Apply spiral pattern mask to convolution weights."""
        batch_size, channels, h, w = weight.shape
        
        # Create spiral mask
        spiral_mask = torch.zeros(h, w, device=weight.device)
        
        if self.learnable_spiral:
            positions = self.spiral_positions
        else:
            positions = self.spiral_positions_buffer
        
        # Convert positions to valid indices
        positions = positions.long()
        positions = torch.clamp(positions, 0, h - 1)
        
        # Set spiral positions to 1
        for i in range(len(positions)):
            x, y = positions[i]
            spiral_mask[x, y] = 1.0
        
        # Expand mask to match weight dimensions
        spiral_mask = spiral_mask.unsqueeze(0).unsqueeze(0)
        spiral_mask = spiral_mask.expand(batch_size, channels, -1, -1)
        
        # Apply mask
        masked_weight = weight * spiral_mask
        
        return masked_weight
    
    def _spiral_interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Interpolate input features along spiral paths.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Interpolated features along spiral paths
        """
        batch_size, channels, height, width = x.shape
        
        # Get spiral positions
        if self.learnable_spiral:
            positions = self.spiral_positions
        else:
            positions = self.spiral_positions_buffer
        
        # Normalize positions to [-1, 1] for grid_sample
        pos_normalized = positions.float() / torch.tensor([height-1, width-1], 
                                                        device=x.device).float() * 2 - 1
        
        # Create sampling grid
        grid = pos_normalized.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, 2)
        grid = grid.expand(batch_size, channels, -1, -1)  # (batch, channels, kernel_size, 2)
        
        # Permute for grid_sample
        grid = grid.permute(0, 3, 1, 2)  # (batch, 2, channels, kernel_size)
        grid = grid.reshape(batch_size, 2, channels * self.kernel_size)
        
        # Reshape input for sampling
        x_reshaped = x.reshape(batch_size, channels, height * width)
        x_reshaped = x_reshaped.unsqueeze(1)  # (batch, 1, channels, height*width)
        
        # Use grid_sample for interpolation
        # Note: We need to handle this differently since grid_sample expects 4D input
        # For simplicity, we'll use manual sampling in this implementation
        sampled_features = []
        
        for b in range(batch_size):
            batch_features = []
            for c in range(channels):
                channel_features = []
                for k in range(self.kernel_size):
                    pos = positions[k]
                    # Clamp positions to valid range
                    pos_x = max(0, min(int(pos[0]), height - 1))
                    pos_y = max(0, min(int(pos[1]), width - 1))
                    
                    # Sample feature
                    feature = x[b, c, pos_x, pos_y]
                    channel_features.append(feature)
                
                batch_features.append(torch.stack(channel_features))
            
            sampled_features.append(torch.stack(batch_features))
        
        sampled = torch.stack(sampled_features)  # (batch, channels, kernel_size)
        
        return sampled
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spiral convolution.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Convolved tensor of shape (batch, out_channels, height_out, width_out)
        """
        batch_size, in_channels, height, width = x.shape
        
        # Apply spiral-aware convolution
        if self.learnable_spiral:
            # Get spiral-masked weights
            masked_weight = self._apply_spiral_mask(self.weight)
            
            # Perform standard convolution with masked weights
            output = F.conv2d(
                x, masked_weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
        else:
            # Alternative approach: Sample along spiral paths and apply 1D convolution
            
            # Sample features along spiral paths for each position
            output_height = (height + 2 * self.padding - self.dilation * 
                           (self.kernel_size - 1) - 1) // self.stride + 1
            output_width = (width + 2 * self.padding - self.dilation * 
                          (self.kernel_size - 1) - 1) // self.stride + 1
            
            output = torch.zeros(batch_size, self.out_channels, 
                                output_height, output_width, device=x.device)
            
            # For each output position
            for i in range(output_height):
                for j in range(output_width):
                    # Get input window
                    h_start = i * self.stride - self.padding
                    w_start = j * self.stride - self.padding
                    
                    # Extract and sample features along spiral
                    window_features = []
                    
                    for b in range(batch_size):
                        # Get the window
                        window = x[b:b+1, :, 
                                  max(0, h_start):min(height, h_start + self.kernel_size),
                                  max(0, w_start):min(width, w_start + self.kernel_size)]
                        
                        # Pad if necessary
                        pad_h = max(0, -h_start) if h_start < 0 else 0
                        pad_w = max(0, -w_start) if w_start < 0 else 0
                        
                        if pad_h > 0 or pad_w > 0:
                            window = F.pad(window, (pad_w, pad_w, pad_h, pad_h))
                        
                        # Sample along spiral
                        sampled = self._spiral_interpolate(window)
                        
                        # Apply weights
                        if self.spiral_weights is not None:
                            weighted = sampled * self.spiral_weights.unsqueeze(0)
                        else:
                            weighted = sampled
                        
                        # Sum along spiral dimension
                        window_features.append(weighted.sum(dim=2))
                    
                    window_features = torch.stack(window_features)
                    
                    # Apply output weights and bias
                    output[:, :, i, j] = window_features.squeeze(2)
        
        return output
    
    def get_spiral_pattern(self) -> torch.Tensor:
        """Get the current spiral pattern."""
        if self.learnable_spiral:
            return self.spiral_positions.detach()
        else:
            return self.spiral_positions_buffer

# ==================== SPIRAL FEATURE EXTRACTOR ====================
class SpiralFeatureExtractor(nn.Module):
    """
    Complete spiral convolution feature extractor for 3-channel images.
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_layers: int = 3,
                 kernel_sizes: List[int] = None,
                 spiral_types: List[str] = None):
        """
        Args:
            in_channels: Input channels (3 for RGB)
            base_channels: Base number of channels
            num_layers: Number of spiral convolution layers
            kernel_sizes: List of kernel sizes for each layer
            spiral_types: List of spiral types for each layer
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        
        if spiral_types is None:
            spiral_types = ['archimedean', 'logarithmic', 'golden']
        
        # Build spiral convolution layers
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            
            layer = SpiralConv2d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[i],
                stride=1 if i == 0 else 2,  # Downsample after first layer
                padding=kernel_sizes[i] // 2,
                spiral_type=spiral_types[i],
                learnable_spiral=True
            )
            
            layers.append(layer)
            
            # Add activation and normalization
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            current_channels = out_channels
        
        # Final pooling
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.layers = nn.Sequential(*layers)
        
        # Feature dimension
        self.feature_dim = current_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input image."""
        features = self.layers(x)
        return features.flatten(1)
    
    def visualize_spiral_patterns(self):
        """Visualize spiral patterns from each layer."""
        fig, axes = plt.subplots(1, self.num_layers, figsize=(15, 5))
        
        if self.num_layers == 1:
            axes = [axes]
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SpiralConv2d):
                pattern = layer.get_spiral_pattern()
                
                ax = axes[i]
                ax.scatter(pattern[:, 0], pattern[:, 1], 
                          c=range(len(pattern)), cmap='viridis',
                          s=50, alpha=0.7)
                ax.plot(pattern[:, 0], pattern[:, 1], 'b-', alpha=0.3)
                ax.scatter([layer.kernel_size // 2], [layer.kernel_size // 2], 
                          c='red', s=100, marker='x', linewidth=2)
                ax.set_xlim(-0.5, layer.kernel_size - 0.5)
                ax.set_ylim(-0.5, layer.kernel_size - 0.5)
                ax.set_aspect('equal')
                ax.set_title(f'Layer {i+1}: {layer.spiral_type} spiral')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==================== SPIRAL CONVOLUTION NETWORK ====================
class SpiralConvNet(nn.Module):
    """
    Complete network using spiral convolution for image classification.
    """
    
    def __init__(self,
                 num_classes: int = 10,
                 in_channels: int = 3,
                 base_channels: int = 32,
                 num_spiral_layers: int = 3):
        """
        Args:
            num_classes: Number of output classes
            in_channels: Input channels
            base_channels: Base channels
            num_spiral_layers: Number of spiral convolution layers
        """
        super().__init__()
        
        # Spiral feature extractor
        self.spiral_extractor = SpiralFeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_spiral_layers
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.spiral_extractor.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.spiral_extractor(x)
        output = self.classifier(features)
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get feature maps from each spiral layer."""
        feature_maps = []
        current = x
        
        for layer in self.spiral_extractor.layers:
            current = layer(current)
            if isinstance(layer, SpiralConv2d):
                feature_maps.append(current.detach())
        
        return feature_maps

# ==================== DATASET LOADERS ====================
class CIFAR10SpiralDataset(Dataset):
    """CIFAR-10 dataset wrapper for spiral convolution."""
    
    def __init__(self, train: bool = True, download: bool = True):
        self.train = train
        self.dataset = datasets.CIFAR10(
            root='./data',
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2470, 0.2435, 0.2616))
            ])
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class SpiralDataAugmentation:
    """Data augmentation specifically designed for spiral convolution."""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2470, 0.2435, 0.2616))
        ])
    
    def __call__(self, img):
        return self.transform(img)

# ==================== TRAINING UTILITIES ====================
class SpiralConvTrainer:
    """Training utilities for spiral convolution networks."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += len(data)
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {correct/len(data):.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                _, predicted = torch.max(output, 1)
                correct = (predicted == target).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += len(data)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50):
        """Full training loop."""
        print(f"Training Spiral Convolution Network for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        axes[0].plot(self.val_losses, label='Val Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.train_accuracies, label='Train Acc', alpha=0.8)
        axes[1].plot(self.val_accuracies, label='Val Acc', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training History - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==================== VISUALIZATION TOOLS ====================
class SpiralVisualizer:
    """Visualization tools for spiral convolution."""
    
    @staticmethod
    def visualize_spiral_on_image(image: torch.Tensor,
                                 spiral_conv: SpiralConv2d,
                                 position: Tuple[int, int] = (16, 16)):
        """Visualize spiral sampling pattern on an image."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Image with spiral pattern
        axes[1].imshow(img_np)
        
        # Get spiral pattern
        pattern = spiral_conv.get_spiral_pattern()
        
        # Plot spiral pattern
        center_x, center_y = position
        
        for i, (dx, dy) in enumerate(pattern):
            x = center_x + dx - spiral_conv.kernel_size // 2
            y = center_y + dy - spiral_conv.kernel_size // 2
            
            # Draw sampling point
            circle = Circle((y, x), radius=0.5, color='red', alpha=0.7)
            axes[1].add_patch(circle)
            
            # Add number
            axes[1].text(y, x, str(i), color='white', 
                        fontsize=8, ha='center', va='center',
                        fontweight='bold')
        
        # Connect points in order
        pattern_np = pattern.cpu().numpy()
        for i in range(len(pattern_np) - 1):
            x1 = center_x + pattern_np[i, 0] - spiral_conv.kernel_size // 2
            y1 = center_y + pattern_np[i, 1] - spiral_conv.kernel_size // 2
            x2 = center_x + pattern_np[i+1, 0] - spiral_conv.kernel_size // 2
            y2 = center_y + pattern_np[i+1, 1] - spiral_conv.kernel_size // 2
            
            axes[1].plot([y1, y2], [x1, x2], 'b-', alpha=0.5, linewidth=1)
        
        axes[1].set_title('Spiral Sampling Pattern Overlay')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_feature_maps(feature_maps: List[torch.Tensor],
                              num_maps: int = 8):
        """Visualize feature maps from spiral convolution."""
        num_layers = len(feature_maps)
        
        fig, axes = plt.subplots(num_layers, min(num_maps, 8), 
                                figsize=(15, 3*num_layers))
        
        if num_layers == 1:
            axes = [axes]
        
        for layer_idx, layer_maps in enumerate(feature_maps):
            # Take first batch
            maps = layer_maps[0]
            
            # Select random channels
            channels = min(maps.shape[0], num_maps)
            selected_channels = torch.randperm(maps.shape[0])[:channels]
            
            for i, channel in enumerate(selected_channels):
                ax = axes[layer_idx][i] if num_layers > 1 else axes[i]
                
                feature_map = maps[channel].cpu().numpy()
                
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'L{layer_idx+1} C{channel}')
                ax.axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Spiral Convolution Feature Maps', fontsize=14)
        plt.tight_layout()
        plt.show()

# ==================== BENCHMARKING ====================
class SpiralConvBenchmark:
    """Benchmark spiral convolution against standard convolution."""
    
    @staticmethod
    def compare_operations(image_size: int = 32,
                          in_channels: int = 3,
                          out_channels: int = 64,
                          kernel_size: int = 3):
        """Compare spiral convolution with standard convolution."""
        import time
        
        # Create random input
        x = torch.randn(1, in_channels, image_size, image_size)
        
        # Standard convolution
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        
        # Spiral convolution
        spiral_conv = SpiralConv2d(in_channels, out_channels, kernel_size, padding=1)
        
        # Warm up
        for _ in range(5):
            _ = conv2d(x)
            _ = spiral_conv(x)
        
        # Benchmark standard convolution
        start = time.time()
        for _ in range(100):
            output_std = conv2d(x)
        std_time = (time.time() - start) / 100
        
        # Benchmark spiral convolution
        start = time.time()
        for _ in range(100):
            output_spiral = spiral_conv(x)
        spiral_time = (time.time() - start) / 100
        
        print("=" * 60)
        print("CONVOLUTION OPERATION BENCHMARK")
        print("=" * 60)
        print(f"Input: {in_channels}x{image_size}x{image_size}")
        print(f"Output: {out_channels} channels")
        print(f"Kernel size: {kernel_size}")
        print(f"\nStandard Conv2d: {std_time*1000:.2f} ms per forward pass")
        print(f"Spiral Conv2d: {spiral_time*1000:.2f} ms per forward pass")
        print(f"Ratio (Spiral/Standard): {spiral_time/std_time:.2f}x")
        print(f"\nOutput shapes are identical: {output_std.shape == output_spiral.shape}")
        
        return std_time, spiral_time

# ==================== MAIN DEMONSTRATION ====================
def demonstrate_spiral_sampling():
    """Demonstrate different spiral sampling patterns."""
    print("=" * 60)
    print("SPIRAL SAMPLING PATTERNS DEMONSTRATION")
    print("=" * 60)
    
    spiral_types = ['archimedean', 'logarithmic', 'golden']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, spiral_type in enumerate(spiral_types):
        sampler = SpiralSampler(
            kernel_size=9,
            num_spirals=8,
            spiral_type=spiral_type
        )
        
        sampler.visualize_sampling_pattern(axes[i])
    
    plt.suptitle('Different Spiral Sampling Patterns', fontsize=14)
    plt.tight_layout()
    plt.show()

def demonstrate_spiral_convolution():
    """Demonstrate spiral convolution on CIFAR-10."""
    print("\n" + "=" * 60)
    print("SPIRAL CONVOLUTION ON CIFAR-10 DATASET")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    
    train_dataset = CIFAR10SpiralDataset(train=True, download=True)
    test_dataset = CIFAR10SpiralDataset(train=False, download=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create spiral convolution network
    model = SpiralConvNet(
        num_classes=10,
        in_channels=3,
        base_channels=32,
        num_spiral_layers=3
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Visualize spiral patterns before training
    print("\nVisualizing spiral patterns...")
    model.spiral_extractor.visualize_spiral_patterns()
    
    # Create trainer
    trainer = SpiralConvTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-4
    )
    
    # Train for a few epochs (for demonstration)
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10  # Reduced for demonstration
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Visualize feature maps
    print("\nVisualizing feature maps...")
    with torch.no_grad():
        sample_batch, _ = next(iter(test_loader))
        sample_batch = sample_batch[:1].to(device)  # Take one sample
        
        feature_maps = model.get_feature_maps(sample_batch)
        SpiralVisualizer.visualize_feature_maps(feature_maps)
    
    # Visualize spiral pattern on sample image
    print("\nVisualizing spiral pattern on sample image...")
    sample_image, _ = test_dataset[0]
    spiral_layer = model.spiral_extractor.layers[0]  # First spiral layer
    
    SpiralVisualizer.visualize_spiral_on_image(
        sample_image,
        spiral_layer,
        position=(16, 16)
    )
    
    return model, trainer

def benchmark_comparison():
    """Benchmark spiral convolution against standard convolution."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK: Spiral vs Standard Convolution")
    print("=" * 60)
    
    benchmark = SpiralConvBenchmark()
    
    # Compare for different configurations
    configurations = [
        (32, 3, 64, 3),   # Small network
        (64, 64, 128, 3), # Medium network
        (128, 128, 256, 5) # Larger network
    ]
    
    results = []
    
    for config in configurations:
        image_size, in_channels, out_channels, kernel_size = config
        std_time, spiral_time = benchmark.compare_operations(
            image_size, in_channels, out_channels, kernel_size
        )
        results.append((config, std_time, spiral_time))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for config, std_time, spiral_time in results:
        image_size, in_channels, out_channels, kernel_size = config
        print(f"Config {in_channels}→{out_channels} @ {image_size}x{image_size}: "
              f"Spiral is {spiral_time/std_time:.2f}x slower")

def demonstrate_edge_cases():
    """Demonstrate edge cases and special features."""
    print("\n" + "=" * 60)
    print("EDGE CASES AND SPECIAL FEATURES")
    print("=" * 60)
    
    # Test with different spiral configurations
    print("\n1. Different Spiral Types in Single Network:")
    
    model = SpiralFeatureExtractor(
        in_channels=3,
        base_channels=16,
        num_layers=3,
        spiral_types=['archimedean', 'logarithmic', 'golden']
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Feature dimension: {model.feature_dim}")
    
    # Test with learnable vs fixed spiral
    print("\n2. Learnable vs Fixed Spiral Patterns:")
    
    learnable_conv = SpiralConv2d(3, 16, 5, learnable_spiral=True)
    fixed_conv = SpiralConv2d(3, 16, 5, learnable_spiral=False)
    
    x = torch.randn(1, 3, 16, 16)
    output_learnable = learnable_conv(x)
    output_fixed = fixed_conv(x)
    
    print(f"  Learnable spiral params: {sum(p.numel() for p in learnable_conv.parameters()):,}")
    print(f"  Fixed spiral params: {sum(p.numel() for p in fixed_conv.parameters()):,}")
    print(f"  Output shapes match: {output_learnable.shape == output_fixed.shape}")

def main():
    """Main demonstration function."""
    print("SPIRAL CONVOLUTION FOR 3-CHANNEL IMAGES")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demonstrate spiral sampling patterns
    demonstrate_spiral_sampling()
    
    # Demonstrate spiral convolution on CIFAR-10
    model, trainer = demonstrate_spiral_convolution()
    
    # Benchmark performance
    benchmark_comparison()
    
    # Demonstrate edge cases
    demonstrate_edge_cases()
    
    print("\n" + "=" * 60)
    print("SPIRAL CONVOLUTION IMPLEMENTATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Implemented:")
    print("1. Multiple spiral types (Archimedean, Logarithmic, Golden)")
    print("2. Learnable spiral patterns")
    print("3. Efficient spiral sampling")
    print("4. Complete feature extractor for 3-channel images")
    print("5. Visualization tools for spiral patterns and features")
    print("6. Benchmarking against standard convolution")

if __name__ == "__main__":
    main()