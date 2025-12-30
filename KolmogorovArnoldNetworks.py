import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Union
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== BASIS FUNCTIONS ====================
class BasisFunction(nn.Module):
    """Base class for basis functions in KAN."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    def in_features(self) -> int:
        return 1
    
    @property
    def out_features(self) -> int:
        return 1

class SplineBasis(BasisFunction):
    """B-spline basis functions for KAN."""
    
    def __init__(self, 
                 grid_size: int = 5, 
                 k: int = 3, 
                 grid_range: Tuple[float, float] = (-1, 1)):
        """
        Args:
            grid_size: Number of grid intervals
            k: Spline degree (k=3 for cubic splines)
            grid_range: Range of the grid
        """
        super().__init__()
        self.grid_size = grid_size
        self.k = k
        self.grid_range = grid_range
        
        # Total number of basis functions
        self.num_basis = grid_size + k
        
        # Create uniform grid points
        self.grid_points = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        
        # Extended knot vector
        self.extended_knots = self._create_knot_vector()
        
    def _create_knot_vector(self) -> torch.Tensor:
        """Create extended knot vector for B-splines."""
        # Repeat boundary knots k+1 times
        left_ext = torch.full((self.k,), self.grid_range[0])
        right_ext = torch.full((self.k,), self.grid_range[1])
        
        # Concatenate extended knots
        knots = torch.cat([
            left_ext,
            self.grid_points,
            right_ext
        ])
        return knots
    
    def _basis_function(self, i: int, k: int, x: torch.Tensor) -> torch.Tensor:
        """Recursive definition of B-spline basis functions (Cox-de Boor algorithm)."""
        knots = self.extended_knots
        
        if k == 0:
            # Base case: piecewise constant
            return ((x >= knots[i]) & (x < knots[i + 1])).float()
        else:
            # Recursive case
            denom1 = knots[i + k] - knots[i]
            term1 = torch.zeros_like(x)
            if denom1 != 0:
                term1 = (x - knots[i]) / denom1 * self._basis_function(i, k - 1, x)
            
            denom2 = knots[i + k + 1] - knots[i + 1]
            term2 = torch.zeros_like(x)
            if denom2 != 0:
                term2 = (knots[i + k + 1] - x) / denom2 * self._basis_function(i + 1, k - 1, x)
            
            return term1 + term2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all basis functions at x.
        
        Args:
            x: Input tensor of shape (batch_size,)
            
        Returns:
            Basis function values of shape (batch_size, num_basis)
        """
        batch_size = x.shape[0]
        basis_values = torch.zeros(batch_size, self.num_basis, device=x.device)
        
        for i in range(self.num_basis):
            basis_values[:, i] = self._basis_function(i, self.k, x)
        
        return basis_values

class FourierBasis(BasisFunction):
    """Fourier basis functions."""
    
    def __init__(self, num_frequencies: int = 5):
        """
        Args:
            num_frequencies: Number of frequency components
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.num_basis = 2 * num_frequencies + 1  # +1 for constant term
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Fourier basis functions.
        
        Returns basis: [1, cos(πx), sin(πx), cos(2πx), sin(2πx), ...]
        """
        batch_size = x.shape[0]
        basis = torch.ones(batch_size, self.num_basis, device=x.device)
        
        for k in range(1, self.num_frequencies + 1):
            basis[:, 2*k-1] = torch.cos(k * math.pi * x)
            basis[:, 2*k] = torch.sin(k * math.pi * x)
        
        return basis

class GaussianBasis(BasisFunction):
    """Gaussian radial basis functions."""
    
    def __init__(self, num_centers: int = 10, sigma: float = 0.2):
        """
        Args:
            num_centers: Number of Gaussian centers
            sigma: Width parameter
        """
        super().__init__()
        self.num_centers = num_centers
        self.sigma = sigma
        self.num_basis = num_centers
        self.centers = torch.linspace(-1, 1, num_centers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Gaussian basis functions."""
        batch_size = x.shape[0]
        basis = torch.zeros(batch_size, self.num_basis, device=x.device)
        
        for i, center in enumerate(self.centers):
            basis[:, i] = torch.exp(-((x - center) ** 2) / (2 * self.sigma ** 2))
        
        return basis

# ==================== KAN LAYER ====================
class KANLayer(nn.Module):
    """
    Single layer of Kolmogorov-Arnold Network.
    Implements: φ(x) = Σ w_i * b_i(x)
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 basis_type: str = 'spline',
                 basis_kwargs: Optional[dict] = None):
        """
        Args:
            in_features: Number of input features
            out_features: Number of output features
            basis_type: Type of basis function ('spline', 'fourier', 'gaussian')
            basis_kwargs: Additional arguments for basis functions
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize basis functions
        if basis_kwargs is None:
            basis_kwargs = {}
        
        self.basis_functions = nn.ModuleList()
        for _ in range(in_features):
            if basis_type == 'spline':
                basis = SplineBasis(**basis_kwargs)
            elif basis_type == 'fourier':
                basis = FourierBasis(**basis_kwargs)
            elif basis_type == 'gaussian':
                basis = GaussianBasis(**basis_kwargs)
            else:
                raise ValueError(f"Unknown basis type: {basis_type}")
            self.basis_functions.append(basis)
        
        # Get number of basis functions from the first basis
        self.num_basis = self.basis_functions[0].num_basis
        
        # Weight parameters: shape (out_features, in_features, num_basis)
        self.weights = nn.Parameter(
            torch.randn(out_features, in_features, self.num_basis) * 0.1
        )
        
        # Optional bias term
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of KAN layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        # Process each input feature
        for i in range(self.in_features):
            # Evaluate basis functions for this input feature
            basis_values = self.basis_functions[i](x[:, i])  # (batch_size, num_basis)
            
            # Weighted sum for all outputs
            for j in range(self.out_features):
                # Get weights for this input-output pair
                weights_ij = self.weights[j, i, :]  # (num_basis,)
                
                # Compute contribution
                contribution = torch.sum(basis_values * weights_ij, dim=1)  # (batch_size,)
                output[:, j] += contribution
        
        # Add bias
        output += self.bias
        
        return output

# ==================== KAN BLOCK ====================
class KANBlock(nn.Module):
    """
    A block of KAN layers with residual connection.
    Implements: y = x + KAN(x)
    """
    
    def __init__(self, 
                 features: int, 
                 basis_type: str = 'spline',
                 basis_kwargs: Optional[dict] = None,
                 dropout: float = 0.0):
        """
        Args:
            features: Number of input/output features
            basis_type: Type of basis function
            basis_kwargs: Additional arguments for basis functions
            dropout: Dropout probability
        """
        super().__init__()
        self.kan_layer = KANLayer(
            in_features=features,
            out_features=features,
            basis_type=basis_type,
            basis_kwargs=basis_kwargs
        )
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.kan_layer(x)
        x = self.dropout(x)
        x = self.norm(x)
        return residual + x

# ==================== COMPLETE KAN NETWORK ====================
class KANNetwork(nn.Module):
    """
    Complete Kolmogorov-Arnold Network.
    
    Based on the Kolmogorov-Arnold Representation Theorem:
    f(x) = Σ_q Φ_q(Σ_p ψ_{q,p}(x_p))
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 basis_type: str = 'spline',
                 basis_kwargs: Optional[dict] = None,
                 num_blocks: int = 2,
                 dropout: float = 0.0,
                 use_residual: bool = True):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output
            basis_type: Type of basis function
            basis_kwargs: Additional arguments for basis functions
            num_blocks: Number of KAN blocks per hidden layer
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        if basis_kwargs is None:
            basis_kwargs = {}
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(
            KANLayer(input_dim, hidden_dims[0], basis_type, basis_kwargs)
        )
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            in_dim, out_dim = hidden_dims[i], hidden_dims[i + 1]
            
            # Add KAN blocks for this hidden layer
            for _ in range(num_blocks):
                layers.append(
                    KANBlock(in_dim, basis_type, basis_kwargs, dropout)
                )
            
            # Transition to next layer dimension
            layers.append(
                KANLayer(in_dim, out_dim, basis_type, basis_kwargs)
            )
        
        # Final KAN blocks for last hidden layer
        for _ in range(num_blocks):
            layers.append(
                KANBlock(hidden_dims[-1], basis_type, basis_kwargs, dropout)
            )
        
        # Output layer
        layers.append(
            KANLayer(hidden_dims[-1], output_dim, basis_type, basis_kwargs)
        )
        
        self.layers = nn.ModuleList(layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.layers:
            if isinstance(layer, KANLayer) or isinstance(layer, KANBlock):
                for param in layer.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        
        # For classification, apply softmax
        if self.output_dim > 1:
            return F.softmax(x, dim=-1)
        return x
    
    def get_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get activations from each layer for visualization."""
        activations = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            activations.append(current.detach().cpu())
        
        return activations

# ==================== ENHANCED KAN WITH ADAPTIVE BASIS ====================
class AdaptiveKANLayer(KANLayer):
    """KAN layer with adaptive basis function selection."""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 basis_types: List[str] = None,
                 basis_kwargs_list: Optional[List[dict]] = None):
        """
        Args:
            basis_types: List of basis types for each input
            basis_kwargs_list: List of kwargs for each basis
        """
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if basis_types is None:
            basis_types = ['spline'] * in_features
        
        if basis_kwargs_list is None:
            basis_kwargs_list = [{}] * in_features
        
        # Initialize different basis functions for each input
        self.basis_functions = nn.ModuleList()
        self.num_basis_list = []
        
        for i in range(in_features):
            basis_type = basis_types[i]
            basis_kwargs = basis_kwargs_list[i]
            
            if basis_type == 'spline':
                basis = SplineBasis(**basis_kwargs)
            elif basis_type == 'fourier':
                basis = FourierBasis(**basis_kwargs)
            elif basis_type == 'gaussian':
                basis = GaussianBasis(**basis_kwargs)
            else:
                raise ValueError(f"Unknown basis type: {basis_type}")
            
            self.basis_functions.append(basis)
            self.num_basis_list.append(basis.num_basis)
        
        # Weight parameters with different basis sizes
        self.weights = nn.ParameterList()
        for j in range(out_features):
            weight_row = []
            for i in range(in_features):
                weight = nn.Parameter(
                    torch.randn(self.num_basis_list[i]) * 0.1
                )
                weight_row.append(weight)
            self.weights.append(nn.ParameterList(weight_row))
        
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive basis."""
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        for i in range(self.in_features):
            basis_values = self.basis_functions[i](x[:, i])
            
            for j in range(self.out_features):
                weights_ij = self.weights[j][i]
                contribution = torch.sum(basis_values * weights_ij, dim=1)
                output[:, j] += contribution
        
        output += self.bias
        return output

# ==================== DATASET GENERATORS ====================
class DatasetGenerator:
    """Generate synthetic datasets for testing KAN."""
    
    @staticmethod
    def generate_function_data(func_name: str, n_samples: int = 1000, 
                              noise: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 1D function data."""
        x = torch.linspace(-2, 2, n_samples).unsqueeze(1)
        
        if func_name == 'sin':
            y = torch.sin(2 * math.pi * x)
        elif func_name == 'abs':
            y = torch.abs(x)
        elif func_name == 'step':
            y = (x > 0).float()
        elif func_name == 'quadratic':
            y = x ** 2
        elif func_name == 'composite':
            y = torch.sin(x) + 0.5 * torch.cos(3 * x)
        else:
            raise ValueError(f"Unknown function: {func_name}")
        
        # Add noise
        y += noise * torch.randn_like(y)
        
        return x, y
    
    @staticmethod
    def generate_2d_classification_data(dataset_name: str = 'moons', 
                                       n_samples: int = 1000):
        """Generate 2D classification datasets."""
        if dataset_name == 'moons':
            X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
        elif dataset_name == 'circles':
            X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
        elif dataset_name == 'xor':
            X = np.random.randn(n_samples, 2)
            y = (X[:, 0] > 0) ^ (X[:, 1] > 0)
            y = y.astype(int)
        elif dataset_name == 'spiral':
            n = n_samples // 2
            theta = np.linspace(0, 4 * np.pi, n)
            r = np.linspace(0, 1, n)
            
            # Class 0
            X0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            y0 = np.zeros(n)
            
            # Class 1
            X1 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
            y1 = np.ones(n)
            
            X = np.vstack([X0, X1])
            y = np.concatenate([y0, y1])
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        return X, y

# ==================== TRAINER AND EVALUATION ====================
class KANTrainer:
    """Training and evaluation utilities for KAN."""
    
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        self.model = model
        self.criterion = nn.CrossEntropyLoss() if model.output_dim > 1 else nn.MSELoss()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
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
            self.optimizer.zero_grad()
            
            output = self.model(data)
            
            if self.model.output_dim > 1:
                loss = self.criterion(output, target)
                _, predicted = torch.max(output, 1)
                correct = (predicted == target).sum().item()
                total_correct += correct
            else:
                loss = self.criterion(output.squeeze(), target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_samples += len(data)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if self.model.output_dim > 1 else None
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                
                if self.model.output_dim > 1:
                    loss = self.criterion(output, target)
                    _, predicted = torch.max(output, 1)
                    correct = (predicted == target).sum().item()
                    total_correct += correct
                else:
                    loss = self.criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                total_samples += len(data)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if self.model.output_dim > 1 else None
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              epochs: int = 100,
              patience: int = 10):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if train_acc is not None:
                self.train_accuracies.append(train_acc)
                self.val_accuracies.append(val_acc)
            
            self.scheduler.step(val_loss)
            
            # Print progress
            if epoch % 10 == 0:
                if train_acc is not None:
                    print(f"Epoch {epoch:3d}: "
                          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch:3d}: "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_kan_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    def plot_training_history(self):
        """Plot training and validation losses."""
        fig, axes = plt.subplots(1, 2 if self.model.output_dim > 1 else 1, 
                                figsize=(12, 4))
        
        if self.model.output_dim > 1:
            # Plot loss
            axes[0].plot(self.train_losses, label='Train Loss', alpha=0.8)
            axes[0].plot(self.val_losses, label='Val Loss', alpha=0.8)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training History')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot accuracy
            axes[1].plot(self.train_accuracies, label='Train Acc', alpha=0.8)
            axes[1].plot(self.val_accuracies, label='Val Acc', alpha=0.8)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy History')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            # Regression: only plot loss
            axes.plot(self.train_losses, label='Train Loss', alpha=0.8)
            axes.plot(self.val_losses, label='Val Loss', alpha=0.8)
            axes.set_xlabel('Epoch')
            axes.set_ylabel('MSE Loss')
            axes.set_title('Training History')
            axes.legend()
            axes.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==================== VISUALIZATION TOOLS ====================
class KANVisualizer:
    """Visualization tools for KAN networks."""
    
    @staticmethod
    def visualize_1d_function(model: KANLayer, 
                             x_range: Tuple[float, float] = (-2, 2),
                             n_points: int = 100):
        """Visualize a 1D KAN layer function."""
        x = torch.linspace(x_range[0], x_range[1], n_points).unsqueeze(1)
        
        model.eval()
        with torch.no_grad():
            y_pred = model(x).squeeze()
        
        plt.figure(figsize=(10, 6))
        plt.plot(x.numpy(), y_pred.numpy(), 'b-', linewidth=2, label='KAN Prediction')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('KAN Function Approximation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def visualize_basis_functions(basis_fn: BasisFunction,
                                 x_range: Tuple[float, float] = (-1, 1),
                                 n_points: int = 200):
        """Visualize basis functions."""
        x = torch.linspace(x_range[0], x_range[1], n_points)
        
        with torch.no_grad():
            basis_values = basis_fn(x.unsqueeze(1))
        
        n_basis = basis_values.shape[1]
        
        fig, axes = plt.subplots(1, min(5, n_basis), figsize=(15, 3))
        if n_basis == 1:
            axes = [axes]
        
        for i in range(min(5, n_basis)):
            ax = axes[i]
            ax.plot(x.numpy(), basis_values[:, i].numpy(), linewidth=2)
            ax.set_title(f'Basis {i+1}')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
        
        plt.suptitle(f'{basis_fn.__class__.__name__} Basis Functions')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_decision_boundary(model: KANNetwork, 
                                   X: torch.Tensor, 
                                   y: torch.Tensor,
                                   resolution: int = 100):
        """Visualize 2D decision boundary."""
        model.eval()
        
        # Create grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Predict on grid
        grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        with torch.no_grad():
            Z = model(grid_points)
            if model.output_dim > 1:
                Z = torch.argmax(Z, dim=1)
            else:
                Z = (Z > 0.5).float()
        
        Z = Z.reshape(xx.shape).numpy()
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', 
                   edgecolors='k', alpha=0.7)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('KAN Decision Boundary')
        plt.colorbar()
        plt.grid(True, alpha=0.3)
        plt.show()

# ==================== MAIN DEMONSTRATION ====================
def demo_1d_regression():
    """Demonstrate KAN on 1D function regression."""
    print("=" * 60)
    print("1D Function Regression with KAN")
    print("=" * 60)
    
    # Generate data
    X, y = DatasetGenerator.generate_function_data('composite', n_samples=1000)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create KAN model
    model = KANNetwork(
        input_dim=1,
        hidden_dims=[16, 32, 16],
        output_dim=1,
        basis_type='spline',
        basis_kwargs={'grid_size': 10, 'k': 3},
        num_blocks=1,
        dropout=0.1
    )
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = KANTrainer(model, learning_rate=0.005, weight_decay=1e-4)
    trainer.train(train_loader, val_loader, epochs=200, patience=20)
    
    # Visualize
    trainer.plot_training_history()
    
    # Test on new data
    model.eval()
    X_test = torch.linspace(-2, 2, 200).unsqueeze(1)
    with torch.no_grad():
        y_pred = model(X_test)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.3, label='Training Data')
    plt.plot(X_test.numpy(), y_pred.numpy(), 'r-', linewidth=3, label='KAN Prediction')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('KAN Function Approximation: sin(x) + 0.5*cos(3x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def demo_2d_classification():
    """Demonstrate KAN on 2D classification."""
    print("\n" + "=" * 60)
    print("2D Classification with KAN")
    print("=" * 60)
    
    # Generate spiral dataset
    X, y = DatasetGenerator.generate_2d_classification_data('spiral', n_samples=1000)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create KAN model
    model = KANNetwork(
        input_dim=2,
        hidden_dims=[32, 64, 32],
        output_dim=2,
        basis_type='spline',
        basis_kwargs={'grid_size': 8, 'k': 3},
        num_blocks=2,
        dropout=0.2
    )
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = KANTrainer(model, learning_rate=0.001)
    trainer.train(train_loader, val_loader, epochs=100, patience=15)
    
    # Visualize
    trainer.plot_training_history()
    
    # Visualize decision boundary
    visualizer = KANVisualizer()
    visualizer.visualize_decision_boundary(model, X_train, y_train)
    
    # Evaluate
    val_loss, val_acc = trainer.validate(val_loader)
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")

def demo_basis_functions():
    """Demonstrate different basis functions."""
    print("\n" + "=" * 60)
    print("Basis Function Visualization")
    print("=" * 60)
    
    # Create basis functions
    spline_basis = SplineBasis(grid_size=5, k=3)
    fourier_basis = FourierBasis(num_frequencies=3)
    gaussian_basis = GaussianBasis(num_centers=10, sigma=0.3)
    
    # Visualize
    visualizer = KANVisualizer()
    print("\nSpline Basis Functions:")
    visualizer.visualize_basis_functions(spline_basis)
    
    print("\nFourier Basis Functions:")
    visualizer.visualize_basis_functions(fourier_basis)
    
    print("\nGaussian Basis Functions:")
    visualizer.visualize_basis_functions(gaussian_basis)

def demo_adaptive_kan():
    """Demonstrate adaptive KAN with different basis functions per input."""
    print("\n" + "=" * 60)
    print("Adaptive KAN Demonstration")
    print("=" * 60)
    
    # Generate XOR dataset
    X, y = DatasetGenerator.generate_2d_classification_data('xor', n_samples=1000)
    
    # Create adaptive KAN model
    model = KANNetwork(
        input_dim=2,
        hidden_dims=[16, 32, 16],
        output_dim=2,
        basis_type='spline',  # Still using single basis type for simplicity
        basis_kwargs={'grid_size': 6, 'k': 2},
        num_blocks=1,
        dropout=0.1
    )
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    trainer = KANTrainer(model, learning_rate=0.003)
    trainer.train(train_loader, val_loader, epochs=50, patience=10)
    
    # Visualize decision boundary
    visualizer = KANVisualizer()
    visualizer.visualize_decision_boundary(model, X_train, y_train)

def main():
    """Main demonstration function."""
    print("KOLMOGOROV-ARNOLD NETWORK (KAN) IMPLEMENTATION")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demo 1: Basis functions
    demo_basis_functions()
    
    # Demo 2: 1D regression
    demo_1d_regression()
    
    # Demo 3: 2D classification
    demo_2d_classification()
    
    # Demo 4: Adaptive KAN
    demo_adaptive_kan()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()