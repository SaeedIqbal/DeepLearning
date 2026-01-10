import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ==================== TANH MATHEMATICAL FOUNDATION ====================
class TanhMathematics:
    """
    Mathematical foundations of the Hyperbolic Tangent (tanh) function.
    Implements all mathematical properties and theorems.
    
    tanh(x) = sinh(x) / cosh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
           = 2σ(2x) - 1 where σ is sigmoid
    """
    
    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic tangent: tanh(x) = sinh(x) / cosh(x)
        
        Mathematical Properties:
        1. Range: (-1, 1)
        2. Domain: (-∞, ∞)
        3. Point of symmetry: (0, 0)
        4. Derivative: tanh'(x) = 1 - tanh²(x)
        5. Inverse: tanh^{-1}(y) = 0.5 * ln((1+y)/(1-y))
        6. Relation to sigmoid: tanh(x) = 2σ(2x) - 1
        """
        # Direct computation using exponential functions
        pos_exp = torch.exp(x)
        neg_exp = torch.exp(-x)
        return (pos_exp - neg_exp) / (pos_exp + neg_exp)
    
    @staticmethod
    def tanh_derivative(x: torch.Tensor, tanh_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Derivative of tanh: tanh'(x) = 1 - tanh²(x)
        
        Can be computed either from:
        1. Direct computation: 1 - tanh²(x)
        2. Using precomputed tanh value
        """
        if tanh_value is None:
            tanh_value = TanhMathematics.tanh(x)
        return 1 - tanh_value ** 2
    
    @staticmethod
    def tanh_inverse(y: torch.Tensor) -> torch.Tensor:
        """
        Inverse hyperbolic tangent (arctanh):
        tanh^{-1}(y) = 0.5 * ln((1+y)/(1-y))
        
        Properties:
        1. Domain: (-1, 1)
        2. Range: (-∞, ∞)
        3. Used in hyperbolic geometry and neural networks
        """
        # Clip to prevent log(0) or division by 0
        y = torch.clamp(y, -1 + 1e-7, 1 - 1e-7)
        return 0.5 * torch.log((1 + y) / (1 - y))
    
    @staticmethod
    def tanh_second_derivative(x: torch.Tensor) -> torch.Tensor:
        """
        Second derivative of tanh: tanh''(x) = -2 * tanh(x) * (1 - tanh²(x))
        """
        tanh_val = TanhMathematics.tanh(x)
        return -2 * tanh_val * (1 - tanh_val ** 2)
    
    @staticmethod
    def tanh_series_expansion(x: torch.Tensor, terms: int = 10) -> torch.Tensor:
        """
        Taylor series expansion of tanh around 0.
        
        tanh(x) ≈ x - x³/3 + 2x⁵/15 - 17x⁷/315 + 62x⁹/2835 - ...
        """
        # Taylor series coefficients for tanh
        coefficients = [1, 0, -1/3, 0, 2/15, 0, -17/315, 0, 62/2835, 0, -1382/155925]
        
        result = torch.zeros_like(x)
        
        for i, coeff in enumerate(coefficients[:terms]):
            power = i
            if coeff != 0:
                result += coeff * (x ** power)
        
        return result
    
    @staticmethod
    def sinh(x: torch.Tensor) -> torch.Tensor:
        """Hyperbolic sine: sinh(x) = (e^x - e^{-x}) / 2"""
        return (torch.exp(x) - torch.exp(-x)) / 2
    
    @staticmethod
    def cosh(x: torch.Tensor) -> torch.Tensor:
        """Hyperbolic cosine: cosh(x) = (e^x + e^{-x}) / 2"""
        return (torch.exp(x) + torch.exp(-x)) / 2
    
    @staticmethod
    def tanh_from_sigmoid(x: torch.Tensor) -> torch.Tensor:
        """
        Compute tanh using sigmoid: tanh(x) = 2σ(2x) - 1
        Useful for numerical stability and relation understanding.
        """
        sigmoid = 1 / (1 + torch.exp(-2 * x))
        return 2 * sigmoid - 1
    
    @staticmethod
    def softsign(x: torch.Tensor) -> torch.Tensor:
        """
        Softsign function: f(x) = x / (1 + |x|)
        Similar shape to tanh but computationally cheaper.
        """
        return x / (1 + torch.abs(x))
    
    @staticmethod
    def compute_gradient_flow(tanh_output: torch.Tensor, 
                            upstream_gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient flow through tanh for backpropagation.
        dL/dx = dL/dtanh * tanh'(x) = dL/dtanh * (1 - tanh²(x))
        """
        return upstream_gradient * (1 - tanh_output ** 2)

# ==================== TANH IMPLEMENTATION CLASSES ====================
@dataclass
class TanhProperties:
    """Mathematical properties of tanh at a specific point."""
    value: float
    derivative: float
    second_derivative: float
    curvature: float
    inflection_points: List[float]
    linear_approximation: float
    hyperbolic_identity: float  # cosh²(x) - sinh²(x)
    
    def __str__(self):
        return (f"Tanh Properties:\n"
                f"  Value: {self.value:.6f}\n"
                f"  Derivative: {self.derivative:.6f}\n"
                f"  Second Derivative: {self.second_derivative:.6f}\n"
                f"  Curvature: {self.curvature:.6f}\n"
                f"  Inflection Points: {self.inflection_points}\n"
                f"  Linear Approx: {self.linear_approximation:.6f}\n"
                f"  Hyperbolic Identity: {self.hyperbolic_identity:.6f}")

class BaseTanh(nn.Module):
    """Base class for all tanh implementations."""
    
    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: Controls steepness (higher = steeper, but range remains (-1,1))
        """
        super().__init__()
        self.scale = scale
        self.math = TanhMathematics()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute derivative at x."""
        raise NotImplementedError
    
    def analyze_at_point(self, x: float) -> TanhProperties:
        """Analyze tanh properties at a specific point."""
        x_tensor = torch.tensor([x])
        value = self.forward(x_tensor).item()
        deriv = self.derivative(x_tensor).item()
        second_deriv = self.math.tanh_second_derivative(x_tensor * self.scale).item()
        
        # Compute curvature
        curvature = abs(second_deriv) / ((1 + deriv ** 2) ** 1.5)
        
        # Find inflection points (where second derivative = 0)
        inflection_points = []
        if abs(second_deriv) < 1e-6:
            inflection_points.append(x)
        
        # Linear approximation around point
        linear_approx = x * self.scale  # For small x, tanh(x) ≈ x
        
        # Verify hyperbolic identity: cosh²(x) - sinh²(x) = 1
        sinh_val = self.math.sinh(x_tensor * self.scale).item()
        cosh_val = self.math.cosh(x_tensor * self.scale).item()
        hyperbolic_identity = cosh_val ** 2 - sinh_val ** 2
        
        return TanhProperties(
            value=value,
            derivative=deriv,
            second_derivative=second_deriv,
            curvature=curvature,
            inflection_points=inflection_points,
            linear_approximation=linear_approx,
            hyperbolic_identity=hyperbolic_identity
        )

class StandardTanh(BaseTanh):
    """Standard tanh implementation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """tanh(x) = (e^{scale*x} - e^{-scale*x}) / (e^{scale*x} + e^{-scale*x})"""
        scaled_x = x * self.scale
        return self.math.tanh(scaled_x)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """tanh'(x) = scale * (1 - tanh²(scale*x))"""
        tanh_val = self.forward(x)
        return self.scale * (1 - tanh_val ** 2)

class StableTanh(BaseTanh):
    """Numerically stable tanh implementation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable tanh computation.
        Avoids overflow for large positive/negative values.
        """
        scaled_x = x * self.scale
        
        # For large positive x: tanh(x) ≈ 1
        # For large negative x: tanh(x) ≈ -1
        # For moderate x: use standard formula
        
        # Use sigmoid relation for stability
        # tanh(x) = 2σ(2x) - 1
        return self.math.tanh_from_sigmoid(scaled_x)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Stable derivative computation."""
        tanh_val = self.forward(x)
        return self.scale * (1 - tanh_val ** 2)

class FastTanh(BaseTanh):
    """Fast approximation of tanh for performance-critical applications."""
    
    def __init__(self, scale: float = 1.0, approximation_type: str = 'piecewise'):
        super().__init__(scale)
        self.approximation_type = approximation_type
        
        # Precompute piecewise approximation parameters
        self._init_piecewise_params()
    
    def _init_piecewise_params(self):
        """Initialize piecewise linear approximation parameters."""
        if self.approximation_type == 'piecewise':
            # Piecewise linear approximation points for tanh
            self.breakpoints = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
            self.slopes = [0.0, 0.07, 0.42, 0.42, 0.07, 0.0]
            self.intercepts = [-1.0, -0.86, -0.42, 0.42, 0.86, 1.0]
        
        elif self.approximation_type == 'rational':
            # Rational approximation parameters
            self.a = 1.0
            self.b = 0.5
            self.c = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fast tanh approximation."""
        scaled_x = x * self.scale
        
        if self.approximation_type == 'piecewise':
            return self._piecewise_approximation(scaled_x)
        elif self.approximation_type == 'rational':
            return self._rational_approximation(scaled_x)
        elif self.approximation_type == 'hard':
            return self._hard_tanh(scaled_x)
        elif self.approximation_type == 'softsign':
            return self.math.softsign(scaled_x)
        else:
            return self.math.tanh(scaled_x)
    
    def _piecewise_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise linear approximation."""
        result = torch.zeros_like(x)
        
        # Left tail (x < -3)
        mask = x < self.breakpoints[0]
        result[mask] = -1.0
        
        # Middle segments
        for i in range(len(self.breakpoints) - 1):
            mask = (x >= self.breakpoints[i]) & (x < self.breakpoints[i + 1])
            if mask.any():
                result[mask] = self.slopes[i] * x[mask] + self.intercepts[i]
        
        # Right tail (x >= 3)
        mask = x >= self.breakpoints[-1]
        result[mask] = 1.0
        
        return result
    
    def _rational_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """Rational approximation: tanh(x) ≈ x / sqrt(1 + x²)"""
        return x / torch.sqrt(1 + x ** 2)
    
    def _hard_tanh(self, x: torch.Tensor) -> torch.Tensor:
        """Hard tanh: clip(x, -1, 1)"""
        return torch.clamp(x, -1, 1)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate derivative."""
        scaled_x = x * self.scale
        
        if self.approximation_type == 'piecewise':
            return self._piecewise_derivative(scaled_x)
        elif self.approximation_type == 'rational':
            denom = torch.pow(1 + scaled_x ** 2, 1.5)
            return self.scale / denom
        elif self.approximation_type == 'hard':
            mask = (scaled_x > -1) & (scaled_x < 1)
            result = torch.zeros_like(scaled_x)
            result[mask] = self.scale
            return result
        elif self.approximation_type == 'softsign':
            denom = (1 + torch.abs(scaled_x)) ** 2
            return self.scale / denom
        else:
            tanh_val = self.forward(x)
            return self.scale * (1 - tanh_val ** 2)
    
    def _piecewise_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise constant derivative."""
        result = torch.zeros_like(x)
        
        for i in range(len(self.breakpoints) - 1):
            mask = (x >= self.breakpoints[i]) & (x < self.breakpoints[i + 1])
            if mask.any():
                result[mask] = self.slopes[i] * self.scale
        
        return result

class LearnableTanh(BaseTanh):
    """Tanh with learnable parameters."""
    
    def __init__(self, scale: float = 1.0, learnable_scale: bool = True):
        """
        Args:
            scale: Initial scale
            learnable_scale: Whether scale is learnable
        """
        super().__init__(scale)
        
        # Make scale learnable if requested
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        
        # Additional learnable parameters
        self.shift = nn.Parameter(torch.tensor(0.0))  # Horizontal shift
        self.output_scale = nn.Parameter(torch.tensor(1.0))  # Vertical scaling
        self.skew = nn.Parameter(torch.tensor(0.0))   # Skew parameter for asymmetry
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Learnable tanh: output_scale * tanh(scale * (x + shift)) + skew * x"""
        linear_component = self.skew * x
        tanh_component = self.math.tanh(self.scale * (x + self.shift))
        
        return self.output_scale * tanh_component + linear_component
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Derivative of learnable tanh."""
        tanh_val = self.math.tanh(self.scale * (x + self.shift))
        tanh_grad = self.scale * (1 - tanh_val ** 2)
        
        return self.output_scale * tanh_grad + self.skew

# ==================== TANH LAYER IMPLEMENTATIONS ====================
class TanhLayer(nn.Module):
    """Neural network layer with tanh activation."""
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 tanh_type: str = 'standard',
                 use_bias: bool = True,
                 scale: float = 1.0):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            tanh_type: Type of tanh ('standard', 'stable', 'fast', 'learnable')
            use_bias: Whether to use bias term
            scale: Tanh scale parameter
        """
        super().__init__()
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        
        # Initialize weights using Xavier/Glorot initialization (good for tanh)
        self._init_weights()
        
        # Tanh activation
        if tanh_type == 'standard':
            self.activation = StandardTanh(scale)
        elif tanh_type == 'stable':
            self.activation = StableTanh(scale)
        elif tanh_type == 'fast':
            self.activation = FastTanh(scale)
        elif tanh_type == 'learnable':
            self.activation = LearnableTanh(scale, learnable_scale=True)
        else:
            raise ValueError(f"Unknown tanh type: {tanh_type}")
    
    def _init_weights(self):
        """Initialize layer weights using Xavier/Glorot initialization."""
        # Xavier initialization is theoretically optimal for tanh
        nn.init.xavier_normal_(self.linear.weight, gain=1.0)  # gain=1 for tanh
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: linear transformation + tanh."""
        linear_out = self.linear(x)
        return self.activation(linear_out)
    
    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate activations for analysis."""
        linear_out = self.linear(x)
        tanh_out = self.activation(linear_out)
        
        return {
            'linear_output': linear_out.detach(),
            'tanh_output': tanh_out.detach(),
            'tanh_gradient': self.activation.derivative(linear_out).detach(),
            'saturation_level': torch.mean(torch.abs(tanh_out)).item()
        }

class TanhNetwork(nn.Module):
    """Multi-layer perceptron with tanh activations."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 tanh_type: str = 'standard',
                 scale: float = 1.0,
                 dropout: float = 0.0,
                 use_batch_norm: bool = True,
                 use_residual: bool = False):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            tanh_type: Type of tanh activation
            scale: Tanh scale parameter
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layer = nn.Linear(prev_dim, hidden_dim)
            
            # Initialize with Xavier (optimal for tanh)
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
            
            layers.append(layer)
            
            # Batch normalization (helps with tanh saturation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Tanh activation
            if tanh_type == 'standard':
                layers.append(StandardTanh(scale))
            elif tanh_type == 'stable':
                layers.append(StableTanh(scale))
            elif tanh_type == 'fast':
                layers.append(FastTanh(scale))
            elif tanh_type == 'learnable':
                layers.append(LearnableTanh(scale, learnable_scale=True))
            
            # Dropout
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression, appropriate for classification)
        output_layer = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_normal_(output_layer.weight, gain=1.0)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.layers = nn.Sequential(*layers)
        
        # Store layer types for analysis
        self.layer_types = []
        for layer in layers:
            if isinstance(layer, (StandardTanh, StableTanh, FastTanh, LearnableTanh)):
                self.layer_types.append(type(layer).__name__)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if self.use_residual and len(self.hidden_dims) > 1:
            # Implement residual connections
            residual = x
            for i, layer in enumerate(self.layers):
                x = layer(x)
                # Add residual connection every 2 layers
                if i % 2 == 1 and i > 0 and x.shape == residual.shape:
                    x = x + residual
                    residual = x
            return x
        else:
            return self.layers(x)
    
    def get_layer_activations(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Get activations from each layer for analysis."""
        activations = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            
            if isinstance(layer, (StandardTanh, StableTanh, FastTanh, LearnableTanh)):
                # Compute saturation metrics
                tanh_output = current.detach()
                mean_abs = torch.mean(torch.abs(tanh_output)).item()
                saturation_ratio = torch.mean((torch.abs(tanh_output) > 0.9).float()).item()
                dead_ratio = torch.mean((torch.abs(tanh_output) < 0.1).float()).item()
                
                activations.append({
                    'type': layer.__class__.__name__,
                    'output': tanh_output,
                    'scale': layer.scale if hasattr(layer, 'scale') else None,
                    'mean_abs': mean_abs,
                    'saturation_ratio': saturation_ratio,
                    'dead_ratio': dead_ratio,
                    'gradient_norm': torch.mean(layer.derivative(torch.zeros_like(tanh_output))).item()
                })
        
        return activations
    
    def analyze_gradient_flow(self, x: torch.Tensor) -> Dict[str, Any]:
        """Analyze gradient flow through the network."""
        # Enable gradient tracking
        x.requires_grad_(True)
        
        # Forward pass
        output = self.forward(x)
        
        # Create dummy loss
        dummy_loss = output.sum()
        
        # Backward pass
        dummy_loss.backward()
        
        # Collect gradient statistics
        gradient_stats = {
            'input_grad_norm': x.grad.norm().item() if x.grad is not None else 0,
            'layer_gradients': []
        }
        
        # Collect gradients for each parameter
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradient_stats['layer_gradients'].append({
                    'name': name,
                    'grad_norm': param.grad.norm().item(),
                    'grad_mean': param.grad.mean().item(),
                    'grad_std': param.grad.std().item()
                })
        
        # Disable gradient tracking
        x.requires_grad_(False)
        
        return gradient_stats

# ==================== DATASET IMPLEMENTATIONS ====================
class TanhDataset(Dataset):
    """Base dataset class for tanh experiments."""
    
    def __init__(self, 
                 num_samples: int = 1000,
                 input_dim: int = 10,
                 noise_level: float = 0.1,
                 function_type: str = 'tanh_based',
                 scale: float = 1.0):
        """
        Args:
            num_samples: Number of samples
            input_dim: Input dimension
            noise_level: Noise level in targets
            function_type: Type of target function
            scale: Scale for tanh-based functions
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.noise_level = noise_level
        self.function_type = function_type
        self.scale = scale
        
        # Generate data
        self.X, self.y = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic dataset."""
        # Generate random input features with zero mean
        X = torch.randn(self.num_samples, self.input_dim)
        
        if self.function_type == 'tanh_based':
            # Target is tanh of linear combination
            true_weights = torch.randn(self.input_dim) * 0.5
            true_bias = torch.randn(1) * 0.5
            linear_output = X @ true_weights + true_bias
            y = torch.tanh(linear_output * self.scale)
            
        elif self.function_type == 'deep_tanh':
            # Deep tanh network as target
            true_weights1 = torch.randn(self.input_dim, self.input_dim // 2) * 0.5
            true_weights2 = torch.randn(self.input_dim // 2, self.input_dim // 4) * 0.5
            true_weights3 = torch.randn(self.input_dim // 4, 1) * 0.5
            
            h1 = torch.tanh(X @ true_weights1)
            h2 = torch.tanh(h1 @ true_weights2)
            y = torch.tanh(h2 @ true_weights3)
            
        elif self.function_type == 'classification':
            # Binary classification with tanh decision boundary
            true_weights = torch.randn(self.input_dim)
            true_bias = torch.randn(1)
            logits = X @ true_weights + true_bias
            probabilities = (torch.tanh(logits * self.scale) + 1) / 2  # Map to [0,1]
            y = (probabilities > 0.5).float()
            
        elif self.function_type == 'regression':
            # Simple regression
            true_weights = torch.randn(self.input_dim)
            true_bias = torch.randn(1)
            y = X @ true_weights + true_bias
            # Normalize to [-1, 1] range
            y = 2 * (y - y.min()) / (y.max() - y.min()) - 1
        
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
        
        # Add noise
        if self.noise_level > 0:
            noise = torch.randn_like(y) * self.noise_level
            y = torch.clamp(y + noise, -1, 1)
        
        return X, y
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FashionMNISTTanhDataset(Dataset):
    """FashionMNIST dataset for tanh classification experiments."""
    
    def __init__(self, train: bool = True, download: bool = True):
        self.dataset = datasets.FashionMNIST(
            root='./data',
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        )
        # Convert labels to [-1, 1] range for tanh output
        # Even classes -> -1, odd classes -> 1 (binary classification)
        self.labels = (self.dataset.targets % 2).float() * 2 - 1  # Map to {-1, 1}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label = self.labels[idx]
        return img.view(-1), label

class CIFAR10TanhDataset(Dataset):
    """CIFAR-10 dataset for tanh experiments."""
    
    def __init__(self, train: bool = True, download: bool = True):
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
        # Convert to binary classification (vehicles vs animals)
        vehicle_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
        animal_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
        
        binary_labels = []
        for target in self.dataset.targets:
            if target in vehicle_classes:
                binary_labels.append(-1.0)  # Vehicles -> -1
            else:
                binary_labels.append(1.0)   # Animals -> 1
        
        self.labels = torch.FloatTensor(binary_labels)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label = self.labels[idx]
        return img.view(-1), label

# ==================== TRAINING AND EVALUATION ====================
class TanhTrainer:
    """Training framework for tanh networks."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 loss_type: str = 'mse',
                 weight_decay: float = 1e-5):
        """
        Args:
            model: Neural network model
            device: Training device
            learning_rate: Learning rate
            loss_type: Loss function type ('mse', 'mae', 'huber', 'tanh_mse')
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type
        
        # Define loss function
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_type == 'huber':
            self.criterion = nn.SmoothL1Loss()
        elif loss_type == 'tanh_mse':
            self.criterion = self._tanh_mse_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Optimizer (Adam works well with tanh)
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Tanh mathematics helper
        self.math = TanhMathematics()
    
    def _tanh_mse_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE loss with tanh output scaling awareness."""
        # Scale targets to [-1, 1] if needed
        if targets.min() >= 0 and targets.max() <= 1:
            targets = targets * 2 - 1
        
        return nn.functional.mse_loss(predictions, targets)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            
            # Ensure output is in correct range for loss
            if self.loss_type == 'tanh_mse' and output.min() > -1.1 and output.max() < 1.1:
                # Output is already in tanh range
                pass
            elif target.min() >= -1 and target.max() <= 1:
                # Targets are in tanh range, ensure output matches
                output = torch.tanh(output)  # Apply tanh if needed
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for tanh networks (helps with saturation)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute accuracy for classification (-1 vs 1)
            if target.dim() == 1 or target.shape[1] == 1:
                predictions = torch.sign(output)  # -1 or 1
                correct = (predictions == target).sum().item()
                total_correct += correct
            
            total_loss += loss.item()
            total_samples += len(data)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
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
                
                # Ensure output is in correct range
                if self.loss_type == 'tanh_mse' and output.min() > -1.1 and output.max() < 1.1:
                    pass
                elif target.min() >= -1 and target.max() <= 1:
                    output = torch.tanh(output)
                
                loss = self.criterion(output, target)
                
                # Compute accuracy
                if target.dim() == 1 or target.shape[1] == 1:
                    predictions = torch.sign(output)
                    correct = (predictions == target).sum().item()
                    total_correct += correct
                
                total_loss += loss.item()
                total_samples += len(data)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              early_stopping_patience: int = 10):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_tanh_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training History - Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        if self.train_accuracies and self.val_accuracies:
            axes[0, 1].plot(self.train_accuracies, label='Train Acc', alpha=0.8)
            axes[0, 1].plot(self.val_accuracies, label='Val Acc', alpha=0.8)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Training History - Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(self.learning_rates, alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss ratio (train/val)
        if len(self.train_losses) > 1 and len(self.val_losses) > 1:
            loss_ratio = [t/v if v > 0 else 1 for t, v in zip(self.train_losses, self.val_losses)]
            axes[1, 1].plot(loss_ratio, alpha=0.8)
            axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Train/Val Loss Ratio')
            axes[1, 1].set_title('Overfitting Indicator')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Tanh Network Training History', fontsize=14)
        plt.tight_layout()
        plt.show()

# ==================== VISUALIZATION TOOLS ====================
class TanhVisualizer:
    """Visualization tools for tanh function."""
    
    @staticmethod
    def plot_tanh_function(tanh_func: BaseTanh,
                          x_range: Tuple[float, float] = (-5, 5),
                          num_points: int = 1000,
                          show_derivatives: bool = True,
                          show_hyperbolic: bool = False):
        """Plot tanh function and its properties."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        # Compute values
        y = tanh_func.forward(x)
        y_prime = tanh_func.derivative(x)
        y_double_prime = TanhMathematics().tanh_second_derivative(x * tanh_func.scale)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Tanh function
        axes[0, 0].plot(x.numpy(), y.numpy(), 'b-', linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('tanh(x)')
        axes[0, 0].set_title(f'Hyperbolic Tangent (scale={tanh_func.scale})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # First derivative
        axes[0, 1].plot(x.numpy(), y_prime.numpy(), 'g-', linewidth=2)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel("tanh'(x)")
        axes[0, 1].set_title('First Derivative')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Second derivative
        axes[1, 0].plot(x.numpy(), y_double_prime.numpy(), 'r-', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel("tanh''(x)")
        axes[1, 0].set_title('Second Derivative')
        axes[1, 0].grid(True, alpha=0.3)
        
        # All together
        axes[1, 1].plot(x.numpy(), y.numpy(), 'b-', label='tanh(x)', linewidth=2)
        axes[1, 1].plot(x.numpy(), y_prime.numpy(), 'g-', label="tanh'(x)", linewidth=2, alpha=0.7)
        axes[1, 1].plot(x.numpy(), y_double_prime.numpy(), 'r-', label="tanh''(x)", linewidth=2, alpha=0.7)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('All Functions Together')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        if show_hyperbolic:
            # Add hyperbolic functions
            math = TanhMathematics()
            sinh_vals = math.sinh(x * tanh_func.scale).numpy()
            cosh_vals = math.cosh(x * tanh_func.scale).numpy()
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(x.numpy(), sinh_vals, 'b-', label='sinh(x)', linewidth=2)
            ax2.plot(x.numpy(), cosh_vals, 'r-', label='cosh(x)', linewidth=2)
            ax2.plot(x.numpy(), y.numpy(), 'g-', label='tanh(x)', linewidth=2)
            ax2.set_xlabel('x')
            ax2.set_ylabel('Value')
            ax2.set_title('Hyperbolic Functions Family')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.show()
        
        plt.suptitle(f'Tanh Analysis - {tanh_func.__class__.__name__}', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_multiple_tanh(tanh_funcs: List[BaseTanh],
                          x_range: Tuple[float, float] = (-5, 5),
                          num_points: int = 1000):
        """Compare multiple tanh implementations."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        plt.figure(figsize=(10, 6))
        
        for tanh_func in tanh_funcs:
            y = tanh_func.forward(x)
            label = f"{tanh_func.__class__.__name__}"
            if hasattr(tanh_func, 'scale'):
                label += f" (scale={tanh_func.scale})"
            plt.plot(x.numpy(), y.numpy(), label=label, linewidth=2)
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('tanh(x)')
        plt.title('Comparison of Tanh Implementations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_activation_statistics(activations: List[Dict[str, torch.Tensor]]):
        """Plot statistics of tanh activations across layers."""
        if not activations:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Mean absolute activation per layer
        layer_indices = range(1, len(activations) + 1)
        mean_abs = [a['mean_abs'] for a in activations]
        
        axes[0, 0].bar(layer_indices, mean_abs, alpha=0.7)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean |tanh(x)|')
        axes[0, 0].set_title('Activation Magnitude per Layer')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Saturation ratio
        saturation_ratios = [a['saturation_ratio'] for a in activations]
        
        axes[0, 1].bar(layer_indices, saturation_ratios, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Saturation Ratio')
        axes[0, 1].set_title('Percentage of |tanh(x)| > 0.9')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dead neuron ratio
        dead_ratios = [a['dead_ratio'] for a in activations]
        
        axes[1, 0].bar(layer_indices, dead_ratios, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Dead Neuron Ratio')
        axes[1, 0].set_title('Percentage of |tanh(x)| < 0.1')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Activation histograms for first and last layer
        if len(activations) >= 2:
            axes[1, 1].hist(activations[0]['output'].flatten().numpy(), 
                          bins=50, alpha=0.5, label='First Layer', density=True)
            axes[1, 1].hist(activations[-1]['output'].flatten().numpy(), 
                          bins=50, alpha=0.5, label='Last Layer', density=True)
            axes[1, 1].set_xlabel('Activation Value')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Activation Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Tanh Activation Statistics Across Layers', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_gradient_analysis(model: TanhNetwork, 
                              dataset: Dataset,
                              num_samples: int = 100):
        """Analyze and visualize gradients in tanh network."""
        # Get sample data
        loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
        data, _ = next(iter(loader))
        
        # Analyze gradient flow
        gradient_stats = model.analyze_gradient_flow(data)
        
        # Plot gradient statistics
        if gradient_stats['layer_gradients']:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Gradient norms per layer
            layer_names = [g['name'] for g in gradient_stats['layer_gradients']]
            grad_norms = [g['grad_norm'] for g in gradient_stats['layer_gradients']]
            
            axes[0].bar(range(len(layer_names)), grad_norms, alpha=0.7)
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel('Gradient Norm')
            axes[0].set_title('Gradient Norms per Layer')
            axes[0].set_xticks(range(len(layer_names)))
            axes[0].set_xticklabels([n.split('.')[0] for n in layer_names], rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # Gradient means
            grad_means = [g['grad_mean'] for g in gradient_stats['layer_gradients']]
            
            axes[1].bar(range(len(layer_names)), grad_means, alpha=0.7, color='green')
            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel('Gradient Mean')
            axes[1].set_title('Gradient Means per Layer')
            axes[1].set_xticks(range(len(layer_names)))
            axes[1].set_xticklabels([n.split('.')[0] for n in layer_names], rotation=45)
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle('Gradient Flow Analysis in Tanh Network', fontsize=14)
            plt.tight_layout()
            plt.show()
        
        return gradient_stats

# ==================== TANH ANALYSIS AND BENCHMARKING ====================
class TanhAnalyzer:
    """Analyzer for tanh function properties."""
    
    def __init__(self, tanh_func: BaseTanh):
        self.tanh_func = tanh_func
        self.math = TanhMathematics()
    
    def analyze_range(self, 
                     x_range: Tuple[float, float] = (-10, 10),
                     num_points: int = 1000) -> Dict[str, Any]:
        """Analyze tanh properties over a range."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        y = self.tanh_func.forward(x)
        y_prime = self.tanh_func.derivative(x)
        
        analysis = {
            'input_range': (x.min().item(), x.max().item()),
            'output_range': (y.min().item(), y.max().item()),
            'output_mean': y.mean().item(),
            'output_std': y.std().item(),
            'max_derivative': y_prime.max().item(),
            'mean_derivative': y_prime.mean().item(),
            'saturation_points': {
                'negative': self._find_saturation_point(x, y, direction='negative'),
                'positive': self._find_saturation_point(x, y, direction='positive')
            },
            'zero_crossing': self._find_zero_crossing(x, y)
        }
        
        return analysis
    
    def _find_saturation_point(self, 
                              x: torch.Tensor, 
                              y: torch.Tensor, 
                              direction: str = 'positive') -> float:
        """Find where tanh saturates (reaches ±0.99)."""
        threshold = 0.99 if direction == 'positive' else -0.99
        
        if direction == 'positive':
            mask = y > threshold
            if mask.any():
                return x[mask].min().item()
        else:
            mask = y < threshold
            if mask.any():
                return x[mask].max().item()
        
        return float('nan')
    
    def _find_zero_crossing(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Find where tanh crosses zero."""
        # Find sign change
        signs = torch.sign(y)
        for i in range(1, len(signs)):
            if signs[i-1] != signs[i]:
                # Linear interpolation for precise zero crossing
                x1, x2 = x[i-1].item(), x[i].item()
                y1, y2 = y[i-1].item(), y[i].item()
                return x1 - y1 * (x2 - x1) / (y2 - y1)
        
        return 0.0  # Default to 0 if no crossing found
    
    def analyze_gradient_properties(self,
                                  x_values: torch.Tensor) -> Dict[str, Any]:
        """Analyze gradient properties of tanh."""
        gradients = self.tanh_func.derivative(x_values)
        
        # Vanishing gradient analysis
        vanishing_mask = gradients < 1e-7
        vanishing_count = vanishing_mask.sum().item()
        
        # Exploding gradient analysis
        exploding_mask = gradients > 10.0
        exploding_count = exploding_mask.sum().item()
        
        analysis = {
            'total_samples': len(x_values),
            'vanishing_gradients': vanishing_count,
            'vanishing_percentage': vanishing_count / len(x_values) * 100,
            'exploding_gradients': exploding_count,
            'exploding_percentage': exploding_count / len(x_values) * 100,
            'mean_gradient': gradients.mean().item(),
            'median_gradient': gradients.median().item(),
            'std_gradient': gradients.std().item(),
            'gradient_histogram': np.histogram(gradients.numpy(), bins=50)
        }
        
        return analysis
    
    def compare_with_sigmoid(self,
                            x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compare tanh with sigmoid."""
        tanh_vals = self.tanh_func.forward(x)
        tanh_grads = self.tanh_func.derivative(x)
        
        # Compute sigmoid values
        sigmoid_vals = 1 / (1 + torch.exp(-x * self.tanh_func.scale))
        sigmoid_grads = sigmoid_vals * (1 - sigmoid_vals)
        
        # Convert sigmoid to tanh range: tanh(x) = 2σ(2x) - 1
        scaled_sigmoid = 2 * (1 / (1 + torch.exp(-2 * x * self.tanh_func.scale))) - 1
        
        return {
            'tanh': tanh_vals,
            'tanh_gradient': tanh_grads,
            'sigmoid': sigmoid_vals,
            'sigmoid_gradient': sigmoid_grads,
            'scaled_sigmoid': scaled_sigmoid,
            'difference': tanh_vals - scaled_sigmoid
        }

class TanhBenchmark:
    """Benchmarking framework for tanh implementations."""
    
    @staticmethod
    def benchmark_forward_pass(tanh_funcs: List[BaseTanh],
                              batch_sizes: List[int] = [1, 10, 100, 1000, 10000],
                              input_dim: int = 100):
        """Benchmark forward pass performance."""
        import time
        
        print("=" * 60)
        print("TANH FORWARD PASS BENCHMARK")
        print("=" * 60)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}, Input dim: {input_dim}")
            print("-" * 50)
            
            # Create random input
            x = torch.randn(batch_size, input_dim)
            
            for tanh_func in tanh_funcs:
                # Warm up
                for _ in range(10):
                    _ = tanh_func.forward(x)
                
                # Benchmark
                start_time = time.time()
                for _ in range(100):
                    _ = tanh_func.forward(x)
                elapsed = time.time() - start_time
                
                # Store result
                func_name = tanh_func.__class__.__name__
                if func_name not in results:
                    results[func_name] = []
                results[func_name].append((batch_size, elapsed / 100))
                
                print(f"  {func_name:20s}: {elapsed/100*1000:6.2f} ms per forward pass")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        for func_name, data in results.items():
            batch_sizes = [d[0] for d in data]
            times = [d[1] * 1000 for d in data]  # Convert to ms
            plt.plot(batch_sizes, times, 'o-', label=func_name, linewidth=2)
        
        plt.xscale('log')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Forward Pass (ms)')
        plt.title('Tanh Forward Pass Performance')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        plt.show()
    
    @staticmethod
    def benchmark_numerical_stability(tanh_funcs: List[BaseTanh],
                                     extreme_values: List[float] = [-100, -50, -20, -10, 10, 20, 50, 100]):
        """Benchmark numerical stability for extreme values."""
        print("\n" + "=" * 60)
        print("NUMERICAL STABILITY BENCHMARK")
        print("=" * 60)
        
        for value in extreme_values:
            print(f"\nInput value: {value}")
            print("-" * 40)
            
            x = torch.tensor([value])
            
            for tanh_func in tanh_funcs:
                try:
                    y = tanh_func.forward(x)
                    grad = tanh_func.derivative(x)
                    
                    # Check if value is correct
                    expected = math.tanh(value * tanh_func.scale) if hasattr(tanh_func, 'scale') else math.tanh(value)
                    error = abs(y.item() - expected)
                    
                    print(f"  {tanh_func.__class__.__name__:20s}: "
                          f"output = {y.item():9.6f}, gradient = {grad.item():9.2e}, "
                          f"error = {error:9.2e}")
                except Exception as e:
                    print(f"  {tanh_func.__class__.__name__:20s}: ERROR - {str(e)}")
    
    @staticmethod
    def benchmark_gradient_properties(network: TanhNetwork,
                                     dataset: Dataset,
                                     num_samples: int = 1000):
        """Benchmark gradient properties in tanh network."""
        print("\n" + "=" * 60)
        print("GRADIENT PROPERTIES BENCHMARK")
        print("=" * 60)
        
        # Create data loader
        loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
        data, targets = next(iter(loader))
        
        # Forward pass
        network.eval()
        with torch.no_grad():
            output = network(data)
        
        # Get layer activations
        activations = network.get_layer_activations(data)
        
        print(f"\nNetwork has {len(activations)} tanh layers")
        
        overall_stats = {
            'total_neurons': 0,
            'saturated_neurons': 0,
            'dead_neurons': 0,
            'mean_gradient': 0.0
        }
        
        for i, activation in enumerate(activations):
            layer_output = activation['output']
            total_neurons = layer_output.numel()
            saturated = (torch.abs(layer_output) > 0.9).sum().item()
            dead = (torch.abs(layer_output) < 0.1).sum().item()
            
            overall_stats['total_neurons'] += total_neurons
            overall_stats['saturated_neurons'] += saturated
            overall_stats['dead_neurons'] += dead
            overall_stats['mean_gradient'] += activation.get('gradient_norm', 0)
            
            print(f"\nLayer {i+1} ({activation['type']}):")
            print(f"  Total neurons: {total_neurons}")
            print(f"  Saturated (|act| > 0.9): {saturated} ({saturated/total_neurons*100:.1f}%)")
            print(f"  Dead (|act| < 0.1): {dead} ({dead/total_neurons*100:.1f}%)")
            print(f"  Mean gradient: {activation.get('gradient_norm', 0):.6f}")
        
        print(f"\nOverall Statistics:")
        print(f"  Total neurons: {overall_stats['total_neurons']}")
        print(f"  Overall saturation: {overall_stats['saturated_neurons']/overall_stats['total_neurons']*100:.1f}%")
        print(f"  Overall dead neurons: {overall_stats['dead_neurons']/overall_stats['total_neurons']*100:.1f}%")
        print(f"  Average gradient: {overall_stats['mean_gradient']/len(activations):.6f}")
        
        return activations, overall_stats

# ==================== MAIN DEMONSTRATION ====================
def demonstrate_tanh_mathematics():
    """Demonstrate mathematical properties of tanh."""
    print("=" * 60)
    print("HYPERBOLIC TANGENT MATHEMATICAL PROPERTIES")
    print("=" * 60)
    
    math = TanhMathematics()
    
    # Test points
    test_points = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    
    print("\nTanh values at test points:")
    print("-" * 40)
    for x in test_points:
        x_tensor = torch.tensor([x])
        tanh_val = math.tanh(x_tensor).item()
        deriv = math.tanh_derivative(x_tensor).item()
        print(f"tanh({x:4.1f}) = {tanh_val:7.4f}, tanh'({x:4.1f}) = {deriv:7.4f}")
    
    print("\nKey Properties:")
    print("1. Range: (-1, 1) - Zero-centered!")
    print("2. tanh(0) = 0 (zero at origin)")
    print("3. tanh(-x) = -tanh(x) (odd function)")
    print("4. lim(x→∞) tanh(x) = 1")
    print("5. lim(x→-∞) tanh(x) = -1")
    print("6. tanh'(x) = 1 - tanh²(x) (maximum at x=0, tanh'(0)=1)")
    print("7. Relation to sigmoid: tanh(x) = 2σ(2x) - 1")
    
    # Demonstrate hyperbolic identities
    print("\nHyperbolic Identities:")
    print("-" * 40)
    x = torch.tensor([1.0])
    sinh_val = math.sinh(x).item()
    cosh_val = math.cosh(x).item()
    identity = cosh_val ** 2 - sinh_val ** 2
    
    print(f"sinh(1) = {sinh_val:.6f}")
    print(f"cosh(1) = {cosh_val:.6f}")
    print(f"cosh²(1) - sinh²(1) = {identity:.6f} (should be 1)")
    
    # Demonstrate inverse
    print("\nInverse Tanh (Arctanh):")
    print("-" * 40)
    for y in [-0.9, -0.5, 0, 0.5, 0.9]:
        y_tensor = torch.tensor([y])
        x_inv = math.tanh_inverse(y_tensor).item()
        print(f"tanh⁻¹({y}) = {x_inv:.6f}")

def demonstrate_tanh_implementations():
    """Demonstrate different tanh implementations."""
    print("\n" + "=" * 60)
    print("TANH IMPLEMENTATIONS")
    print("=" * 60)
    
    # Create different tanh implementations
    tanh_impls = [
        StandardTanh(scale=1.0),
        StandardTanh(scale=2.0),
        StableTanh(scale=1.0),
        FastTanh(scale=1.0, approximation_type='piecewise'),
        FastTanh(scale=1.0, approximation_type='hard'),
        FastTanh(scale=1.0, approximation_type='softsign'),
        LearnableTanh(scale=1.0, learnable_scale=True)
    ]
    
    # Visualize them
    visualizer = TanhVisualizer()
    visualizer.plot_multiple_tanh(tanh_impls, x_range=(-5, 5))
    
    # Analyze properties at x=0
    print("\nProperties at x=0 (should be exactly 0 for all):")
    print("-" * 60)
    for impl in tanh_impls:
        props = impl.analyze_at_point(0)
        print(f"\n{impl.__class__.__name__}:")
        print(f"  Value: {props.value:.6f} (should be 0.000000)")
        print(f"  Derivative: {props.derivative:.6f} (should be {impl.scale:.6f})")
        print(f"  Hyperbolic Identity: {props.hyperbolic_identity:.6f} (should be 1.000000)")

def train_tanh_network_example():
    """Train a tanh network on a classification task."""
    print("\n" + "=" * 60)
    print("TANH NETWORK TRAINING EXAMPLE")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset (using FashionMNIST for binary classification)
    print("\nLoading FashionMNIST dataset for binary classification...")
    try:
        dataset = FashionMNISTTanhDataset(train=True)
        test_dataset = FashionMNISTTanhDataset(train=False)
    except:
        print("FashionMNIST not available, using synthetic dataset...")
        dataset = TanhDataset(num_samples=5000, input_dim=20, 
                            function_type='classification', scale=1.0)
        test_dataset = TanhDataset(num_samples=1000, input_dim=20,
                                 function_type='classification', scale=1.0)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create tanh network
    print("\nCreating tanh network...")
    model = TanhNetwork(
        input_dim=784 if hasattr(dataset, 'dataset') else 20,
        hidden_dims=[256, 128, 64],
        output_dim=1,
        tanh_type='standard',
        scale=1.0,
        dropout=0.3,
        use_batch_norm=True,
        use_residual=True
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = TanhTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        loss_type='tanh_mse',
        weight_decay=1e-5
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        early_stopping_patience=10
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Test
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"\nTest Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Analyze activations
    print("\nAnalyzing network activations...")
    visualizer = TanhVisualizer()
    
    # Get sample data for analysis
    sample_data, _ = next(iter(test_loader))
    sample_data = sample_data[:100]  # Use 100 samples
    
    activations = model.get_layer_activations(sample_data)
    visualizer.plot_activation_statistics(activations)
    
    # Analyze gradients
    gradient_stats = visualizer.plot_gradient_analysis(model, test_dataset, 100)
    
    return model, trainer

def benchmark_tanh_performance():
    """Benchmark tanh implementations."""
    print("\n" + "=" * 60)
    print("TANH PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create tanh implementations
    tanh_funcs = [
        StandardTanh(scale=1.0),
        StableTanh(scale=1.0),
        FastTanh(scale=1.0, approximation_type='piecewise'),
        FastTanh(scale=1.0, approximation_type='hard'),
        FastTanh(scale=1.0, approximation_type='softsign')
    ]
    
    # Run benchmarks
    benchmark = TanhBenchmark()
    
    # Forward pass benchmark
    benchmark.benchmark_forward_pass(tanh_funcs)
    
    # Numerical stability benchmark
    benchmark.benchmark_numerical_stability(tanh_funcs)
    
    # Create analyzer for detailed analysis
    print("\n" + "=" * 60)
    print("TANH PROPERTY ANALYSIS")
    print("=" * 60)
    
    analyzer = TanhAnalyzer(tanh_funcs[0])
    
    # Generate test data
    x_test = torch.randn(10000) * 5  # Values in approx [-15, 15]
    
    # Analyze gradient properties
    gradient_analysis = analyzer.analyze_gradient_properties(x_test)
    print(f"\nGradient Properties Analysis:")
    print(f"  Vanishing gradients (<1e-7): {gradient_analysis['vanishing_gradients']} "
          f"({gradient_analysis['vanishing_percentage']:.1f}%)")
    print(f"  Mean gradient: {gradient_analysis['mean_gradient']:.6f}")
    print(f"  Median gradient: {gradient_analysis['median_gradient']:.6f}")
    print(f"  Gradient std: {gradient_analysis['std_gradient']:.6f}")
    
    # Compare with sigmoid
    print("\nComparison with Sigmoid:")
    print("-" * 40)
    comparison = analyzer.compare_with_sigmoid(torch.tensor([-2, -1, 0, 1, 2]))
    
    for i, x in enumerate([-2, -1, 0, 1, 2]):
        tanh_val = comparison['tanh'][i].item()
        sigmoid_val = comparison['sigmoid'][i].item()
        scaled_sigmoid = comparison['scaled_sigmoid'][i].item()
        
        print(f"x={x:2d}: tanh={tanh_val:6.3f}, σ(x)={sigmoid_val:6.3f}, "
              f"2σ(2x)-1={scaled_sigmoid:6.3f}, diff={tanh_val-scaled_sigmoid:8.3e}")

def demonstrate_tanh_applications():
    """Demonstrate practical applications of tanh."""
    print("\n" + "=" * 60)
    print("TANH PRACTICAL APPLICATIONS")
    print("=" * 60)
    
    print("\n1. Neural Network Activation:")
    print("   • Zero-centered: Better gradient flow than sigmoid")
    print("   • Range (-1,1): Can represent both positive and negative features")
    print("   • Smooth: Everywhere differentiable")
    
    print("\n2. Recurrent Neural Networks (RNNs):")
    print("   • Historically popular in vanilla RNNs")
    print("   • Less prone to vanishing gradients than sigmoid (but still has issues)")
    print("   • Used in LSTM/GRU gates along with sigmoid")
    
    print("\n3. Hyperbolic Geometry:")
    print("   • Maps real numbers to hyperbolic space")
    print("   • Used in hyperbolic neural networks")
    print("   • Preserves certain geometric properties")
    
    print("\n4. Normalization and Scaling:")
    print("   • Can be used to normalize data to (-1,1) range")
    print("   • Used in attention mechanisms for soft weighting")
    
    print("\n5. Physics and Engineering:")
    print("   • Models wave propagation")
    print("   • Used in solutions to differential equations")
    print("   • Appears in special relativity")
    
    # Demonstrate RNN with tanh
    print("\n" + "-" * 40)
    print("Simple RNN with Tanh Example:")
    
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
            self.h2o = nn.Linear(hidden_size, output_size)
            self.tanh = StandardTanh()
        
        def forward(self, input_seq):
            batch_size = input_seq.size(0)
            seq_len = input_seq.size(1)
            
            # Initialize hidden state
            hidden = torch.zeros(batch_size, self.hidden_size)
            
            outputs = []
            for t in range(seq_len):
                # Combine input and hidden state
                combined = torch.cat((input_seq[:, t, :], hidden), dim=1)
                
                # Update hidden state with tanh
                hidden = self.tanh(self.i2h(combined))
                
                # Generate output
                output = self.h2o(hidden)
                outputs.append(output.unsqueeze(1))
            
            return torch.cat(outputs, dim=1), hidden
    
    # Create a simple RNN
    rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
    
    # Test with random input
    input_seq = torch.randn(4, 8, 10)  # batch=4, seq_len=8, input_size=10
    output_seq, final_hidden = rnn(input_seq)
    
    print(f"RNN Input shape: {input_seq.shape}")
    print(f"RNN Output shape: {output_seq.shape}")
    print(f"Final hidden state shape: {final_hidden.shape}")
    print(f"Hidden state range: [{final_hidden.min():.3f}, {final_hidden.max():.3f}]")

def main():
    """Main demonstration function."""
    print("HYPERBOLIC TANGENT ACTIVATION FUNCTION IMPLEMENTATION")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demonstrate mathematical properties
    demonstrate_tanh_mathematics()
    
    # Demonstrate different implementations
    demonstrate_tanh_implementations()
    
    # Train tanh network
    model, trainer = train_tanh_network_example()
    
    # Benchmark performance
    benchmark_tanh_performance()
    
    # Demonstrate applications
    demonstrate_tanh_applications()
    
    print("\n" + "=" * 60)
    print("TANH IMPLEMENTATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Implemented:")
    print("1. Standard Tanh: tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)")
    print("2. Stable Tanh: Numerically stable implementation")
    print("3. Fast Tanh: Multiple approximations for performance")
    print("4. Learnable Tanh: With learnable scale, shift, and skew")
    print("5. Complete mathematical analysis tools")
    print("6. Tanh network with batch norm and residual connections")
    print("7. Gradient flow analysis and visualization")
    print("8. Training framework for tanh networks")
    print("9. Benchmarking and comparison tools")
    print("10. Practical applications demonstration")

if __name__ == "__main__":
    main()