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

# ==================== SIGMOID MATHEMATICAL FOUNDATION ====================
class SigmoidMathematics:
    """
    Mathematical foundations of the Sigmoid function.
    Implements all mathematical properties and theorems.
    """
    
    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        """
        Sigmoid function: σ(x) = 1 / (1 + e^{-x})
        
        Mathematical Properties:
        1. Range: (0, 1)
        2. Domain: (-∞, ∞)
        3. Point of symmetry: (0, 0.5)
        4. Derivative: σ'(x) = σ(x) * (1 - σ(x))
        5. Inverse: σ^{-1}(y) = ln(y / (1 - y))
        """
        # Standard implementation
        return 1 / (1 + torch.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: torch.Tensor, sigmoid_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
        
        Can be computed either from:
        1. Direct computation: σ(x) * (1 - σ(x))
        2. Using precomputed sigmoid value
        """
        if sigmoid_value is None:
            sigmoid_value = SigmoidMathematics.sigmoid(x)
        return sigmoid_value * (1 - sigmoid_value)
    
    @staticmethod
    def sigmoid_inverse(y: torch.Tensor) -> torch.Tensor:
        """
        Inverse sigmoid (logit function): σ^{-1}(y) = ln(y / (1 - y))
        
        Properties:
        1. Domain: (0, 1)
        2. Range: (-∞, ∞)
        3. Used in logistic regression
        """
        # Clip to prevent log(0) or division by 0
        y = torch.clamp(y, 1e-7, 1 - 1e-7)
        return torch.log(y / (1 - y))
    
    @staticmethod
    def sigmoid_second_derivative(x: torch.Tensor) -> torch.Tensor:
        """
        Second derivative of sigmoid: σ''(x) = σ(x) * (1 - σ(x)) * (1 - 2σ(x))
        """
        sig = SigmoidMathematics.sigmoid(x)
        return sig * (1 - sig) * (1 - 2 * sig)
    
    @staticmethod
    def sigmoid_series_expansion(x: torch.Tensor, terms: int = 10) -> torch.Tensor:
        """
        Taylor series expansion of sigmoid around 0.
        
        σ(x) ≈ 1/2 + x/4 - x³/48 + x⁵/480 - 17x⁷/80640 + ...
        """
        result = torch.ones_like(x) * 0.5
        
        # Taylor series coefficients
        coefficients = [1/4, 0, -1/48, 0, 1/480, 0, -17/80640]
        
        for i, coeff in enumerate(coefficients[:terms]):
            power = i + 1
            if coeff != 0:
                result += coeff * (x ** power)
        
        return result
    
    @staticmethod
    def logistic_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Logistic loss (binary cross-entropy):
        L = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
        """
        predictions = torch.clamp(predictions, 1e-7, 1 - 1e-7)
        return - (targets * torch.log(predictions) + 
                 (1 - targets) * torch.log(1 - predictions)).mean()
    
    @staticmethod
    def softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Softplus function: f(x) = log(1 + exp(βx)) / β
        Smooth approximation of ReLU, related to sigmoid.
        """
        return torch.log(1 + torch.exp(beta * x)) / beta
    
    @staticmethod
    def compute_gradient_flow(sigmoid_output: torch.Tensor, 
                            upstream_gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient flow through sigmoid for backpropagation.
        dL/dx = dL/dσ * σ'(x) = dL/dσ * σ(x) * (1 - σ(x))
        """
        return upstream_gradient * sigmoid_output * (1 - sigmoid_output)

# ==================== SIGMOID IMPLEMENTATION CLASSES ====================
@dataclass
class SigmoidProperties:
    """Mathematical properties of sigmoid at a specific point."""
    value: float
    derivative: float
    second_derivative: float
    curvature: float
    inflection_point: bool
    linear_approximation: float
    
    def __str__(self):
        return (f"Sigmoid Properties:\n"
                f"  Value: {self.value:.6f}\n"
                f"  Derivative: {self.derivative:.6f}\n"
                f"  Second Derivative: {self.second_derivative:.6f}\n"
                f"  Curvature: {self.curvature:.6f}\n"
                f"  Inflection Point: {self.inflection_point}\n"
                f"  Linear Approx: {self.linear_approximation:.6f}")

class BaseSigmoid(nn.Module):
    """Base class for all sigmoid implementations."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Controls steepness (higher = steeper)
        """
        super().__init__()
        self.temperature = temperature
        self.math = SigmoidMathematics()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute derivative at x."""
        raise NotImplementedError
    
    def analyze_at_point(self, x: float) -> SigmoidProperties:
        """Analyze sigmoid properties at a specific point."""
        x_tensor = torch.tensor([x])
        value = self.forward(x_tensor).item()
        deriv = self.derivative(x_tensor).item()
        second_deriv = self.math.sigmoid_second_derivative(x_tensor * self.temperature).item()
        
        # Compute curvature
        curvature = abs(second_deriv) / ((1 + deriv ** 2) ** 1.5)
        
        # Check if inflection point (second derivative = 0)
        inflection_point = abs(second_deriv) < 1e-6
        
        # Linear approximation around point
        linear_approx = 0.5 + 0.25 * x * self.temperature
        
        return SigmoidProperties(
            value=value,
            derivative=deriv,
            second_derivative=second_deriv,
            curvature=curvature,
            inflection_point=inflection_point,
            linear_approximation=linear_approx
        )

class StandardSigmoid(BaseSigmoid):
    """Standard sigmoid implementation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """σ(x) = 1 / (1 + exp(-temperature * x))"""
        return self.math.sigmoid(x * self.temperature)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """σ'(x) = σ(x) * (1 - σ(x)) * temperature"""
        sig = self.forward(x)
        return sig * (1 - sig) * self.temperature

class StableSigmoid(BaseSigmoid):
    """Numerically stable sigmoid implementation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable sigmoid computation.
        Avoids overflow for large positive/negative values.
        """
        # For positive x: 1 / (1 + exp(-x))
        # For negative x: exp(x) / (1 + exp(x))
        x_scaled = x * self.temperature
        
        # Use stable computation
        positive_mask = x_scaled >= 0
        negative_mask = ~positive_mask
        
        result = torch.zeros_like(x_scaled)
        
        # For positive values
        if positive_mask.any():
            exp_neg = torch.exp(-x_scaled[positive_mask])
            result[positive_mask] = 1 / (1 + exp_neg)
        
        # For negative values
        if negative_mask.any():
            exp_pos = torch.exp(x_scaled[negative_mask])
            result[negative_mask] = exp_pos / (1 + exp_pos)
        
        return result
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Stable derivative computation."""
        sig = self.forward(x)
        return sig * (1 - sig) * self.temperature

class FastSigmoid(BaseSigmoid):
    """Fast approximation of sigmoid for performance-critical applications."""
    
    def __init__(self, temperature: float = 1.0, approximation_type: str = 'piecewise'):
        super().__init__(temperature)
        self.approximation_type = approximation_type
        
        # Precompute piecewise approximation parameters
        self._init_piecewise_params()
    
    def _init_piecewise_params(self):
        """Initialize piecewise linear approximation parameters."""
        if self.approximation_type == 'piecewise':
            # Piecewise linear approximation points
            self.breakpoints = [-6.0, -4.0, -2.0, 2.0, 4.0, 6.0]
            self.slopes = [0.0, 0.006, 0.119, 0.881, 0.994, 1.0]
            self.intercepts = [0.0, 0.024, 0.238, 0.762, 1.024, 1.0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fast sigmoid approximation."""
        x_scaled = x * self.temperature
        
        if self.approximation_type == 'piecewise':
            return self._piecewise_approximation(x_scaled)
        elif self.approximation_type == 'rational':
            return self._rational_approximation(x_scaled)
        elif self.approximation_type == 'hard':
            return self._hard_sigmoid(x_scaled)
        else:
            return self.math.sigmoid(x_scaled)
    
    def _piecewise_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise linear approximation."""
        result = torch.zeros_like(x)
        
        # Left tail (x < -6)
        mask = x < self.breakpoints[0]
        result[mask] = 0.0
        
        # Middle segments
        for i in range(len(self.breakpoints) - 1):
            mask = (x >= self.breakpoints[i]) & (x < self.breakpoints[i + 1])
            if mask.any():
                result[mask] = self.slopes[i] * x[mask] + self.intercepts[i]
        
        # Right tail (x >= 6)
        mask = x >= self.breakpoints[-1]
        result[mask] = 1.0
        
        return result
    
    def _rational_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """Rational approximation: σ(x) ≈ 0.5 + x/(4√(x²+1))"""
        return 0.5 + x / (4 * torch.sqrt(x ** 2 + 1))
    
    def _hard_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Hard sigmoid: clip(0.5 + x/2, 0, 1)"""
        return torch.clamp(0.5 + x / 2, 0, 1)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate derivative."""
        if self.approximation_type == 'piecewise':
            return self._piecewise_derivative(x * self.temperature)
        elif self.approximation_type == 'rational':
            x_scaled = x * self.temperature
            denom = 4 * torch.pow(x_scaled ** 2 + 1, 1.5)
            return 1 / denom
        elif self.approximation_type == 'hard':
            x_scaled = x * self.temperature
            mask = (x_scaled > -1) & (x_scaled < 1)
            result = torch.zeros_like(x_scaled)
            result[mask] = 0.5 * self.temperature
            return result
        else:
            sig = self.forward(x)
            return sig * (1 - sig) * self.temperature
    
    def _piecewise_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Piecewise constant derivative."""
        result = torch.zeros_like(x)
        
        for i in range(len(self.breakpoints) - 1):
            mask = (x >= self.breakpoints[i]) & (x < self.breakpoints[i + 1])
            if mask.any():
                result[mask] = self.slopes[i] * self.temperature
        
        return result

class LearnableSigmoid(BaseSigmoid):
    """Sigmoid with learnable parameters."""
    
    def __init__(self, temperature: float = 1.0, learnable_temp: bool = True):
        """
        Args:
            temperature: Initial temperature
            learnable_temp: Whether temperature is learnable
        """
        super().__init__(temperature)
        
        # Make temperature learnable if requested
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        
        # Additional learnable parameters
        self.shift = nn.Parameter(torch.tensor(0.0))  # Horizontal shift
        self.scale = nn.Parameter(torch.tensor(1.0))  # Vertical scaling
        self.skew = nn.Parameter(torch.tensor(0.0))   # Skew parameter
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Learnable sigmoid: scale * σ(temperature * (x + shift)) + skew * x"""
        linear_component = self.skew * x
        sigmoid_component = self.math.sigmoid(self.temperature * (x + self.shift))
        
        return self.scale * sigmoid_component + linear_component
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Derivative of learnable sigmoid."""
        sig = self.math.sigmoid(self.temperature * (x + self.shift))
        sigmoid_grad = self.temperature * sig * (1 - sig)
        
        return self.scale * sigmoid_grad + self.skew

# ==================== SIGMOID LAYER IMPLEMENTATIONS ====================
class SigmoidLayer(nn.Module):
    """Neural network layer with sigmoid activation."""
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 sigmoid_type: str = 'standard',
                 use_bias: bool = True,
                 temperature: float = 1.0):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            sigmoid_type: Type of sigmoid ('standard', 'stable', 'fast', 'learnable')
            use_bias: Whether to use bias term
            temperature: Sigmoid temperature
        """
        super().__init__()
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        
        # Initialize weights
        self._init_weights()
        
        # Sigmoid activation
        if sigmoid_type == 'standard':
            self.activation = StandardSigmoid(temperature)
        elif sigmoid_type == 'stable':
            self.activation = StableSigmoid(temperature)
        elif sigmoid_type == 'fast':
            self.activation = FastSigmoid(temperature)
        elif sigmoid_type == 'learnable':
            self.activation = LearnableSigmoid(temperature, learnable_temp=True)
        else:
            raise ValueError(f"Unknown sigmoid type: {sigmoid_type}")
    
    def _init_weights(self):
        """Initialize layer weights using Xavier/Glorot initialization."""
        nn.init.xavier_normal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: linear transformation + sigmoid."""
        linear_out = self.linear(x)
        return self.activation(linear_out)
    
    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate activations for analysis."""
        linear_out = self.linear(x)
        sigmoid_out = self.activation(linear_out)
        
        return {
            'linear_output': linear_out.detach(),
            'sigmoid_output': sigmoid_out.detach(),
            'sigmoid_gradient': self.activation.derivative(linear_out).detach()
        }

class SigmoidNetwork(nn.Module):
    """Multi-layer perceptron with sigmoid activations."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 sigmoid_type: str = 'standard',
                 temperature: float = 1.0,
                 dropout: float = 0.0,
                 use_batch_norm: bool = False):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            sigmoid_type: Type of sigmoid activation
            temperature: Sigmoid temperature
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Sigmoid activation
            if sigmoid_type == 'standard':
                layers.append(StandardSigmoid(temperature))
            elif sigmoid_type == 'stable':
                layers.append(StableSigmoid(temperature))
            elif sigmoid_type == 'fast':
                layers.append(FastSigmoid(temperature))
            elif sigmoid_type == 'learnable':
                layers.append(LearnableSigmoid(temperature, learnable_temp=True))
            
            # Dropout
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression, softmax for classification)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.layers(x)
    
    def get_layer_activations(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Get activations from each layer for analysis."""
        activations = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            
            if isinstance(layer, (StandardSigmoid, StableSigmoid, 
                                FastSigmoid, LearnableSigmoid)):
                activations.append({
                    'type': layer.__class__.__name__,
                    'output': current.detach(),
                    'temperature': layer.temperature if hasattr(layer, 'temperature') else None
                })
        
        return activations

# ==================== DATASET IMPLEMENTATIONS ====================
class SigmoidDataset(Dataset):
    """Base dataset class for sigmoid experiments."""
    
    def __init__(self, 
                 num_samples: int = 1000,
                 input_dim: int = 10,
                 noise_level: float = 0.1,
                 function_type: str = 'linear'):
        """
        Args:
            num_samples: Number of samples
            input_dim: Input dimension
            noise_level: Noise level in targets
            function_type: Type of target function ('linear', 'nonlinear', 'classification')
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.noise_level = noise_level
        self.function_type = function_type
        
        # Generate data
        self.X, self.y = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic dataset."""
        # Generate random input features
        X = torch.randn(self.num_samples, self.input_dim)
        
        if self.function_type == 'linear':
            # Linear regression with sigmoid output
            true_weights = torch.randn(self.input_dim)
            true_bias = torch.randn(1)
            linear_output = X @ true_weights + true_bias
            y = torch.sigmoid(linear_output)  # Output in [0, 1]
            
        elif self.function_type == 'nonlinear':
            # Nonlinear regression
            true_weights1 = torch.randn(self.input_dim, self.input_dim // 2)
            true_weights2 = torch.randn(self.input_dim // 2)
            hidden = torch.sigmoid(X @ true_weights1)
            linear_output = hidden @ true_weights2
            y = torch.sigmoid(linear_output)  # Output in [0, 1]
            
        elif self.function_type == 'classification':
            # Binary classification
            true_weights = torch.randn(self.input_dim)
            true_bias = torch.randn(1)
            logits = X @ true_weights + true_bias
            probabilities = torch.sigmoid(logits)
            y = (probabilities > 0.5).float()  # Binary labels
        
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
        
        # Add noise
        if self.noise_level > 0:
            noise = torch.randn_like(y) * self.noise_level
            y = torch.clamp(y + noise, 0, 1)
        
        return X, y
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MNISTSigmoidDataset(Dataset):
    """MNIST dataset for sigmoid classification experiments."""
    
    def __init__(self, train: bool = True, download: bool = True):
        self.train = train
        self.dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        # Convert to binary classification (even vs odd)
        self.labels = (self.dataset.targets % 2).float()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        label = self.labels[idx]
        return img.view(-1), label

class BinaryClassificationDataset(Dataset):
    """Binary classification dataset from sklearn datasets."""
    
    def __init__(self, 
                 dataset_name: str = 'moons',
                 num_samples: int = 1000,
                 noise: float = 0.1):
        from sklearn.datasets import make_moons, make_circles, make_classification
        
        self.dataset_name = dataset_name
        
        if dataset_name == 'moons':
            X, y = make_moons(n_samples=num_samples, noise=noise, random_state=42)
        elif dataset_name == 'circles':
            X, y = make_circles(n_samples=num_samples, noise=noise, factor=0.5, random_state=42)
        elif dataset_name == 'linear':
            X, y = make_classification(n_samples=num_samples, n_features=2, 
                                      n_informative=2, n_redundant=0,
                                      random_state=42)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== TRAINING AND EVALUATION ====================
class SigmoidTrainer:
    """Training framework for sigmoid networks."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.01,
                 loss_type: str = 'mse'):
        """
        Args:
            model: Neural network model
            device: Training device
            learning_rate: Learning rate
            loss_type: Loss function type ('mse', 'bce', 'logistic')
        """
        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type
        
        # Define loss function
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'bce':
            self.criterion = nn.BCELoss()
        elif loss_type == 'logistic':
            self.criterion = self._logistic_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Sigmoid mathematics helper
        self.math = SigmoidMathematics()
    
    def _logistic_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Logistic loss (binary cross-entropy)."""
        return self.math.logistic_loss(predictions, targets)
    
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
            
            # For classification, apply sigmoid if not already in output
            if self.loss_type in ['bce', 'logistic'] and output.shape == target.shape:
                # Ensure output is in [0, 1]
                output = torch.sigmoid(output)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy for classification
            if target.dim() == 1 or target.shape[1] == 1:
                predictions = (output > 0.5).float()
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
                
                # Apply sigmoid for classification
                if self.loss_type in ['bce', 'logistic'] and output.shape == target.shape:
                    output = torch.sigmoid(output)
                
                loss = self.criterion(output, target)
                
                # Compute accuracy
                if target.dim() == 1 or target.shape[1] == 1:
                    predictions = (output > 0.5).float()
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
              epochs: int = 50,
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
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_sigmoid_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        axes[0].plot(self.val_losses, label='Val Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot (if available)
        if self.train_accuracies and self.val_accuracies:
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
class SigmoidVisualizer:
    """Visualization tools for sigmoid function."""
    
    @staticmethod
    def plot_sigmoid_function(sigmoid_func: BaseSigmoid,
                             x_range: Tuple[float, float] = (-10, 10),
                             num_points: int = 1000,
                             show_derivatives: bool = True):
        """Plot sigmoid function and its derivatives."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        # Compute values
        y = sigmoid_func.forward(x)
        y_prime = sigmoid_func.derivative(x)
        y_double_prime = SigmoidMathematics().sigmoid_second_derivative(x * sigmoid_func.temperature)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Sigmoid function
        axes[0, 0].plot(x.numpy(), y.numpy(), 'b-', linewidth=2)
        axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('σ(x)')
        axes[0, 0].set_title(f'Sigmoid Function (temperature={sigmoid_func.temperature})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # First derivative
        axes[0, 1].plot(x.numpy(), y_prime.numpy(), 'g-', linewidth=2)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel("σ'(x)")
        axes[0, 1].set_title('First Derivative')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Second derivative
        axes[1, 0].plot(x.numpy(), y_double_prime.numpy(), 'r-', linewidth=2)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel("σ''(x)")
        axes[1, 0].set_title('Second Derivative')
        axes[1, 0].grid(True, alpha=0.3)
        
        # All together
        axes[1, 1].plot(x.numpy(), y.numpy(), 'b-', label='σ(x)', linewidth=2)
        axes[1, 1].plot(x.numpy(), y_prime.numpy(), 'g-', label="σ'(x)", linewidth=2)
        axes[1, 1].plot(x.numpy(), y_double_prime.numpy(), 'r-', label="σ''(x)", linewidth=2)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('All Functions Together')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Sigmoid Analysis - {sigmoid_func.__class__.__name__}', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_multiple_sigmoids(sigmoid_funcs: List[BaseSigmoid],
                              x_range: Tuple[float, float] = (-10, 10),
                              num_points: int = 1000):
        """Compare multiple sigmoid implementations."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        plt.figure(figsize=(10, 6))
        
        for sigmoid_func in sigmoid_funcs:
            y = sigmoid_func.forward(x)
            label = f"{sigmoid_func.__class__.__name__}"
            if hasattr(sigmoid_func, 'temperature'):
                label += f" (temp={sigmoid_func.temperature})"
            plt.plot(x.numpy(), y.numpy(), label=label, linewidth=2)
        
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('σ(x)')
        plt.title('Comparison of Sigmoid Implementations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_gradient_flow(sigmoid_output: torch.Tensor,
                          upstream_gradient: torch.Tensor):
        """Visualize gradient flow through sigmoid."""
        # Compute gradient
        local_gradient = sigmoid_output * (1 - sigmoid_output)
        total_gradient = upstream_gradient * local_gradient
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Sigmoid output
        axes[0].hist(sigmoid_output.numpy().flatten(), bins=50, alpha=0.7)
        axes[0].set_xlabel('Sigmoid Output')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Sigmoid Outputs')
        axes[0].grid(True, alpha=0.3)
        
        # Local gradient
        axes[1].hist(local_gradient.numpy().flatten(), bins=50, alpha=0.7, color='green')
        axes[1].set_xlabel('Local Gradient σ(x)(1-σ(x))')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Local Gradients')
        axes[1].grid(True, alpha=0.3)
        
        # Total gradient
        axes[2].hist(total_gradient.numpy().flatten(), bins=50, alpha=0.7, color='red')
        axes[2].set_xlabel('Total Gradient')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Distribution of Total Gradients')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(model: nn.Module,
                              dataset: BinaryClassificationDataset,
                              resolution: int = 100):
        """Plot decision boundary for 2D classification."""
        model.eval()
        
        # Create grid
        x_min, x_max = dataset.X[:, 0].min() - 0.5, dataset.X[:, 0].max() + 0.5
        y_min, y_max = dataset.X[:, 1].min() - 0.5, dataset.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Predict on grid
        grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        with torch.no_grad():
            Z = model(grid_points)
            Z = torch.sigmoid(Z) if Z.min() < 0 or Z.max() > 1 else Z
            Z = Z.reshape(xx.shape).numpy()
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdBu', levels=50)
        plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y.squeeze(),
                   cmap='RdBu', edgecolors='k', alpha=0.6)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary with Sigmoid Activation')
        plt.colorbar(label='Probability')
        plt.grid(True, alpha=0.3)
        plt.show()

# ==================== SIGMOID ANALYSIS AND BENCHMARKING ====================
class SigmoidAnalyzer:
    """Analyzer for sigmoid function properties."""
    
    def __init__(self, sigmoid_func: BaseSigmoid):
        self.sigmoid_func = sigmoid_func
        self.math = SigmoidMathematics()
    
    def analyze_range(self, 
                     x_range: Tuple[float, float] = (-10, 10),
                     num_points: int = 1000) -> Dict[str, Any]:
        """Analyze sigmoid properties over a range."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        y = self.sigmoid_func.forward(x)
        y_prime = self.sigmoid_func.derivative(x)
        
        analysis = {
            'input_range': (x.min().item(), x.max().item()),
            'output_range': (y.min().item(), y.max().item()),
            'output_mean': y.mean().item(),
            'output_std': y.std().item(),
            'max_derivative': y_prime.max().item(),
            'mean_derivative': y_prime.mean().item(),
            'saturation_points': {
                'left': self._find_saturation_point(x, y, direction='left'),
                'right': self._find_saturation_point(x, y, direction='right')
            }
        }
        
        return analysis
    
    def _find_saturation_point(self, 
                              x: torch.Tensor, 
                              y: torch.Tensor, 
                              direction: str = 'left') -> float:
        """Find where sigmoid saturates (reaches 0.01 or 0.99)."""
        if direction == 'left':
            mask = y < 0.01
            if mask.any():
                return x[mask].max().item()
        else:  # right
            mask = y > 0.99
            if mask.any():
                return x[mask].min().item()
        return float('nan')
    
    def analyze_gradient_vanishing(self,
                                  x_values: torch.Tensor) -> Dict[str, Any]:
        """Analyze gradient vanishing problem."""
        gradients = self.sigmoid_func.derivative(x_values)
        
        # Count vanishing gradients
        vanishing_mask = gradients < 1e-7
        vanishing_count = vanishing_mask.sum().item()
        vanishing_percentage = vanishing_count / len(x_values) * 100
        
        analysis = {
            'total_samples': len(x_values),
            'vanishing_gradients_count': vanishing_count,
            'vanishing_gradients_percentage': vanishing_percentage,
            'mean_gradient': gradients.mean().item(),
            'min_gradient': gradients.min().item(),
            'max_gradient': gradients.max().item(),
            'gradient_histogram': np.histogram(gradients.numpy(), bins=50)
        }
        
        return analysis
    
    def compare_with_other_activations(self,
                                      x: torch.Tensor,
                                      other_activations: List[Callable]) -> Dict[str, torch.Tensor]:
        """Compare sigmoid with other activation functions."""
        results = {
            'sigmoid': self.sigmoid_func.forward(x),
            'sigmoid_gradient': self.sigmoid_func.derivative(x)
        }
        
        for i, activation in enumerate(other_activations):
            name = activation.__name__ if hasattr(activation, '__name__') else f'activation_{i}'
            results[name] = activation(x)
        
        return results

class SigmoidBenchmark:
    """Benchmarking framework for sigmoid implementations."""
    
    @staticmethod
    def benchmark_forward_pass(sigmoid_funcs: List[BaseSigmoid],
                              batch_sizes: List[int] = [1, 10, 100, 1000, 10000]):
        """Benchmark forward pass performance."""
        import time
        
        print("=" * 60)
        print("SIGMOID FORWARD PASS BENCHMARK")
        print("=" * 60)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}")
            print("-" * 40)
            
            # Create random input
            x = torch.randn(batch_size, 100)
            
            for sigmoid_func in sigmoid_funcs:
                # Warm up
                for _ in range(10):
                    _ = sigmoid_func.forward(x)
                
                # Benchmark
                start_time = time.time()
                for _ in range(100):
                    _ = sigmoid_func.forward(x)
                elapsed = time.time() - start_time
                
                # Store result
                func_name = sigmoid_func.__class__.__name__
                if func_name not in results:
                    results[func_name] = []
                results[func_name].append((batch_size, elapsed / 100))
                
                print(f"  {func_name:20s}: {elapsed/100*1000:.2f} ms per forward pass")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        for func_name, data in results.items():
            batch_sizes = [d[0] for d in data]
            times = [d[1] * 1000 for d in data]  # Convert to ms
            plt.plot(batch_sizes, times, 'o-', label=func_name, linewidth=2)
        
        plt.xscale('log')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Forward Pass (ms)')
        plt.title('Sigmoid Forward Pass Performance')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        plt.show()
    
    @staticmethod
    def benchmark_numerical_stability(sigmoid_funcs: List[BaseSigmoid],
                                     extreme_values: List[float] = [-100, -50, -20, 20, 50, 100]):
        """Benchmark numerical stability for extreme values."""
        print("\n" + "=" * 60)
        print("NUMERICAL STABILITY BENCHMARK")
        print("=" * 60)
        
        for value in extreme_values:
            print(f"\nInput value: {value}")
            print("-" * 40)
            
            x = torch.tensor([value])
            
            for sigmoid_func in sigmoid_funcs:
                try:
                    y = sigmoid_func.forward(x)
                    grad = sigmoid_func.derivative(x)
                    print(f"  {sigmoid_func.__class__.__name__:20s}: "
                          f"output = {y.item():.6f}, gradient = {grad.item():.6e}")
                except Exception as e:
                    print(f"  {sigmoid_func.__class__.__name__:20s}: ERROR - {str(e)}")
    
    @staticmethod
    def benchmark_gradient_flow(network: SigmoidNetwork,
                               dataset: Dataset,
                               num_samples: int = 1000):
        """Benchmark gradient flow through sigmoid network."""
        print("\n" + "=" * 60)
        print("GRADIENT FLOW BENCHMARK")
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
        
        print(f"\nNetwork has {len(activations)} sigmoid layers")
        
        for i, activation in enumerate(activations):
            layer_output = activation['output']
            mean_activation = layer_output.mean().item()
            std_activation = layer_output.std().item()
            dead_neurons = (layer_output < 1e-7).sum().item()
            saturated_neurons = (layer_output > 1 - 1e-7).sum().item()
            
            print(f"\nLayer {i+1} ({activation['type']}):")
            print(f"  Mean activation: {mean_activation:.6f}")
            print(f"  Std activation: {std_activation:.6f}")
            print(f"  Dead neurons: {dead_neurons} ({dead_neurons/layer_output.numel()*100:.1f}%)")
            print(f"  Saturated neurons: {saturated_neurons} ({saturated_neurons/layer_output.numel()*100:.1f}%)")
        
        return activations

# ==================== MAIN DEMONSTRATION ====================
def demonstrate_sigmoid_mathematics():
    """Demonstrate mathematical properties of sigmoid."""
    print("=" * 60)
    print("SIGMOID MATHEMATICAL PROPERTIES")
    print("=" * 60)
    
    math = SigmoidMathematics()
    
    # Test points
    test_points = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    
    print("\nSigmoid values at test points:")
    print("-" * 40)
    for x in test_points:
        x_tensor = torch.tensor([x])
        sig = math.sigmoid(x_tensor).item()
        deriv = math.sigmoid_derivative(x_tensor).item()
        print(f"σ({x:3d}) = {sig:.6f}, σ'({x:3d}) = {deriv:.6f}")
    
    print("\nKey Properties:")
    print("1. Range: (0, 1)")
    print("2. σ(0) = 0.5 (point of symmetry)")
    print("3. σ(-x) = 1 - σ(x) (odd symmetry about (0, 0.5))")
    print("4. lim(x→∞) σ(x) = 1")
    print("5. lim(x→-∞) σ(x) = 0")
    print("6. σ'(x) = σ(x) * (1 - σ(x)) (maximum at x=0, σ'(0)=0.25)")
    
    # Demonstrate inverse
    print("\nInverse Sigmoid (Logit):")
    print("-" * 40)
    for y in [0.1, 0.25, 0.5, 0.75, 0.9]:
        y_tensor = torch.tensor([y])
        x = math.sigmoid_inverse(y_tensor).item()
        print(f"σ⁻¹({y}) = {x:.6f}")
    
    # Demonstrate logistic loss
    print("\nLogistic Loss (Binary Cross-Entropy):")
    print("-" * 40)
    predictions = torch.tensor([0.1, 0.5, 0.9])
    targets = torch.tensor([0, 1, 1])
    loss = math.logistic_loss(predictions, targets)
    print(f"Predictions: {predictions.numpy()}")
    print(f"Targets: {targets.numpy()}")
    print(f"Loss: {loss.item():.6f}")

def demonstrate_sigmoid_implementations():
    """Demonstrate different sigmoid implementations."""
    print("\n" + "=" * 60)
    print("SIGMOID IMPLEMENTATIONS")
    print("=" * 60)
    
    # Create different sigmoid implementations
    sigmoid_impls = [
        StandardSigmoid(temperature=1.0),
        StandardSigmoid(temperature=2.0),
        StableSigmoid(temperature=1.0),
        FastSigmoid(temperature=1.0, approximation_type='piecewise'),
        FastSigmoid(temperature=1.0, approximation_type='hard'),
        LearnableSigmoid(temperature=1.0, learnable_temp=True)
    ]
    
    # Visualize them
    visualizer = SigmoidVisualizer()
    visualizer.plot_multiple_sigmoids(sigmoid_impls, x_range=(-10, 10))
    
    # Analyze properties
    print("\nProperties at x=0:")
    print("-" * 40)
    for impl in sigmoid_impls:
        props = impl.analyze_at_point(0)
        print(f"\n{impl.__class__.__name__}:")
        print(f"  Value: {props.value:.6f}")
        print(f"  Derivative: {props.derivative:.6f}")
        print(f"  Inflection: {props.inflection_point}")

def train_sigmoid_network_example():
    """Train a sigmoid network on a classification task."""
    print("\n" + "=" * 60)
    print("SIGMOID NETWORK TRAINING EXAMPLE")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("\nCreating binary classification dataset...")
    dataset = BinaryClassificationDataset(dataset_name='moons', num_samples=1000)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create sigmoid network
    print("\nCreating sigmoid network...")
    model = SigmoidNetwork(
        input_dim=2,
        hidden_dims=[16, 16, 8],
        output_dim=1,
        sigmoid_type='standard',
        temperature=1.0,
        dropout=0.2,
        use_batch_norm=True
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = SigmoidTrainer(
        model=model,
        device=device,
        learning_rate=0.01,
        loss_type='bce'
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
    
    # Visualize decision boundary
    print("\nVisualizing decision boundary...")
    visualizer = SigmoidVisualizer()
    visualizer.plot_decision_boundary(model, dataset)
    
    return model, trainer

def benchmark_sigmoid_performance():
    """Benchmark sigmoid implementations."""
    print("\n" + "=" * 60)
    print("SIGMOID PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create sigmoid implementations
    sigmoid_funcs = [
        StandardSigmoid(temperature=1.0),
        StableSigmoid(temperature=1.0),
        FastSigmoid(temperature=1.0, approximation_type='piecewise'),
        FastSigmoid(temperature=1.0, approximation_type='hard')
    ]
    
    # Run benchmarks
    benchmark = SigmoidBenchmark()
    
    # Forward pass benchmark
    benchmark.benchmark_forward_pass(sigmoid_funcs)
    
    # Numerical stability benchmark
    benchmark.benchmark_numerical_stability(sigmoid_funcs)
    
    # Analyze properties
    print("\n" + "=" * 60)
    print("SIGMOID PROPERTY ANALYSIS")
    print("=" * 60)
    
    analyzer = SigmoidAnalyzer(sigmoid_funcs[0])
    
    # Generate test data
    x_test = torch.randn(10000) * 10  # Values in approx [-30, 30]
    
    # Analyze gradient vanishing
    gradient_analysis = analyzer.analyze_gradient_vanishing(x_test)
    print(f"\nGradient Vanishing Analysis:")
    print(f"  Samples with gradient < 1e-7: {gradient_analysis['vanishing_gradients_count']}")
    print(f"  Percentage: {gradient_analysis['vanishing_gradients_percentage']:.1f}%")
    print(f"  Mean gradient: {gradient_analysis['mean_gradient']:.6f}")
    print(f"  Min gradient: {gradient_analysis['min_gradient']:.6e}")
    print(f"  Max gradient: {gradient_analysis['max_gradient']:.6f}")

def demonstrate_sigmoid_applications():
    """Demonstrate practical applications of sigmoid."""
    print("\n" + "=" * 60)
    print("SIGMOID PRACTICAL APPLICATIONS")
    print("=" * 60)
    
    print("\n1. Logistic Regression:")
    print("   Sigmoid converts linear combination to probability")
    print("   P(y=1|x) = σ(w·x + b)")
    
    print("\n2. Neural Network Activation:")
    print("   Historically popular activation function")
    print("   Outputs in (0,1) suitable for probability-like outputs")
    
    print("\n3. Gating Mechanisms:")
    print("   Used in LSTMs and GRUs for forget/input/output gates")
    
    print("\n4. Attention Mechanisms:")
    print("   Can be used in attention weights computation")
    
    print("\n5. Binary Classification:")
    print("   Natural choice for output layer in binary classification")
    
    print("\n6. Reinforcement Learning:")
    print("   Used in policy gradients for probability outputs")
    
    # Demonstrate logistic regression
    print("\n" + "-" * 40)
    print("Logistic Regression Example:")
    
    # Create synthetic data
    torch.manual_seed(42)
    n_samples = 100
    X = torch.randn(n_samples, 2)
    true_weights = torch.tensor([2.0, -1.0])
    true_bias = 0.5
    logits = X @ true_weights + true_bias
    probabilities = torch.sigmoid(logits)
    y = (probabilities > 0.5).float()
    
    # Create logistic regression model
    class LogisticRegression(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            self.sigmoid = StandardSigmoid()
        
        def forward(self, x):
            return self.sigmoid(self.linear(x))
    
    model = LogisticRegression(2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Train briefly
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(X).squeeze()
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  True weights: {true_weights.numpy()}")
    print(f"  Learned weights: {model.linear.weight.detach().numpy().flatten()}")

def main():
    """Main demonstration function."""
    print("SIGMOID ACTIVATION FUNCTION IMPLEMENTATION FROM SCRATCH")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demonstrate mathematical properties
    demonstrate_sigmoid_mathematics()
    
    # Demonstrate different implementations
    demonstrate_sigmoid_implementations()
    
    # Train sigmoid network
    model, trainer = train_sigmoid_network_example()
    
    # Benchmark performance
    benchmark_sigmoid_performance()
    
    # Demonstrate applications
    demonstrate_sigmoid_applications()
    
    print("\n" + "=" * 60)
    print("SIGMOID IMPLEMENTATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Implemented:")
    print("1. Standard Sigmoid: σ(x) = 1/(1+e^{-x})")
    print("2. Stable Sigmoid: Numerically stable implementation")
    print("3. Fast Sigmoid: Approximations for performance")
    print("4. Learnable Sigmoid: With learnable parameters")
    print("5. Complete mathematical analysis tools")
    print("6. Gradient flow visualization")
    print("7. Training framework for sigmoid networks")
    print("8. Benchmarking and comparison tools")
    print("9. Practical applications demonstration")

if __name__ == "__main__":
    main()