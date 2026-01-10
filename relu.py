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

# ==================== ReLU MATHEMATICAL FOUNDATION ====================
class ReLUMathematics:
    """
    Mathematical foundations of the Rectified Linear Unit (ReLU) function.
    Implements all mathematical properties and theorems.
    
    ReLU(x) = max(0, x) = x⁺ (positive part)
    """
    
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        """
        Rectified Linear Unit: ReLU(x) = max(0, x)
        
        Mathematical Properties:
        1. Range: [0, ∞)
        2. Domain: (-∞, ∞)
        3. Piecewise linear: f(x) = 0 for x ≤ 0, f(x) = x for x > 0
        4. Derivative: ReLU'(x) = 0 for x < 0, 1 for x > 0, undefined at 0
        5. Convex: Both convex and non-decreasing
        6. Sparsity inducing: Produces sparse activations
        """
        return torch.maximum(torch.zeros_like(x), x)
    
    @staticmethod
    def relu_derivative(x: torch.Tensor) -> torch.Tensor:
        """
        Derivative of ReLU: ReLU'(x) = 
        0 for x < 0
        1 for x > 0
        undefined at x = 0 (subgradient in [0,1])
        
        In practice, we use subgradient at 0 (either 0 or 1)
        PyTorch convention: ReLU'(0) = 0
        """
        # Using PyTorch convention: derivative at 0 is 0
        return (x > 0).float()
    
    @staticmethod
    def relu_subgradient(x: torch.Tensor, at_zero: float = 0.0) -> torch.Tensor:
        """
        Subgradient of ReLU at non-differentiable point (x=0).
        The subgradient at 0 is any value in [0, 1].
        
        Args:
            at_zero: Value to use for gradient at x=0 (default 0, PyTorch convention)
        """
        gradient = (x > 0).float()
        zero_mask = (x == 0)
        gradient[zero_mask] = at_zero
        return gradient
    
    @staticmethod
    def relu_second_derivative(x: torch.Tensor) -> torch.Tensor:
        """
        Second derivative of ReLU (distributional derivative).
        ReLU''(x) = δ(x) (Dirac delta function at 0)
        
        In practice, we treat it as 0 everywhere except at 0.
        """
        return torch.zeros_like(x)
    
    @staticmethod
    def relu_integral(x: torch.Tensor) -> torch.Tensor:
        """
        Integral of ReLU: ∫ReLU(t)dt = 
        0 for x ≤ 0
        x²/2 for x > 0
        
        Known as the "softplus" function up to scaling.
        """
        return torch.where(x > 0, x ** 2 / 2, torch.zeros_like(x))
    
    @staticmethod
    def softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Softplus function: f(x) = log(1 + exp(βx)) / β
        Smooth approximation of ReLU.
        
        Properties:
        1. lim(β→∞) softplus(x) = ReLU(x)
        2. Always positive
        3. Everywhere differentiable
        """
        return torch.log(1 + torch.exp(beta * x)) / beta
    
    @staticmethod
    def compute_sparsity(activations: torch.Tensor) -> float:
        """
        Compute sparsity of ReLU activations.
        Sparsity = percentage of zeros in activations.
        
        ReLU naturally induces sparsity by setting negative values to 0.
        """
        if activations.numel() == 0:
            return 0.0
        zero_mask = (activations == 0)
        return zero_mask.float().mean().item()
    
    @staticmethod
    def compute_dead_neurons(activations: torch.Tensor, 
                           threshold: float = 1e-7) -> float:
        """
        Compute percentage of "dead neurons" (always outputting 0).
        
        A neuron is dead if all its activations are ≤ threshold.
        """
        if activations.numel() == 0:
            return 0.0
        
        # Check if activations are all close to zero
        max_activation = activations.abs().max()
        return (max_activation <= threshold).float().item()
    
    @staticmethod
    def compute_gradient_flow(relu_output: torch.Tensor, 
                            upstream_gradient: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient flow through ReLU for backpropagation.
        dL/dx = dL/dReLU * ReLU'(x) = dL/dReLU * I(x > 0)
        
        This is the gradient stopping property: gradients only flow
        through active (positive) neurons.
        """
        # Mask for active neurons (x > 0)
        active_mask = (relu_output > 0).float()
        
        # Gradient only flows through active neurons
        return upstream_gradient * active_mask

# ==================== ReLU IMPLEMENTATION CLASSES ====================
@dataclass
class ReLUProperties:
    """Mathematical properties of ReLU at a specific point."""
    value: float
    derivative: float
    subgradient_at_zero: float
    is_active: bool
    sparsity_contribution: float
    linear_region: str  # 'negative', 'zero', 'positive'
    
    def __str__(self):
        return (f"ReLU Properties:\n"
                f"  Value: {self.value:.6f}\n"
                f"  Derivative: {self.derivative:.6f}\n"
                f"  Subgradient at 0: {self.subgradient_at_zero:.6f}\n"
                f"  Is Active: {self.is_active}\n"
                f"  Sparsity Contribution: {self.sparsity_contribution:.6f}\n"
                f"  Linear Region: {self.linear_region}")

class BaseReLU(nn.Module):
    """Base class for all ReLU implementations."""
    
    def __init__(self, leak: float = 0.0):
        """
        Args:
            leak: Leakage parameter for LeakyReLU variants
                  0 = standard ReLU, >0 = LeakyReLU
        """
        super().__init__()
        self.leak = leak
        self.math = ReLUMathematics()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute derivative at x."""
        raise NotImplementedError
    
    def analyze_at_point(self, x: float) -> ReLUProperties:
        """Analyze ReLU properties at a specific point."""
        x_tensor = torch.tensor([x])
        value = self.forward(x_tensor).item()
        deriv = self.derivative(x_tensor).item()
        
        # Determine linear region
        if x < 0:
            linear_region = 'negative'
        elif x == 0:
            linear_region = 'zero'
        else:
            linear_region = 'positive'
        
        # Determine if neuron is active
        is_active = (value > 0)
        
        # Sparsity contribution (1 if inactive, 0 if active)
        sparsity_contribution = 0.0 if is_active else 1.0
        
        # Subgradient at zero
        subgradient_at_zero = self._get_subgradient_at_zero()
        
        return ReLUProperties(
            value=value,
            derivative=deriv,
            subgradient_at_zero=subgradient_at_zero,
            is_active=is_active,
            sparsity_contribution=sparsity_contribution,
            linear_region=linear_region
        )
    
    def _get_subgradient_at_zero(self) -> float:
        """Get subgradient value at x=0."""
        return self.leak if hasattr(self, 'leak') else 0.0
    
    def compute_sparsity(self, x: torch.Tensor) -> float:
        """Compute sparsity induced by this ReLU on input x."""
        output = self.forward(x)
        return self.math.compute_sparsity(output)
    
    def compute_dead_neurons(self, x: torch.Tensor, threshold: float = 1e-7) -> float:
        """Compute percentage of dead neurons in batch."""
        output = self.forward(x)
        return self.math.compute_dead_neurons(output, threshold)

class StandardReLU(BaseReLU):
    """Standard ReLU implementation: max(0, x)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU(x) = max(0, x)"""
        # Implementation from scratch
        return torch.where(x > 0, x, torch.zeros_like(x))
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU'(x) = 1 if x > 0, else 0 (PyTorch convention)"""
        return (x > 0).float()

class LeakyReLU(BaseReLU):
    """Leaky ReLU: f(x) = x if x > 0, else αx."""
    
    def __init__(self, negative_slope: float = 0.01):
        """
        Args:
            negative_slope: Slope for negative inputs (α)
        """
        super().__init__(leak=negative_slope)
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LeakyReLU(x) = max(αx, x)"""
        # Implementation from scratch
        return torch.where(x > 0, x, self.negative_slope * x)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """LeakyReLU'(x) = 1 if x > 0, else α"""
        return torch.where(x > 0, 
                          torch.ones_like(x), 
                          torch.full_like(x, self.negative_slope))

class ParametricReLU(BaseReLU):
    """Parametric ReLU (PReLU): f(x) = x if x > 0, else a*x where a is learnable."""
    
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        """
        Args:
            num_parameters: Number of a parameters to learn
                          1 = shared across channels
                          >1 = per channel
            init: Initial value for a
        """
        super().__init__(leak=init)
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.full((num_parameters,), init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PReLU(x) = max(0, x) + a * min(0, x)"""
        # Implementation from scratch
        positive = torch.maximum(torch.zeros_like(x), x)
        negative = torch.minimum(torch.zeros_like(x), x)
        
        # Apply learned weight to negative part
        if self.num_parameters == 1:
            negative = negative * self.weight
        else:
            # Reshape weight for channel-wise multiplication
            weight_shape = [1] * x.dim()
            weight_shape[1] = self.num_parameters  # Assuming channels dimension
            weight = self.weight.view(*weight_shape)
            negative = negative * weight
        
        return positive + negative
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """PReLU'(x) = 1 if x > 0, else a"""
        if self.num_parameters == 1:
            weight = self.weight
        else:
            weight_shape = [1] * x.dim()
            weight_shape[1] = self.num_parameters
            weight = self.weight.view(*weight_shape)
        
        return torch.where(x > 0, 
                          torch.ones_like(x), 
                          weight.expand_as(x))

class RandomizedReLU(BaseReLU):
    """Randomized ReLU (RReLU): leak parameter is random during training."""
    
    def __init__(self, lower: float = 0.125, upper: float = 0.333, 
                 training: bool = True):
        """
        Args:
            lower: Lower bound for random α
            upper: Upper bound for random α
            training: Whether in training mode (random) or eval mode (average)
        """
        super().__init__(leak=(lower + upper) / 2)
        self.lower = lower
        self.upper = upper
        self.training = training
        
        # Store current random value
        self.current_alpha = (lower + upper) / 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RReLU(x) = x if x > 0, else αx where α ~ Uniform(lower, upper) during training"""
        if self.training:
            # Sample new α for each call during training
            alpha = torch.empty(1).uniform_(self.lower, self.upper)
            self.current_alpha = alpha.item()
        else:
            # Use average α during evaluation
            alpha = torch.tensor([(self.lower + self.upper) / 2])
        
        return torch.where(x > 0, x, alpha * x)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """RReLU'(x) = 1 if x > 0, else current α"""
        current_alpha = torch.tensor([self.current_alpha]).to(x.device)
        return torch.where(x > 0, 
                          torch.ones_like(x), 
                          current_alpha.expand_as(x))

class ExponentialReLU(BaseReLU):
    """Exponential ReLU (ELU): f(x) = x if x > 0, else α(exp(x)-1)."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Scaling parameter for negative values
        """
        super().__init__(leak=alpha)
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ELU(x) = x if x > 0, else α(exp(x)-1)"""
        # Implementation from scratch
        positive = torch.maximum(torch.zeros_like(x), x)
        negative = torch.minimum(torch.zeros_like(x), x)
        
        # Apply exponential to negative part
        negative = self.alpha * (torch.exp(negative) - 1)
        
        return positive + negative
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """ELU'(x) = 1 if x > 0, else α * exp(x)"""
        return torch.where(x > 0, 
                          torch.ones_like(x), 
                          self.alpha * torch.exp(x))

class ScaledExponentialReLU(BaseReLU):
    """Scaled Exponential Linear Unit (SELU): self-normalizing variant."""
    
    def __init__(self):
        """
        SELU has fixed parameters that ensure self-normalizing properties.
        α ≈ 1.6733, λ ≈ 1.0507
        """
        super().__init__(leak=0.0)
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SELU(x) = λ * (x if x > 0, else α(exp(x)-1))"""
        # Implementation from scratch
        positive = torch.maximum(torch.zeros_like(x), x)
        negative = torch.minimum(torch.zeros_like(x), x)
        
        negative = self.alpha * (torch.exp(negative) - 1)
        
        return self.scale * (positive + negative)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """SELU'(x) = λ * (1 if x > 0, else α * exp(x))"""
        grad = torch.where(x > 0, 
                          torch.ones_like(x), 
                          self.alpha * torch.exp(x))
        return self.scale * grad

class SwishReLU(BaseReLU):
    """Swish/SiLU: f(x) = x * sigmoid(x)."""
    
    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Scaling parameter (β in x * σ(βx))
        """
        super().__init__(leak=0.0)
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Swish(x) = x * σ(βx) where σ is sigmoid"""
        # Implementation from scratch
        sigmoid = 1 / (1 + torch.exp(-self.beta * x))
        return x * sigmoid
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Swish'(x) = σ(βx) + βx * σ(βx) * (1 - σ(βx))"""
        sigmoid = 1 / (1 + torch.exp(-self.beta * x))
        return sigmoid + self.beta * x * sigmoid * (1 - sigmoid)

class GaussianReLU(BaseReLU):
    """Gaussian Error Linear Unit (GELU): f(x) = x * Φ(x) where Φ is Gaussian CDF."""
    
    def __init__(self, approximate: bool = True):
        """
        Args:
            approximate: Whether to use approximation for speed
        """
        super().__init__(leak=0.0)
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GELU(x) = x * Φ(x) where Φ is Gaussian CDF"""
        if self.approximate:
            # Fast approximation: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
            return 0.5 * x * (1 + torch.tanh(
                math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            ))
        else:
            # Exact computation using error function
            return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """GELU'(x) = Φ(x) + x * φ(x) where φ is Gaussian PDF"""
        if self.approximate:
            # Derivative of approximation
            tanh_term = torch.tanh(
                math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
            sech_squared = 1 - tanh_term ** 2
            inner_deriv = math.sqrt(2 / math.pi) * (1 + 0.134145 * x ** 2)
            return 0.5 * (1 + tanh_term) + 0.5 * x * sech_squared * inner_deriv
        else:
            # Exact derivative
            cdf = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
            pdf = torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
            return cdf + x * pdf

class SparseReLU(BaseReLU):
    """Sparse ReLU with explicit sparsity control."""
    
    def __init__(self, sparsity_target: float = 0.5, temperature: float = 1.0):
        """
        Args:
            sparsity_target: Target sparsity level (0 to 1)
            temperature: Controls sharpness of sparsity enforcement
        """
        super().__init__(leak=0.0)
        self.sparsity_target = sparsity_target
        self.temperature = temperature
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse ReLU: f(x) = max(0, x - threshold)"""
        # Apply threshold
        shifted = x - self.threshold
        
        # Apply ReLU
        return torch.maximum(torch.zeros_like(shifted), shifted)
    
    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse ReLU'(x) = 1 if x > threshold, else 0"""
        return (x > self.threshold).float()
    
    def compute_sparsity_loss(self, activations: torch.Tensor) -> torch.Tensor:
        """Compute loss to encourage target sparsity."""
        current_sparsity = self.math.compute_sparsity(activations)
        sparsity_diff = current_sparsity - self.sparsity_target
        return sparsity_diff ** 2

# ==================== ReLU LAYER IMPLEMENTATIONS ====================
class ReLULayer(nn.Module):
    """Neural network layer with ReLU activation."""
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 relu_type: str = 'standard',
                 use_bias: bool = True,
                 **relu_kwargs):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            relu_type: Type of ReLU ('standard', 'leaky', 'prelu', 'elu', 'selu', 'swish', 'gelu')
            use_bias: Whether to use bias term
            relu_kwargs: Additional arguments for ReLU
        """
        super().__init__()
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        
        # Initialize weights using Kaiming/He initialization (optimal for ReLU)
        self._init_weights()
        
        # ReLU activation
        self.relu_type = relu_type
        self.activation = self._create_activation(relu_type, **relu_kwargs)
        
        # Statistics tracking
        self.activation_stats = {
            'sparsity_history': [],
            'mean_activation_history': [],
            'max_activation_history': [],
            'dead_neuron_history': []
        }
    
    def _init_weights(self):
        """Initialize layer weights using Kaiming/He initialization."""
        # Kaiming initialization is optimal for ReLU
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def _create_activation(self, relu_type: str, **kwargs) -> BaseReLU:
        """Create ReLU activation function."""
        if relu_type == 'standard':
            return StandardReLU()
        elif relu_type == 'leaky':
            return LeakyReLU(negative_slope=kwargs.get('negative_slope', 0.01))
        elif relu_type == 'prelu':
            return ParametricReLU(
                num_parameters=kwargs.get('num_parameters', 1),
                init=kwargs.get('init', 0.25)
            )
        elif relu_type == 'rrelu':
            return RandomizedReLU(
                lower=kwargs.get('lower', 0.125),
                upper=kwargs.get('upper', 0.333)
            )
        elif relu_type == 'elu':
            return ExponentialReLU(alpha=kwargs.get('alpha', 1.0))
        elif relu_type == 'selu':
            return ScaledExponentialReLU()
        elif relu_type == 'swish':
            return SwishReLU(beta=kwargs.get('beta', 1.0))
        elif relu_type == 'gelu':
            return GaussianReLU(approximate=kwargs.get('approximate', True))
        elif relu_type == 'sparse':
            return SparseReLU(
                sparsity_target=kwargs.get('sparsity_target', 0.5),
                temperature=kwargs.get('temperature', 1.0)
            )
        else:
            raise ValueError(f"Unknown ReLU type: {relu_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: linear transformation + ReLU."""
        linear_out = self.linear(x)
        activated = self.activation(linear_out)
        
        # Track statistics (only in training mode)
        if self.training:
            self._track_statistics(activated)
        
        return activated
    
    def _track_statistics(self, activations: torch.Tensor):
        """Track activation statistics."""
        sparsity = self.activation.compute_sparsity(activations)
        mean_activation = activations.mean().item()
        max_activation = activations.max().item()
        dead_neurons = self.activation.compute_dead_neurons(activations)
        
        self.activation_stats['sparsity_history'].append(sparsity)
        self.activation_stats['mean_activation_history'].append(mean_activation)
        self.activation_stats['max_activation_history'].append(max_activation)
        self.activation_stats['dead_neuron_history'].append(dead_neurons)
    
    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate activations for analysis."""
        linear_out = self.linear(x)
        relu_out = self.activation(linear_out)
        
        return {
            'linear_output': linear_out.detach(),
            'relu_output': relu_out.detach(),
            'relu_gradient': self.activation.derivative(linear_out).detach(),
            'active_mask': (relu_out > 0).float().detach(),
            'sparsity': self.activation.compute_sparsity(relu_out)
        }

class ReLUNetwork(nn.Module):
    """Multi-layer perceptron with ReLU activations."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 relu_type: str = 'standard',
                 use_batch_norm: bool = True,
                 dropout: float = 0.0,
                 residual: bool = False,
                 **relu_kwargs):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            relu_type: Type of ReLU activation
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
            residual: Whether to use residual connections
            relu_kwargs: Additional arguments for ReLU
        """
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.relu_type = relu_type
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        self.residual = residual
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear_layer = nn.Linear(prev_dim, hidden_dim)
            
            # Kaiming initialization for ReLU
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='relu')
            nn.init.zeros_(linear_layer.bias)
            
            layers.append(linear_layer)
            
            # Batch normalization (helps with training deep networks)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # ReLU activation
            layers.append(self._create_activation_layer(relu_type, **relu_kwargs))
            
            # Dropout
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        output_layer = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_normal_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.layers = nn.Sequential(*layers)
        
        # Store layer information for analysis
        self.layer_types = []
        for layer in layers:
            if isinstance(layer, (BaseReLU, nn.ReLU)):
                self.layer_types.append(type(layer).__name__)
            elif isinstance(layer, nn.Linear):
                self.layer_types.append('Linear')
            elif isinstance(layer, nn.BatchNorm1d):
                self.layer_types.append('BatchNorm')
            elif isinstance(layer, nn.Dropout):
                self.layer_types.append('Dropout')
    
    def _create_activation_layer(self, relu_type: str, **kwargs) -> nn.Module:
        """Create ReLU activation layer."""
        if relu_type == 'standard':
            return StandardReLU()
        elif relu_type == 'leaky':
            return LeakyReLU(negative_slope=kwargs.get('negative_slope', 0.01))
        elif relu_type == 'prelu':
            return ParametricReLU(
                num_parameters=kwargs.get('num_parameters', 1),
                init=kwargs.get('init', 0.25)
            )
        elif relu_type == 'elu':
            return ExponentialReLU(alpha=kwargs.get('alpha', 1.0))
        elif relu_type == 'selu':
            return ScaledExponentialReLU()
        elif relu_type == 'swish':
            return SwishReLU(beta=kwargs.get('beta', 1.0))
        elif relu_type == 'gelu':
            return GaussianReLU(approximate=kwargs.get('approximate', True))
        else:
            return StandardReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if self.residual and len(self.hidden_dims) > 1:
            # Implement residual connections
            residual = x
            residual_idx = 0
            
            for i, layer in enumerate(self.layers):
                x = layer(x)
                
                # Add residual connection every 3 layers (Linear -> BN -> ReLU)
                if (i + 1) % 3 == 0 and residual_idx < len(self.hidden_dims) - 1:
                    # Ensure dimensions match
                    if x.shape == residual.shape:
                        x = x + residual
                        residual = x
                    residual_idx += 1
            return x
        else:
            return self.layers(x)
    
    def get_layer_activations(self, x: torch.Tensor) -> List[Dict[str, Any]]:
        """Get activations from each layer for analysis."""
        activations = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            
            if isinstance(layer, BaseReLU):
                # Compute ReLU-specific statistics
                output = current.detach()
                sparsity = layer.compute_sparsity(output)
                dead_neurons = layer.compute_dead_neurons(output)
                mean_activation = output.mean().item()
                
                activations.append({
                    'type': layer.__class__.__name__,
                    'output': output,
                    'sparsity': sparsity,
                    'dead_neurons': dead_neurons,
                    'mean_activation': mean_activation,
                    'max_activation': output.max().item(),
                    'min_activation': output.min().item()
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
            'input_gradient_norm': x.grad.norm().item() if x.grad is not None else 0,
            'layer_gradients': [],
            'vanishing_layers': 0,
            'exploding_layers': 0
        }
        
        # Collect gradients for each parameter
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                
                gradient_stats['layer_gradients'].append({
                    'name': name,
                    'grad_norm': grad_norm,
                    'grad_mean': grad_mean,
                    'is_vanishing': grad_norm < 1e-7,
                    'is_exploding': grad_norm > 100.0
                })
                
                if grad_norm < 1e-7:
                    gradient_stats['vanishing_layers'] += 1
                if grad_norm > 100.0:
                    gradient_stats['exploding_layers'] += 1
        
        # Disable gradient tracking
        x.requires_grad_(False)
        
        return gradient_stats

# ==================== DATASET IMPLEMENTATIONS ====================
class CIFAR10ReLUDataset(Dataset):
    """CIFAR-10 dataset for ReLU experiments."""
    
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
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img.view(-1), label

class FashionMNISTReLUDataset(Dataset):
    """FashionMNIST dataset for ReLU experiments."""
    
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
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img.view(-1), label

class ReLUSyntheticDataset(Dataset):
    """Synthetic dataset for ReLU experiments."""
    
    def __init__(self, 
                 num_samples: int = 10000,
                 input_dim: int = 100,
                 num_classes: int = 10,
                 complexity: str = 'linear'):
        """
        Args:
            num_samples: Number of samples
            input_dim: Input dimension
            num_classes: Number of classes
            complexity: Dataset complexity ('linear', 'nonlinear', 'deep')
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.complexity = complexity
        
        # Generate data
        self.X, self.y = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic dataset."""
        # Generate random input features
        X = torch.randn(self.num_samples, self.input_dim)
        
        if self.complexity == 'linear':
            # Linear classification
            true_weights = torch.randn(self.input_dim, self.num_classes)
            true_bias = torch.randn(self.num_classes)
            logits = X @ true_weights + true_bias
            y = torch.argmax(logits, dim=1)
            
        elif self.complexity == 'nonlinear':
            # Nonlinear classification with ReLU-like structure
            W1 = torch.randn(self.input_dim, 50) * 0.1
            W2 = torch.randn(50, self.num_classes) * 0.1
            
            # Apply ReLU-like nonlinearity
            h = torch.relu(X @ W1)
            logits = h @ W2
            y = torch.argmax(logits, dim=1)
            
        elif self.complexity == 'deep':
            # Deep ReLU network as data generator
            W1 = torch.randn(self.input_dim, 64) * math.sqrt(2 / self.input_dim)
            W2 = torch.randn(64, 32) * math.sqrt(2 / 64)
            W3 = torch.randn(32, 16) * math.sqrt(2 / 32)
            W4 = torch.randn(16, self.num_classes) * math.sqrt(2 / 16)
            
            h1 = torch.relu(X @ W1)
            h2 = torch.relu(h1 @ W2)
            h3 = torch.relu(h2 @ W3)
            logits = h3 @ W4
            y = torch.argmax(logits, dim=1)
        
        else:
            raise ValueError(f"Unknown complexity: {self.complexity}")
        
        return X, y
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== TRAINING AND EVALUATION ====================
class ReLUTrainer:
    """Training framework for ReLU networks."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 optimizer_type: str = 'adam'):
        """
        Args:
            model: Neural network model
            device: Training device
            learning_rate: Learning rate
            weight_decay: L2 regularization
            optimizer_type: Optimizer type ('adam', 'sgd', 'rmsprop')
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function (cross-entropy for classification)
        self.criterion = nn.CrossEntropyLoss()
        
        # Create optimizer
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
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
        self.gradient_norms = []
        
        # ReLU mathematics helper
        self.math = ReLUMathematics()
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_gradient_norm = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (helps with training stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Track gradient norm before update
            grad_norm = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = math.sqrt(grad_norm)
            total_gradient_norm += grad_norm
            
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum().item()
            total_correct += correct
            total_loss += loss.item()
            total_samples += len(data)
            num_batches += 1
            
            # Print batch progress
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'  Batch {batch_idx:4d}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {correct/len(data):.4f}, '
                      f'Grad: {grad_norm:.4f}, '
                      f'LR: {current_lr:.6f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_grad_norm = total_gradient_norm / num_batches if num_batches > 0 else 0
        
        self.gradient_norms.append(avg_grad_norm)
        
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
        
        print(f"Training {self.model.__class__.__name__} for {epochs} epochs...")
        print(f"ReLU type: {getattr(self.model, 'relu_type', 'standard')}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
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
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  Grad Norm: {self.gradient_norms[-1]:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_relu_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training History - Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Train Acc', alpha=0.8)
        axes[0, 1].plot(self.val_accuracies, label='Val Acc', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training History - Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 2].plot(self.learning_rates, alpha=0.8)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Gradient norms
        axes[1, 0].plot(self.gradient_norms, alpha=0.8, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norms During Training')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss ratio (train/val)
        if len(self.train_losses) > 1 and len(self.val_losses) > 1:
            loss_ratio = [t/v if v > 0 else 1 for t, v in zip(self.train_losses, self.val_losses)]
            axes[1, 1].plot(loss_ratio, alpha=0.8, color='purple')
            axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Train/Val Loss Ratio')
            axes[1, 1].set_title('Overfitting Indicator')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Accuracy difference
        if len(self.train_accuracies) > 1 and len(self.val_accuracies) > 1:
            acc_diff = [t - v for t, v in zip(self.train_accuracies, self.val_accuracies)]
            axes[1, 2].plot(acc_diff, alpha=0.8, color='orange')
            axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Train - Val Accuracy')
            axes[1, 2].set_title('Generalization Gap')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('ReLU Network Training History', fontsize=14)
        plt.tight_layout()
        plt.show()

# ==================== VISUALIZATION TOOLS ====================
class ReLUVisualizer:
    """Visualization tools for ReLU function."""
    
    @staticmethod
    def plot_relu_function(relu_func: BaseReLU,
                          x_range: Tuple[float, float] = (-3, 3),
                          num_points: int = 1000,
                          show_derivative: bool = True,
                          show_integral: bool = False):
        """Plot ReLU function and its properties."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        # Compute values
        y = relu_func.forward(x)
        y_prime = relu_func.derivative(x)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # ReLU function
        axes[0].plot(x.numpy(), y.numpy(), 'b-', linewidth=2)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('ReLU(x)')
        axes[0].set_title(f'ReLU Function ({relu_func.__class__.__name__})')
        axes[0].grid(True, alpha=0.3)
        
        # Derivative
        axes[1].step(x.numpy(), y_prime.numpy(), 'g-', linewidth=2, where='mid')
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel("ReLU'(x)")
        axes[1].set_title('Derivative (Step Function)')
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].grid(True, alpha=0.3)
        
        # Both together
        axes[2].plot(x.numpy(), y.numpy(), 'b-', label='ReLU(x)', linewidth=2)
        axes[2].step(x.numpy(), y_prime.numpy(), 'g-', label="ReLU'(x)", 
                    linewidth=2, where='mid', alpha=0.7)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('Value')
        axes[2].set_title('Function and Derivative')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        if show_integral:
            # Add integral plot
            y_integral = ReLUMathematics().relu_integral(x)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.plot(x.numpy(), y.numpy(), 'b-', label='ReLU(x)', linewidth=2)
            ax2.plot(x.numpy(), y_integral.numpy(), 'r-', label='∫ReLU(x)dx', linewidth=2)
            ax2.set_xlabel('x')
            ax2.set_ylabel('Value')
            ax2.set_title('ReLU and its Integral')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.show()
        
        plt.suptitle(f'ReLU Analysis - {relu_func.__class__.__name__}', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_multiple_relu(relu_funcs: List[BaseReLU],
                          x_range: Tuple[float, float] = (-3, 3),
                          num_points: int = 1000):
        """Compare multiple ReLU implementations."""
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot functions
        for relu_func in relu_funcs:
            y = relu_func.forward(x)
            label = relu_func.__class__.__name__
            if hasattr(relu_func, 'negative_slope'):
                label += f" (α={relu_func.negative_slope})"
            elif hasattr(relu_func, 'alpha'):
                label += f" (α={relu_func.alpha})"
            
            axes[0].plot(x.numpy(), y.numpy(), label=label, linewidth=2)
        
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('f(x)')
        axes[0].set_title('ReLU Variants Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot derivatives
        for relu_func in relu_funcs:
            y_prime = relu_func.derivative(x)
            label = relu_func.__class__.__name__
            
            axes[1].step(x.numpy(), y_prime.numpy(), label=label, 
                        linewidth=2, where='mid', alpha=0.7)
        
        axes[1].set_xlabel('x')
        axes[1].set_ylabel("f'(x)")
        axes[1].set_title('Derivatives Comparison')
        axes[1].set_ylim([-0.1, 1.5])
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('ReLU Variants and Their Derivatives', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_activation_statistics(activations: List[Dict[str, Any]]):
        """Plot statistics of ReLU activations across layers."""
        if not activations:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Sparsity per layer
        layer_indices = range(1, len(activations) + 1)
        sparsities = [a['sparsity'] for a in activations]
        
        axes[0, 0].bar(layer_indices, sparsities, alpha=0.7)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Sparsity')
        axes[0, 0].set_title('Activation Sparsity per Layer')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dead neurons per layer
        dead_neurons = [a['dead_neurons'] for a in activations]
        
        axes[0, 1].bar(layer_indices, dead_neurons, alpha=0.7, color='red')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Dead Neurons Ratio')
        axes[0, 1].set_title('Dead Neurons per Layer')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mean activation per layer
        mean_activations = [a['mean_activation'] for a in activations]
        
        axes[0, 2].bar(layer_indices, mean_activations, alpha=0.7, color='green')
        axes[0, 2].set_xlabel('Layer')
        axes[0, 2].set_ylabel('Mean Activation')
        axes[0, 2].set_title('Mean Activation per Layer')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Max activation per layer
        max_activations = [a['max_activation'] for a in activations]
        
        axes[1, 0].bar(layer_indices, max_activations, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Max Activation')
        axes[1, 0].set_title('Maximum Activation per Layer')
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
        
        # Activation heatmap for first layer
        if len(activations) >= 1:
            im = axes[1, 2].imshow(activations[0]['output'][:50, :50].numpy(), 
                                 aspect='auto', cmap='hot')
            axes[1, 2].set_xlabel('Neuron')
            axes[1, 2].set_ylabel('Sample')
            axes[1, 2].set_title('First Layer Activations (50x50)')
            plt.colorbar(im, ax=axes[1, 2])
        
        plt.suptitle('ReLU Activation Statistics Across Layers', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_gradient_analysis(gradient_stats: Dict[str, Any]):
        """Plot gradient analysis results."""
        if not gradient_stats.get('layer_gradients'):
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Gradient norms per layer
        layer_names = [g['name'].split('.')[0] for g in gradient_stats['layer_gradients']]
        grad_norms = [g['grad_norm'] for g in gradient_stats['layer_gradients']]
        
        axes[0].bar(range(len(layer_names)), grad_norms, alpha=0.7)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Gradient Norm')
        axes[0].set_title('Gradient Norms per Layer')
        axes[0].set_xticks(range(len(layer_names)))
        axes[0].set_xticklabels(layer_names, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Gradient means
        grad_means = [g['grad_mean'] for g in gradient_stats['layer_gradients']]
        
        axes[1].bar(range(len(layer_names)), grad_means, alpha=0.7, color='green')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Gradient Mean')
        axes[1].set_title('Gradient Means per Layer')
        axes[1].set_xticks(range(len(layer_names)))
        axes[1].set_xticklabels(layer_names, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Vanishing/exploding indicators
        vanishing = [1 if g['is_vanishing'] else 0 for g in gradient_stats['layer_gradients']]
        exploding = [1 if g['is_exploding'] else 0 for g in gradient_stats['layer_gradients']]
        
        x = range(len(layer_names))
        width = 0.35
        axes[2].bar(x, vanishing, width, label='Vanishing', alpha=0.7, color='red')
        axes[2].bar([i + width for i in x], exploding, width, label='Exploding', 
                   alpha=0.7, color='orange')
        axes[2].set_xlabel('Layer')
        axes[2].set_ylabel('Indicator')
        axes[2].set_title('Gradient Problems per Layer')
        axes[2].set_xticks([i + width/2 for i in x])
        axes[2].set_xticklabels(layer_names, rotation=45)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Gradient Flow Analysis\n'
                    f'Vanishing layers: {gradient_stats["vanishing_layers"]}, '
                    f'Exploding layers: {gradient_stats["exploding_layers"]}', 
                    fontsize=14)
        plt.tight_layout()
        plt.show()

# ==================== ReLU ANALYSIS AND BENCHMARKING ====================
class ReLUBenchmark:
    """Benchmarking framework for ReLU implementations."""
    
    @staticmethod
    def benchmark_forward_pass(relu_funcs: List[BaseReLU],
                              batch_sizes: List[int] = [1, 10, 100, 1000, 10000],
                              input_dim: int = 100):
        """Benchmark forward pass performance."""
        import time
        
        print("=" * 60)
        print("ReLU FORWARD PASS BENCHMARK")
        print("=" * 60)
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}, Input dim: {input_dim}")
            print("-" * 50)
            
            # Create random input
            x = torch.randn(batch_size, input_dim)
            
            for relu_func in relu_funcs:
                # Warm up
                for _ in range(10):
                    _ = relu_func.forward(x)
                
                # Benchmark
                start_time = time.time()
                iterations = 100 if batch_size < 1000 else 10
                for _ in range(iterations):
                    _ = relu_func.forward(x)
                elapsed = time.time() - start_time
                
                # Store result
                func_name = relu_func.__class__.__name__
                if func_name not in results:
                    results[func_name] = []
                results[func_name].append((batch_size, elapsed / iterations))
                
                print(f"  {func_name:20s}: {elapsed/iterations*1000:6.2f} ms per forward pass")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        for func_name, data in results.items():
            batch_sizes = [d[0] for d in data]
            times = [d[1] * 1000 for d in data]  # Convert to ms
            plt.plot(batch_sizes, times, 'o-', label=func_name, linewidth=2)
        
        plt.xscale('log')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Forward Pass (ms)')
        plt.title('ReLU Forward Pass Performance')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        plt.show()
        
        return results
    
    @staticmethod
    def benchmark_training_comparison(dataset: Dataset,
                                     relu_types: List[str] = ['standard', 'leaky', 'elu', 'swish'],
                                     epochs: int = 20):
        """Compare different ReLU types in training."""
        print("\n" + "=" * 60)
        print("ReLU TRAINING COMPARISON")
        print("=" * 60)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        results = {}
        
        for relu_type in relu_types:
            print(f"\nTraining with {relu_type} ReLU...")
            print("-" * 40)
            
            # Create model
            if hasattr(dataset, 'input_dim'):
                input_dim = dataset.input_dim
            else:
                # Try to infer input dimension
                sample, _ = dataset[0]
                input_dim = sample.shape[0]
            
            model = ReLUNetwork(
                input_dim=input_dim,
                hidden_dims=[256, 128, 64],
                output_dim=10,
                relu_type=relu_type,
                use_batch_norm=True,
                dropout=0.3
            )
            
            # Create trainer
            trainer = ReLUTrainer(
                model=model,
                device=device,
                learning_rate=0.001,
                weight_decay=1e-4,
                optimizer_type='adam'
            )
            
            # Train
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                early_stopping_patience=5
            )
            
            # Store results
            results[relu_type] = {
                'trainer': trainer,
                'final_val_acc': trainer.val_accuracies[-1] if trainer.val_accuracies else 0,
                'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else 0,
                'best_val_acc': max(trainer.val_accuracies) if trainer.val_accuracies else 0
            }
            
            print(f"  Final Validation Accuracy: {results[relu_type]['final_val_acc']:.4f}")
            print(f"  Best Validation Accuracy: {results[relu_type]['best_val_acc']:.4f}")
        
        # Plot comparison
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        
        relu_names = list(results.keys())
        final_accs = [results[name]['final_val_acc'] for name in relu_names]
        best_accs = [results[name]['best_val_acc'] for name in relu_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Final accuracy comparison
        bars1 = axes[0].bar(relu_names, final_accs, alpha=0.7)
        axes[0].set_ylabel('Final Validation Accuracy')
        axes[0].set_title('Final Accuracy Comparison')
        axes[0].set_ylim([0, 1])
        
        for bar, acc in zip(bars1, final_accs):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.4f}', ha='center', va='bottom')
        
        # Best accuracy comparison
        bars2 = axes[1].bar(relu_names, best_accs, alpha=0.7, color='green')
        axes[1].set_ylabel('Best Validation Accuracy')
        axes[1].set_title('Best Accuracy Comparison')
        axes[1].set_ylim([0, 1])
        
        for bar, acc in zip(bars2, best_accs):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    @staticmethod
    def analyze_dead_neuron_problem(network: ReLUNetwork,
                                   dataset: Dataset,
                                   num_samples: int = 1000):
        """Analyze the dead neuron problem in ReLU networks."""
        print("\n" + "=" * 60)
        print("DEAD NEURON ANALYSIS")
        print("=" * 60)
        
        # Create data loader
        loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
        data, _ = next(iter(loader))
        
        # Get layer activations
        activations = network.get_layer_activations(data)
        
        print(f"\nNetwork has {len(activations)} ReLU layers")
        
        total_neurons = 0
        total_dead = 0
        
        for i, activation in enumerate(activations):
            layer_output = activation['output']
            num_neurons = layer_output.shape[1] if len(layer_output.shape) > 1 else 1
            dead_ratio = activation['dead_neurons']
            dead_count = int(dead_ratio * num_neurons)
            
            total_neurons += num_neurons
            total_dead += dead_count
            
            print(f"\nLayer {i+1} ({activation['type']}):")
            print(f"  Total neurons: {num_neurons}")
            print(f"  Dead neurons: {dead_count} ({dead_ratio*100:.1f}%)")
            print(f"  Mean activation: {activation['mean_activation']:.6f}")
            print(f"  Sparsity: {activation['sparsity']*100:.1f}%")
        
        print(f"\nOverall Statistics:")
        print(f"  Total neurons in network: {total_neurons}")
        print(f"  Total dead neurons: {total_dead} ({total_dead/total_neurons*100:.1f}%)")
        print(f"  Overall sparsity: {sum(a['sparsity'] for a in activations)/len(activations)*100:.1f}%")
        
        return activations

# ==================== MAIN DEMONSTRATION ====================
def demonstrate_relu_mathematics():
    """Demonstrate mathematical properties of ReLU."""
    print("=" * 60)
    print("RECTIFIED LINEAR UNIT MATHEMATICAL PROPERTIES")
    print("=" * 60)
    
    math = ReLUMathematics()
    
    # Test points
    test_points = [-3, -2, -1, 0, 1, 2, 3]
    
    print("\nReLU values at test points:")
    print("-" * 40)
    for x in test_points:
        x_tensor = torch.tensor([x])
        relu_val = math.relu(x_tensor).item()
        deriv = math.relu_derivative(x_tensor).item()
        print(f"ReLU({x:2d}) = {relu_val:4.1f}, ReLU'({x:2d}) = {deriv:4.1f}")
    
    print("\nKey Properties:")
    print("1. Piecewise linear: f(x) = max(0, x)")
    print("2. Range: [0, ∞) - Non-negative output")
    print("3. ReLU(0) = 0 (exact zero)")
    print("4. ReLU'(x) = 1 if x > 0, else 0 (step function)")
    print("5. Convex: Both convex and non-decreasing")
    print("6. Sparsity inducing: Sets negative inputs to 0")
    print("7. Computational efficiency: No exponentials, just max operation")
    
    # Demonstrate sparsity
    print("\nSparsity Demonstration:")
    print("-" * 40)
    x = torch.randn(1000) * 2  # Standard normal scaled by 2
    relu_output = math.relu(x)
    sparsity = math.compute_sparsity(relu_output)
    dead_neurons = math.compute_dead_neurons(x.unsqueeze(1))
    
    print(f"Input shape: {x.shape}")
    print(f"Sparsity in ReLU output: {sparsity*100:.1f}% zeros")
    print(f"Dead neurons in batch: {dead_neurons*100:.1f}%")
    
    # Demonstrate gradient flow
    print("\nGradient Flow Demonstration:")
    print("-" * 40)
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=False)
    relu_out = math.relu(x)
    upstream_grad = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    downstream_grad = math.compute_gradient_flow(relu_out, upstream_grad)
    
    for i, (xi, ri, gi) in enumerate(zip(x, relu_out, downstream_grad)):
        print(f"x={xi:4.1f}: ReLU(x)={ri:4.1f}, Gradient flow={gi:4.1f}")

def demonstrate_relu_implementations():
    """Demonstrate different ReLU implementations."""
    print("\n" + "=" * 60)
    print("ReLU IMPLEMENTATIONS")
    print("=" * 60)
    
    # Create different ReLU implementations
    relu_impls = [
        StandardReLU(),
        LeakyReLU(negative_slope=0.01),
        LeakyReLU(negative_slope=0.1),
        ParametricReLU(num_parameters=1, init=0.25),
        ExponentialReLU(alpha=1.0),
        ScaledExponentialReLU(),
        SwishReLU(beta=1.0),
        GaussianReLU(approximate=True),
        SparseReLU(sparsity_target=0.7, temperature=1.0)
    ]
    
    # Visualize them
    visualizer = ReLUVisualizer()
    visualizer.plot_multiple_relu(relu_impls, x_range=(-3, 3))
    
    # Analyze properties at key points
    print("\nProperties at key points:")
    print("-" * 60)
    test_points = [-2, -1, 0, 1, 2]
    
    for point in test_points:
        print(f"\nAt x = {point}:")
        for relu in relu_impls[:3]:  # Show first 3 for clarity
            props = relu.analyze_at_point(point)
            print(f"  {relu.__class__.__name__:20s}: value={props.value:6.3f}, "
                  f"derivative={props.derivative:6.3f}, active={props.is_active}")

def train_relu_network_example():
    """Train a ReLU network on CIFAR-10."""
    print("\n" + "=" * 60)
    print("ReLU NETWORK TRAINING EXAMPLE (CIFAR-10)")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CIFAR-10 dataset
    print("\nLoading CIFAR-10 dataset...")
    try:
        train_dataset = CIFAR10ReLUDataset(train=True, download=True)
        test_dataset = CIFAR10ReLUDataset(train=False, download=True)
    except:
        print("CIFAR-10 not available, using synthetic dataset...")
        train_dataset = ReLUSyntheticDataset(
            num_samples=10000, input_dim=3072, num_classes=10, complexity='deep'
        )
        test_dataset = ReLUSyntheticDataset(
            num_samples=2000, input_dim=3072, num_classes=10, complexity='deep'
        )
    
    # Split training data
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create ReLU network
    print("\nCreating ReLU network...")
    input_dim = 3072 if hasattr(train_dataset, 'dataset') else 100
    model = ReLUNetwork(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],
        output_dim=10,
        relu_type='standard',
        use_batch_norm=True,
        dropout=0.3,
        residual=True
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ReLUTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-4,
        optimizer_type='adam'
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        early_stopping_patience=5
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Test
    test_loss, test_accuracy = trainer.validate(test_loader)
    print(f"\nTest Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    
    # Analyze activations
    print("\nAnalyzing network activations...")
    visualizer = ReLUVisualizer()
    
    # Get sample data for analysis
    sample_data, _ = next(iter(test_loader))
    sample_data = sample_data[:100]  # Use 100 samples
    
    activations = model.get_layer_activations(sample_data)
    visualizer.plot_activation_statistics(activations)
    
    # Analyze gradients
    gradient_stats = model.analyze_gradient_flow(sample_data)
    visualizer.plot_gradient_analysis(gradient_stats)
    
    # Analyze dead neurons
    print("\nAnalyzing dead neuron problem...")
    benchmark = ReLUBenchmark()
    benchmark.analyze_dead_neuron_problem(model, test_dataset, 1000)
    
    return model, trainer

def benchmark_relu_performance():
    """Benchmark ReLU implementations."""
    print("\n" + "=" * 60)
    print("ReLU PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create ReLU implementations
    relu_funcs = [
        StandardReLU(),
        LeakyReLU(negative_slope=0.01),
        ExponentialReLU(alpha=1.0),
        SwishReLU(beta=1.0),
        GaussianReLU(approximate=True)
    ]
    
    # Run benchmarks
    benchmark = ReLUBenchmark()
    
    # Forward pass benchmark
    benchmark.benchmark_forward_pass(relu_funcs)
    
    # Create synthetic dataset for training comparison
    print("\n" + "=" * 60)
    print("TRAINING COMPARISON ON SYNTHETIC DATASET")
    print("=" * 60)
    
    synthetic_dataset = ReLUSyntheticDataset(
        num_samples=5000, input_dim=100, num_classes=10, complexity='deep'
    )
    
    benchmark.benchmark_training_comparison(
        synthetic_dataset,
        relu_types=['standard', 'leaky', 'elu', 'swish'],
        epochs=15
    )

def demonstrate_relu_applications():
    """Demonstrate practical applications of ReLU."""
    print("\n" + "=" * 60)
    print("ReLU PRACTICAL APPLICATIONS")
    print("=" * 60)
    
    print("\n1. Deep Neural Networks:")
    print("   • Default choice for most modern architectures")
    print("   • Used in CNNs (AlexNet, VGG, ResNet, etc.)")
    print("   • Used in Transformers (MLP layers)")
    
    print("\n2. Computer Vision:")
    print("   • Convolutional layers followed by ReLU")
    print("   • Enables learning complex visual features")
    print("   • Sparse activations help with feature selection")
    
    print("\n3. Natural Language Processing:")
    print("   • Used in feed-forward networks in Transformers")
    print("   • Position-wise feed-forward networks")
    print("   • GELU (Gaussian Error Linear Unit) common in BERT/GPT")
    
    print("\n4. Generative Models:")
    print("   • Used in GAN generators and discriminators")
    print("   • Variational Autoencoders")
    print("   • Normalizing flows")
    
    print("\n5. Reinforcement Learning:")
    print("   • Policy networks")
    print("   • Value networks")
    print("   • Q-networks in Deep Q-Learning")
    
    # Demonstrate CNN with ReLU
    print("\n" + "-" * 40)
    print("Simple CNN with ReLU Example:")
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.relu = StandardReLU()
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create a simple CNN
    cnn = SimpleCNN()
    
    # Test with random input (simulating 32x32 RGB image)
    input_tensor = torch.randn(4, 3, 32, 32)  # batch=4, channels=3, height=32, width=32
    output = cnn(input_tensor)
    
    print(f"CNN Input shape: {input_tensor.shape}")
    print(f"CNN Output shape: {output.shape}")
    print(f"Number of ReLU activations: 3")
    print(f"ReLU implementation: {cnn.relu.__class__.__name__}")

def main():
    """Main demonstration function."""
    print("RECTIFIED LINEAR UNIT ACTIVATION FUNCTION IMPLEMENTATION")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demonstrate mathematical properties
    demonstrate_relu_mathematics()
    
    # Demonstrate different implementations
    demonstrate_relu_implementations()
    
    # Train ReLU network
    model, trainer = train_relu_network_example()
    
    # Benchmark performance
    benchmark_relu_performance()
    
    # Demonstrate applications
    demonstrate_relu_applications()
    
    print("\n" + "=" * 60)
    print("ReLU IMPLEMENTATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Implemented:")
    print("1. Standard ReLU: max(0, x)")
    print("2. LeakyReLU: Non-zero gradient for negative inputs")
    print("3. PReLU: Learnable leak parameter")
    print("4. ELU: Exponential for negative values")
    print("5. SELU: Self-normalizing variant")
    print("6. Swish/SiLU: x * sigmoid(x)")
    print("7. GELU: Gaussian Error Linear Unit")
    print("8. SparseReLU: Explicit sparsity control")
    print("9. Complete mathematical analysis tools")
    print("10. Kaiming initialization (optimal for ReLU)")
    print("11. Gradient flow analysis")
    print("12. Dead neuron problem analysis")
    print("13. Training framework for ReLU networks")
    print("14. Benchmarking and comparison tools")
    print("15. Practical applications demonstration")

if __name__ == "__main__":
    main()