import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any, Callable
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# ==================== RMSProp OPTIMIZER (FROM SCRATCH) ====================
class RMSPropOptimizer:
    """
    Root Mean Square Propagation (RMSProp) optimizer implemented from scratch.
    
    RMSProp algorithm:
    1. Compute gradient: g_t = ∇f(θ_t)
    2. Update squared gradient estimate: v_t = β * v_{t-1} + (1-β) * g_t²
    3. Update parameters: θ_{t+1} = θ_t - (α / √(v_t + ε)) * g_t
    """
    
    def __init__(self, 
                 params: List[nn.Parameter],
                 lr: float = 0.001,
                 alpha: float = 0.99,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 momentum: float = 0.0,
                 centered: bool = False):
        """
        Args:
            params: List of parameters to optimize
            lr: Learning rate (α)
            alpha: Decay rate (β) for moving average
            eps: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
            momentum: Momentum factor (0 to disable)
            centered: Whether to use centered RMSProp
        """
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        
        # Initialize state
        self.state: Dict[nn.Parameter, Dict[str, torch.Tensor]] = {}
        self._initialize_state()
        
        # Track statistics
        self.history = {
            'grad_norms': [],
            'param_norms': [],
            'learning_rates': [],
            'velocity_norms': []
        }
        
        self.step_count = 0
    
    def _initialize_state(self):
        """Initialize optimizer state for each parameter."""
        for param in self.params:
            if param not in self.state:
                self.state[param] = {
                    'square_avg': torch.zeros_like(param.data),
                    'momentum_buffer': torch.zeros_like(param.data) if self.momentum > 0 else None,
                    'grad_avg': torch.zeros_like(param.data) if self.centered else None
                }
    
    def zero_grad(self):
        """Clear gradients for all parameters."""
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
    
    def _compute_effective_lr(self, square_avg: torch.Tensor) -> torch.Tensor:
        """Compute effective learning rate for each parameter."""
        # RMSProp update: lr / sqrt(v + eps)
        return self.lr / torch.sqrt(square_avg + self.eps)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model
                    and returns the loss
            
        Returns:
            Loss if closure provided, else None
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        self.step_count += 1
        
        # Update each parameter
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)
            
            # Get state for this parameter
            state = self.state[param]
            square_avg = state['square_avg']
            
            # Update squared gradient estimate
            # v_t = β * v_{t-1} + (1-β) * g_t²
            square_avg.mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
            
            # Centered RMSProp: maintain average of gradients
            if self.centered:
                grad_avg = state['grad_avg']
                grad_avg.mul_(self.alpha).add_(grad, alpha=1 - self.alpha)
                
                # Compute centered variance
                avg = square_avg - grad_avg.pow(2)
                denom = avg.sqrt().add_(self.eps)
            else:
                denom = square_avg.sqrt().add_(self.eps)
            
            # Compute update
            if self.momentum > 0:
                buf = state['momentum_buffer']
                
                # Momentum update
                buf.mul_(self.momentum).addcdiv_(grad, denom, value=-self.lr)
                
                # Apply update
                param.data.add_(buf)
            else:
                # Standard RMSProp update
                param.data.addcdiv_(grad, denom, value=-self.lr)
            
            # Track statistics
            self._track_statistics(param, grad, square_avg, denom)
        
        return loss
    
    def _track_statistics(self, 
                         param: nn.Parameter,
                         grad: torch.Tensor,
                         square_avg: torch.Tensor,
                         denom: torch.Tensor):
        """Track optimization statistics."""
        self.history['grad_norms'].append(grad.norm().item())
        self.history['param_norms'].append(param.data.norm().item())
        
        # Compute effective learning rate for this parameter
        effective_lr = self.lr / torch.sqrt(square_avg + self.eps)
        self.history['learning_rates'].append(effective_lr.mean().item())
        
        if self.momentum > 0:
            buf = self.state[param]['momentum_buffer']
            self.history['velocity_norms'].append(buf.norm().item())
    
    def get_current_lr(self) -> float:
        """Get current effective learning rate."""
        if self.history['learning_rates']:
            return self.history['learning_rates'][-1]
        return self.lr
    
    def set_lr(self, lr: float):
        """Update learning rate."""
        self.lr = lr
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dictionary."""
        return {
            'state': {id(p): s for p, s in self.state.items()},
            'param_groups': [{'params': [id(p) for p in self.params]}],
            'step_count': self.step_count,
            'lr': self.lr,
            'alpha': self.alpha,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'centered': self.centered,
            'history': self.history
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from dictionary."""
        self.step_count = state_dict['step_count']
        self.lr = state_dict['lr']
        self.alpha = state_dict['alpha']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.momentum = state_dict['momentum']
        self.centered = state_dict['centered']
        self.history = state_dict['history']

# ==================== ENHANCED RMSProp VARIANTS ====================
class RMSPropW(RMSPropOptimizer):
    """
    RMSProp with weight decay fix (AdamW style weight decay).
    Applies weight decay separately from gradient update.
    """
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step with decoupled weight decay."""
        loss = None
        if closure is not None:
            loss = closure()
        
        self.step_count += 1
        
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Apply weight decay (AdamW style)
            if self.weight_decay != 0:
                param.data.mul_(1 - self.lr * self.weight_decay)
            
            # Get state
            state = self.state[param]
            square_avg = state['square_avg']
            
            # Update squared gradient estimate
            square_avg.mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
            
            # Compute denominator
            if self.centered:
                grad_avg = state['grad_avg']
                grad_avg.mul_(self.alpha).add_(grad, alpha=1 - self.alpha)
                denom = (square_avg - grad_avg.pow(2)).sqrt().add_(self.eps)
            else:
                denom = square_avg.sqrt().add_(self.eps)
            
            # Apply update
            if self.momentum > 0:
                buf = state['momentum_buffer']
                buf.mul_(self.momentum).addcdiv_(grad, denom, value=-self.lr)
                param.data.add_(buf)
            else:
                param.data.addcdiv_(grad, denom, value=-self.lr)
            
            # Track statistics
            self._track_statistics(param, grad, square_avg, denom)
        
        return loss

class NesterovRMSProp(RMSPropOptimizer):
    """
    RMSProp with Nesterov momentum.
    """
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step with Nesterov momentum."""
        loss = None
        if closure is not None:
            loss = closure()
        
        self.step_count += 1
        
        for param in self.params:
            if param.grad is None:
                continue
            
            # Nesterov: look ahead
            if self.momentum > 0:
                state = self.state[param]
                buf = state['momentum_buffer']
                
                # Apply momentum to get lookahead position
                param.data.add_(buf, alpha=self.momentum)
            
            # Compute gradient at lookahead position
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)
            
            # Update state
            state = self.state[param]
            square_avg = state['square_avg']
            
            # Update squared gradient estimate
            square_avg.mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
            
            # Compute denominator
            if self.centered:
                grad_avg = state['grad_avg']
                grad_avg.mul_(self.alpha).add_(grad, alpha=1 - self.alpha)
                denom = (square_avg - grad_avg.pow(2)).sqrt().add_(self.eps)
            else:
                denom = square_avg.sqrt().add_(self.eps)
            
            # Apply update with Nesterov correction
            if self.momentum > 0:
                buf = state['momentum_buffer']
                
                # Save old momentum
                old_buf = buf.clone()
                
                # Update momentum
                buf.mul_(self.momentum).addcdiv_(grad, denom, value=-self.lr)
                
                # Apply Nesterov correction
                param.data.add_(buf.mul_(1 + self.momentum).sub_(old_buf))
            else:
                param.data.addcdiv_(grad, denom, value=-self.lr)
            
            # Track statistics
            self._track_statistics(param, grad, square_avg, denom)
        
        return loss

# ==================== LEARNING RATE SCHEDULERS ====================
class RMSPropScheduler:
    """Learning rate scheduler for RMSProp."""
    
    def __init__(self, 
                 optimizer: RMSPropOptimizer,
                 schedule_type: str = 'exponential',
                 **kwargs):
        """
        Args:
            optimizer: RMSProp optimizer
            schedule_type: Type of schedule ('exponential', 'cosine', 'step', 'cyclic')
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.base_lr = optimizer.lr
        
        if schedule_type == 'exponential':
            self.gamma = kwargs.get('gamma', 0.95)
            self.step_size = kwargs.get('step_size', 10)
        elif schedule_type == 'cosine':
            self.T_max = kwargs.get('T_max', 50)
            self.eta_min = kwargs.get('eta_min', 1e-6)
        elif schedule_type == 'step':
            self.step_size = kwargs.get('step_size', 30)
            self.gamma = kwargs.get('gamma', 0.1)
        elif schedule_type == 'cyclic':
            self.base_lr = kwargs.get('base_lr', 1e-4)
            self.max_lr = kwargs.get('max_lr', 1e-2)
            self.step_size = kwargs.get('step_size', 2000)
            self.mode = kwargs.get('mode', 'triangular')
    
    def step(self, epoch: int = None):
        """Update learning rate based on schedule."""
        if epoch is None:
            epoch = self.optimizer.step_count
        
        if self.schedule_type == 'exponential':
            # Exponential decay: lr = base_lr * gamma^(epoch/step_size)
            new_lr = self.base_lr * (self.gamma ** (epoch / self.step_size))
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing: lr = eta_min + 0.5*(base_lr - eta_min)*(1 + cos(π*epoch/T_max))
            if epoch < self.T_max:
                new_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
                        (1 + math.cos(math.pi * epoch / self.T_max))
            else:
                new_lr = self.eta_min
        
        elif self.schedule_type == 'step':
            # Step decay: lr = base_lr * gamma^(epoch // step_size)
            new_lr = self.base_lr * (self.gamma ** (epoch // self.step_size))
        
        elif self.schedule_type == 'cyclic':
            # Cyclical learning rates
            cycle = math.floor(1 + epoch / (2 * self.step_size))
            x = abs(epoch / self.step_size - 2 * cycle + 1)
            
            if self.mode == 'triangular':
                lr_range = self.max_lr - self.base_lr
                new_lr = self.base_lr + lr_range * max(0, (1 - x))
            elif self.mode == 'triangular2':
                lr_range = (self.max_lr - self.base_lr) / (2 ** (cycle - 1))
                new_lr = self.base_lr + lr_range * max(0, (1 - x))
            else:
                new_lr = self.base_lr
        
        else:
            new_lr = self.base_lr
        
        self.optimizer.set_lr(new_lr)
        return new_lr

# ==================== TEST FUNCTIONS FOR OPTIMIZATION ====================
class TestFunctions:
    """Collection of test functions for optimizer benchmarking."""
    
    @staticmethod
    def rosenbrock(x: torch.Tensor, y: torch.Tensor, a: float = 1, b: float = 100) -> torch.Tensor:
        """
        Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
        Global minimum at (a, a²)
        """
        return (a - x) ** 2 + b * (y - x ** 2) ** 2
    
    @staticmethod
    def rastrigin(x: torch.Tensor, y: torch.Tensor, A: float = 10) -> torch.Tensor:
        """
        Rastrigin function: f(x,y) = A*2 + Σ[x_i² - A*cos(2πx_i)]
        Global minimum at (0,0)
        """
        return A * 2 + (x ** 2 - A * torch.cos(2 * math.pi * x)) + \
                      (y ** 2 - A * torch.cos(2 * math.pi * y))
    
    @staticmethod
    def himmelblau(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Himmelblau's function: f(x,y) = (x²+y-11)² + (x+y²-7)²
        4 minima at (3,2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
        """
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    
    @staticmethod
    def ackley(x: torch.Tensor, y: torch.Tensor, a: float = 20, 
               b: float = 0.2, c: float = 2 * math.pi) -> torch.Tensor:
        """
        Ackley function: f(x,y) = -a*exp(-b*sqrt(0.5*(x²+y²))) - 
                                   exp(0.5*(cos(cx)+cos(cy))) + a + exp(1)
        Global minimum at (0,0)
        """
        return -a * torch.exp(-b * torch.sqrt(0.5 * (x ** 2 + y ** 2))) - \
               torch.exp(0.5 * (torch.cos(c * x) + torch.cos(c * y))) + a + math.e
    
    @staticmethod
    def beale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
        Global minimum at (3, 0.5)
        """
        return (1.5 - x + x * y) ** 2 + \
               (2.25 - x + x * y ** 2) ** 2 + \
               (2.625 - x + x * y ** 3) ** 2
    
    @staticmethod
    def get_function(name: str) -> Callable:
        """Get test function by name."""
        functions = {
            'rosenbrock': TestFunctions.rosenbrock,
            'rastrigin': TestFunctions.rastrigin,
            'himmelblau': TestFunctions.himmelblau,
            'ackley': TestFunctions.ackley,
            'beale': TestFunctions.beale
        }
        return functions.get(name, TestFunctions.rosenbrock)

# ==================== OPTIMIZER COMPARISON FRAMEWORK ====================
class OptimizerComparison:
    """Framework for comparing different optimizers."""
    
    def __init__(self, 
                 test_function: str = 'rosenbrock',
                 bounds: Tuple[float, float, float, float] = (-5, 5, -5, 5)):
        """
        Args:
            test_function: Name of test function
            bounds: (x_min, x_max, y_min, y_max) for visualization
        """
        self.test_function = test_function
        self.func = TestFunctions.get_function(test_function)
        self.bounds = bounds
        self.results = {}
        
    def optimize(self,
                 optimizer_class: type,
                 init_params: Tuple[float, float],
                 iterations: int = 1000,
                 **optimizer_kwargs) -> Dict[str, Any]:
        """
        Run optimization with given optimizer.
        
        Returns:
            Dictionary with optimization results
        """
        # Initialize parameters
        x = torch.tensor(init_params[0], requires_grad=True)
        y = torch.tensor(init_params[1], requires_grad=True)
        
        # Create optimizer
        optimizer = optimizer_class([x, y], **optimizer_kwargs)
        
        # Track optimization path
        path = [init_params]
        losses = []
        
        # Optimization loop
        for i in range(iterations):
            # Compute loss
            loss = self.func(x, y)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Optimization step
            optimizer.step()
            
            # Track progress
            path.append((x.item(), y.item()))
            losses.append(loss.item())
            
            # Early stopping if converged
            if i > 10 and abs(losses[-1] - losses[-2]) < 1e-8:
                break
        
        return {
            'path': np.array(path),
            'losses': losses,
            'final_params': (x.item(), y.item()),
            'final_loss': losses[-1],
            'optimizer_state': optimizer.state_dict(),
            'optimizer_history': optimizer.history
        }
    
    def compare_optimizers(self,
                          optimizers: Dict[str, Dict],
                          init_params: Tuple[float, float] = (0, 0),
                          iterations: int = 500) -> Dict[str, Any]:
        """
        Compare multiple optimizers.
        
        Args:
            optimizers: Dictionary of {name: {class: OptimizerClass, kwargs: {...}}}
            init_params: Initial parameters
            iterations: Number of iterations
        
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for name, opt_info in optimizers.items():
            print(f"Running {name}...")
            
            result = self.optimize(
                optimizer_class=opt_info['class'],
                init_params=init_params,
                iterations=iterations,
                **opt_info.get('kwargs', {})
            )
            
            comparison_results[name] = result
        
        self.results = comparison_results
        return comparison_results
    
    def visualize_comparison(self, figsize: Tuple[int, int] = (15, 10)):
        """Visualize comparison of optimizers."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Contour plot with optimization paths
        ax = axes[0]
        self._plot_contour(ax)
        
        for name, result in self.results.items():
            path = result['path']
            ax.plot(path[:, 0], path[:, 1], 'o-', markersize=3, 
                   linewidth=1, label=name, alpha=0.7)
        
        ax.set_title(f'{self.test_function.capitalize()} Function\nOptimization Paths')
        ax.legend()
        
        # 2. Loss convergence
        ax = axes[1]
        for name, result in self.results.items():
            losses = result['losses']
            ax.semilogy(losses, label=name, alpha=0.7)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Loss Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Gradient norm evolution
        ax = axes[2]
        for name, result in self.results.items():
            if 'optimizer_history' in result:
                grad_norms = result['optimizer_history']['grad_norms']
                ax.plot(grad_norms, label=name, alpha=0.7)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Learning rate evolution
        ax = axes[3]
        for name, result in self.results.items():
            if 'optimizer_history' in result and 'learning_rates' in result['optimizer_history']:
                lrs = result['optimizer_history']['learning_rates']
                ax.plot(lrs, label=name, alpha=0.7)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Parameter norm evolution
        ax = axes[4]
        for name, result in self.results.items():
            if 'optimizer_history' in result and 'param_norms' in result['optimizer_history']:
                param_norms = result['optimizer_history']['param_norms']
                ax.plot(param_norms, label=name, alpha=0.7)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Norm')
        ax.set_title('Parameter Norm Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Final performance comparison
        ax = axes[5]
        names = list(self.results.keys())
        final_losses = [self.results[name]['final_loss'] for name in names]
        
        bars = ax.bar(names, final_losses)
        ax.set_ylabel('Final Loss')
        ax.set_title('Final Loss Comparison')
        ax.set_yscale('log')
        
        # Add value labels on bars
        for bar, loss in zip(bars, final_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{loss:.2e}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_contour(self, ax):
        """Plot contour of test function."""
        x_min, x_max, y_min, y_max = self.bounds
        
        # Create grid
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute function values
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.func(torch.tensor(X[i, j]), torch.tensor(Y[i, j])).item()
        
        # Plot contour
        levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 20)
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        ax.contour(X, Y, Z, levels=levels, colors='k', alpha=0.3, linewidths=0.5)
        
        # Mark global minimum if known
        if self.test_function == 'rosenbrock':
            ax.plot(1, 1, 'r*', markersize=15, label='Global Minimum (1,1)')
        elif self.test_function == 'rastrigin' or self.test_function == 'ackley':
            ax.plot(0, 0, 'r*', markersize=15, label='Global Minimum (0,0)')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

# ==================== NEURAL NETWORK FOR REAL DATASET ====================
class RMSPropNetwork(nn.Module):
    """Neural network for testing RMSProp on real datasets."""
    
    def __init__(self, 
                 input_dim: int = 784,
                 hidden_dims: List[int] = [256, 128, 64],
                 output_dim: int = 10,
                 activation: str = 'relu',
                 dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)

# ==================== DATASET LOADERS ====================
class MNISTRMSPropDataset(Dataset):
    """MNIST dataset wrapper for RMSProp testing."""
    
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
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class FashionMNISTDataset(Dataset):
    """FashionMNIST dataset for testing."""
    
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
        return self.dataset[idx]

# ==================== TRAINING FRAMEWORK ====================
class RMSPropTrainer:
    """Training framework for RMSProp optimizer."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 optimizer_class: type = RMSPropOptimizer,
                 **optimizer_kwargs):
        """
        Args:
            model: Neural network model
            device: Training device
            optimizer_class: RMSProp optimizer class
            optimizer_kwargs: Arguments for optimizer
        """
        self.model = model.to(device)
        self.device = device
        
        # Create optimizer
        self.optimizer = optimizer_class(
            model.parameters(),
            **optimizer_kwargs
        )
        
        # Learning rate scheduler
        self.scheduler = RMSPropScheduler(
            self.optimizer,
            schedule_type='cosine',
            T_max=50,
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
    
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
            loss = self.criterion(output, target)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += len(data)
        
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
              epochs: int = 20):
        """Full training loop."""
        print(f"Training with {self.optimizer.__class__.__name__}...")
        
        for epoch in range(epochs):
            # Update learning rate
            current_lr = self.scheduler.step(epoch)
            self.learning_rates.append(current_lr)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.6f}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train', alpha=0.8)
        axes[0, 0].plot(self.val_losses, label='Val', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.train_accuracies, label='Train', alpha=0.8)
        axes[0, 1].plot(self.val_accuracies, label='Val', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(self.learning_rates, alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient norms from optimizer history
        if hasattr(self.optimizer, 'history') and 'grad_norms' in self.optimizer.history:
            axes[1, 1].plot(self.optimizer.history['grad_norms'], alpha=0.8)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Norm Evolution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==================== BENCHMARKING UTILITIES ====================
class RMSPropBenchmark:
    """Benchmark RMSProp against other optimizers."""
    
    @staticmethod
    def benchmark_on_test_functions():
        """Benchmark optimizers on test functions."""
        print("=" * 60)
        print("BENCHMARKING RMSProp ON TEST FUNCTIONS")
        print("=" * 60)
        
        # Define optimizers to compare
        optimizers = {
            'RMSProp': {
                'class': RMSPropOptimizer,
                'kwargs': {'lr': 0.01, 'alpha': 0.99}
            },
            'RMSPropW': {
                'class': RMSPropW,
                'kwargs': {'lr': 0.01, 'alpha': 0.99, 'weight_decay': 0.01}
            },
            'NesterovRMSProp': {
                'class': NesterovRMSProp,
                'kwargs': {'lr': 0.01, 'alpha': 0.99, 'momentum': 0.9}
            },
            'CenteredRMSProp': {
                'class': RMSPropOptimizer,
                'kwargs': {'lr': 0.01, 'alpha': 0.99, 'centered': True}
            }
        }
        
        # Test functions to use
        test_functions = ['rosenbrock', 'rastrigin', 'himmelblau']
        
        for func_name in test_functions:
            print(f"\nTesting on {func_name.capitalize()} function:")
            print("-" * 40)
            
            comparator = OptimizerComparison(
                test_function=func_name,
                bounds=(-5, 5, -5, 5)
            )
            
            results = comparator.compare_optimizers(
                optimizers=optimizers,
                init_params=(0, 0),
                iterations=200
            )
            
            # Print results
            for name, result in results.items():
                final_params = result['final_params']
                final_loss = result['final_loss']
                print(f"{name:20s}: Final loss = {final_loss:.2e}, "
                      f"Params = ({final_params[0]:.4f}, {final_params[1]:.4f})")
            
            # Visualize
            comparator.visualize_comparison(figsize=(12, 8))
    
    @staticmethod
    def benchmark_on_mnist():
        """Benchmark RMSProp on MNIST dataset."""
        print("\n" + "=" * 60)
        print("BENCHMARKING RMSProp ON MNIST DATASET")
        print("=" * 60)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load MNIST dataset
        print("\nLoading MNIST dataset...")
        train_dataset = MNISTRMSPropDataset(train=True, download=True)
        test_dataset = MNISTRMSPropDataset(train=False, download=True)
        
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
        
        # Define optimizers to compare
        optimizer_configs = {
            'RMSProp (lr=0.001)': {
                'class': RMSPropOptimizer,
                'kwargs': {'lr': 0.001, 'alpha': 0.99, 'eps': 1e-8}
            },
            'RMSProp (lr=0.01)': {
                'class': RMSPropOptimizer,
                'kwargs': {'lr': 0.01, 'alpha': 0.99, 'eps': 1e-8}
            },
            'RMSPropW': {
                'class': RMSPropW,
                'kwargs': {'lr': 0.001, 'alpha': 0.99, 'eps': 1e-8, 'weight_decay': 0.01}
            },
            'NesterovRMSProp': {
                'class': NesterovRMSProp,
                'kwargs': {'lr': 0.001, 'alpha': 0.99, 'eps': 1e-8, 'momentum': 0.9}
            }
        }
        
        results = {}
        
        for name, config in optimizer_configs.items():
            print(f"\nTraining with {name}...")
            
            # Create model
            model = RMSPropNetwork(
                input_dim=784,
                hidden_dims=[256, 128, 64],
                output_dim=10,
                dropout=0.3
            )
            
            # Create trainer
            trainer = RMSPropTrainer(
                model=model,
                device=device,
                optimizer_class=config['class'],
                **config['kwargs']
            )
            
            # Train
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=15  # Reduced for demonstration
            )
            
            # Test
            test_loss, test_acc = trainer.validate(test_loader)
            print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
            
            # Store results
            results[name] = {
                'trainer': trainer,
                'test_loss': test_loss,
                'test_acc': test_acc
            }
        
        # Plot comparison
        print("\n" + "=" * 60)
        print("FINAL COMPARISON RESULTS")
        print("=" * 60)
        
        names = list(results.keys())
        test_accs = [results[name]['test_acc'] for name in names]
        test_losses = [results[name]['test_loss'] for name in names]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        bars1 = axes[0].bar(names, test_accs)
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy Comparison')
        axes[0].set_ylim(0.8, 1.0)
        
        for bar, acc in zip(bars1, test_accs):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.4f}', ha='center', va='bottom')
        
        # Loss comparison
        bars2 = axes[1].bar(names, test_losses)
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('Test Loss Comparison')
        
        for bar, loss in zip(bars2, test_losses):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{loss:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return results

# ==================== VISUALIZATION TOOLS ====================
class RMSPropVisualizer:
    """Visualization tools for RMSProp optimizer."""
    
    @staticmethod
    def visualize_gradient_accumulation(optimizer: RMSPropOptimizer):
        """Visualize gradient accumulation in RMSProp."""
        if not optimizer.history['grad_norms']:
            print("No history data available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Gradient norms
        axes[0, 0].plot(optimizer.history['grad_norms'])
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].set_title('Gradient Norm Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter norms
        axes[0, 1].plot(optimizer.history['param_norms'])
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Parameter Norm')
        axes[0, 1].set_title('Parameter Norm Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rates
        if optimizer.history['learning_rates']:
            axes[1, 0].plot(optimizer.history['learning_rates'])
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Effective Learning Rate')
            axes[1, 0].set_title('Learning Rate Adaptation')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Velocity norms (if momentum is used)
        if optimizer.history['velocity_norms']:
            axes[1, 1].plot(optimizer.history['velocity_norms'])
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Velocity Norm')
            axes[1, 1].set_title('Momentum Buffer Norm')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'RMSProp Optimizer Statistics (α={optimizer.alpha}, ε={optimizer.eps})', 
                    fontsize=14)
        plt.show()
    
    @staticmethod
    def create_optimization_animation(comparator: OptimizerComparison,
                                     optimizer_name: str,
                                     interval: int = 50):
        """Create animation of optimization process."""
        result = comparator.results.get(optimizer_name)
        if result is None:
            print(f"No results for {optimizer_name}")
            return
        
        path = result['path']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot contour
        comparator._plot_contour(ax)
        
        # Initialize line and point
        line, = ax.plot([], [], 'o-', markersize=4, linewidth=1, color='red')
        point, = ax.plot([], [], 'ro', markersize=8)
        
        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point
        
        def update(frame):
            # Update line up to current frame
            line.set_data(path[:frame+1, 0], path[:frame+1, 1])
            point.set_data(path[frame:frame+1, 0], path[frame:frame+1, 1])
            ax.set_title(f'{optimizer_name} - Iteration {frame+1}/{len(path)}')
            return line, point
        
        anim = FuncAnimation(fig, update, frames=len(path),
                            init_func=init, interval=interval, blit=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim

# ==================== MAIN DEMONSTRATION ====================
def demonstrate_rmsprop_math():
    """Demonstrate RMSProp mathematical properties."""
    print("=" * 60)
    print("RMSProp MATHEMATICAL FOUNDATION")
    print("=" * 60)
    
    print("\nRMSProp Algorithm:")
    print("1. Compute gradient: g_t = ∇f(θ_t)")
    print("2. Update squared gradient estimate: v_t = β * v_{t-1} + (1-β) * g_t²")
    print("3. Update parameters: θ_{t+1} = θ_t - (α / √(v_t + ε)) * g_t")
    print("\nWhere:")
    print("  α = learning rate")
    print("  β = decay rate (typically 0.99)")
    print("  ε = small constant for numerical stability (1e-8)")
    print("\nKey Properties:")
    print("  • Adapts learning rate per parameter")
    print("  • Accumulates squared gradients (moving average)")
    print("  • Works well for non-stationary objectives")
    print("  • Good for online and batch learning")
    
    # Demonstrate update calculation
    print("\n" + "-" * 40)
    print("EXAMPLE CALCULATION:")
    
    # Simulate a simple optimization
    params = [torch.tensor(2.0, requires_grad=True)]
    
    # Create RMSProp optimizer
    optimizer = RMSPropOptimizer(params, lr=0.1, alpha=0.9, eps=1e-8)
    
    print(f"Initial parameter: {params[0].item():.4f}")
    
    for i in range(5):
        # Simple quadratic loss
        loss = params[0] ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Get state
        state = optimizer.state[params[0]]
        square_avg = state['square_avg'].item()
        grad = params[0].grad.item() if params[0].grad is not None else 0
        
        print(f"\nStep {i+1}:")
        print(f"  Gradient: {grad:.4f}")
        print(f"  Squared avg: {square_avg:.6f}")
        print(f"  Effective LR: {0.1/math.sqrt(square_avg + 1e-8):.6f}")
        print(f"  New param: {params[0].item():.4f}")

def demonstrate_rmsprop_variants():
    """Demonstrate different RMSProp variants."""
    print("\n" + "=" * 60)
    print("RMSProp VARIANTS")
    print("=" * 60)
    
    variants = [
        ('Standard RMSProp', RMSPropOptimizer, 
         {'lr': 0.01, 'alpha': 0.99}),
        ('RMSProp with Momentum', RMSPropOptimizer,
         {'lr': 0.01, 'alpha': 0.99, 'momentum': 0.9}),
        ('Centered RMSProp', RMSPropOptimizer,
         {'lr': 0.01, 'alpha': 0.99, 'centered': True}),
        ('RMSPropW', RMSPropW,
         {'lr': 0.01, 'alpha': 0.99, 'weight_decay': 0.01}),
        ('Nesterov RMSProp', NesterovRMSProp,
         {'lr': 0.01, 'alpha': 0.99, 'momentum': 0.9})
    ]
    
    for name, cls, kwargs in variants:
        print(f"\n{name}:")
        print(f"  Class: {cls.__name__}")
        print(f"  Parameters: {kwargs}")

def main():
    """Main demonstration function."""
    print("RMSProp OPTIMIZER IMPLEMENTATION FROM SCRATCH")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demonstrate mathematical foundation
    demonstrate_rmsprop_math()
    
    # Demonstrate variants
    demonstrate_rmsprop_variants()
    
    # Benchmark on test functions
    RMSPropBenchmark.benchmark_on_test_functions()
    
    # Benchmark on MNIST
    results = RMSPropBenchmark.benchmark_on_mnist()
    
    print("\n" + "=" * 60)
    print("RMSProp IMPLEMENTATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Implemented:")
    print("1. Standard RMSProp with adaptive learning rates")
    print("2. RMSPropW with decoupled weight decay")
    print("3. Nesterov RMSProp with momentum")
    print("4. Centered RMSProp variant")
    print("5. Learning rate schedulers")
    print("6. Comprehensive visualization tools")
    print("7. Benchmarking framework")
    print("8. Real dataset testing (MNIST)")

if __name__ == "__main__":
    main()