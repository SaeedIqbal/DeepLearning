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

# ==================== MATHEMATICAL FOUNDATION ====================
class AdamMathematics:
    """
    Mathematical foundations of the Adam optimizer.
    Implements all core equations from the original paper.
    """
    
    @staticmethod
    def compute_moving_average(current: float, 
                               new_value: float, 
                               beta: float) -> float:
        """
        Compute exponential moving average.
        EMA_t = β * EMA_{t-1} + (1-β) * value_t
        """
        return beta * current + (1 - beta) * new_value
    
    @staticmethod
    def compute_bias_correction(raw_estimate: float, 
                               beta: float, 
                               timestep: int) -> float:
        """
        Compute bias correction for moving averages.
        corrected = raw / (1 - β^t)
        """
        if timestep == 0:
            return 0.0
        return raw_estimate / (1 - beta ** timestep)
    
    @staticmethod
    def compute_adaptive_learning_rate(base_lr: float,
                                      m_hat: torch.Tensor,
                                      v_hat: torch.Tensor,
                                      epsilon: float) -> torch.Tensor:
        """
        Compute adaptive learning rate for each parameter.
        η_hat = η / (√v_hat + ε)
        """
        return base_lr / (torch.sqrt(v_hat) + epsilon)
    
    @staticmethod
    def compute_parameter_update(m_hat: torch.Tensor,
                                adaptive_lr: torch.Tensor) -> torch.Tensor:
        """
        Compute parameter update.
        Δθ = -η_hat * m_hat
        """
        return -adaptive_lr * m_hat

# ==================== ADAM STATE ====================
@dataclass
class AdamParameterState:
    """State for a single parameter in Adam optimizer."""
    m: torch.Tensor  # First moment estimate
    v: torch.Tensor  # Second moment estimate
    timestep: int = 0  # Time step for bias correction
    
    def update_moment(self, 
                     grad: torch.Tensor, 
                     beta1: float, 
                     beta2: float) -> None:
        """Update moments with new gradient."""
        # Update first moment (mean)
        self.m = beta1 * self.m + (1 - beta1) * grad
        
        # Update second moment (uncentered variance)
        self.v = beta2 * self.v + (1 - beta2) * grad * grad
        
        self.timestep += 1
    
    def get_bias_corrected(self, 
                          beta1: float, 
                          beta2: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get bias-corrected moment estimates."""
        if self.timestep == 0:
            return torch.zeros_like(self.m), torch.zeros_like(self.v)
        
        m_hat = self.m / (1 - beta1 ** self.timestep)
        v_hat = self.v / (1 - beta2 ** self.timestep)
        
        return m_hat, v_hat

# ==================== ADAM OPTIMIZER (FROM SCRATCH) ====================
class AdamOptimizer:
    """
    Adaptive Moment Estimation (Adam) optimizer implemented from scratch.
    
    Algorithm from "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014):
    1. m_t = β1 * m_{t-1} + (1-β1) * g_t
    2. v_t = β2 * v_{t-1} + (1-β2) * g_t²
    3. m_hat_t = m_t / (1-β1^t)
    4. v_hat_t = v_t / (1-β2^t)
    5. θ_{t+1} = θ_t - η * m_hat_t / (√v_hat_t + ε)
    """
    
    def __init__(self, 
                 params: List[nn.Parameter],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 amsgrad: bool = False,
                 maximize: bool = False):
        """
        Args:
            params: List of parameters to optimize
            lr: Learning rate (η)
            betas: Coefficients for computing running averages (β1, β2)
            eps: Term for numerical stability (ε)
            weight_decay: Weight decay (L2 regularization) coefficient
            amsgrad: Whether to use AMSGrad variant
            maximize: Maximize the objective instead of minimization
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        
        # Initialize states
        self.states: Dict[nn.Parameter, AdamParameterState] = {}
        self._initialize_states()
        
        # For AMSGrad
        if amsgrad:
            self.max_v: Dict[nn.Parameter, torch.Tensor] = {}
            for param in self.params:
                self.max_v[param] = torch.zeros_like(param.data)
        
        # Tracking and analysis
        self.timestep = 0
        self.history = {
            'grad_norms': [],
            'param_norms': [],
            'm_norms': [],
            'v_norms': [],
            'effective_lrs': [],
            'update_norms': [],
            'beta1_powers': [],
            'beta2_powers': []
        }
        
        # Mathematical helper
        self.math = AdamMathematics()
    
    def _initialize_states(self):
        """Initialize optimizer states for all parameters."""
        for param in self.params:
            if param not in self.states:
                self.states[param] = AdamParameterState(
                    m=torch.zeros_like(param.data),
                    v=torch.zeros_like(param.data),
                    timestep=0
                )
    
    def zero_grad(self):
        """Clear gradients for all parameters."""
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
    
    def _apply_weight_decay(self, 
                           param: nn.Parameter, 
                           grad: torch.Tensor) -> torch.Tensor:
        """Apply weight decay to gradient."""
        if self.weight_decay != 0:
            # L2 regularization: add weight_decay * param to gradient
            grad = grad.add(param.data, alpha=self.weight_decay)
        return grad
    
    def _compute_adaptive_update(self,
                                m_hat: torch.Tensor,
                                v_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive update using Adam formula.
        update = -lr * m_hat / (sqrt(v_hat) + eps)
        """
        # Compute denominator with numerical stability
        denominator = torch.sqrt(v_hat) + self.eps
        
        # Compute update
        update = -self.lr * m_hat / denominator
        
        # Flip sign if maximizing
        if self.maximize:
            update = -update
        
        return update
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model
                    and returns the loss
            
        Returns:
            Loss if closure provided, else None
        """
        self.timestep += 1
        
        loss = None
        if closure is not None:
            loss = closure()
        
        # Update each parameter
        for param in self.params:
            if param.grad is None:
                continue
            
            # Get gradient
            grad = param.grad.data
            
            # Apply weight decay
            grad = self._apply_weight_decay(param, grad)
            
            # Get state for this parameter
            state = self.states[param]
            
            # Update moments
            state.update_moment(grad, self.beta1, self.beta2)
            
            # Get bias-corrected moments
            m_hat, v_hat = state.get_bias_corrected(self.beta1, self.beta2)
            
            # For AMSGrad, keep running maximum of v_hat
            if self.amsgrad:
                max_v = self.max_v[param]
                max_v = torch.max(max_v, v_hat)
                self.max_v[param] = max_v
                v_hat = max_v
            
            # Compute update
            update = self._compute_adaptive_update(m_hat, v_hat)
            
            # Apply update
            param.data.add_(update)
            
            # Track statistics
            self._track_statistics(param, grad, state, update, m_hat, v_hat)
        
        return loss
    
    def _track_statistics(self,
                         param: nn.Parameter,
                         grad: torch.Tensor,
                         state: AdamParameterState,
                         update: torch.Tensor,
                         m_hat: torch.Tensor,
                         v_hat: torch.Tensor):
        """Track optimization statistics for analysis."""
        self.history['grad_norms'].append(grad.norm().item())
        self.history['param_norms'].append(param.data.norm().item())
        self.history['m_norms'].append(m_hat.norm().item())
        self.history['v_norms'].append(v_hat.norm().item())
        self.history['update_norms'].append(update.norm().item())
        
        # Compute effective learning rate
        effective_lr = self.lr / (torch.sqrt(v_hat) + self.eps)
        self.history['effective_lrs'].append(effective_lr.mean().item())
        
        # Track beta powers
        if state.timestep > 0:
            self.history['beta1_powers'].append(self.beta1 ** state.timestep)
            self.history['beta2_powers'].append(self.beta2 ** state.timestep)
    
    def get_effective_learning_rates(self) -> Dict[str, float]:
        """Get current effective learning rates for each parameter."""
        effective_lrs = {}
        for param in self.params:
            state = self.states[param]
            m_hat, v_hat = state.get_bias_corrected(self.beta1, self.beta2)
            eff_lr = self.lr / (torch.sqrt(v_hat) + self.eps)
            effective_lrs[str(param.shape)] = eff_lr.mean().item()
        return effective_lrs
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dictionary."""
        state_dict = {
            'states': {id(p): {
                'm': s.m.clone(),
                'v': s.v.clone(),
                'timestep': s.timestep
            } for p, s in self.states.items()},
            'timestep': self.timestep,
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'amsgrad': self.amsgrad,
            'maximize': self.maximize,
            'history': self.history
        }
        
        if self.amsgrad:
            state_dict['max_v'] = {id(p): v.clone() 
                                  for p, v in self.max_v.items()}
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state from dictionary."""
        self.timestep = state_dict['timestep']
        self.lr = state_dict['lr']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']
        self.amsgrad = state_dict['amsgrad']
        self.maximize = state_dict['maximize']
        self.history = state_dict['history']
        
        # Reconstruct states
        for p in self.params:
            pid = id(p)
            if pid in state_dict['states']:
                s = state_dict['states'][pid]
                self.states[p] = AdamParameterState(
                    m=s['m'].to(p.device),
                    v=s['v'].to(p.device),
                    timestep=s['timestep']
                )
        
        if self.amsgrad and 'max_v' in state_dict:
            for p in self.params:
                pid = id(p)
                if pid in state_dict['max_v']:
                    self.max_v[p] = state_dict['max_v'][pid].to(p.device)

# ==================== ADAM VARIANTS ====================
class AdamW(AdamOptimizer):
    """
    Adam with decoupled weight decay (AdamW).
    Applies weight decay after the adaptive learning rate update.
    """
    
    def _apply_weight_decay(self, 
                           param: nn.Parameter, 
                           grad: torch.Tensor) -> torch.Tensor:
        """AdamW: Don't add weight decay to gradient here."""
        return grad
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step with decoupled weight decay."""
        self.timestep += 1
        
        loss = None
        if closure is not None:
            loss = closure()
        
        for param in self.params:
            if param.grad is None:
                continue
            
            # Apply weight decay before gradient update (AdamW style)
            if self.weight_decay != 0:
                param.data.mul_(1 - self.lr * self.weight_decay)
            
            # Get gradient
            grad = param.grad.data
            
            # Get state
            state = self.states[param]
            
            # Update moments
            state.update_moment(grad, self.beta1, self.beta2)
            
            # Get bias-corrected moments
            m_hat, v_hat = state.get_bias_corrected(self.beta1, self.beta2)
            
            # Compute update
            update = self._compute_adaptive_update(m_hat, v_hat)
            
            # Apply update
            param.data.add_(update)
            
            # Track statistics
            self._track_statistics(param, grad, state, update, m_hat, v_hat)
        
        return loss

class NAdam(AdamOptimizer):
    """
    Nesterov-accelerated Adaptive Moment Estimation.
    Incorporates Nesterov momentum into Adam.
    """
    
    def _compute_adaptive_update(self,
                                m_hat: torch.Tensor,
                                v_hat: torch.Tensor) -> torch.Tensor:
        """Compute update with Nesterov momentum."""
        # Nesterov update: lookahead momentum
        m_hat_nesterov = self.beta1 * m_hat + (1 - self.beta1) * m_hat
        
        denominator = torch.sqrt(v_hat) + self.eps
        update = -self.lr * m_hat_nesterov / denominator
        
        if self.maximize:
            update = -update
        
        return update

class AdaBound(AdamOptimizer):
    """
    Adaptive Gradient Methods with Dynamic Bound.
    Clips the adaptive learning rates with dynamic bounds.
    """
    
    def __init__(self, 
                 params: List[nn.Parameter],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 final_lr: float = 0.1,
                 gamma: float = 0.001,
                 amsbound: bool = False):
        """
        Args:
            final_lr: Final (SGD) learning rate
            gamma: Convergence speed of the bound functions
            amsbound: Whether to use AMSBound variant
        """
        super().__init__(params, lr, betas, eps, weight_decay, False, False)
        self.final_lr = final_lr
        self.gamma = gamma
        self.amsbound = amsbound
        
        if self.amsbound:
            self.max_v: Dict[nn.Parameter, torch.Tensor] = {}
            for param in self.params:
                self.max_v[param] = torch.zeros_like(param.data)
    
    def _compute_adaptive_update(self,
                                m_hat: torch.Tensor,
                                v_hat: torch.Tensor) -> torch.Tensor:
        """Compute update with dynamic bounds."""
        # For AMSBound, use maximum of v_hat
        if self.amsbound:
            max_v = self.max_v[self.current_param]
            max_v = torch.max(max_v, v_hat)
            self.max_v[self.current_param] = max_v
            v_hat = max_v
        
        # Compute base step size
        step_size = self.lr / (torch.sqrt(v_hat) + self.eps)
        
        # Compute lower and upper bounds
        lower_bound = self.final_lr * (1 - 1 / (self.gamma * self.timestep + 1))
        upper_bound = self.final_lr * (1 + 1 / (self.gamma * self.timestep))
        
        # Clip step size
        step_size = torch.clamp(step_size, lower_bound, upper_bound)
        
        # Compute update
        update = -step_size * m_hat
        
        if self.maximize:
            update = -update
        
        return update
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step with dynamic bounds."""
        self.timestep += 1
        
        loss = None
        if closure is not None:
            loss = closure()
        
        for param in self.params:
            if param.grad is None:
                continue
            
            self.current_param = param
            
            # Apply weight decay
            grad = self._apply_weight_decay(param, param.grad.data)
            
            # Get state
            state = self.states[param]
            
            # Update moments
            state.update_moment(grad, self.beta1, self.beta2)
            
            # Get bias-corrected moments
            m_hat, v_hat = state.get_bias_corrected(self.beta1, self.beta2)
            
            # Compute update with bounds
            update = self._compute_adaptive_update(m_hat, v_hat)
            
            # Apply update
            param.data.add_(update)
            
            # Track statistics
            self._track_statistics(param, grad, state, update, m_hat, v_hat)
        
        return loss

class RAdam(AdamOptimizer):
    """
    Rectified Adam (RAdam).
    Rectifies the variance of adaptive learning rate.
    """
    
    def __init__(self, 
                 params: List[nn.Parameter],
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(params, lr, betas, eps, weight_decay)
        # Additional parameters for RAdam
        self.rho_inf = 2 / (1 - self.beta2) - 1
    
    def _compute_adaptive_update(self,
                                m_hat: torch.Tensor,
                                v_hat: torch.Tensor) -> torch.Tensor:
        """Compute update with rectification."""
        # Compute rectification term
        beta2_t = self.beta2 ** self.timestep
        rho_t = self.rho_inf - 2 * self.timestep * beta2_t / (1 - beta2_t)
        
        if rho_t > 4:  # Apply rectification
            # Compute rectification term
            r_t = math.sqrt((rho_t - 4) * (rho_t - 2) * self.rho_inf / 
                          ((self.rho_inf - 4) * (self.rho_inf - 2) * rho_t))
            
            # Compute adaptive learning rate with rectification
            adaptive_lr = self.lr * r_t / (torch.sqrt(v_hat) + self.eps)
            
            # Compute update
            update = -adaptive_lr * m_hat
        else:
            # Use unrectified update
            update = -self.lr * m_hat
        
        if self.maximize:
            update = -update
        
        return update

# ==================== LEARNING RATE SCHEDULERS ====================
class AdamScheduler:
    """Learning rate schedulers designed for Adam."""
    
    def __init__(self, 
                 optimizer: AdamOptimizer,
                 schedule_type: str = 'cosine',
                 **kwargs):
        """
        Args:
            schedule_type: 'cosine', 'linear', 'exponential', 'cyclic', 'onecycle'
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        self.base_lr = optimizer.lr
        
        # Initialize based on schedule type
        if schedule_type == 'cosine':
            self.T_max = kwargs.get('T_max', 100)
            self.eta_min = kwargs.get('eta_min', 1e-6)
        elif schedule_type == 'linear':
            self.start_lr = kwargs.get('start_lr', self.base_lr)
            self.end_lr = kwargs.get('end_lr', 1e-6)
            self.total_steps = kwargs.get('total_steps', 1000)
        elif schedule_type == 'onecycle':
            self.max_lr = kwargs.get('max_lr', 0.01)
            self.total_steps = kwargs.get('total_steps', 1000)
            self.pct_start = kwargs.get('pct_start', 0.3)
            self.div_factor = kwargs.get('div_factor', 25)
            self.final_div_factor = kwargs.get('final_div_factor', 1e4)
        
        self.step_count = 0
        self.lr_history = []
    
    def step(self):
        """Update learning rate based on schedule."""
        self.step_count += 1
        
        if self.schedule_type == 'cosine':
            # Cosine annealing
            if self.step_count <= self.T_max:
                new_lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
                        (1 + math.cos(math.pi * self.step_count / self.T_max))
            else:
                new_lr = self.eta_min
        
        elif self.schedule_type == 'linear':
            # Linear decay
            progress = min(self.step_count / self.total_steps, 1.0)
            new_lr = self.start_lr + (self.end_lr - self.start_lr) * progress
        
        elif self.schedule_type == 'exponential':
            # Exponential decay
            gamma = self.kwargs.get('gamma', 0.95)
            step_size = self.kwargs.get('step_size', 10)
            new_lr = self.base_lr * (gamma ** (self.step_count / step_size))
        
        elif self.schedule_type == 'onecycle':
            # One-cycle policy
            total_steps = self.total_steps
            pct_start = self.pct_start
            div_factor = self.div_factor
            final_div_factor = self.final_div_factor
            
            # Initial learning rate
            initial_lr = self.base_lr / div_factor
            
            # Peak learning rate
            peak_lr = self.max_lr
            
            # Final learning rate
            final_lr = initial_lr / final_div_factor
            
            # Calculate step counts
            up_steps = int(total_steps * pct_start)
            down_steps = total_steps - up_steps
            
            if self.step_count <= up_steps:
                # Warm-up phase
                progress = self.step_count / up_steps
                new_lr = initial_lr + (peak_lr - initial_lr) * progress
            elif self.step_count <= total_steps:
                # Cool-down phase
                progress = (self.step_count - up_steps) / down_steps
                new_lr = peak_lr - (peak_lr - final_lr) * progress
            else:
                new_lr = final_lr
        
        else:
            new_lr = self.base_lr
        
        self.optimizer.lr = new_lr
        self.lr_history.append(new_lr)
        
        return new_lr
    
    def plot_schedule(self, total_steps: int = 1000):
        """Plot the learning rate schedule."""
        # Temporarily save current state
        current_step = self.step_count
        current_lr = self.optimizer.lr
        
        # Simulate schedule
        lrs = []
        for _ in range(total_steps):
            lrs.append(self.step())
        
        # Restore state
        self.step_count = current_step
        self.optimizer.lr = current_lr
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{self.schedule_type.capitalize()} Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()

# ==================== TEST FUNCTIONS ====================
class AdamTestFunctions:
    """Test functions for Adam optimizer benchmarking."""
    
    @staticmethod
    def quadratic_bowl(x: torch.Tensor, 
                      A: torch.Tensor = None,
                      b: torch.Tensor = None) -> torch.Tensor:
        """
        Quadratic bowl: f(x) = 0.5 * x^T A x + b^T x
        Minimum at x = -A^{-1}b
        """
        if A is None:
            A = torch.eye(x.shape[0], device=x.device)
        if b is None:
            b = torch.ones(x.shape[0], device=x.device)
        
        return 0.5 * x.T @ A @ x + b.T @ x
    
    @staticmethod
    def rosenbrock(x: torch.Tensor, a: float = 1, b: float = 100) -> torch.Tensor:
        """Rosenbrock function for 2D optimization."""
        if x.shape[0] == 2:
            return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
        else:
            # Multi-dimensional Rosenbrock
            result = 0
            for i in range(x.shape[0] - 1):
                result += (a - x[i]) ** 2 + b * (x[i+1] - x[i] ** 2) ** 2
            return result
    
    @staticmethod
    def stochastic_quadratic(x: torch.Tensor, 
                           noise_scale: float = 0.1) -> torch.Tensor:
        """Quadratic with stochastic noise."""
        main_loss = 0.5 * torch.sum(x ** 2)
        noise = noise_scale * torch.randn_like(main_loss)
        return main_loss + noise
    
    @staticmethod
    def create_test_problem(name: str, 
                          dim: int = 2) -> Callable:
        """Create test problem function."""
        if name == 'quadratic':
            A = torch.eye(dim)
            b = torch.ones(dim)
            return lambda x: AdamTestFunctions.quadratic_bowl(x, A, b)
        elif name == 'rosenbrock':
            return lambda x: AdamTestFunctions.rosenbrock(x)
        elif name == 'stochastic':
            return lambda x: AdamTestFunctions.stochastic_quadratic(x)
        else:
            raise ValueError(f"Unknown test problem: {name}")

# ==================== OPTIMIZER ANALYZER ====================
class AdamAnalyzer:
    """Analyzer for Adam optimizer behavior."""
    
    def __init__(self, optimizer: AdamOptimizer):
        self.optimizer = optimizer
        self.analysis_results = {}
    
    def analyze_convergence(self, 
                           param_history: List[torch.Tensor],
                           loss_history: List[float]) -> Dict[str, Any]:
        """Analyze convergence properties."""
        if len(loss_history) < 2:
            return {}
        
        # Compute convergence metrics
        final_loss = loss_history[-1]
        initial_loss = loss_history[0]
        loss_reduction = (initial_loss - final_loss) / initial_loss
        
        # Compute convergence rate
        if len(loss_history) > 10:
            last_10 = loss_history[-10:]
            convergence_rate = np.mean(np.diff(np.log(last_10)))
        else:
            convergence_rate = 0
        
        # Compute parameter movement
        if len(param_history) > 1:
            total_movement = sum(
                torch.norm(param_history[i] - param_history[i-1]).item()
                for i in range(1, len(param_history))
            )
        else:
            total_movement = 0
        
        analysis = {
            'final_loss': final_loss,
            'loss_reduction_ratio': loss_reduction,
            'convergence_rate': convergence_rate,
            'total_parameter_movement': total_movement,
            'num_iterations': len(loss_history),
            'final_gradient_norm': self.optimizer.history['grad_norms'][-1] 
                                  if self.optimizer.history['grad_norms'] else 0
        }
        
        self.analysis_results = analysis
        return analysis
    
    def analyze_learning_rate_adaptation(self) -> Dict[str, Any]:
        """Analyze how learning rates adapt during optimization."""
        if not self.optimizer.history['effective_lrs']:
            return {}
        
        eff_lrs = self.optimizer.history['effective_lrs']
        
        analysis = {
            'mean_effective_lr': np.mean(eff_lrs),
            'std_effective_lr': np.std(eff_lrs),
            'min_effective_lr': np.min(eff_lrs),
            'max_effective_lr': np.max(eff_lrs),
            'lr_adaptation_ratio': np.max(eff_lrs) / (np.min(eff_lrs) + 1e-10)
        }
        
        return analysis
    
    def plot_optimization_dynamics(self, figsize: Tuple[int, int] = (15, 10)):
        """Plot detailed optimization dynamics."""
        history = self.optimizer.history
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Gradient norms
        if history['grad_norms']:
            axes[0].plot(history['grad_norms'])
            axes[0].set_title('Gradient Norms')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Parameter norms
        if history['param_norms']:
            axes[1].plot(history['param_norms'])
            axes[1].set_title('Parameter Norms')
            axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Moment norms
        if history['m_norms']:
            axes[2].plot(history['m_norms'], label='m (1st moment)')
        if history['v_norms']:
            axes[2].plot(history['v_norms'], label='v (2nd moment)')
        axes[2].set_title('Moment Norms')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Effective learning rates
        if history['effective_lrs']:
            axes[3].plot(history['effective_lrs'])
            axes[3].set_title('Effective Learning Rates')
            axes[3].set_yscale('log')
            axes[3].grid(True, alpha=0.3)
        
        # Plot 5: Update norms
        if history['update_norms']:
            axes[4].plot(history['update_norms'])
            axes[4].set_title('Update Norms')
            axes[4].set_yscale('log')
            axes[4].grid(True, alpha=0.3)
        
        # Plot 6: Beta powers
        if history['beta1_powers']:
            axes[5].plot(history['beta1_powers'], label='β1^t')
        if history['beta2_powers']:
            axes[5].plot(history['beta2_powers'], label='β2^t')
        axes[5].set_title('Beta Powers')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        
        # Plot 7: Learning rate vs gradient (scatter)
        if history['grad_norms'] and history['effective_lrs']:
            axes[6].scatter(history['grad_norms'], history['effective_lrs'], 
                          alpha=0.5, s=10)
            axes[6].set_xlabel('Gradient Norm')
            axes[6].set_ylabel('Effective LR')
            axes[6].set_title('LR vs Gradient')
            axes[6].set_xscale('log')
            axes[6].set_yscale('log')
            axes[6].grid(True, alpha=0.3)
        
        # Plot 8: Histogram of effective learning rates
        if history['effective_lrs']:
            axes[7].hist(history['effective_lrs'], bins=50, alpha=0.7)
            axes[7].set_title('Effective LR Distribution')
            axes[7].set_xscale('log')
            axes[7].grid(True, alpha=0.3)
        
        # Plot 9: Ratio of moments
        if history['m_norms'] and history['v_norms']:
            ratios = []
            for m, v in zip(history['m_norms'], history['v_norms']):
                if v > 0:
                    ratios.append(m / math.sqrt(v))
            if ratios:
                axes[8].plot(ratios)
                axes[8].set_title('m / sqrt(v) Ratio')
                axes[8].grid(True, alpha=0.3)
        
        plt.suptitle(f'Adam Optimizer Dynamics Analysis\n'
                    f'β1={self.optimizer.beta1}, β2={self.optimizer.beta2}, ε={self.optimizer.eps}', 
                    fontsize=14)
        plt.tight_layout()
        plt.show()

# ==================== NEURAL NETWORK FOR REAL DATASET ====================
class AdamNetwork(nn.Module):
    """Neural network for testing Adam on real datasets."""
    
    def __init__(self, 
                 input_dim: int = 784,
                 hidden_dims: List[int] = [512, 256, 128],
                 output_dim: int = 10,
                 activation: str = 'relu',
                 dropout: float = 0.3,
                 batch_norm: bool = True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', 
                                       nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)

# ==================== DATASET LOADERS ====================
class CIFAR10AdamDataset(Dataset):
    """CIFAR-10 dataset wrapper for Adam testing."""
    
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
        return self.dataset[idx]

class DataAugmentation:
    """Data augmentation for better training."""
    
    def __init__(self, dataset: str = 'cifar10'):
        if dataset == 'cifar10':
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2470, 0.2435, 0.2616))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    
    def __call__(self, img):
        return self.transform(img)

# ==================== TRAINING FRAMEWORK ====================
class AdamTrainer:
    """Training framework for Adam optimizer."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 optimizer_class: type = AdamOptimizer,
                 **optimizer_kwargs):
        """
        Args:
            model: Neural network model
            device: Training device
            optimizer_class: Adam optimizer class
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
        self.scheduler = AdamScheduler(
            self.optimizer,
            schedule_type='onecycle',
            max_lr=0.01,
            total_steps=1000
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Analyzer
        self.analyzer = AdamAnalyzer(self.optimizer)
    
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
            
            # Update learning rate
            self.scheduler.step()
            self.learning_rates.append(self.optimizer.lr)
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += len(data)
            
            # Print batch progress
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.lr
                print(f'  Batch {batch_idx:4d}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {correct/len(data):.4f}, '
                      f'LR: {current_lr:.6f}')
        
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
        print(f"Optimizer parameters: lr={self.optimizer.lr}, "
              f"betas={self.optimizer.beta1, self.optimizer.beta2}, "
              f"eps={self.optimizer.eps}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  Current LR: {self.optimizer.lr:.6f}")
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
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
        axes[0, 2].plot(self.learning_rates, alpha=0.8)
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Effective learning rates from optimizer
        if self.optimizer.history['effective_lrs']:
            axes[1, 0].plot(self.optimizer.history['effective_lrs'], alpha=0.8)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Effective LR')
            axes[1, 0].set_title('Effective Learning Rates')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient norms
        if self.optimizer.history['grad_norms']:
            axes[1, 1].plot(self.optimizer.history['grad_norms'], alpha=0.8)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Gradient Norms')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Update norms
        if self.optimizer.history['update_norms']:
            axes[1, 2].plot(self.optimizer.history['update_norms'], alpha=0.8)
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Update Norm')
            axes[1, 2].set_title('Parameter Update Norms')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {self.optimizer.__class__.__name__}', 
                    fontsize=14)
        plt.tight_layout()
        plt.show()

# ==================== BENCHMARKING FRAMEWORK ====================
class AdamBenchmark:
    """Benchmarking framework for Adam optimizer."""
    
    @staticmethod
    def benchmark_test_functions():
        """Benchmark Adam on test functions."""
        print("=" * 60)
        print("BENCHMARKING ADAM ON TEST FUNCTIONS")
        print("=" * 60)
        
        # Define test problems
        test_problems = [
            ('Quadratic', 'quadratic', 10),
            ('Rosenbrock (2D)', 'rosenbrock', 2),
            ('Rosenbrock (10D)', 'rosenbrock', 10),
            ('Stochastic', 'stochastic', 20)
        ]
        
        # Define optimizers to compare
        optimizers = {
            'Adam': {
                'class': AdamOptimizer,
                'kwargs': {'lr': 0.01, 'betas': (0.9, 0.999)}
            },
            'AdamW': {
                'class': AdamW,
                'kwargs': {'lr': 0.01, 'betas': (0.9, 0.999), 'weight_decay': 0.01}
            },
            'NAdam': {
                'class': NAdam,
                'kwargs': {'lr': 0.01, 'betas': (0.9, 0.999)}
            },
            'RAdam': {
                'class': RAdam,
                'kwargs': {'lr': 0.01, 'betas': (0.9, 0.999)}
            }
        }
        
        results = {}
        
        for prob_name, prob_type, dim in test_problems:
            print(f"\n{prob_name} (dim={dim}):")
            print("-" * 40)
            
            # Create test function
            test_func = AdamTestFunctions.create_test_problem(prob_type, dim)
            
            # Initialize parameters
            init_params = [torch.randn(dim, requires_grad=True) * 2]
            
            prob_results = {}
            
            for opt_name, opt_config in optimizers.items():
                # Reset parameters
                params = [torch.randn(dim, requires_grad=True) * 2]
                
                # Create optimizer
                optimizer = opt_config['class'](params, **opt_config['kwargs'])
                
                # Optimize
                losses = []
                for i in range(1000):
                    loss = test_func(params[0])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    
                    # Early stopping
                    if i > 10 and abs(losses[-1] - losses[-2]) < 1e-8:
                        break
                
                prob_results[opt_name] = {
                    'final_loss': losses[-1],
                    'convergence_steps': len(losses),
                    'optimizer': optimizer
                }
                
                print(f"  {opt_name:10s}: Final loss = {losses[-1]:.2e}, "
                      f"Steps = {len(losses):4d}")
            
            results[prob_name] = prob_results
        
        return results
    
    @staticmethod
    def benchmark_cifar10():
        """Benchmark Adam on CIFAR-10."""
        print("\n" + "=" * 60)
        print("BENCHMARKING ADAM ON CIFAR-10")
        print("=" * 60)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load CIFAR-10 dataset
        print("\nLoading CIFAR-10 dataset...")
        train_dataset = CIFAR10AdamDataset(train=True, download=True)
        test_dataset = CIFAR10AdamDataset(train=False, download=True)
        
        # Split training data
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, 
                                 num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        print(f"Training samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Define optimizers to compare
        optimizer_configs = {
            'Adam': {
                'class': AdamOptimizer,
                'kwargs': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8}
            },
            'AdamW': {
                'class': AdamW,
                'kwargs': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8, 
                          'weight_decay': 0.01}
            },
            'NAdam': {
                'class': NAdam,
                'kwargs': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8}
            },
            'RAdam': {
                'class': RAdam,
                'kwargs': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8}
            }
        }
        
        results = {}
        
        for name, config in optimizer_configs.items():
            print(f"\n{'='*40}")
            print(f"Training with {name}")
            print(f"{'='*40}")
            
            # Create model
            model = AdamNetwork(
                input_dim=32*32*3,
                hidden_dims=[512, 256, 128],
                output_dim=10,
                dropout=0.3
            )
            
            # Create trainer
            trainer = AdamTrainer(
                model=model,
                device=device,
                optimizer_class=config['class'],
                **config['kwargs']
            )
            
            # Train
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=10  # Reduced for demonstration
            )
            
            # Test
            test_loss, test_acc = trainer.validate(test_loader)
            print(f"\nTest Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
            
            # Store results
            results[name] = {
                'trainer': trainer,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'final_train_acc': trainer.train_accuracies[-1] if trainer.train_accuracies else 0,
                'final_val_acc': trainer.val_accuracies[-1] if trainer.val_accuracies else 0
            }
            
            # Plot training history
            trainer.plot_training_history()
            
            # Analyze optimizer dynamics
            print("\nAnalyzing optimizer dynamics...")
            analyzer = AdamAnalyzer(trainer.optimizer)
            analyzer.plot_optimization_dynamics()
        
        # Final comparison
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
        axes[0].set_ylim(0.5, 0.9)
        
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
class AdamVisualizer:
    """Visualization tools for Adam optimizer."""
    
    @staticmethod
    def visualize_adaptive_lr(optimizer: AdamOptimizer):
        """Visualize adaptive learning rates across parameters."""
        eff_lrs = optimizer.get_effective_learning_rates()
        
        if not eff_lrs:
            return
        
        plt.figure(figsize=(10, 6))
        
        shapes = list(eff_lrs.keys())
        values = list(eff_lrs.values())
        
        bars = plt.bar(range(len(shapes)), values)
        plt.xlabel('Parameter Shape')
        plt.ylabel('Effective Learning Rate')
        plt.title('Adaptive Learning Rates Across Parameters')
        plt.xticks(range(len(shapes)), shapes, rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2e}', ha='center', va='bottom', fontsize=8)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_bias_correction(optimizer: AdamOptimizer):
        """Visualize bias correction effect."""
        if not optimizer.history['beta1_powers']:
            return
        
        plt.figure(figsize=(10, 6))
        
        steps = range(1, len(optimizer.history['beta1_powers']) + 1)
        beta1_powers = optimizer.history['beta1_powers']
        beta2_powers = optimizer.history['beta2_powers']
        
        plt.plot(steps, beta1_powers, label='β1^t', linewidth=2)
        plt.plot(steps, beta2_powers, label='β2^t', linewidth=2)
        
        # Add bias correction lines
        plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10% threshold')
        plt.axhline(y=0.01, color='g', linestyle='--', alpha=0.5, label='1% threshold')
        
        plt.xlabel('Time Step (t)')
        plt.ylabel('Bias Correction Factor')
        plt.title('Bias Correction in Adam (1 - β^t)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()

# ==================== MAIN DEMONSTRATION ====================
def demonstrate_adam_algorithm():
    """Demonstrate Adam algorithm step-by-step."""
    print("=" * 60)
    print("ADAM ALGORITHM STEP-BY-STEP DEMONSTRATION")
    print("=" * 60)
    
    print("\nAdam Algorithm Steps:")
    print("1. Initialize: m_0 = 0, v_0 = 0, t = 0")
    print("2. For each iteration t:")
    print("   a. Compute gradient: g_t = ∇f(θ_t)")
    print("   b. Update biased first moment: m_t = β1 * m_{t-1} + (1-β1) * g_t")
    print("   c. Update biased second moment: v_t = β2 * v_{t-1} + (1-β2) * g_t²")
    print("   d. Compute bias-corrected moments:")
    print("      m_hat_t = m_t / (1-β1^t)")
    print("      v_hat_t = v_t / (1-β2^t)")
    print("   e. Update parameters:")
    print("      θ_{t+1} = θ_t - η * m_hat_t / (√v_hat_t + ε)")
    
    # Create a simple example
    print("\n" + "-" * 40)
    print("Numerical Example:")
    
    # Initialize parameter
    param = torch.tensor([2.0], requires_grad=True)
    
    # Create Adam optimizer
    optimizer = AdamOptimizer([param], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    
    print(f"Initial parameter: {param.item():.4f}")
    
    for i in range(3):
        # Simple quadratic loss
        loss = (param - 1.0) ** 2
        optimizer.zero_grad()
        loss.backward()
        
        # Get gradient
        grad = param.grad.item()
        
        # Get state before update
        state = optimizer.states[param]
        m_before = state.m.item()
        v_before = state.v.item()
        timestep_before = state.timestep
        
        # Perform update
        optimizer.step()
        
        # Get state after update
        state = optimizer.states[param]
        m_after = state.m.item()
        v_after = state.v.item()
        m_hat, v_hat = state.get_bias_corrected(0.9, 0.999)
        
        print(f"\nStep {i+1}:")
        print(f"  Gradient: {grad:.4f}")
        print(f"  Moments before update: m={m_before:.6f}, v={v_before:.6f}")
        print(f"  Moments after update: m={m_after:.6f}, v={v_after:.6f}")
        print(f"  Bias-corrected: m_hat={m_hat.item():.6f}, v_hat={v_hat.item():.6f}")
        print(f"  New parameter: {param.item():.4f}")

def demonstrate_adam_variants():
    """Demonstrate different Adam variants."""
    print("\n" + "=" * 60)
    print("ADAM VARIANTS COMPARISON")
    print("=" * 60)
    
    variants = [
        ('Adam', AdamOptimizer, 
         'Standard Adam with bias correction'),
        ('AdamW', AdamW,
         'Adam with decoupled weight decay'),
        ('NAdam', NAdam,
         'Adam with Nesterov momentum'),
        ('AMSGrad', lambda params, **kwargs: AdamOptimizer(params, amsgrad=True, **kwargs),
         'Adam with maximum of second moments'),
        ('RAdam', RAdam,
         'Rectified Adam for variance rectification'),
        ('AdaBound', AdaBound,
         'Adam with dynamic learning rate bounds')
    ]
    
    for name, cls, description in variants:
        print(f"\n{name}:")
        print(f"  Description: {description}")
        if hasattr(cls, '__name__'):
            print(f"  Class: {cls.__name__}")

def main():
    """Main demonstration function."""
    print("ADAM OPTIMIZER IMPLEMENTATION FROM SCRATCH")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demonstrate algorithm
    demonstrate_adam_algorithm()
    
    # Demonstrate variants
    demonstrate_adam_variants()
    
    # Benchmark on test functions
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARKS")
    print("=" * 60)
    
    # Benchmark on test functions
    test_results = AdamBenchmark.benchmark_test_functions()
    
    # Benchmark on CIFAR-10
    cifar_results = AdamBenchmark.benchmark_cifar10()
    
    print("\n" + "=" * 60)
    print("ADAM OPTIMIZER IMPLEMENTATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Implemented:")
    print("1. Standard Adam with bias correction")
    print("2. AdamW with decoupled weight decay")
    print("3. NAdam with Nesterov momentum")
    print("4. AMSGrad variant")
    print("5. RAdam with variance rectification")
    print("6. AdaBound with dynamic bounds")
    print("7. Learning rate schedulers")
    print("8. Comprehensive analysis tools")
    print("9. Real dataset testing (CIFAR-10)")
    print("10. Mathematical foundation visualization")

if __name__ == "__main__":
    main()