import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

def get_parser(parser):
    return parser


class ADABOP(nn.Module):
    """ADABOP (Adaptive Bound Optimization) for continuous task learning (domain-il)
    Core idea: Adaptively adjust optimization boundaries (gradient clipping, loss constraints) 
    during training to stabilize continuous task learning without buffer
    """
    NAME = 'adabop'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args):
        """Initialize ADABOP module
        
        Args:
            backbone: Backbone network (lane/ trajectory prediction model)
            loss: Loss function for the task
            args: Command line arguments with hyperparameters
        """
        super(ADABOP, self).__init__()
        self.net = backbone
        self.loss = loss
        self.args = args
        self.device = args.device
        self.transform = None

        # Initialize optimizer (consistent with Vanilla, base Adam)
        self.opt = Adam(self.net.parameters(), lr=self.args.lr)

        # ADABOP core adaptive boundary parameters
        self.current_clip_bound = args.adabop_clip_bound  # Adaptive gradient clip boundary
        self.bound_decay = args.adabop_bound_decay        # Decay rate for boundary
        self.min_bound = args.adabop_min_bound            # Minimum boundary value
        self.loss_scale = args.adabop_loss_scale          # Loss scaling for boundary adjustment

    def _update_adaptive_bound(self, loss_value):
        """Update the adaptive gradient clipping boundary based on current loss
        
        Args:
            loss_value: Current batch loss value (scalar)
        """
        # Adjust boundary by loss magnitude (larger loss → smaller boundary for stable training)
        boundary_update = self.current_clip_bound * self.bound_decay * (1 + self.loss_scale * loss_value)
        # Ensure boundary does not drop below minimum value
        self.current_clip_bound = max(boundary_update, self.min_bound)

    def observe(self, inputs, labels, task_id=None, record_list=None):
        """Forward pass, loss calculation, adaptive boundary optimization, and backward pass
        
        Args:
            inputs: Input tensor (batch of data for current task)
            labels: Ground truth labels for the batch
            task_id: ID of current task (unused for domain-il but kept for compatibility)
            record_list: List to record training metrics (optional)
        
        Returns:
            Scalar loss value for current batch
        """
        # Zero out gradients (consistent with Vanilla)
        self.opt.zero_grad()

        # Forward pass through backbone network
        outputs = self.net(inputs)
        log_lanescore, heatmap, heatmap_reg = outputs
        outputs_prediction = [log_lanescore, heatmap, heatmap_reg]

        # Calculate loss (same as Vanilla, with optional scaling for ADABOP)
        loss = self.loss(outputs_prediction, labels) * self.loss_scale

        # Backward pass to compute gradients
        loss.backward()

        # ADABOP core: Adaptive gradient clipping based on current boundary
        torch.nn.utils.clip_grad_norm_(
            parameters=self.net.parameters(),
            max_norm=self.current_clip_bound,
            norm_type=2  # L2 norm for gradient clipping
        )

        # Update optimizer parameters
        self.opt.step()

        # Update adaptive boundary for next iteration
        self._update_adaptive_bound(loss.item())

        # Record metrics if needed (compatible with Vanilla's record_list)
        if record_list is not None:
            record_list.append({
                'loss': loss.item(),
                'current_clip_bound': self.current_clip_bound
            })

        return loss.item()