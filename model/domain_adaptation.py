# model/domain_adaptation.py
"""
Domain Adaptation components for GCN multi-label classification
Supports both MMD (Maximum Mean Discrepancy) and adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============= Feature Encoder =============
class DomainAdaptationEncoder(nn.Module):
    """
    Deep feature encoder that learns domain-invariant representations
    Architecture: Input -> FC layers with dropout -> Latent features
    """
    def __init__(
        self,
        input_dim,
        latent_dim=128,
        hidden_dims=[256, 128],
        dropout=0.3,
        use_batchnorm=True
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final projection to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: [N, input_dim] raw features
        Returns:
            latent: [N, latent_dim] domain-invariant features
        """
        return self.encoder(x)


# ============= Domain Discriminator =============
class DomainDiscriminator(nn.Module):
    """
    Binary classifier to distinguish source vs target domain
    Used in adversarial domain adaptation (DANN-style)
    """
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification
        )
        
    def forward(self, latent_features):
        """
        Args:
            latent_features: [N, latent_dim]
        Returns:
            domain_logits: [N, 1] logits for domain classification
        """
        return self.discriminator(latent_features)


# ============= Domain Adaptation Losses =============
class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss
    Measures distribution distance between source and target domains
    Uses Gaussian RBF kernel
    """
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        
    def gaussian_kernel(self, source, target):
        """
        Compute Gaussian RBF kernel matrix
        """
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        
        # Compute pairwise L2 distances
        total0 = total.unsqueeze(0).expand(n_samples, n_samples, total.size(1))
        total1 = total.unsqueeze(1).expand(n_samples, n_samples, total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        # Compute bandwidth (median heuristic)
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        
        # Compute multi-scale Gaussian kernel
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        """
        Args:
            source: [N_s, latent_dim] source domain features
            target: [N_t, latent_dim] target domain features
        Returns:
            mmd_loss: scalar tensor
        """
        batch_size_source = source.size(0)
        batch_size_target = target.size(0)
        
        kernels = self.gaussian_kernel(source, target)
        
        # Split kernel matrix into 4 blocks
        XX = kernels[:batch_size_source, :batch_size_source]
        YY = kernels[batch_size_source:, batch_size_source:]
        XY = kernels[:batch_size_source, batch_size_source:]
        YX = kernels[batch_size_source:, :batch_size_source]
        
        # Compute MMD
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss


class CoralLoss(nn.Module):
    """
    CORAL (CORrelation ALignment) loss
    Aligns second-order statistics (covariance) between domains
    Simpler and faster than MMD
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, source, target):
        """
        Args:
            source: [N_s, latent_dim]
            target: [N_t, latent_dim]
        Returns:
            coral_loss: scalar
        """
        d = source.size(1)  # feature dimension
        
        # Compute covariance matrices
        source_centered = source - source.mean(0, keepdim=True)
        target_centered = target - target.mean(0, keepdim=True)
        
        cov_source = torch.matmul(source_centered.t(), source_centered) / (source.size(0) - 1)
        cov_target = torch.matmul(target_centered.t(), target_centered) / (target.size(0) - 1)
        
        # Frobenius norm of difference
        loss = torch.sum((cov_source - cov_target) ** 2) / (4 * d * d)
        return loss


# ============= Gradient Reversal Layer =============
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial domain adaptation
    Forward: identity function
    Backward: multiply gradient by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Wrapper for gradient reversal
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ============= Combined Domain Adaptation Model =============
class DomainAdaptiveGCN(nn.Module):
    """
    Complete model: Encoder + GCN + Domain Discriminator
    Supports multiple domain adaptation strategies
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        num_labels,
        encoder_hidden=[256, 128],
        dropout=0.3,
        da_method='mmd',  # 'mmd', 'coral', 'dann', or None
    ):
        super().__init__()
        
        self.da_method = da_method
        
        # Feature encoder (domain adaptation)
        self.encoder = DomainAdaptationEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden,
            dropout=dropout
        )
        
        # GCN classifier (same as your ResidualGCN but takes latent features)
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(latent_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_labels)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Domain discriminator (for DANN)
        if da_method == 'dann':
            self.domain_discriminator = DomainDiscriminator(latent_dim)
            self.grl = GradientReversalLayer()
        
    def forward(self, x, edge_index, edge_weight=None, return_latent=False):
        """
        Args:
            x: [N, input_dim] raw features
            edge_index: [2, E]
            edge_weight: [E] or None
            return_latent: if True, return latent features for DA loss
        Returns:
            logits: [N, num_labels]
            latent: [N, latent_dim] (if return_latent=True)
        """
        # Encode to latent space
        latent = self.encoder(x)
        
        # GCN forward
        h1 = self.conv1(latent, edge_index, edge_weight=edge_weight)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        
        h2 = self.conv2(h1, edge_index, edge_weight=edge_weight)
        h2 = self.bn2(h2)
        h2 = F.relu(h2 + h1)
        h2 = self.dropout(h2)
        
        h3 = self.conv3(h2, edge_index, edge_weight=edge_weight)
        h3 = self.bn3(h3)
        h3 = F.relu(h3 + h2)
        h3 = self.dropout(h3)
        
        logits = self.head(h3)
        
        if return_latent:
            return logits, latent
        return logits
    
    def get_domain_logits(self, latent_features):
        """
        Get domain classification logits (for DANN)
        """
        if self.da_method == 'dann':
            reversed_latent = self.grl(latent_features)
            return self.domain_discriminator(reversed_latent)
        else:
            raise ValueError("Domain discriminator only available with da_method='dann'")


# ============= Helper Functions =============
def prepare_domain_data(data, domain_indices):
    """
    Split data into source and target domains
    
    Args:
        data: PyG Data object
        domain_indices: dict with keys 'source' and 'target', values are node indices
    Returns:
        source_x, target_x: feature tensors
    """
    source_idx = domain_indices['source']
    target_idx = domain_indices['target']
    
    source_x = data.x[source_idx]
    target_x = data.x[target_idx]
    
    return source_x, target_x


def compute_domain_adaptation_loss(
    model,
    source_x,
    target_x,
    method='mmd',
    mmd_loss_fn=None,
    coral_loss_fn=None
):
    """
    Compute domain adaptation loss
    
    Args:
        model: DomainAdaptiveGCN
        source_x: [N_s, input_dim]
        target_x: [N_t, input_dim]
        method: 'mmd', 'coral', or 'dann'
        mmd_loss_fn: MMDLoss instance
        coral_loss_fn: CoralLoss instance
    Returns:
        da_loss: scalar tensor
    """
    # Encode to latent space
    source_latent = model.encoder(source_x)
    target_latent = model.encoder(target_x)
    
    if method == 'mmd':
        if mmd_loss_fn is None:
            mmd_loss_fn = MMDLoss()
        return mmd_loss_fn(source_latent, target_latent)
    
    elif method == 'coral':
        if coral_loss_fn is None:
            coral_loss_fn = CoralLoss()
        return coral_loss_fn(source_latent, target_latent)
    
    elif method == 'dann':
        # For DANN, domain loss is computed separately in training loop
        # This is because gradient reversal happens automatically
        batch_size_source = source_latent.size(0)
        batch_size_target = target_latent.size(0)
        
        # Create domain labels
        domain_label_source = torch.zeros(batch_size_source, 1).to(source_latent.device)
        domain_label_target = torch.ones(batch_size_target, 1).to(target_latent.device)
        
        # Get domain predictions
        domain_pred_source = model.get_domain_logits(source_latent)
        domain_pred_target = model.get_domain_logits(target_latent)
        
        # Binary cross entropy loss
        domain_loss = F.binary_cross_entropy_with_logits(
            torch.cat([domain_pred_source, domain_pred_target], dim=0),
            torch.cat([domain_label_source, domain_label_target], dim=0)
        )
        return domain_loss
    
    else:
        raise ValueError(f"Unknown DA method: {method}")


# ============= Example Usage =============
if __name__ == '__main__':
    # Example configuration
    input_dim = 100
    latent_dim = 64
    hidden_dim = 128
    num_labels = 5
    
    # Create model
    model = DomainAdaptiveGCN(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels,
        da_method='mmd'
    )
    
    # Dummy data
    x = torch.randn(50, input_dim)
    edge_index = torch.randint(0, 50, (2, 100))
    
    # Forward pass
    logits, latent = model(x, edge_index, return_latent=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test MMD loss
    source_x = x[:25]
    target_x = x[25:]
    mmd_loss = MMDLoss()
    source_latent = model.encoder(source_x)
    target_latent = model.encoder(target_x)
    loss = mmd_loss(source_latent, target_latent)
    print(f"MMD loss: {loss.item():.4f}")