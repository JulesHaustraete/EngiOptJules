"""This code is largely based on the excellent PyTorch GAN repo: https://github.com/eriklindernoren/PyTorch-GAN.

We essentially refreshed the Python style, use wandb for logging, and made a few little improvements.
There are also a couple of code parts that are problem dependent and need to be adjusted for the specific problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import TYPE_CHECKING, Callable

from engibench.utils.all_problems import BUILTIN_PROBLEMS
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as f
from torchvision import transforms
import tqdm

from engiopt.transforms import flatten_dict_factory
import wandb

if TYPE_CHECKING:
    from engibench.utils.problem import Problem


class Normalizer:
    """Base normalizer class."""

    def __init__(self, min_val: th.Tensor, max_val: th.Tensor, eps: float = 1e-7):
        self.eps = eps
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalizes the input tensor."""
        raise NotImplementedError

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Denormalizes the input tensor."""
        raise NotImplementedError


class MinMaxNormalizer(Normalizer):
    """Min-Max normalization to [0, 1]."""

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalizes the input tensor to [0, 1]."""
        device = x.device
        min_val = self.min_val.to(device)
        max_val = self.max_val.to(device)
        return (x - min_val) / (max_val - min_val + self.eps)

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Denormalizes the input tensor from [0, 1]."""
        device = x.device
        min_val = self.min_val.to(device)
        max_val = self.max_val.to(device)
        return x * (max_val - min_val + self.eps) + min_val


class StandardScalerNormalizer(Normalizer):
    """Standard scaler normalization (z-score)."""

    def __init__(self, mean_val: th.Tensor, std_val: th.Tensor, eps: float = 1e-7):
        super().__init__(mean_val, std_val, eps)  # Use mean/std as min/max for consistency
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalizes the input tensor using z-score."""
        device = x.device
        mean_val = self.mean_val.to(device)
        std_val = self.std_val.to(device)
        return (x - mean_val) / (std_val + self.eps)

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Denormalizes the input tensor from z-score."""
        device = x.device
        mean_val = self.mean_val.to(device)
        std_val = self.std_val.to(device)
        return x * (std_val + self.eps) + mean_val


class NoNormalizer(Normalizer):
    """No normalization (identity)."""

    def __init__(self):
        # Dummy values for compatibility
        super().__init__(th.tensor(0.0), th.tensor(1.0))

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Returns input unchanged."""
        return x

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Returns input unchanged."""
        return x


class MultiNormalizer:
    """Normalizer that can handle multiple features separately."""
    
    def __init__(self, normalizers: list[Normalizer]):
        self.normalizers = normalizers
    
    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize each feature with its own normalizer."""
        if len(self.normalizers) == 1:
            # Common normalizer for all features
            return self.normalizers[0].normalize(x)
        else:
            # Separate normalizer per feature
            normalized_features = []
            for i, normalizer in enumerate(self.normalizers):
                if i < x.shape[-1]:  # Avoid index out of bounds
                    normalized_features.append(normalizer.normalize(x[..., i:i+1]))
            return th.cat(normalized_features, dim=-1)
    
    def denormalize(self, x: th.Tensor) -> th.Tensor:
        """Denormalize each feature with its own normalizer."""
        if len(self.normalizers) == 1:
            # Common normalizer for all features
            return self.normalizers[0].denormalize(x)
        else:
            # Separate normalizer per feature
            denormalized_features = []
            for i, normalizer in enumerate(self.normalizers):
                if i < x.shape[-1]:  # Avoid index out of bounds
                    denormalized_features.append(normalizer.denormalize(x[..., i:i+1]))
            return th.cat(denormalized_features, dim=-1)


def create_normalizer(normalization_type: str, data_tensor: th.Tensor, device: th.device, strategy: str = "common") -> Normalizer:
    """Factory function to create the appropriate normalizer."""
    if normalization_type == "MinMax":
        if strategy == "common":
            # Common mode: calculate across all but first dimension (original behavior)
            min_val = data_tensor.amin(dim=tuple(range(1, data_tensor.ndim))).to(device)
            max_val = data_tensor.amax(dim=tuple(range(1, data_tensor.ndim))).to(device)
        else:
            # Separate mode: calculate across samples (dim 0) for each feature
            min_val = data_tensor.amin(dim=0).to(device)
            max_val = data_tensor.amax(dim=0).to(device)
        return MinMaxNormalizer(min_val, max_val)
    
    elif normalization_type == "StandardScaler":
        if strategy == "common":
            # Common mode: calculate across all but first dimension (original behavior)
            mean_val = data_tensor.mean(dim=tuple(range(1, data_tensor.ndim))).to(device)
            std_val = data_tensor.std(dim=tuple(range(1, data_tensor.ndim))).to(device)
        else:
            # Separate mode: calculate across samples (dim 0) for each feature
            mean_val = data_tensor.mean(dim=0).to(device)
            std_val = data_tensor.std(dim=0).to(device)
        return StandardScalerNormalizer(mean_val, std_val)
    
    elif normalization_type == "No Norm":
        return NoNormalizer()
    
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

def create_multi_normalizer(
    normalization_type: str, 
    data_tensor: th.Tensor, 
    device: th.device,
    strategy: str = "common"
) -> MultiNormalizer:
    """Factory function to create multi-normalizer with different strategies.
    
    Args:
        normalization_type: Type of normalization ('MinMax', 'StandardScaler', 'No Norm')
        data_tensor: Input data tensor with shape [..., n_features]
        device: Device to put normalizers on
        strategy: 'common' (single scaler) or 'separate' (per-feature scaler)
    
    Returns:
        MultiNormalizer instance
    """
    if strategy == "common":
        # Single normalizer for all features
        normalizer = create_normalizer(normalization_type, data_tensor, device, strategy)
        return MultiNormalizer([normalizer])
    
    elif strategy == "separate":
        # Separate normalizer for each feature
        normalizers = []
        n_features = data_tensor.shape[-1]
        
        for i in range(n_features):
            # Extract data for feature i
            feature_data = data_tensor[..., i:i+1]
            normalizer = create_normalizer(normalization_type, feature_data, device, strategy)
            normalizers.append(normalizer)
        
        return MultiNormalizer(normalizers)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Available: ['common', 'separate']")





@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "airfoil"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    # Tracking
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    seed: int = 1
    """Random seed."""
    save_model: bool = False
    """Saves the model to disk."""

    # Algorithm specific
    n_epochs: int = 5000
    """number of epochs of training"""
    batch_size: int = 32
    """size of the batches"""
    lr_gen: float = 0.00005
    """learning rate for the generator"""
    lr_disc: float = 0.0002
    """learning rate for the discriminator"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 4
    """dimensionality of the latent space"""
    noise_dim: int = 10
    """dimensionality of the noise space for design diversity"""
    sample_interval: int = 400
    """interval between image samples"""
    
    # === ABLATION STUDY BOOLEANS ===
    # Step 1: Noise z for design diversity
    use_noise_z: bool = False
    """If True, use noise z + latent c (v1.1), else use only latent z (baseline)"""
    
    # Step 2: Normalization improvements
    normalization_type: str = "MinMax"
    """Type of normalization: 'StandardScaler', 'MinMax', or 'No Norm'"""
    
    normalization_strategy: str = "common"
    """Normalization strategy: 'common' (single scaler) or 'separate' (per-feature scaler)"""
    
    # Step 3: MLP features
    use_mlp_features: bool = False
    """If True, add MLP feature generator for abstract features"""
    
    # Step 4: Scalar decoder
    use_scalar_decoder: bool = False
    """If True, add scalar decoder/generator for design scalars"""
    
    # Step 5: CPW Generator for control points
    use_cpw_generator: bool = False
    """If True, add CPWGenerator for control points and weights (replaces standard MLP)"""
    
    # CPW Generator parameters
    n_control_points: int = 32
    """Number of control points for CPW generator"""
    n_data_points: int = 192
    """Number of data points for output design (airfoil default)"""
    use_cpw_interpolation: bool = False
    """If True, use interpolation for CPW (step 5), else use linear transform"""
    cpw_interpolation_type: str = "linear"
    """Type of interpolation: 'linear', 'cubic_spline', 'bspline'"""
    
    # Step 6: Separate normalization
    use_separate_normalization: bool = False
    """If True, use separate normalizers for conditions vs scalars (step 6)"""
    conditions_normalization_type: str = "StandardScaler"
    """Type of normalization for conditions: 'StandardScaler', 'MinMax', or 'No Norm'"""
    scalars_normalization_type: str = "MinMax"
    """Type of normalization for design scalars: 'StandardScaler', 'MinMax', or 'No Norm'"""
    
    # Step 7: Coordinate decoder (test without strong geometric constraints)
    use_coord_decoder: bool = False
    """If True, use coordinate decoder instead of simple MLP (step 7)"""
    
    # Step 8: Advanced discriminator
    use_advanced_discriminator: bool = False
    """If True, use CNN+InfoGAN discriminator instead of simple MLP"""
    discriminator_type: str = "mlp"
    """Type of discriminator: 'mlp' (simple), 'cnn' (convolutional), 'cnn_infogan' (CNN + Q-Network)"""
    
    # Step 9: Bezier layer (geometric constraints)
    use_bezier_layer: bool = False
    """If True, add BezierLayer for geometric constraints (step 9)"""
    
    # Step 10: Advanced loss functions
    use_advanced_losses: bool = False
    """If True, add Q-Loss, R-Loss, Curvature-Loss"""
    q_loss_weight: float = 1.0
    """Weight for Q-Loss (latent code inference loss)"""
    r_loss_weight: float = 10.0
    """Weight for R-Loss (regularization loss on control points and weights)"""
    curvature_loss_weight: float = 1.0
    """Weight for Curvature-Loss (smoothness constraint)"""


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        noise_dim: int,
        n_conds: int,
        design_shape: tuple[int, ...],
        design_normalizer: MinMaxNormalizer,
        conds_normalizer: MinMaxNormalizer,
        scalar_normalizer: MinMaxNormalizer = None,
        use_noise_z: bool = True,
        use_mlp_features: bool = False,
        use_scalar_decoder: bool = False,
        use_cpw_generator: bool = False,
        use_cpw_interpolation: bool = True,
        cpw_interpolation_type: str = "linear",
        use_separate_normalization: bool = False,
        use_coord_decoder: bool = False,
        # Step 9: Bezier layer parameters
        use_bezier_layer: bool = False,
        m_features: int = 256,
        feature_gen_layers: tuple[int, ...] = (1024,),
        scalar_features: int = 1,
        scalar_layers: tuple[int, ...] = (128, 128, 128, 128),
        # CPW Generator parameters
        n_control_points: int = 32,
        cpw_dense_layers: tuple[int, ...] = (1024,),
        cpw_deconv_channels: tuple[int, ...] = (768, 384, 192, 96),
        n_data_points: int = 192,  # Default for airfoil
        bezier_eps: float = 1e-7,  # Epsilon for BezierLayer numerical stability
    ):
        super().__init__()
        self.design_shape = design_shape  # Store design shape
        self.design_normalizer = design_normalizer
        self.conds_normalizer = conds_normalizer
        self.scalar_normalizer = scalar_normalizer  # Step 6: Separate scalar normalizer
        self.use_noise_z = use_noise_z
        self.use_mlp_features = use_mlp_features
        self.use_scalar_decoder = use_scalar_decoder
        self.use_cpw_generator = use_cpw_generator
        self.use_cpw_interpolation = use_cpw_interpolation
        self.cpw_interpolation_type = cpw_interpolation_type
        self.use_separate_normalization = use_separate_normalization  # Step 6
        self.use_coord_decoder = use_coord_decoder  # Step 7
        self.use_bezier_layer = use_bezier_layer  # Step 9
        self.m_features = m_features
        self.scalar_features = scalar_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.bezier_eps = bezier_eps

        def block(in_feat: int, out_feat: int, *, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Calculate input dimension based on whether we use noise_z
        input_dim = latent_dim + n_conds
        if self.use_noise_z:
            input_dim += noise_dim

        if self.use_coord_decoder:
            # Step 7: CoordDecoder - flexible MLP without geometric constraints
            # This directly generates coordinates without strong geometric priors
            expected_coord_size = int(np.prod(design_shape))
            if self.use_scalar_decoder:
                expected_coord_size -= scalar_features  # Exclude scalar features from coord decoder
            
            self.coord_decoder = nn.Sequential(
                *block(input_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, expected_coord_size),
                nn.Tanh(),
            )
            
            if self.use_scalar_decoder:
                # If using MLPs features, scalar generator takes features, otherwise takes input directly
                scalar_input_dim = m_features if self.use_mlp_features else input_dim
                
                if self.use_mlp_features:
                    # Create feature generator for scalar decoder
                    self.feature_generator = nn.Sequential(
                        *block(input_dim, 256, normalize=False),
                        *block(256, 512),
                        *block(512, m_features),
                    )
                
                # Scalar generator (separate decoder for scalar features)
                self.scalar_generator = nn.Sequential(
                    *block(scalar_input_dim, scalar_layers[0], normalize=False),
                    *[layer for i in range(len(scalar_layers) - 1) for layer in block(scalar_layers[i], scalar_layers[i+1])],
                    nn.Linear(scalar_layers[-1], scalar_features),
                    # No activation here - sigmoid is applied later
                )
        elif self.use_cpw_generator:
            # Step 5: Add CPWGenerator for control points and weights
            self.cpw_generator = CPWGenerator(
                input_dim, n_control_points, cpw_dense_layers, cpw_deconv_channels
            )
            
            # Step 9: Add BezierLayer for geometric constraints
            if self.use_bezier_layer:
                # BezierLayer requires features from feature generator
                if not self.use_mlp_features:
                    # Force feature generation for BezierLayer
                    self.use_mlp_features = True
                    print("Warning: Forcing use_mlp_features=True for BezierLayer compatibility")
                
                self.feature_generator = MLP(input_dim, m_features, feature_gen_layers)
                self.bezier_layer = BezierLayer(m_features, n_control_points, n_data_points, bezier_eps)
            else:
                # Need feature generator for scalar generation when using CPW
                if self.use_mlp_features:
                    self.feature_generator = MLP(input_dim, m_features, feature_gen_layers)
            
            if self.use_scalar_decoder:
                # Step 4: Add scalar generator for design scalars
                input_for_scalar = m_features if (self.use_mlp_features or self.use_bezier_layer) else input_dim
                self.scalar_generator = MLP(input_for_scalar, scalar_features, scalar_layers)
                
        elif self.use_mlp_features:
            # Step 3: Add feature generator for abstract features
            self.feature_generator = MLP(input_dim, m_features, feature_gen_layers)
            
            if self.use_scalar_decoder:
                # Step 4: Add scalar generator for design scalars
                self.scalar_generator = MLP(m_features, scalar_features, scalar_layers)
                # Adjust main generator output size to exclude scalars
                coord_features = int(np.prod(design_shape)) - scalar_features
                self.model = nn.Sequential(
                    *block(m_features, 128, normalize=False),
                    *block(128, 256),
                    *block(256, 512),
                    *block(512, 1024),
                    nn.Linear(1024, coord_features),
                    nn.Tanh(),
                )
            else:
                # Step 3 only: Main generator takes features as input
                self.model = nn.Sequential(
                    *block(m_features, 128, normalize=False),
                    *block(128, 256),
                    *block(256, 512),
                    *block(512, 1024),
                    nn.Linear(1024, int(np.prod(design_shape))),
                    nn.Tanh(),
                )
        else:
            # Standard MLP generator (baseline) or CPW without MLP features
            if not self.use_cpw_generator:
                self.model = nn.Sequential(
                    *block(input_dim, 128, normalize=False),
                    *block(128, 256),
                    *block(256, 512),
                    *block(512, 1024),
                    nn.Linear(1024, int(np.prod(design_shape))),
                    nn.Tanh(),
                )

    def forward(self, c: th.Tensor, z: th.Tensor = None, conds: th.Tensor = None) -> th.Tensor:
        """Forward pass for the generator.

        Args:
            c (th.Tensor): Latent code input tensor.
            z (th.Tensor, optional): Noise input tensor. Used only if use_noise_z=True.
            conds (th.Tensor): Condition tensor.

        Returns:
            th.Tensor: Generated design tensor.
        """
        normalized_conds = self.conds_normalizer.normalize(conds)
        
        if self.use_noise_z and z is not None:
            # Version 1.1+: use both c and z
            gen_input = th.cat((c, z, normalized_conds), -1)
        else:
            # Baseline version: use only c
            gen_input = th.cat((c, normalized_conds), -1)
        
        if self.use_coord_decoder:
            # Step 7: Use CoordDecoder for flexible coordinate generation without geometric constraints
            coords = self.coord_decoder(gen_input)  # Generate coordinate part directly
            
            if self.use_scalar_decoder:
                # Generate scalars separately
                if self.use_mlp_features:
                    features = self.feature_generator(gen_input)
                    scalars = th.sigmoid(self.scalar_generator(features))
                else:
                    scalars = th.sigmoid(self.scalar_generator(gen_input))
                
                # Combine scalars and coordinates (alphabetical order: angle_of_attack first, then coords)
                design = th.cat([scalars, coords], dim=-1)
            else:
                design = coords
                
        elif self.use_cpw_generator:
            # Step 5: Use CPWGenerator for control points and weights
            cp, w = self.cpw_generator(gen_input)  # Generate control points and weights
            
            if self.use_bezier_layer:
                # Step 9: Use BezierLayer for geometrically constrained curve generation
                features = self.feature_generator(gen_input)
                # BezierLayer generates [N, 2, n_data_points], then flatten to [N, 2*n_data_points]
                bezier_coords, _, _ = self.bezier_layer(features, cp, w)  # [N, 2, n_data_points]
                design = bezier_coords.view(bezier_coords.size(0), -1)  # [N, 2*n_data_points]
                
            elif self.use_cpw_interpolation:
                # Step 5: Use different interpolation methods between control points
                design = self.interpolate_control_points(cp, self.n_data_points, self.cpw_interpolation_type)  # [N, 2*n_data_points]
            else:
                # Fallback: Use control points directly with linear transformation
                # Flatten control points to match expected design shape
                design = cp.view(cp.size(0), -1)  # [N, 2*n_control_points]
                
                # If design doesn't match expected shape, adjust with linear transformation
                expected_size = int(np.prod(self.design_shape))
                if design.size(1) != expected_size:
                    if not hasattr(self, 'cp_to_design'):
                        self.cp_to_design = nn.Linear(design.size(1), expected_size).to(design.device)
                    design = self.cp_to_design(design)
            
            # Ensure design matches expected shape (for interpolation and bezier paths)
            if self.use_cpw_interpolation or self.use_bezier_layer:
                expected_size = int(np.prod(self.design_shape))
                if design.size(1) != expected_size:
                    # If sizes don't match, truncate or pad
                    if design.size(1) > expected_size:
                        design = design[:, :expected_size]  # Truncate
                    else:
                        # Pad with zeros if needed
                        pad_size = expected_size - design.size(1)
                        padding = th.zeros(design.size(0), pad_size, device=design.device)
                        design = th.cat([design, padding], dim=1)
            
            # Handle scalar generation if needed
            if self.use_scalar_decoder and hasattr(self, 'scalar_generator'):
                if hasattr(self, 'feature_generator'):
                    features = self.feature_generator(gen_input)
                    scalars = th.sigmoid(self.scalar_generator(features))
                else:
                    scalars = th.sigmoid(self.scalar_generator(gen_input))
                
                # Combine with scalars and coordinates (alphabetical order: scalars first, then coords)
                if design.size(1) > self.scalar_features:
                    coord_part = design[:, :-self.scalar_features]
                    design = th.cat([scalars, coord_part], dim=-1)
                else:
                    design = th.cat([scalars, design], dim=-1)
                    
        elif self.use_mlp_features:
            # Step 3: Generate abstract features first, then use them for design
            features = self.feature_generator(gen_input)
            
            if self.use_scalar_decoder:
                # Step 4: Generate coordinates and scalars separately
                coords = self.model(features)  # Generate coordinate part
                scalars = th.sigmoid(self.scalar_generator(features))  # Generate scalar part
                
                # Combine scalars and coordinates (alphabetical order: angle_of_attack first, then coords)
                design = th.cat([scalars, coords], dim=-1)
            else:
                # Step 3 only: Generate all features from abstract features
                design = self.model(features)
        else:
            # Standard generation (baseline)
            design = self.model(gen_input)
            
        design = design.view(design.size(0), *self.design_shape)
        
        # Step 6: Apply separate normalization for coordinates and scalars
        if self.use_separate_normalization and self.scalar_normalizer is not None:
            # Separate coordinates and scalars for denormalization
            # For airfoil: first feature is scalar (angle_of_attack), rest are coordinates
            scalar_part = design[..., :self.scalar_features]  # First feature(s)
            coord_part = design[..., self.scalar_features:]   # Rest of features
            
            # Denormalize coordinates and scalars separately
            denorm_scalars = self.scalar_normalizer.denormalize(scalar_part.view(scalar_part.size(0), -1))
            denorm_coords = self.design_normalizer.denormalize(coord_part.view(coord_part.size(0), -1))
            
            # Concatenate denormalized scalars and coordinates (preserve order)
            final_design = th.cat([
                denorm_scalars, 
                denorm_coords
            ], dim=-1)
            
            return final_design.view(final_design.size(0), *self.design_shape)
        else:
            # Standard denormalization (backward compatibility)
            return self.design_normalizer.denormalize(design)
    
    def interpolate_control_points(self, cp: th.Tensor, n_data_points: int, interpolation_type: str = "linear") -> th.Tensor:
        """
        Interpolation between control points with multiple methods for Step 5.
        
        Args:
            cp: Control points tensor [N, 2, n_control_points]
            n_data_points: Number of output data points
            interpolation_type: Type of interpolation ('linear', 'cubic_spline', 'bspline')
            
        Returns:
            Interpolated design [N, 2*n_data_points] (flattened for compatibility)
        """
        if interpolation_type == "linear":
            return self._linear_interpolation(cp, n_data_points)
        elif interpolation_type == "cubic_spline":
            return self._cubic_spline_interpolation(cp, n_data_points)
        elif interpolation_type == "bspline":
            return self._bspline_interpolation(cp, n_data_points)
        else:
            raise ValueError(f"Unknown interpolation type: {interpolation_type}. Available: ['linear', 'cubic_spline', 'bspline']")
    
    def _linear_interpolation(self, cp: th.Tensor, n_data_points: int) -> th.Tensor:
        """
        Linear interpolation between control points (original method).
        """
        batch_size = cp.size(0)
        device = cp.device
        
        # Extract x and y coordinates
        cp_x = cp[:, 0, :]  # [N, n_control_points]
        cp_y = cp[:, 1, :]  # [N, n_control_points]
        
        # Create parameter space for control points and output points
        t_control = th.linspace(0, 1, self.n_control_points, device=device)
        t_output = th.linspace(0, 1, n_data_points, device=device)
        
        x_interp = th.zeros(batch_size, n_data_points, device=device)
        y_interp = th.zeros(batch_size, n_data_points, device=device)
        
        for i in range(batch_size):
            # Use searchsorted to find indices for interpolation
            indices = th.searchsorted(t_control, t_output, right=False)
            indices = th.clamp(indices, 1, self.n_control_points - 1)
            
            # Get the left and right points for interpolation
            indices_left = indices - 1
            indices_right = indices
            
            # Get parameter values
            t_left = t_control[indices_left]
            t_right = t_control[indices_right]
            
            # Get control point values
            x_left = cp_x[i][indices_left]
            x_right = cp_x[i][indices_right]
            y_left = cp_y[i][indices_left]
            y_right = cp_y[i][indices_right]
            
            # Linear interpolation: v = v_left + (t - t_left) * (v_right - v_left) / (t_right - t_left)
            t_diff = t_right - t_left
            t_diff = th.where(t_diff == 0, th.ones_like(t_diff), t_diff)  # Avoid division by zero
            
            alpha = (t_output - t_left) / t_diff
            
            x_interp[i] = x_left + alpha * (x_right - x_left)
            y_interp[i] = y_left + alpha * (y_right - y_left)
        
        # Stack and flatten: [N, 2, n_data_points] -> [N, 2*n_data_points]
        design_2d = th.stack([x_interp, y_interp], dim=1)  # [N, 2, n_data_points]
        design_flat = design_2d.view(batch_size, -1)  # [N, 2*n_data_points]
        
        return design_flat
    
    def _cubic_spline_interpolation(self, cp: th.Tensor, n_data_points: int) -> th.Tensor:
        """
        Cubic spline interpolation between control points.
        More smooth than linear, better for airfoil aerodynamics.
        """
        batch_size = cp.size(0)
        device = cp.device
        
        # Extract x and y coordinates
        cp_x = cp[:, 0, :]  # [N, n_control_points]
        cp_y = cp[:, 1, :]  # [N, n_control_points]
        
        # Create parameter space for control points and output points
        t_control = th.linspace(0, 1, self.n_control_points, device=device)
        t_output = th.linspace(0, 1, n_data_points, device=device)
        
        x_interp = th.zeros(batch_size, n_data_points, device=device)
        y_interp = th.zeros(batch_size, n_data_points, device=device)
        
        for i in range(batch_size):
            # Cubic spline interpolation for x and y separately
            x_interp[i] = self._cubic_spline_1d(t_control, cp_x[i], t_output)
            y_interp[i] = self._cubic_spline_1d(t_control, cp_y[i], t_output)
        
        # Stack and flatten: [N, 2, n_data_points] -> [N, 2*n_data_points]
        design_2d = th.stack([x_interp, y_interp], dim=1)  # [N, 2, n_data_points]
        design_flat = design_2d.view(batch_size, -1)  # [N, 2*n_data_points]
        
        return design_flat
    
    def _cubic_spline_1d(self, t_control: th.Tensor, values: th.Tensor, t_output: th.Tensor) -> th.Tensor:
        """
        1D cubic spline interpolation using finite differences for derivatives.
        """
        n = len(t_control)
        device = t_control.device
        
        # Calculate finite differences for derivatives (natural spline boundary conditions)
        h = t_control[1:] - t_control[:-1]  # [n-1]
        
        # Build tridiagonal system for second derivatives
        # Using natural spline (second derivative = 0 at boundaries)
        A = th.zeros(n, n, device=device)
        b = th.zeros(n, device=device)
        
        # Natural boundary conditions
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        
        # Interior points
        for j in range(1, n-1):
            A[j, j-1] = h[j-1]
            A[j, j] = 2 * (h[j-1] + h[j])
            A[j, j+1] = h[j]
            b[j] = 6 * ((values[j+1] - values[j]) / h[j] - (values[j] - values[j-1]) / h[j-1])
        
        # Solve for second derivatives
        try:
            second_derivs = th.linalg.solve(A, b)
        except:
            # Fallback to pseudo-inverse if singular
            second_derivs = th.pinverse(A) @ b
        
        # Interpolate using cubic spline formula
        result = th.zeros_like(t_output)
        
        for k, t in enumerate(t_output):
            # Find interval
            if t <= t_control[0]:
                idx = 0
            elif t >= t_control[-1]:
                idx = n - 2
            else:
                idx = th.searchsorted(t_control[1:], t).item()
            
            # Cubic spline coefficients
            h_k = h[idx]
            t_diff = t - t_control[idx]
            
            a = values[idx]
            b_coeff = (values[idx+1] - values[idx]) / h_k - h_k * (2 * second_derivs[idx] + second_derivs[idx+1]) / 6
            c_coeff = second_derivs[idx] / 2
            d_coeff = (second_derivs[idx+1] - second_derivs[idx]) / (6 * h_k)
            
            # Evaluate cubic polynomial
            result[k] = a + b_coeff * t_diff + c_coeff * t_diff**2 + d_coeff * t_diff**3
        
        return result
    
    def _bspline_interpolation(self, cp: th.Tensor, n_data_points: int, degree: int = 3) -> th.Tensor:
        """
        B-spline interpolation between control points.
        Provides local control and smoothness, very close to Bezier curves.
        """
        batch_size = cp.size(0)
        device = cp.device
        
        # Extract x and y coordinates
        cp_x = cp[:, 0, :]  # [N, n_control_points]
        cp_y = cp[:, 1, :]  # [N, n_control_points]
        
        x_interp = th.zeros(batch_size, n_data_points, device=device)
        y_interp = th.zeros(batch_size, n_data_points, device=device)
        
        for i in range(batch_size):
            # B-spline interpolation for x and y separately
            x_interp[i] = self._bspline_1d(cp_x[i], n_data_points, degree)
            y_interp[i] = self._bspline_1d(cp_y[i], n_data_points, degree)
        
        # Stack and flatten: [N, 2, n_data_points] -> [N, 2*n_data_points]
        design_2d = th.stack([x_interp, y_interp], dim=1)  # [N, 2, n_data_points]
        design_flat = design_2d.view(batch_size, -1)  # [N, 2*n_data_points]
        
        return design_flat
    
    def _bspline_1d(self, control_points: th.Tensor, n_output: int, degree: int = 3) -> th.Tensor:
        """
        1D B-spline interpolation using De Boor's algorithm.
        """
        n_control = len(control_points)
        device = control_points.device
        
        # Create uniform knot vector
        n_knots = n_control + degree + 1
        knots = th.zeros(n_knots, device=device)
        for i in range(degree + 1, n_knots - degree - 1):
            knots[i] = (i - degree) / (n_control - degree)
        knots[n_knots - degree - 1:] = 1.0
        
        # Parameter values for output points
        u_values = th.linspace(0, 1, n_output, device=device)
        result = th.zeros(n_output, device=device)
        
        for j, u in enumerate(u_values):
            # Clamp u to valid range
            u = th.clamp(u, 0.0, 1.0 - 1e-7)
            
            # Find knot span
            span = self._find_span(n_control - 1, degree, u, knots)
            
            # Compute B-spline basis functions
            basis = self._basis_funs(span, u, degree, knots)
            
            # Evaluate curve point
            curve_point = 0.0
            for k in range(degree + 1):
                if span - degree + k >= 0 and span - degree + k < n_control:
                    curve_point += basis[k] * control_points[span - degree + k]
            
            result[j] = curve_point
        
        return result
    
    def _find_span(self, n: int, degree: int, u: float, knots: th.Tensor) -> int:
        """Find the knot span index for B-spline evaluation."""
        if u >= knots[n + 1]:
            return n
        if u <= knots[degree]:
            return degree
        
        low = degree
        high = n + 1
        mid = (low + high) // 2
        
        while u < knots[mid] or u >= knots[mid + 1]:
            if u < knots[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
        
        return mid
    
    def _basis_funs(self, i: int, u: float, degree: int, knots: th.Tensor) -> th.Tensor:
        """Compute B-spline basis functions using Cox-de Boor recursion."""
        device = knots.device
        basis = th.zeros(degree + 1, device=device)
        left = th.zeros(degree + 1, device=device)
        right = th.zeros(degree + 1, device=device)
        
        basis[0] = 1.0
        
        for j in range(1, degree + 1):
            left[j] = u - knots[i + 1 - j]
            right[j] = knots[i + j] - u
            saved = 0.0
            
            for r in range(j):
                temp = basis[r] / (right[r + 1] + left[j - r])
                basis[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            
            basis[j] = saved
        
        return basis


class Discriminator(nn.Module):
    """Step 8: Advanced discriminator with MLP, CNN, and CNN+InfoGAN (Q-Network) support."""
    
    def __init__(self, 
                 conds_normalizer: MinMaxNormalizer, 
                 design_normalizer: MinMaxNormalizer, 
                 design_shape: tuple[int, ...], 
                 n_conds: int, 
                 scalar_normalizer: MinMaxNormalizer = None, 
                 use_separate_normalization: bool = False,
                 scalar_features: int = 1,
                 discriminator_type: str = "mlp",
                 latent_dim: int = 4,
                 dropout: float = 0.4,
                 momentum: float = 0.9):
        super().__init__()
        self.conds_normalizer = conds_normalizer
        self.design_normalizer = design_normalizer
        self.scalar_normalizer = scalar_normalizer  # Step 6: Separate scalar normalizer
        self.use_separate_normalization = use_separate_normalization  # Step 6
        self.scalar_features = scalar_features
        self.discriminator_type = discriminator_type
        self.latent_dim = latent_dim
        self.n_conds = n_conds
        self.design_shape = design_shape
        
        if discriminator_type == "mlp":
            # Simple MLP discriminator (baseline)
            self.model = nn.Sequential(
                nn.Linear(int(np.prod(design_shape)) + n_conds, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )
            
        elif discriminator_type == "cnn":
            # CNN discriminator
            # Check if design is flattened (1D) or already 2D
            if len(design_shape) == 1:
                # Flattened design case - angle_of_attack at position 0, coords at positions 1-384
                # For airfoil: angle_of_attack at position 0, coords at positions 1-384 as [2, 192]
                coords_2d_shape = (2, 192)
            else:
                # Already 2D case - use design shape directly
                coords_2d_shape = design_shape
            
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 4), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(64, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout),
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(128, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout),
            )

            # Calculate conv output shape with dummy pass using coordinate shape
            test_in = th.zeros(1, 1, coords_2d_shape[0], coords_2d_shape[1])
            out = self.conv1(test_in)
            out = self.conv2(out)
            flat_dim = out.numel()
            
            # Store coordinate shape for forward pass
            self.coords_2d_shape = coords_2d_shape

            self.post_conv_fc = nn.Sequential(
                nn.Linear(flat_dim + n_conds, 256),
                nn.BatchNorm1d(256, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
            )

            # Final discriminator head
            self.d_out = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
            
        elif discriminator_type == "cnn_infogan":
            # CNN + InfoGAN discriminator (with Q-Network)
            # Check if design is flattened (1D) or already 2D
            if len(design_shape) == 1:
                # Flattened design case - angle_of_attack at position 0, coords at positions 1-384
                # For airfoil: angle_of_attack at position 0, coords at positions 1-384 as [2, 192]
                coords_2d_shape = (2, 192)
            else:
                # Already 2D case - use design shape directly
                coords_2d_shape = design_shape
            
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 4), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(64, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout),
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(128, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout),
            )

            # Calculate conv output shape with dummy pass using coordinate shape
            test_in = th.zeros(1, 1, coords_2d_shape[0], coords_2d_shape[1])
            out = self.conv1(test_in)
            out = self.conv2(out)
            flat_dim = out.numel()
            
            # Store coordinate shape for forward pass
            self.coords_2d_shape = coords_2d_shape

            self.post_conv_fc = nn.Sequential(
                nn.Linear(flat_dim + n_conds, 256),
                nn.BatchNorm1d(256, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
            )

            # Discriminator head
            self.d_out = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
            
            # Q-Network head (for InfoGAN)
            self.q_fc = nn.Sequential(
                nn.Linear(256, 128),
                nn.BatchNorm1d(128, momentum=momentum),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.q_mean = nn.Linear(128, latent_dim)
            self.q_logstd = nn.Linear(128, latent_dim)
            
        else:
            raise ValueError(f"Unknown discriminator type: {discriminator_type}. Available: ['mlp', 'cnn', 'cnn_infogan']")

    def forward(self, design: th.Tensor, conds: th.Tensor) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """Forward pass for the discriminator.
        
        Returns:
            - For MLP and CNN: single tensor (discriminator output)
            - For CNN+InfoGAN: tuple (discriminator output, q_output)
        """
        design_flat = design.view(design.size(0), -1)
        
        # Step 6: Apply separate normalization for coordinates and scalars
        if self.use_separate_normalization and self.scalar_normalizer is not None:
            # Separate coordinates and scalars for normalization
            # For airfoil: first feature is scalar (angle_of_attack), rest are coordinates
            scalar_part = design_flat[:, :self.scalar_features]  # First feature(s)
            coord_part = design_flat[:, self.scalar_features:]   # Rest of features
            
            # Normalize coordinates and scalars separately
            norm_scalars = self.scalar_normalizer.normalize(scalar_part)
            norm_coords = self.design_normalizer.normalize(coord_part)
            
            # Concatenate normalized scalars and coordinates (preserve order)
            normalized_design = th.cat([norm_scalars, norm_coords], dim=-1)
        else:
            # Standard normalization (backward compatibility)
            normalized_design = self.design_normalizer.normalize(design_flat)
        
        normalized_conds = self.conds_normalizer.normalize(conds)
        
        if self.discriminator_type == "mlp":
            # MLP path
            d_in = th.cat((normalized_design, normalized_conds), -1)
            return self.model(d_in)
            
        elif self.discriminator_type == "cnn":
            # CNN path - coordinates are at positions 1-384 (angle_of_attack at position 0)
            if len(self.design_shape) == 1:
                # Flattened case: extract coordinates from positions 1-384
                coords_only = normalized_design[:, 1:385]  # Extract positions 1-384 (384 features)
                # Reshape to [N, 1, 2, 192] for CNN
                x = coords_only.view(-1, 2, 192).unsqueeze(1)
            else:
                # Already 2D case - use all design
                x = normalized_design.view(-1, *self.design_shape).unsqueeze(1)
            
            out = self.conv1(x)
            out = self.conv2(out)
            out = out.view(out.size(0), -1)
            
            # Concatenate with conditions
            out = th.cat((out, normalized_conds), dim=1)
            out = self.post_conv_fc(out)
            
            return self.d_out(out)
            
        elif self.discriminator_type == "cnn_infogan":
            # CNN + InfoGAN path - coordinates are at positions 1-384 (angle_of_attack at position 0)
            if len(self.design_shape) == 1:
                # Flattened case: extract coordinates from positions 1-384
                coords_only = normalized_design[:, 1:385]  # Extract positions 1-384 (384 features)
                # Reshape to [N, 1, 2, 192] for CNN
                x = coords_only.view(-1, 2, 192).unsqueeze(1)
            else:
                # Already 2D case - use all design
                x = normalized_design.view(-1, *self.design_shape).unsqueeze(1)
            
            out = self.conv1(x)
            out = self.conv2(out)
            out = out.view(out.size(0), -1)
            
            # Concatenate with conditions
            out = th.cat((out, normalized_conds), dim=1)
            out = self.post_conv_fc(out)

            # Discriminator output
            d = self.d_out(out)

            # Q-Network output
            q_int = self.q_fc(out)
            q_mean = self.q_mean(q_int)
            q_logstd = self.q_logstd(q_int)
            q_logstd = th.clamp(q_logstd, min=-16)  # Prevent numerical issues

            # q shape [N, 2, latent_dim] - [mean, logstd]
            q = th.stack([q_mean, q_logstd], dim=1)
            return d, q


def compute_q_loss(q_mean: th.Tensor, q_logstd: th.Tensor, q_target: th.Tensor, eps: float = 1e-7) -> th.Tensor:
    """Computes latent code inference loss for InfoGAN.
    
    NOTE: This function is implemented for Step 8 (architecture) but will only be USED in Step 10 (advanced losses).
    For Step 8, we only test the CNN+Q-Network architecture without using the Q-Loss in training.
    
    Args:
        q_mean: Predicted mean of latent code [N, latent_dim]
        q_logstd: Predicted log std of latent code [N, latent_dim]  
        q_target: Target latent code [N, latent_dim]
        eps: Small epsilon for numerical stability
        
    Returns:
        Q-loss scalar tensor
    """
    epsilon = (q_target - q_mean) / (q_logstd.exp() + eps)
    q_loss_elem = q_logstd + 0.5 * (epsilon**2)
    return q_loss_elem.mean()


def bce_with_logits(pred: th.Tensor, target: th.Tensor) -> th.Tensor:
    """Replicates tensorflow sigmoid_cross_entropy_with_logits for consistency with cgan_bezier."""
    return f.binary_cross_entropy_with_logits(pred, target, reduction="mean")


def compute_r_loss(cp: th.Tensor, w: th.Tensor) -> th.Tensor:
    """Computes regularization loss on control points and weights (Step 10).
    
    This loss encourages:
    1. Sparsity in weights (except endpoints)
    2. Smooth transitions between control points
    3. Proper closure of airfoil shape
    
    Args:
        cp: Control points tensor [N, 2, n_control_points]
        w: Weights tensor [N, 1, n_control_points]
        
    Returns:
        R-loss scalar tensor
    """
    # Weight regularization: encourage sparsity (except endpoints)
    r_w_loss = w[:, :, 1:-1].mean()

    # Control point smoothness: minimize distance between consecutive points
    cp_diff = cp[:, :, 1:] - cp[:, :, :-1]
    cp_dist = cp_diff.norm(dim=1)
    r_cp_loss = cp_dist.mean()

    # Closure constraint: first and last points should be close for airfoil
    ends = cp[:, :, 0] - cp[:, :, -1]
    end_norm = ends.norm(dim=1)
    # Penalty for negative y-coordinates at closure (airfoil should close above x-axis)
    penal = th.clamp(-10 * ends[:, 1], min=0.0)
    r_ends_loss = end_norm + penal
    r_ends_loss_mean = r_ends_loss.mean()

    return r_w_loss + r_cp_loss + r_ends_loss_mean


def compute_curvature_loss(design: th.Tensor, eps: float = 1e-7) -> th.Tensor:
    """Computes curvature-based smoothness loss (Step 10).
    
    This loss penalizes high curvature to encourage smooth airfoil shapes.
    For 2D curves, curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    
    Args:
        design: Design tensor [N, 2*n_points] (flattened coordinates)
        eps: Small epsilon for numerical stability
        
    Returns:
        Curvature-loss scalar tensor
    """
    batch_size = design.shape[0]
    n_points = design.shape[1] // 2
    
    # Reshape to [N, 2, n_points]
    coords = design.view(batch_size, 2, n_points)
    x = coords[:, 0, :]  # [N, n_points]
    y = coords[:, 1, :]  # [N, n_points]
    
    # Compute first derivatives (central differences for interior points)
    dx = th.zeros_like(x)
    dy = th.zeros_like(y)
    
    # Forward difference for first point
    dx[:, 0] = x[:, 1] - x[:, 0]
    dy[:, 0] = y[:, 1] - y[:, 0]
    
    # Central difference for interior points
    dx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2.0
    dy[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / 2.0
    
    # Backward difference for last point
    dx[:, -1] = x[:, -1] - x[:, -2]
    dy[:, -1] = y[:, -1] - y[:, -2]
    
    # Compute second derivatives (central differences)
    d2x = th.zeros_like(x)
    d2y = th.zeros_like(y)
    
    # For interior points only (need 3 points for second derivative)
    if n_points >= 3:
        d2x[:, 1:-1] = x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]
        d2y[:, 1:-1] = y[:, 2:] - 2*y[:, 1:-1] + y[:, :-2]
    
    # Compute curvature: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    numerator = th.abs(dx * d2y - dy * d2x)
    denominator = th.pow(dx**2 + dy**2 + eps, 1.5)
    curvature = numerator / denominator
    
    # Return mean curvature as loss (penalize high curvature)
    return curvature.mean()


def prepare_data_with_separate_normalization(
    problem: Problem, 
    device: th.device, 
    use_separate_normalization: bool = False,
    conditions_normalization_type: str = "StandardScaler",
    scalars_normalization_type: str = "MinMax",
    normalization_type: str = "MinMax",
    normalization_strategy: str = "common"
) -> tuple[th.utils.data.TensorDataset, MinMaxNormalizer, MinMaxNormalizer, MinMaxNormalizer]:
    """Prepares the data with separate normalization for conditions and scalars (Step 6).

    Args:
        problem (Problem): The problem to prepare the data for.
        device (th.device): The device to prepare the data on.
        use_separate_normalization (bool): If True, use separate normalizers for conditions vs scalars.
        conditions_normalization_type (str): Type of normalization for conditions.
        scalars_normalization_type (str): Type of normalization for design scalars.
        normalization_type (str): Default normalization type for backward compatibility.
        normalization_strategy (str): Strategy ('common', 'separate').

    Returns:
        tuple: (training_dataset, condition_normalizer, design_normalizer, scalar_normalizer)
    """
    training_ds = problem.dataset.with_format("torch", device=device)["train"]

    # Flatten the designs if they are a Dict
    if isinstance(problem.design_space, spaces.Box):
        transform = transforms.Lambda(lambda x: x.flatten(1))
    elif isinstance(problem.design_space, spaces.Dict):
        transform = flatten_dict_factory(problem, device)

    training_ds = th.utils.data.TensorDataset(
        transform(training_ds["optimal_design"]),
        *[training_ds[key] for key in problem.conditions_keys],
    )

    if use_separate_normalization:
        # Step 6: Separate normalization for conditions vs scalars
        
        # Create condition normalizer with specific type
        if normalization_strategy == "common":
            cond_tensors = th.stack(training_ds.tensors[1:])
        else:
            cond_list = [tensor.unsqueeze(1) for tensor in training_ds.tensors[1:]]
            cond_tensors = th.cat(cond_list, dim=1)
        
        conds_normalizer = create_multi_normalizer(
            conditions_normalization_type, cond_tensors, device, normalization_strategy
        )

        # Separate design into coordinates and scalars  
        designs_tensor = training_ds.tensors[0]  # [n_samples, n_features]
        
        # For airfoil: first feature is scalar (angle_of_attack), rest are coordinates
        scalar_tensor = designs_tensor[:, 0:1]        # First feature (angle_of_attack)
        coordinate_tensor = designs_tensor[:, 1:]     # Rest of features (coordinates)
        
        # Create separate normalizers for coordinates and scalars
        if normalization_strategy == "common":
            coord_data_for_norm = coordinate_tensor.T
            scalar_data_for_norm = scalar_tensor.T
        else:
            coord_data_for_norm = coordinate_tensor
            scalar_data_for_norm = scalar_tensor
        
        design_normalizer = create_multi_normalizer(
            normalization_type, coord_data_for_norm, device, normalization_strategy
        )
        
        scalar_normalizer = create_multi_normalizer(
            scalars_normalization_type, scalar_data_for_norm, device, normalization_strategy
        )
        
    else:
        # Standard normalization (backward compatibility)
        scalar_normalizer = None
        
        # Create condition normalizer using the multi-normalizer factory
        if normalization_strategy == "common":
            cond_tensors = th.stack(training_ds.tensors[1:])
        else:
            cond_list = [tensor.unsqueeze(1) for tensor in training_ds.tensors[1:]]
            cond_tensors = th.cat(cond_list, dim=1)
        
        conds_normalizer = create_multi_normalizer(
            normalization_type, cond_tensors, device, normalization_strategy
        )

        # Create design normalizer using the multi-normalizer factory
        if normalization_strategy == "common":
            design_tensors = training_ds.tensors[0].T
        else:
            design_tensors = training_ds.tensors[0]
        
        design_normalizer = create_multi_normalizer(
            normalization_type, design_tensors, device, normalization_strategy
        )

    return training_ds, conds_normalizer, design_normalizer, scalar_normalizer


def prepare_data(
    problem: Problem, 
    device: th.device, 
    normalization_type: str = "MinMax",
    normalization_strategy: str = "common"
) -> tuple[th.utils.data.TensorDataset, MinMaxNormalizer, MinMaxNormalizer]:
    """Prepares the data for the generator and discriminator (backward compatibility).

    Args:
        problem (Problem): The problem to prepare the data for.
        device (th.device): The device to prepare the data on.
        normalization_type (str): Type of normalization ('MinMax', 'StandardScaler', 'No Norm').
        normalization_strategy (str): Strategy ('common', 'separate').

    Returns:
        tuple[th.utils.data.TensorDataset, MinMaxNormalizer, MinMaxNormalizer]: The training dataset, condition normalizer, and design normalizer.
    """
    training_ds, conds_normalizer, design_normalizer, _ = prepare_data_with_separate_normalization(
        problem, device, 
        use_separate_normalization=False,
        normalization_type=normalization_type, 
        normalization_strategy=normalization_strategy
    )
    return training_ds, conds_normalizer, design_normalizer


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_width: tuple[int, ...],
        activation_block: type[nn.Module] = nn.LeakyReLU,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.model = self._build_model(layer_width, activation_block, alpha)

    def _build_model(
        self,
        layer_width: tuple[int, ...],
        activation_block: type[nn.Module],
        alpha: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_sizes = (self.in_features, *layer_width)
        out_sizes = (*layer_width, self.out_features)
        for idx, (in_f, out_f) in enumerate(zip(in_sizes, out_sizes)):
            layers.append(nn.Linear(in_f, out_f))
            if idx < len(layer_width):  # Hidden layers, not the final one
                layers.append(nn.BatchNorm1d(out_f))
                layers.append(activation_block(alpha))
        return nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass for the MLP."""
        return self.model(x)


class Deconv1DCombo(nn.Module):
    """1D deconvolutional block with BatchNorm and LeakyReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(alpha),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass for Deconv1DCombo."""
        return self.seq(x)


class CPWGenerator(nn.Module):
    """Control Points and Weights Generator from Bezier GAN."""
    
    def __init__(
        self,
        in_features: int,
        n_control_points: int,
        dense_layers: tuple[int, ...] = (1024,),
        deconv_channels: tuple[int, ...] = (96 * 8, 96 * 4, 96 * 2, 96),
    ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points

        self.in_chnl, self.in_width = self._calculate_parameters(n_control_points, deconv_channels)
        self.dense = MLP(in_features, self.in_chnl * self.in_width, dense_layers)
        self.deconv = self._build_deconv(deconv_channels)
        self.cp_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 2, 1), nn.Tanh())
        self.w_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 1, 1), nn.Sigmoid())

    def _calculate_parameters(self, n_control_points: int, channels: tuple[int, ...]) -> tuple[int, int]:
        n_l = len(channels) - 1
        in_chnl = channels[0]
        in_width = n_control_points // (2**n_l)
        assert in_width >= 4, (
            f"Too many deconvolutional layers ({n_l}) for the {self.n_control_points} control points."
        )
        return in_chnl, in_width

    def _build_deconv(self, channels: tuple[int, ...]) -> nn.Sequential:
        deconv_blocks = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            block = Deconv1DCombo(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            deconv_blocks.append(block)
        return nn.Sequential(*deconv_blocks)

    def forward(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass for the CPWGenerator."""
        x = self.dense(z).view(-1, self.in_chnl, self.in_width)  # [N, in_chnl, in_width]
        x = self.deconv(x)  # [N, out_chnl, width_out]
        cp = self.cp_gen(x)  # [N, 2, n_control_points]
        w = self.w_gen(x)  # [N, 1, n_control_points]
        return cp, w


class BezierLayer(nn.Module):
    """Bezier curve generation layer."""
    
    def __init__(self, in_features: int, n_control_points: int, n_data_points: int, eps: float = 1e-7):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.eps = eps

        self.generate_intervals = nn.Sequential(
            nn.Linear(in_features, n_data_points - 1),
            nn.Softmax(dim=1),
            nn.ConstantPad1d((1, 0), 0),  # leading zero
        )

    def forward(
        self,
        features: th.Tensor,
        control_points: th.Tensor,
        weights: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass for the Bezier layer."""
        cp, w = self._check_consistency(control_points, weights)
        bs, pv, intvls = self._generate_bernstein_polynomial(features)
        dp = (cp * w) @ bs / (w @ bs)  # [N, 2, n_data_points]
        return dp, pv, intvls

    def _check_consistency(self, control_points: th.Tensor, weights: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        assert control_points.shape[-1] == self.n_control_points
        assert weights.shape[-1] == self.n_control_points
        return control_points, weights

    def _generate_bernstein_polynomial(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        intvls = self.generate_intervals(features)  # [N, n_data_points]
        pv = th.cumsum(intvls, dim=-1).clamp(0, 1).unsqueeze(1)  # [N, 1, n_data_points]

        pw1 = th.arange(0.0, self.n_control_points, device=features.device).view(1, -1, 1)
        pw2 = th.flip(pw1, (1,))
        lbs = (
            pw1 * th.log(pv + self.eps)
            + pw2 * th.log(1 - pv + self.eps)
            + th.lgamma(th.tensor(self.n_control_points, device=features.device) + self.eps).view(1, -1, 1)
            - th.lgamma(pw1 + 1 + self.eps)
            - th.lgamma(pw2 + 1 + self.eps)
        )
        bs = th.exp(lbs)  # [N, n_control_points, n_data_points]
        return bs, pv, intvls


def debug_print(msg: str, force: bool = False):
    """Print debug message."""
    pass  # Debug disabled for simplicity


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    if not isinstance(problem.design_space, (spaces.Box, spaces.Dict)):
        raise ValueError("This algorithm only works with Box or Dict spaces.")

    if isinstance(problem.design_space, spaces.Box):
        design_shape = problem.design_space.shape
    else:
        dummy_design, _ = problem.random_design()
        design_shape = spaces.flatten(problem.design_space, dummy_design).shape
    conditions = problem.conditions
    n_conds = len(conditions)

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), save_code=True, name=run_name)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    training_ds, conds_normalizer, design_normalizer, scalar_normalizer = prepare_data_with_separate_normalization(
        problem, device, 
        use_separate_normalization=args.use_separate_normalization,
        conditions_normalization_type=args.conditions_normalization_type,
        scalars_normalization_type=args.scalars_normalization_type,
        normalization_type=args.normalization_type, 
        normalization_strategy=args.normalization_strategy
    )

    # DEBUG: Print design space information
    debug_print(f"=== DEBUG: Design Space Information ===")
    debug_print(f"Problem: {args.problem_id}")
    debug_print(f"Design space type: {type(problem.design_space)}")
    debug_print(f"Design space: {problem.design_space}")
    debug_print(f"Design shape: {design_shape}")
    
    # DEBUG: Print sample data shapes
    sample_design, *sample_conds = training_ds[0]
    debug_print(f"=== DEBUG: Sample Data Shapes ===")
    debug_print(f"Sample design shape: {sample_design.shape}")
    debug_print(f"Sample design dtype: {sample_design.dtype}")
    debug_print(f"Number of conditions: {len(sample_conds)}")
    for i, cond in enumerate(sample_conds):
        debug_print(f"  Condition {i} shape: {cond.shape}")
    debug_print(f"Expected design_shape for pipeline: {design_shape}")
    debug_print(f"Total design features: {int(np.prod(design_shape))}")

    print(f"Using normalization: {args.normalization_type} with strategy: {args.normalization_strategy}")
    print(f"Using separate normalization: {args.use_separate_normalization}")
    if args.use_separate_normalization:
        print(f"  - Conditions normalization: {args.conditions_normalization_type}")
        print(f"  - Scalars normalization: {args.scalars_normalization_type}")
    print(f"Conditions normalizers: {len(conds_normalizer.normalizers)}")
    print(f"Design normalizers: {len(design_normalizer.normalizers)}")
    if scalar_normalizer:
        print(f"Scalar normalizers: {len(scalar_normalizer.normalizers)}")
    print(f"Using MLP features: {args.use_mlp_features}")
    print(f"Using scalar decoder: {args.use_scalar_decoder}")
    print(f"Using CPW generator: {args.use_cpw_generator}")
    if args.use_cpw_generator:
        print(f"  - Control points: {args.n_control_points}")
        print(f"  - Data points: {args.n_data_points}")
        print(f"  - Interpolation: {args.cpw_interpolation_type if args.use_cpw_interpolation else 'Linear transform'}")
        print(f"  - Bezier layer: {'ENABLED' if args.use_bezier_layer else 'DISABLED'}")
        if args.use_bezier_layer:
            print(f"    * Geometric constraints via Bernstein polynomials")
            print(f"    * Epsilon for numerical stability: {1e-7}")  # Default bezier_eps
    print(f"Using noise z: {args.use_noise_z}")
    print(f"Using coordinate decoder: {args.use_coord_decoder}")
    print(f"Design shape: {design_shape} (total features: {int(np.prod(design_shape))})")
    
    # Step 8: Discriminator type information
    print(f"Using discriminator type: {args.discriminator_type}")
    if args.discriminator_type == "mlp":
        print("  - Simple MLP discriminator (baseline)")
    elif args.discriminator_type == "cnn":
        print("  - CNN discriminator with 2D convolutions")
    elif args.discriminator_type == "cnn_infogan":
        print("  - CNN + InfoGAN discriminator with Q-Network")
        print(f"  - Latent code inference for {args.latent_dim}D latent space")
    
    # Step 10: Advanced loss functions information
    print(f"Using advanced losses: {args.use_advanced_losses}")
    if args.use_advanced_losses:
        print("  - Q-Loss: Latent code inference loss (InfoGAN)")
        print(f"    * Weight: {args.q_loss_weight}")
        print("  - R-Loss: Regularization loss on control points and weights")
        print(f"    * Weight: {args.r_loss_weight}")
        print("  - Curvature-Loss: Smoothness constraint on generated curves")
        print(f"    * Weight: {args.curvature_loss_weight}")
        if args.discriminator_type != "cnn_infogan":
            print("    Note: Q-Loss requires InfoGAN discriminator, only Curvature-Loss will be used")
        if not args.use_cpw_generator:
            print("    Note: R-Loss requires CPW generator, will be set to zero")

    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Loss function
    adversarial_loss = th.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(
        latent_dim=args.latent_dim,
        noise_dim=args.noise_dim,
        n_conds=n_conds,
        design_shape=design_shape,
        design_normalizer=design_normalizer,
        conds_normalizer=conds_normalizer,
        scalar_normalizer=scalar_normalizer,  # Step 6: Pass scalar normalizer
        use_noise_z=args.use_noise_z,
        use_mlp_features=args.use_mlp_features,
        use_scalar_decoder=args.use_scalar_decoder,
        use_cpw_generator=args.use_cpw_generator,
        use_cpw_interpolation=args.use_cpw_interpolation,
        cpw_interpolation_type=args.cpw_interpolation_type,
        use_separate_normalization=args.use_separate_normalization,  # Step 6
        use_coord_decoder=args.use_coord_decoder,  # Step 7
        use_bezier_layer=args.use_bezier_layer,  # Step 9
        m_features=256,  # Default from cgan_bezier
        feature_gen_layers=(1024,),  # Default from cgan_bezier
        scalar_features=1,  # Default: 1 scalar (angle of attack for airfoil)
        scalar_layers=(128, 128, 128, 128),  # Default from cgan_bezier
        # CPW Generator parameters
        n_control_points=args.n_control_points,
        n_data_points=args.n_data_points,
    )
    discriminator = Discriminator(
        conds_normalizer=conds_normalizer, 
        design_normalizer=design_normalizer,
        scalar_normalizer=scalar_normalizer,  # Step 6: Pass scalar normalizer
        design_shape=design_shape,
        n_conds=n_conds,
        use_separate_normalization=args.use_separate_normalization,  # Step 6
        scalar_features=1,  # Default: 1 scalar (angle of attack for airfoil)
        discriminator_type=args.discriminator_type,  # Step 8: Discriminator type
        latent_dim=args.latent_dim,  # Step 8: For InfoGAN Q-Network
    )

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Optimizers
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2))
    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(args.b1, args.b2))

    # Bounds for latent code c (aligned with cgan_bezier)
    bounds = (0.0, 1.0)

    @th.no_grad()
    def sample_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Samples n_designs from the generator."""
        if args.use_noise_z:
            # Version 1.1: Sample latent code c (uniform) and noise z (normal with std=0.5)
            c = (bounds[1] - bounds[0]) * th.rand((n_designs, args.latent_dim), device=device, dtype=th.float) + bounds[0]
            z = 0.5 * th.randn((n_designs, args.noise_dim), device=device, dtype=th.float)
        else:
            # Baseline version: Only latent code (called z for compatibility)
            c = th.randn((n_designs, args.latent_dim), device=device, dtype=th.float)
            z = None

        linspaces = [
            th.linspace(conds[:, i].min(), conds[:, i].max(), n_designs, device=device) for i in range(conds.shape[1])
        ]

        desired_conds = th.stack(linspaces, dim=1)
        gen_designs = generator(c, z, desired_conds)
        return desired_conds, gen_designs

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            designs = data[0]
            conds = th.stack((data[1:]), dim=1)

            # Adversarial ground truths
            valid = th.ones((designs.size(0), 1), requires_grad=False, device=device)
            fake = th.zeros((designs.size(0), 1), requires_grad=False, device=device)

            # Generate latent codes for this batch
            if args.use_noise_z:
                # Version 1.1: Sample latent code and noise as generator input
                c = (bounds[1] - bounds[0]) * th.rand((designs.size(0), args.latent_dim), device=device, dtype=th.float) + bounds[0]
                z = 0.5 * th.randn((designs.size(0), args.noise_dim), device=device, dtype=th.float)
            else:
                # Baseline version: Only latent code
                c = th.randn((designs.size(0), args.latent_dim), device=device, dtype=th.float)
                z = None

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_generator.zero_grad()

            gen_designs = generator(c, z, conds)

            # Step 10: Advanced loss functions integration
            if args.discriminator_type == "cnn_infogan":
                # InfoGAN architecture with advanced losses
                d_fake, q_fake = discriminator(gen_designs, conds)
                g_loss_base = adversarial_loss(d_fake, valid)
                
                # Initialize advanced losses
                q_loss = th.tensor(0.0, device=device)
                r_loss = th.tensor(0.0, device=device)
                curvature_loss = th.tensor(0.0, device=device)
                
                if args.use_advanced_losses:
                    # Q-Loss (latent code inference)
                    q_mean = q_fake[:, 0, :]  # [N, latent_dim]
                    q_logstd = q_fake[:, 1, :]  # [N, latent_dim]
                    q_loss = compute_q_loss(q_mean, q_logstd, q_target=c)
                    
                    # R-Loss (regularization on control points) - only if using CPW generator
                    if args.use_cpw_generator and hasattr(generator, 'cpw_generator'):
                        # We need to get control points and weights from generator
                        # For now, set to zero as we don't have direct access
                        # This could be improved by modifying generator to return CP and weights
                        r_loss = th.tensor(0.0, device=device)
                    
                    # Curvature-Loss (smoothness constraint)
                    curvature_loss = compute_curvature_loss(gen_designs[:, 1:])
                
                # Total generator loss
                g_loss = (g_loss_base + 
                         args.q_loss_weight * q_loss + 
                         args.r_loss_weight * r_loss + 
                         args.curvature_loss_weight * curvature_loss)
                         
            else:
                # Standard GAN training (MLP or CNN discriminator)
                d_fake = discriminator(gen_designs, conds)
                g_loss_base = adversarial_loss(d_fake, valid)
                
                # Initialize advanced losses
                q_loss = th.tensor(0.0, device=device)
                r_loss = th.tensor(0.0, device=device)
                curvature_loss = th.tensor(0.0, device=device)
                
                if args.use_advanced_losses:
                    # Only curvature loss available for non-InfoGAN discriminators
                    curvature_loss = compute_curvature_loss(gen_designs[:, 1:])
                
                # Total generator loss
                g_loss = g_loss_base + args.curvature_loss_weight * curvature_loss

            g_loss.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_discriminator.zero_grad()

            if args.discriminator_type == "cnn_infogan":
                # InfoGAN discriminator training with advanced losses
                
                # Real samples
                d_real, q_real = discriminator(designs, conds)
                d_loss_real = adversarial_loss(d_real, valid)
                
                # Fake samples
                d_fake_detached, q_fake_detached = discriminator(gen_designs.detach(), conds)
                d_loss_fake = adversarial_loss(d_fake_detached, fake)
                
                # Initialize Q-loss for discriminator
                d_q_loss = th.tensor(0.0, device=device)
                
                if args.use_advanced_losses:
                    # Q-Loss on fake samples (discriminator trains to predict correct latent code)
                    q_mean_detached = q_fake_detached[:, 0, :]  # [N, latent_dim]
                    q_logstd_detached = q_fake_detached[:, 1, :]  # [N, latent_dim]
                    d_q_loss = compute_q_loss(q_mean_detached, q_logstd_detached, q_target=c)
                
                # Total discriminator loss
                d_loss = (d_loss_real + d_loss_fake) / 2 + args.q_loss_weight * d_q_loss
                
            else:
                # Standard GAN discriminator training (MLP or CNN)
                d_real = discriminator(designs, conds)
                d_fake_detached = discriminator(gen_designs.detach(), conds)
                
                d_loss_real = adversarial_loss(d_real, valid)
                d_loss_fake = adversarial_loss(d_fake_detached, fake)
                d_loss = (d_loss_real + d_loss_fake) / 2
                
                # No Q-loss for non-InfoGAN discriminators
                d_q_loss = th.tensor(0.0, device=device)

            d_loss.backward()
            optimizer_discriminator.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                
                # Prepare logging data based on discriminator type and advanced losses
                log_data = {
                    "d_loss": d_loss.item(),
                    "g_loss": g_loss.item(),
                    "g_loss_base": g_loss_base.item(),
                    "epoch": epoch,
                    "batch": batches_done,
                    "discriminator_type": args.discriminator_type,
                    "use_advanced_losses": args.use_advanced_losses,
                }
                
                # Add advanced loss components to logging
                if args.use_advanced_losses:
                    log_data.update({
                        "q_loss": q_loss.item(),
                        "r_loss": r_loss.item(),
                        "curvature_loss": curvature_loss.item(),
                        "d_q_loss": d_q_loss.item(),
                        "q_loss_weight": args.q_loss_weight,
                        "r_loss_weight": args.r_loss_weight,
                        "curvature_loss_weight": args.curvature_loss_weight,
                    })
                
                wandb.log(log_data)
                
                # Enhanced logging for advanced losses
                if args.use_advanced_losses:
                    print(
                        f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                        f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                        f"[Q loss: {q_loss.item():.4f}] [R loss: {r_loss.item():.4f}] "
                        f"[Curvature loss: {curvature_loss.item():.4f}] "
                        f"[Discriminator: {args.discriminator_type}]"
                    )
                else:
                    print(
                        f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                        f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                        f"[Discriminator: {args.discriminator_type}]"
                    )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Extract 25 designs
                    desired_conds, designs = sample_designs(25)
                    fig, axes = plt.subplots(5, 5, figsize=(20, 20))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(designs):
                        if isinstance(problem.design_space, spaces.Dict):
                            design = spaces.unflatten(problem.design_space, tensor.cpu().numpy())
                        else:
                            design = tensor.cpu().numpy()
                        dc = desired_conds[j].cpu()
                        # use problem's render method to get the image
                        fig, ax = problem.render(design)
                        ax.figure.canvas.draw()
                        img = np.array(fig.canvas.renderer.buffer_rgba())
                        axes[j].imshow(img)
                        title = [(conditions[i][0], f"{dc[i]:.2f}") for i in range(n_conds)]
                        title_string = "\n ".join(f"{condition}: {value}" for condition, value in title)
                        axes[j].title.set_text(title_string)  # Set title
                        axes[j].set_xticks([])  # Hide x ticks
                        axes[j].set_yticks([])  # Hide y ticks
                        plt.close(fig)  # Close the original figure to free memory

                    plt.tight_layout()
                    img_fname = f"images/{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"designs": wandb.Image(img_fname)})

                # --------------
                #  Save models
                # --------------
                if args.save_model and epoch == args.n_epochs - 1 and i == len(dataloader) - 1:
                    ckpt_gen = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "generator": generator.state_dict(),
                        "optimizer_generator": optimizer_generator.state_dict(),
                        "loss": g_loss.item(),
                        # Save configuration for evaluation compatibility
                        "use_noise_z": args.use_noise_z,
                        "use_mlp_features": args.use_mlp_features,
                        "use_scalar_decoder": args.use_scalar_decoder,
                        "use_cpw_generator": args.use_cpw_generator,
                        "use_cpw_interpolation": args.use_cpw_interpolation,
                        "cpw_interpolation_type": args.cpw_interpolation_type,
                        "use_separate_normalization": args.use_separate_normalization,
                        "use_coord_decoder": args.use_coord_decoder,  # Step 7
                        "use_bezier_layer": args.use_bezier_layer,  # Step 9
                        "use_advanced_losses": args.use_advanced_losses,  # Step 10
                        "q_loss_weight": args.q_loss_weight,  # Step 10
                        "r_loss_weight": args.r_loss_weight,  # Step 10
                        "curvature_loss_weight": args.curvature_loss_weight,  # Step 10
                        "conditions_normalization_type": args.conditions_normalization_type,
                        "scalars_normalization_type": args.scalars_normalization_type,
                        "n_control_points": args.n_control_points,
                        "n_data_points": args.n_data_points,
                        "normalization_type": args.normalization_type,
                        "normalization_strategy": args.normalization_strategy,
                        "latent_dim": args.latent_dim,
                        "noise_dim": args.noise_dim,
                    }
                    ckpt_disc = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "discriminator": discriminator.state_dict(),
                        "optimizer_discriminator": optimizer_discriminator.state_dict(),
                        "loss": d_loss.item(),
                        "discriminator_type": args.discriminator_type,  # Step 8
                        "use_advanced_losses": args.use_advanced_losses,  # Step 10
                        "q_loss_weight": args.q_loss_weight,  # Step 10
                    }

                    th.save(ckpt_gen, "generator.pth")
                    th.save(ckpt_disc, "discriminator.pth")
                    artifact_gen = wandb.Artifact(f"{args.problem_id}_{args.algo}_generator", type="model")
                    artifact_gen.add_file("generator.pth")
                    artifact_disc = wandb.Artifact(f"{args.problem_id}_{args.algo}_discriminator", type="model")
                    artifact_disc.add_file("discriminator.pth")

                    wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                    wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

    wandb.finish()