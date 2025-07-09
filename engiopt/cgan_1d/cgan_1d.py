"""This code is largely based on the excellent PyTorch GAN repo: https://github.com/eriklindernoren/PyTorch-GAN.

We essentially refreshed the Python style, use wandb for logging, and made a few little improvements.
There are also a couple of code parts that are problem dependent and need to be adjusted for the specific problem.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time
from typing import TYPE_CHECKING

from engibench.utils.all_problems import BUILTIN_PROBLEMS
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
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


class CPWGenerator(nn.Module):
    """Control Points and Weights Generator.
    
    Generates control points and weights for parametric curves,
    then interpolates to create the exact number of output features needed.
    """
    
    def __init__(self, input_dim: int, n_output_features: int, n_points: int = 10):
        super().__init__()
        self.n_points = n_points
        self.n_output_features = n_output_features
        
        # Generate control points (we use 2D points for variety)
        self.control_points_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_points * 2),  # n_points with x,y coordinates
            nn.Tanh()  # Control points in [-1, 1]
        )
        
        # Generate weights
        self.weights_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_points),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            output_features: [batch_size, n_output_features]
        """
        # Generate control points
        cp_flat = self.control_points_net(x)
        control_points = cp_flat.view(-1, self.n_points, 2)  # [batch, n_points, 2]
        
        # Generate weights
        weights = self.weights_net(x)  # [batch, n_points]
        
        # Interpolate to generate exact number of output features
        output_features = interpolate_from_control_points(
            control_points, weights, self.n_output_features
        )
        
        return output_features


def interpolate_from_control_points(control_points: th.Tensor, weights: th.Tensor, n_output_features: int) -> th.Tensor:
    """Interpolate features from control points and weights.
    
    Args:
        control_points: [batch_size, n_points, n_dims]
        weights: [batch_size, n_points]
        n_output_features: Number of features to generate
        
    Returns:
        interpolated_features: [batch_size, n_output_features]
    """
    batch_size = control_points.shape[0]
    n_points = control_points.shape[1]
    n_dims = control_points.shape[2]
    
    # Create parameter values t in [0, 1] for each output feature
    t_values = th.linspace(0, 1, n_output_features, device=control_points.device)
    
    # Simple weighted interpolation to generate features
    interpolated = []
    
    for i, t_val in enumerate(t_values):
        # Calculate basis functions (simple distance-based weighting)
        basis = th.zeros_like(weights)
        for j in range(n_points):
            # Simple distance-based weighting
            center = j / (n_points - 1) if n_points > 1 else 0.5
            sigma = 1.0 / n_points
            basis[:, j] = th.exp(-((t_val - center) ** 2) / (2 * sigma ** 2))
        
        # Normalize basis functions
        basis = basis / (basis.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weight the basis functions
        weighted_basis = basis * weights
        weighted_basis = weighted_basis / (weighted_basis.sum(dim=1, keepdim=True) + 1e-8)
        
        # Interpolate feature value (use average of all dims from control points)
        feature_value = th.sum(weighted_basis.unsqueeze(-1) * control_points, dim=1)
        # Take mean across dimensions to get a single feature value
        feature_value = feature_value.mean(dim=-1, keepdim=True)
        interpolated.append(feature_value)
    
    # Concatenate all features
    interpolated = th.cat(interpolated, dim=-1)  # [batch_size, n_output_features]
    return interpolated


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
    use_noise_z: bool = True
    """If True, use noise z + latent c (v1.1), else use only latent z (baseline)"""
    
    # Step 2: Normalization improvements
    normalization_type: str = "MinMax"
    """Type of normalization: 'StandardScaler', 'MinMax', or 'No Norm'"""
    
    normalization_strategy: str = "common"
    """Normalization strategy: 'common' (single scaler) or 'separate' (per-feature scaler)"""
    
    # Step 3: CPW Generator for control points
    use_cpw_generator: bool = False
    """If True, add CPWGenerator for control points and weights"""
    
    # Step 4: Scalar decoder
    use_scalar_decoder: bool = False
    """If True, add scalar decoder/generator for design scalars"""
    
    # Step 5: Separate normalization
    use_separate_normalization: bool = False
    """If True, use separate normalizers for conditions vs scalars"""
    
    # Step 6: MLP features
    use_mlp_features: bool = False
    """If True, add MLP feature generator for abstract features"""
    
    # Step 7: Coordinate decoder (alternative to Bezier)
    use_coord_decoder: bool = False
    """If True, use coordinate decoder instead of simple MLP"""
    
    # Step 8: Advanced discriminator
    use_advanced_discriminator: bool = False
    """If True, use CNN+InfoGAN discriminator instead of simple MLP"""
    
    # Step 9: Bezier layer
    use_bezier_layer: bool = False
    """If True, add BezierLayer for geometric constraints"""
    
    # Step 10: Advanced loss functions
    use_advanced_losses: bool = False
    """If True, add Q-Loss, R-Loss, Curvature-Loss"""


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        noise_dim: int,
        n_conds: int,
        design_shape: tuple[int, ...],
        design_normalizer: MultiNormalizer,
        conds_normalizer: MultiNormalizer,
        use_noise_z: bool = True,
        use_cpw_generator: bool = False,
    ):
        super().__init__()
        self.design_shape = design_shape  # Store design shape
        self.design_normalizer = design_normalizer
        self.conds_normalizer = conds_normalizer
        self.use_noise_z = use_noise_z
        self.use_cpw_generator = use_cpw_generator

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

        if self.use_cpw_generator:
            # Step 3: Use CPWGenerator for control points and weights
            n_output_features = int(np.prod(design_shape))  # Exact number needed (e.g., 385)
            self.cpw_generator = CPWGenerator(
                input_dim=input_dim, 
                n_output_features=n_output_features,
                n_points=10
            )
        else:
            # Standard MLP generator
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
        
        if self.use_cpw_generator:
            # Step 3: Generate using control points and weights
            design_flat = self.cpw_generator(gen_input)  # Returns [batch, n_features] directly
            design = design_flat.view(design_flat.size(0), *self.design_shape)
        else:
            # Standard generation
            design = self.model(gen_input)
            design = design.view(design.size(0), *self.design_shape)
            
        return self.design_normalizer.denormalize(design)


class Discriminator(nn.Module):
    def __init__(self, conds_normalizer: MultiNormalizer, design_normalizer: MultiNormalizer, 
                 design_shape: tuple[int, ...], n_conds: int):
        super().__init__()
        self.conds_normalizer = conds_normalizer
        self.design_normalizer = design_normalizer

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(design_shape)) + n_conds, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, design: th.Tensor, conds: th.Tensor) -> th.Tensor:
        design_flat = design.view(design.size(0), -1)
        normalized_design = self.design_normalizer.normalize(design_flat)
        normalized_conds = self.conds_normalizer.normalize(conds)
        d_in = th.cat((normalized_design, normalized_conds), -1)
        return self.model(d_in)


def prepare_data(
    problem: Problem, 
    device: th.device, 
    normalization_type: str = "MinMax",
    normalization_strategy: str = "common"
) -> tuple[th.utils.data.TensorDataset, MultiNormalizer, MultiNormalizer]:
    """Prepares the data for the generator and discriminator.

    Args:
        problem (Problem): The problem to prepare the data for.
        device (th.device): The device to prepare the data on.
        normalization_type (str): Type of normalization ('MinMax', 'StandardScaler', 'No Norm').
        normalization_strategy (str): Strategy ('common', 'separate').

    Returns:
        tuple[th.utils.data.TensorDataset, MultiNormalizer, MultiNormalizer]: The training dataset, condition normalizer, and design normalizer.
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

    # Create condition normalizer using the multi-normalizer factory
    if normalization_strategy == "common":
        # Mode common: garde le comportement original
        cond_tensors = th.stack(training_ds.tensors[1:])
    else:
        # Mode separate: organise correctement les conditions par feature
        # Les conditions sont dans training_ds.tensors[1:], chacune a shape [n_samples]
        # On veut les organiser en [n_samples, n_conditions]
        cond_list = [tensor.unsqueeze(1) for tensor in training_ds.tensors[1:]]
        cond_tensors = th.cat(cond_list, dim=1)
    
    conds_normalizer = create_multi_normalizer(
        normalization_type, cond_tensors, device, normalization_strategy
    )

    # Create design normalizer using the multi-normalizer factory
    if normalization_strategy == "common":
        # Mode common: garde le comportement original
        design_tensors = training_ds.tensors[0].T
    else:
        # Mode separate: utilise les designs avec la forme [n_samples, n_features]
        design_tensors = training_ds.tensors[0]
    
    design_normalizer = create_multi_normalizer(
        normalization_type, design_tensors, device, normalization_strategy
    )

    return training_ds, conds_normalizer, design_normalizer


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

    training_ds, conds_normalizer, design_normalizer = prepare_data(
        problem, device, args.normalization_type, args.normalization_strategy
    )

    print(f"Using normalization: {args.normalization_type} with strategy: {args.normalization_strategy}")
    print(f"Conditions normalizers: {len(conds_normalizer.normalizers)}")
    print(f"Design normalizers: {len(design_normalizer.normalizers)}")
    print(f"Using CPWGenerator: {args.use_cpw_generator}")
    print(f"Using noise z: {args.use_noise_z}")
    print(f"Design shape: {design_shape} (total features: {int(np.prod(design_shape))})")

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
        use_noise_z=args.use_noise_z,
        use_cpw_generator=args.use_cpw_generator,
    )
    discriminator = Discriminator(
        conds_normalizer=conds_normalizer, 
        design_normalizer=design_normalizer,
        design_shape=design_shape,
        n_conds=n_conds
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

            # -----------------
            #  Train Generator
            # min log(1 - D(G(z))) <==> max log(D(G(z)))
            # -----------------
            optimizer_generator.zero_grad()

            if args.use_noise_z:
                # Version 1.1: Sample latent code and noise as generator input
                c = (bounds[1] - bounds[0]) * th.rand((designs.size(0), args.latent_dim), device=device, dtype=th.float) + bounds[0]
                z = 0.5 * th.randn((designs.size(0), args.noise_dim), device=device, dtype=th.float)
                gen_designs = generator(c, z, conds)
            else:
                # Baseline version: Only latent code
                c = th.randn((designs.size(0), args.latent_dim), device=device, dtype=th.float)
                gen_designs = generator(c, None, conds)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_designs, conds), valid)

            g_loss.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # max log(D(real)) + log(1 - D(G(z)))
            # ---------------------
            optimizer_discriminator.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(designs, conds), valid)
            fake_loss = adversarial_loss(discriminator(gen_designs.detach(), conds), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_discriminator.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                wandb.log(
                    {
                        "d_loss": d_loss.item(),
                        "g_loss": g_loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
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
                        "use_cpw_generator": args.use_cpw_generator,
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
