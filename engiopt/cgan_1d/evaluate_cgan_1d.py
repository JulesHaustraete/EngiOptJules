"""Evaluation for the CGAN 1D."""

from __future__ import annotations

import argparse
import dataclasses
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch as th

from engiopt import metrics
from engiopt.cgan_1d.cgan_1d import Generator, MultiNormalizer, prepare_data
from engiopt.dataset_sample_conditions import sample_conditions
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "airfoil"
    """Problem identifier."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "cgan_1d_{problem_id}_metrics.csv"
    """Chemin du fichier CSV de sortie ; peut inclure {problem_id}."""
    version: str = "baseline"
    """Label de version pour identifier l'expérience (informatif seulement)."""


def parse_args() -> Args:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CGAN 1D model")
    
    parser.add_argument("--problem-id", default="airfoil", help="Problem identifier")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--wandb-project", default="engiopt", help="Wandb project name")
    parser.add_argument("--wandb-entity", default=None, help="Wandb entity name")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of generated samples")
    parser.add_argument("--sigma", type=float, default=10.0, help="Kernel bandwidth for MMD and DPP")
    parser.add_argument("--output-csv", default="cgan_1d_{problem_id}_metrics.csv", help="Output CSV path")
    parser.add_argument("--version", default="baseline", help="Label de version pour identifier l'expérience (informatif seulement)")
    
    parsed_args = parser.parse_args()
    
    return Args(
        problem_id=parsed_args.problem_id,
        seed=parsed_args.seed,
        wandb_project=parsed_args.wandb_project,
        wandb_entity=parsed_args.wandb_entity,
        n_samples=parsed_args.n_samples,
        sigma=parsed_args.sigma,
        output_csv=parsed_args.output_csv,
        version=parsed_args.version,
    )


if __name__ == "__main__":
    args = parse_args()

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    # Seeding for reproducibility
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)
    th.backends.cudnn.deterministic = True

    # Select device
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    if isinstance(problem.design_space, spaces.Box):
        design_shape = problem.design_space.shape
    else:
        dummy_design, _ = problem.random_design()
        design_shape = spaces.flatten(problem.design_space, dummy_design).shape

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=seed,
    )

    ### Set Up Generator ###
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_cgan_1d_generator:latest"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_cgan_1d_generator:latest"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, "generator.pth")
    ckpt = th.load(ckpt_path, map_location=device)

    _, conds_normalizer, design_normalizer = prepare_data(
        problem, 
        device, 
        run.config.get("normalization_type", "MinMax"),
        run.config.get("normalization_strategy", "common")
    )

    print(f"📋 Configuration du modèle W&B:")
    print(f"   latent_dim: {run.config['latent_dim']}")
    print(f"   noise_dim: {run.config.get('noise_dim', 10)}")
    print(f"   use_noise_z: {run.config.get('use_noise_z', False)}")
    print(f"   use_cpw_generator: {run.config.get('use_cpw_generator', False)}")
    print(f"   normalization_type: {run.config.get('normalization_type', 'MinMax')}")
    print(f"   normalization_strategy: {run.config.get('normalization_strategy', 'common')}")

    model = Generator(
        latent_dim=run.config["latent_dim"],
        noise_dim=run.config.get("noise_dim", 10),  # Default to 10 if not in config
        n_conds=len(problem.conditions),
        design_shape=design_shape,
        design_normalizer=design_normalizer,
        conds_normalizer=conds_normalizer,
        use_noise_z=run.config.get("use_noise_z", False),  # Add this parameter
        use_cpw_generator=run.config.get("use_cpw_generator", False),  # Add this parameter
    ).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()

    # Sample latent code and noise based on model configuration from W&B
    use_noise_z = run.config.get("use_noise_z", False)  # Use whatever the model was trained with
    
    print(f"🎲 Mode de sampling basé sur la config W&B: use_noise_z={use_noise_z}")
    print(f"📋 Version label (informatif): {args.version}")
    
    try:
        if use_noise_z:
            # Version avec noise: Sample latent code c (uniform) and noise z (normal with std=0.5)
            bounds = (0.0, 1.0)
            c = (bounds[1] - bounds[0]) * th.rand((args.n_samples, run.config["latent_dim"]), device=device) + bounds[0]
            z = 0.5 * th.randn((args.n_samples, run.config.get("noise_dim", 10)), device=device)
            print(f"🎲 Sampling avec noise: c.shape={c.shape}, z.shape={z.shape}")
            gen_designs = model(c, z, conditions_tensor)
        else:
            # Version baseline: Only latent code
            c = th.randn((args.n_samples, run.config["latent_dim"]), device=device)
            print(f"🎲 Sampling baseline: c.shape={c.shape}")
            gen_designs = model(c, None, conditions_tensor)
    except Exception as e:
        print(f"❌ ERREUR lors de la génération")
        print(f"   Config W&B: use_noise_z={use_noise_z}")
        print(f"   Label version: {args.version}")
        print(f"   Erreur: {e}")
        print("\n🚫 ARRÊT: Impossible de générer des designs")
        raise SystemExit(f"Échec de génération")
    gen_designs_np = gen_designs.detach().cpu().numpy()

    print("\n=== SIMPLE SIMULATION TEST (FAILURE RATIO) ===")
    fail_ratio = metrics.simulate_failure_ratio(
        problem=problem,
        gen_designs=gen_designs_np,
        sampled_conditions=sampled_conditions,
    )
    print(f"Failure ratio: {fail_ratio}")
    
    print("\n=== ADDITIONAL METRICS ===")
    # Convert generated designs and sampled designs to numpy arrays
    gen_designs_np = np.asarray(gen_designs_np, dtype=np.float32)
    sampled_designs_np = np.array([d["coords"] for d in sampled_designs_np], dtype=np.float32)
    gen_designs_np = gen_designs_np[:, 1:]  # Remove the first value of each design (angle of attack)

    # Reshape because MMD expects gen_designs and sampled_designs to be 2D arrays and same dimensions 
    gen_designs_np = gen_designs_np.reshape(len(gen_designs_np), -1)
    sampled_designs_np = sampled_designs_np.reshape(len(sampled_designs_np), -1)

    print("After reshape:")
    print("gen_designs_np shape:", gen_designs_np.shape)
    print("sampled_designs_np shape:", sampled_designs_np.shape)

    # Compute curvature statistics for generated designs
    def compute_curvature_stats(coords):
        # coords: (N, 2) ou (2, N)
        if coords.shape[0] == 2:
            x, y = coords[0], coords[1]
        else:
            x, y = coords[:, 0], coords[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        return np.mean(np.abs(curvature)), np.std(curvature)

    # Calculate mean and standard deviation of curvature for each generated design
    mean_curvatures = []
    std_curvatures = []
    for design in gen_designs_np:
        # reshape si besoin pour avoir (N, 2)
        coords = design.reshape(-1, 2)
        mean_c, std_c = compute_curvature_stats(coords)
        mean_curvatures.append(mean_c)
        std_curvatures.append(std_c)

    print("Moyenne absolue de la courbure (générés):", np.mean(mean_curvatures))
    print("Ecart-type de la courbure (générés):", np.mean(std_curvatures))

    def compute_smoothness(xy: np.ndarray) -> np.ndarray:
        """
        Calcule la "lissitude" comme la norme L2 du gradient le long de la courbe.

        Args:
            xy: array de forme (B, N, 2) où B = batch, N = points, 2 = (x, y)

        Returns:
            np.ndarray de forme (B,) avec un score de lissitude par échantillon
        """
        diffs = xy[:, 1:, :] - xy[:, :-1, :]         # (B, N-1, 2)
        squared_norms = np.sum(diffs**2, axis=-1)    # (B, N-1)
        smoothness = np.sum(squared_norms, axis=1)   # (B,)
        return smoothness
    
    # Compute smoothness for generated designs
    gen_designs_np_reshaped = gen_designs_np.reshape(len(gen_designs_np), -1, 2)
    smoothness_scores = compute_smoothness(gen_designs_np_reshaped)
    print("Smoothness (moyenne sur tout les designs):", np.mean(smoothness_scores))

    def compute_cp_regularity(control_points: np.ndarray) -> np.ndarray:
        """
        Mesure la régularité des points de contrôle basée sur l'écart à la moyenne des distances entre points.
        Args:
            control_points: array (B, N, 2) où N est le nombre de points de contrôle
        Returns:
            np.ndarray (B,) avec le score de régularité (plus bas = plus régulier)
        """
        diffs = control_points[:, 1:, :] - control_points[:, :-1, :]  # (B, N-1, 2)
        dists = np.linalg.norm(diffs, axis=-1)  # (B, N-1)
        mean_dists = np.mean(dists, axis=1, keepdims=True)  # (B, 1)
        regularity = np.mean((dists - mean_dists) ** 2, axis=1)  # variance
        return regularity

    # Compute CP regularity for generated designs
    cp_regularity_scores = compute_cp_regularity(gen_designs_np_reshaped)
    print("CP Regularity (moyenne sur tous les designs):", np.mean(cp_regularity_scores))
    

    # Simple MMD and DPP calculations without complex processing
    print("\n=== BASIC SHAPE METRICS ===")
    
    # Calculate MMD between generated designs and sampled designs
    mmd_value = metrics.mmd(gen_designs_np, sampled_designs_np, sigma=args.sigma)
    print(f"MMD: {mmd_value}")

    # Calculation of DPP diversity using the generated designs
    dpp_value = metrics.dpp_diversity(gen_designs_np, sigma=args.sigma)
    print(f"DPP diversity: {dpp_value}")

    # Create simple results
    results_dict = {
        "problem_id": args.problem_id,
        "version": args.version,
        "model_id": "cgan_1d",
        "seed": seed,
        "n_samples": args.n_samples,
        "fail_ratio": fail_ratio,
        "mmd": mmd_value,
        "dpp": dpp_value,
        "mean_curvature": np.mean(mean_curvatures),
        "std_curvature": np.mean(std_curvatures),
        "smoothness": np.mean(smoothness_scores),
        "cp_regularity": np.mean(cp_regularity_scores), 
    }
    
    metrics_df = pd.DataFrame(results_dict, index=[0])
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"Seed {seed} done; appended to {out_path}")
