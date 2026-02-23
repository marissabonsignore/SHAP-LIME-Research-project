import numpy as np
import pandas as pd


def load_spx_data(csv_path="filtered_spx_options_with_features.csv"):
    
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Define the three features
    FEATURES = ["Log_Moneyness", "Time_to_Maturity", "IV"]
    
    # Extract the feature columns
    feature_data = df[FEATURES].copy()
    
    # Calculate statistics for perturbations
    feature_stds = feature_data.std().values
    feature_mins = feature_data.min().values.copy()
    feature_maxs = feature_data.max().values.copy()
    
    # Handle features with zero variance (constant values)
    # Add small epsilon to max values where min == max to avoid clipping issues
    epsilon = 1e-6
    zero_variance_mask = (feature_maxs - feature_mins) < epsilon
    if np.any(zero_variance_mask):
        print(f"Warning: Found {np.sum(zero_variance_mask)} constant feature(s). Adding small epsilon for clipping bounds.")
        feature_maxs[zero_variance_mask] = feature_mins[zero_variance_mask] + epsilon
    
    print(f"Loaded {len(feature_data)} samples with features: {FEATURES}")
    print(f"Feature statistics:")
    for i, feature in enumerate(FEATURES):
        print(f"  {feature}: mean={feature_data[feature].mean():.4f}, "
              f"std={feature_stds[i]:.4f}, "
              f"range=[{feature_mins[i]:.4f}, {feature_maxs[i]:.4f}]")
    
    return feature_data, feature_stds, (feature_mins, feature_maxs)


def generate_gaussian_perturbations(x_instance, n_samples, noise_scale=0.1, feature_stds=None, clip_bounds=None):
    """
    Generate Gaussian noise perturbations for tabular input data to be used with LIME or SHAP local explanations.
    
    This function creates perturbed versions of a single input instance by adding Gaussian noise
    scaled by feature-specific standard deviations. The perturbations are useful for generating
    local neighborhoods around an instance for explainability methods.
    
    """
    # Convert input to numpy array and ensure it's 1D
    x_instance = np.asarray(x_instance).flatten()
    n_features = len(x_instance)
    
    # Validate inputs
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")
    
    # Set default feature standard deviations if not provided
    if feature_stds is None:
        feature_stds = np.ones(n_features)
    else:
        feature_stds = np.asarray(feature_stds).flatten()
        if len(feature_stds) != n_features:
            raise ValueError(f"feature_stds length ({len(feature_stds)}) must match "
                           f"number of features ({n_features})")
    
    # Validate clip_bounds if provided
    if clip_bounds is not None:
        if len(clip_bounds) != 2:
            raise ValueError("clip_bounds must be a tuple of (min_vals, max_vals)")
        
        min_vals, max_vals = clip_bounds
        min_vals = np.asarray(min_vals).flatten()
        max_vals = np.asarray(max_vals).flatten()
        
        if len(min_vals) != n_features or len(max_vals) != n_features:
            raise ValueError("clip_bounds arrays must have the same length as number of features")
        
        if np.any(min_vals >= max_vals):
            raise ValueError("All min_vals must be less than corresponding max_vals")
    
    # Calculate noise standard deviations for each feature
    # noise_std = noise_scale * feature_std for each feature
    noise_stds = noise_scale * feature_stds
    
    # Generate Gaussian noise with mean 0 and feature-specific standard deviations
    # Shape: (n_samples, n_features)
    # Each column corresponds to noise for one feature
    noise = np.random.normal(
        loc=0.0,  # mean = 0
        scale=noise_stds.reshape(1, -1),  # broadcast feature-specific stds
        size=(n_samples, n_features)
    )
    
    # Add noise to the original instance (broadcast x_instance across all samples)
    # x_instance shape: (n_features,) -> broadcast to (n_samples, n_features)
    perturbed_samples = x_instance.reshape(1, -1) + noise
    
    # Apply clipping if bounds are provided
    if clip_bounds is not None:
        min_vals, max_vals = clip_bounds
        # Clip each feature to its respective bounds
        perturbed_samples = np.clip(
            perturbed_samples,
            a_min=min_vals.reshape(1, -1),  # broadcast min bounds
            a_max=max_vals.reshape(1, -1)   # broadcast max bounds
        )
    
    return perturbed_samples


def generate_perturbations_for_instance(instance_index, csv_path="filtered_spx_options_with_features.csv", 
                                       n_samples=100, noise_scale=0.1, use_clipping=True):
    
    # Load the data
    feature_data, feature_stds, feature_bounds = load_spx_data(csv_path)
    
    # Get the specific instance
    if instance_index >= len(feature_data):
        raise ValueError(f"Instance index {instance_index} is out of range. "
                        f"Dataset has {len(feature_data)} samples.")
    
    original_instance = feature_data.iloc[instance_index].values
    
    # Generate perturbations
    clip_bounds = feature_bounds if use_clipping else None
    perturbations = generate_gaussian_perturbations(
        x_instance=original_instance,
        n_samples=n_samples,
        noise_scale=noise_scale,
        feature_stds=feature_stds,
        clip_bounds=clip_bounds
    )
    
    return original_instance, perturbations, feature_data.columns.tolist()


def demo_spx_perturbations():
    """
    Professional demonstration of Gaussian perturbations for SPX options data.
    """
    print("=" * 80)
    print("SPX OPTIONS GAUSSIAN PERTURBATIONS DEMO")
    print("=" * 80)
    
    try:
        # Load dataset once
        print("\n📊 DATASET OVERVIEW:")
        feature_data, feature_stds, feature_bounds = load_spx_data("filtered_spx_options_with_features.csv")
        
        # Demo with multiple instances
        print(f"\n🎯 PERTURBATION EXAMPLES:")
        for instance_idx in [0, 100, 1000]:
            print(f"\n--- Instance {instance_idx} ---")
            
            # Get original instance
            original_instance = feature_data.iloc[instance_idx].values
            
            # Generate perturbations directly
            perturbations = generate_gaussian_perturbations(
                x_instance=original_instance,
                n_samples=3,
                noise_scale=0.1,
                feature_stds=feature_stds,
                clip_bounds=feature_bounds
            )
            
            print(f"Original: Log_Moneyness={original_instance[0]:.4f}, Time_to_Maturity={original_instance[1]:.4f}, IV={original_instance[2]:.4f}")
            for i, perturbed in enumerate(perturbations):
                print(f"Perturb {i+1}: Log_Moneyness={perturbed[0]:.4f}, Time_to_Maturity={perturbed[1]:.4f}, IV={perturbed[2]:.4f}")
        
        print(f"\n✅ SUCCESS: Generated perturbations for LIME/SHAP local explanations")
        print(f"📈 Ready for model explainability analysis!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    demo_spx_perturbations()