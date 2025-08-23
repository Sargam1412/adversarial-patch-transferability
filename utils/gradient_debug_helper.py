import torch
import numpy as np
import matplotlib.pyplot as plt

def debug_gradient_flow(loss, feature_maps_adv, feature_maps_rand, adv_patch, logger, iteration):
    """
    Debug gradient flow to identify where the problem is in your entropy trainer
    
    Args:
        loss: The computed loss tensor
        feature_maps_adv: Feature maps from adversarial image
        feature_maps_rand: Feature maps from random image  
        adv_patch: The adversarial patch tensor
        logger: Logger object for output
        iteration: Current iteration number
    """
    
    logger.info(f"=== GRADIENT DEBUG ITERATION {iteration} ===")
    
    # 1. Check feature maps
    if feature_maps_adv is not None:
        logger.info(f"Feature Maps Adv - Shape: {feature_maps_adv.shape}")
        logger.info(f"Feature Maps Adv - Norm: {torch.norm(feature_maps_adv).item():.6f}")
        logger.info(f"Feature Maps Adv - Mean: {feature_maps_adv.mean().item():.6f}")
        logger.info(f"Feature Maps Adv - Std: {feature_maps_adv.std().item():.6f}")
        logger.info(f"Feature Maps Adv - Min: {feature_maps_adv.min().item():.6f}")
        logger.info(f"Feature Maps Adv - Max: {feature_maps_adv.max().item():.6f}")
    else:
        logger.warning("Feature maps are None!")
    
    # 2. Check loss
    logger.info(f"Loss - Value: {loss.item():.6f}")
    logger.info(f"Loss - Shape: {loss.shape}")
    logger.info(f"Loss - Requires Grad: {loss.requires_grad}")
    
    # 3. Check patch
    logger.info(f"Patch - Shape: {adv_patch.shape}")
    logger.info(f"Patch - Norm: {torch.norm(adv_patch).item():.6f}")
    logger.info(f"Patch - Requires Grad: {adv_patch.requires_grad}")
    logger.info(f"Patch - Grad: {adv_patch.grad is not None}")
    
    # 4. Check gradients before backward pass
    if adv_patch.grad is not None:
        logger.info(f"Patch Grad (before) - Norm: {torch.norm(adv_patch.grad).item():.6f}")
        logger.info(f"Patch Grad (before) - Mean: {adv_patch.grad.mean().item():.6f}")
        logger.info(f"Patch Grad (before) - Std: {adv_patch.grad.std().item():.6f}")
    else:
        logger.info("Patch Grad (before) - None")
    
    logger.info("==========================================")
    
    return {
        'feature_maps_valid': feature_maps_adv is not None,
        'loss_valid': loss is not None and loss.requires_grad,
        'patch_requires_grad': adv_patch.requires_grad,
        'has_gradients': adv_patch.grad is not None
    }

def check_gradients_after_backward(loss, feature_maps_adv, adv_patch, logger, iteration):
    """
    Check gradients after backward pass to see if they're flowing correctly
    """
    logger.info(f"=== AFTER BACKWARD PASS - ITERATION {iteration} ===")
    
    # 1. Check if feature maps have gradients
    if feature_maps_adv is not None and feature_maps_adv.grad is not None:
        logger.info(f"Feature Maps Grad - Norm: {torch.norm(feature_maps_adv.grad).item():.6f}")
        logger.info(f"Feature Maps Grad - Mean: {feature_maps_adv.grad.mean().item():.6f}")
        logger.info(f"Feature Maps Grad - Std: {feature_maps_adv.grad.std().item():.6f}")
    else:
        logger.warning("Feature maps have no gradients!")
    
    # 2. Check patch gradients
    if adv_patch.grad is not None:
        logger.info(f"Patch Grad (after) - Norm: {torch.norm(adv_patch.grad).item():.6f}")
        logger.info(f"Patch Grad (after) - Mean: {adv_patch.grad.mean().item():.6f}")
        logger.info(f"Patch Grad (after) - Std: {adv_patch.grad.std().item():.6f}")
        
        # Check if gradients are meaningful
        grad_norm = torch.norm(adv_patch.grad).item()
        if grad_norm > 1e-8:
            logger.info("✓ Patch gradients are meaningful (> 1e-8)")
        else:
            logger.warning("✗ Patch gradients are too small (< 1e-8) - this explains no updates!")
    else:
        logger.warning("Patch has no gradients after backward pass!")
    
    # 3. Check loss gradients
    if loss.grad is not None:
        logger.info(f"Loss Grad - Norm: {torch.norm(loss.grad).item():.6f}")
    else:
        logger.info("Loss has no gradients")
    
    logger.info("================================================")

def analyze_entropy_loss_issue(feature_maps_adv, feature_maps_rand, logger):
    """
    Analyze why entropy loss might not be working with feature maps
    """
    logger.info("=== ENTROPY LOSS ANALYSIS ===")
    
    if feature_maps_adv is None or feature_maps_rand is None:
        logger.error("Feature maps are None - hooks not working!")
        return False
    
    # Check if feature maps have the right shape for entropy loss
    logger.info(f"Feature Maps Adv Shape: {feature_maps_adv.shape}")
    logger.info(f"Feature Maps Rand Shape: {feature_maps_rand.shape}")
    
    # Entropy loss expects [B, C, H, W] or [C, H, W] format
    if feature_maps_adv.dim() == 3:  # [C, H, W]
        logger.info("Feature maps are 3D - this should work with entropy loss")
    elif feature_maps_adv.dim() == 4:  # [B, C, H, W]
        logger.info("Feature maps are 4D - this should work with entropy loss")
    else:
        logger.error(f"Feature maps have unexpected dimensions: {feature_maps_adv.dim()}")
        return False
    
    # Check if feature maps have reasonable values
    adv_norm = torch.norm(feature_maps_adv).item()
    rand_norm = torch.norm(feature_maps_rand).item()
    
    logger.info(f"Feature Maps Adv Norm: {adv_norm:.6f}")
    logger.info(f"Feature Maps Rand Norm: {rand_norm:.6f}")
    
    if adv_norm < 1e-6 or rand_norm < 1e-6:
        logger.warning("Feature maps have very small norms - this might cause numerical issues!")
    
    # Check if feature maps are all zeros or constants
    if torch.allclose(feature_maps_adv, torch.zeros_like(feature_maps_adv)):
        logger.error("Feature maps are all zeros - this will cause entropy loss to fail!")
        return False
    
    if torch.allclose(feature_maps_adv, feature_maps_adv.mean()):
        logger.warning("Feature maps are constant - this might cause gradient issues!")
    
    logger.info("=================================")
    return True

def create_gradient_debug_plots(gradient_history, save_path=None):
    """
    Create plots to visualize gradient flow over time
    
    Args:
        gradient_history: List of dictionaries with gradient info
        save_path: Path to save the plot
    """
    if not gradient_history:
        print("No gradient history to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gradient Flow Debug Analysis', fontsize=16)
    
    iterations = range(len(gradient_history))
    
    # Extract data
    patch_grads = [info.get('patch_grad_norm', 0) for info in gradient_history]
    feature_grads = [info.get('feature_grad_norm', 0) for info in gradient_history]
    loss_values = [info.get('loss_value', 0) for info in gradient_history]
    patch_updates = [info.get('patch_update', 0) for info in gradient_history]
    
    # Plot 1: Patch gradients over time
    axes[0, 0].plot(iterations, patch_grads, 'b-', label='Patch Gradient Norm')
    axes[0, 0].set_title('Patch Gradients Over Time')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Gradient Norm')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Feature map gradients over time
    axes[0, 1].plot(iterations, feature_grads, 'r-', label='Feature Map Gradient Norm')
    axes[0, 1].set_title('Feature Map Gradients Over Time')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Loss values over time
    axes[1, 0].plot(iterations, loss_values, 'g-', label='Loss Value')
    axes[1, 0].set_title('Loss Values Over Time')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Patch updates over time
    axes[1, 1].plot(iterations, patch_updates, 'm-', label='Patch Update Magnitude')
    axes[1, 1].set_title('Patch Updates Over Time')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Update Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gradient debug plots saved to {save_path}")
    
    plt.show()

def suggest_fixes(logger, gradient_analysis):
    """
    Suggest fixes based on gradient analysis
    """
    logger.info("=== SUGGESTED FIXES ===")
    
    if not gradient_analysis['feature_maps_valid']:
        logger.info("1. FIX HOOKS: Feature maps are not being captured")
        logger.info("   - Check if layer_name is correct")
        logger.info("   - Verify hooks are registered")
        logger.info("   - Ensure forward pass reaches the hooked layer")
    
    if not gradient_analysis['loss_valid']:
        logger.info("2. FIX LOSS: Loss is not differentiable")
        logger.info("   - Ensure loss.requires_grad = True")
        logger.info("   - Check if loss computation preserves gradients")
    
    if not gradient_analysis['patch_requires_grad']:
        logger.info("3. FIX PATCH: Patch doesn't require gradients")
        logger.info("   - Set adv_patch.requires_grad_(True)")
        logger.info("   - Don't call .detach() on the patch")
    
    if not gradient_analysis['has_gradients']:
        logger.info("4. FIX GRADIENTS: No gradients flowing to patch")
        logger.info("   - Check if loss.backward() is called")
        logger.info("   - Verify computational graph is connected")
        logger.info("   - Consider using final logits instead of feature maps")
    
    logger.info("=================================")
