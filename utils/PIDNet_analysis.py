import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class BranchAnalyzer:
    """
    Utility class to analyze PIDNet branch gradients and determine which branch is most crucial
    for adversarial patch attacks.
    """
    
    def __init__(self, pickle_file_path):
        """
        Initialize with the path to the saved pickle file containing branch analysis data.
        
        Args:
            pickle_file_path (str): Path to the pickle file saved by the modified trainer
        """
        self.pickle_file_path = pickle_file_path
        self.data = self.load_data()
        
    def load_data(self):
        """Load the saved data from pickle file."""
        try:
            with open(self.pickle_file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def analyze_branch_importance(self):
        """
        Analyze the importance of each branch based on gradient magnitudes and losses.
        
        Returns:
            dict: Analysis results for each branch
        """
        if self.data is None:
            return None
            
        # Unpack data: (patch, IoU, branch_losses, branch_gradients)
        patch, iou, branch_losses, branch_gradients = self.data
        
        analysis = {}
        
        for branch_name in ['P', 'I', 'D']:
            if branch_name in branch_losses and len(branch_losses[branch_name]) > 0:
                # Calculate average loss
                avg_loss = np.mean(branch_losses[branch_name])
                
                # Calculate average gradient magnitude
                grad_magnitudes = []
                for grad_tensor in branch_gradients[branch_name]:
                    if isinstance(grad_tensor, torch.Tensor):
                        grad_magnitudes.append(torch.norm(grad_tensor).item())
                    else:
                        grad_magnitudes.append(np.linalg.norm(grad_tensor))
                
                avg_grad_magnitude = np.mean(grad_magnitudes)
                std_grad_magnitude = np.std(grad_magnitudes)
                
                # Calculate gradient variance (stability measure)
                grad_variance = np.var(grad_magnitudes)
                
                analysis[branch_name] = {
                    'avg_loss': avg_loss,
                    'avg_grad_magnitude': avg_grad_magnitude,
                    'std_grad_magnitude': std_grad_magnitude,
                    'grad_variance': grad_variance,
                    'grad_magnitudes': grad_magnitudes,
                    'losses': branch_losses[branch_name]
                }
        
        return analysis
    
    def plot_branch_comparison(self, save_path=None):
        """
        Create comprehensive plots comparing the three branches.
        
        Args:
            save_path (str, optional): Path to save the plots
        """
        analysis = self.analyze_branch_importance()
        if analysis is None:
            print("No analysis data available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PIDNet Branch Analysis for Adversarial Patch Attack', fontsize=16)
        
        # Plot 1: Average gradient magnitudes
        branches = list(analysis.keys())
        avg_grads = [analysis[branch]['avg_grad_magnitude'] for branch in branches]
        std_grads = [analysis[branch]['std_grad_magnitude'] for branch in branches]
        
        bars = axes[0, 0].bar(branches, avg_grads, yerr=std_grads, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Average Gradient Magnitudes')
        axes[0, 0].set_ylabel('Gradient Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Color bars based on importance
        colors = ['red' if g == max(avg_grads) else 'blue' for g in avg_grads]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Plot 2: Loss progression over iterations
        for branch in branches:
            if len(analysis[branch]['losses']) > 0:
                axes[0, 1].plot(analysis[branch]['losses'], label=f'{branch} Branch', alpha=0.8)
        
        axes[0, 1].set_title('Loss Progression Over Iterations')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cross-Entropy Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient magnitude progression
        for branch in branches:
            if len(analysis[branch]['grad_magnitudes']) > 0:
                axes[1, 0].plot(analysis[branch]['grad_magnitudes'], label=f'{branch} Branch', alpha=0.8)
        
        axes[1, 0].set_title('Gradient Magnitude Progression Over Iterations')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Gradient Magnitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Gradient stability (variance)
        grad_variances = [analysis[branch]['grad_variance'] for branch in branches]
        bars = axes[1, 1].bar(branches, grad_variances, alpha=0.7)
        axes[1, 1].set_title('Gradient Stability (Lower = More Stable)')
        axes[1, 1].set_ylabel('Gradient Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Color bars based on stability
        colors = ['green' if v == min(grad_variances) else 'orange' for v in grad_variances]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
    
    def get_crucial_branch_ranking(self):
        """
        Rank branches by their importance for adversarial attacks.
        
        Returns:
            list: Ranked list of branches from most to least crucial
        """
        analysis = self.analyze_branch_importance()
        if analysis is None:
            return []
            
        # Create ranking based on multiple criteria
        branch_scores = {}
        
        for branch_name, branch_data in analysis.items():
            # Score based on gradient magnitude (higher = more important)
            grad_score = branch_data['avg_grad_magnitude']
            
            # Score based on gradient stability (lower variance = more stable = more important)
            stability_score = 1.0 / (1.0 + branch_data['grad_variance'])
            
            # Combined score (weighted average)
            combined_score = 0.7 * grad_score + 0.3 * stability_score
            
            branch_scores[branch_name] = {
                'combined_score': combined_score,
                'grad_score': grad_score,
                'stability_score': stability_score,
                'avg_loss': branch_data['avg_loss']
            }
        
        # Sort by combined score
        ranked_branches = sorted(branch_scores.items(), 
                               key=lambda x: x[1]['combined_score'], 
                               reverse=True)
        
        return ranked_branches
    
    def print_detailed_analysis(self):
        """Print a detailed analysis report."""
        analysis = self.analyze_branch_importance()
        if analysis is None:
            print("No analysis data available")
            return
            
        print("=" * 60)
        print("PIDNet Branch Analysis Report")
        print("=" * 60)
        
        # Print summary statistics
        for branch_name, branch_data in analysis.items():
            print(f"\n{branch_name} Branch Analysis:")
            print(f"  Average Loss: {branch_data['avg_loss']:.4f}")
            print(f"  Average Gradient Magnitude: {branch_data['avg_grad_magnitude']:.4f}")
            print(f"  Gradient Standard Deviation: {branch_data['std_grad_magnitude']:.4f}")
            print(f"  Gradient Variance: {branch_data['grad_variance']:.4f}")
        
        # Print ranking
        print("\n" + "=" * 40)
        print("Branch Importance Ranking")
        print("=" * 40)
        
        ranked_branches = self.get_crucial_branch_ranking()
        for i, (branch_name, scores) in enumerate(ranked_branches, 1):
            print(f"{i}. {branch_name} Branch")
            print(f"   Combined Score: {scores['combined_score']:.4f}")
            print(f"   Gradient Score: {scores['grad_score']:.4f}")
            print(f"   Stability Score: {scores['stability_score']:.4f}")
            print(f"   Average Loss: {scores['avg_loss']:.4f}")
        
        # Print recommendations
        print("\n" + "=" * 40)
        print("Recommendations")
        print("=" * 40)
        
        most_crucial = ranked_branches[0][0]
        print(f"• The {most_crucial} branch appears to be most crucial for adversarial attacks")
        print(f"• Focus on optimizing the patch to maximize impact on the {most_crucial} branch")
        print(f"• Consider using branch-specific loss functions to target the {most_crucial} branch")
        
        if len(ranked_branches) > 1:
            second_crucial = ranked_branches[1][0]
            print(f"• The {second_crucial} branch is the second most important")
            print(f"• Consider multi-branch targeting strategies")
    
    def export_analysis_to_csv(self, save_path):
        """
        Export the analysis results to CSV format for further analysis.
        
        Args:
            save_path (str): Path to save the CSV file
        """
        import pandas as pd
        
        analysis = self.analyze_branch_importance()
        if analysis is None:
            print("No analysis data available")
            return
            
        # Prepare data for CSV
        data_rows = []
        for branch_name, branch_data in analysis.items():
            for i in range(len(branch_data['losses'])):
                data_rows.append({
                    'branch': branch_name,
                    'iteration': i,
                    'loss': branch_data['losses'][i],
                    'gradient_magnitude': branch_data['grad_magnitudes'][i]
                })
        
        df = pd.DataFrame(data_rows)
        df.to_csv(save_path, index=False)
        print(f"Analysis data exported to {save_path}")


def main():
    """Example usage of the BranchAnalyzer."""
    
    # Example usage - replace with your actual pickle file path
    pickle_file = "path/to/your/branch_analysis.pickle"
    
    if Path(pickle_file).exists():
        analyzer = BranchAnalyzer(pickle_file)
        
        # Print detailed analysis
        analyzer.print_detailed_analysis()
        
        # Create plots
        analyzer.plot_branch_comparison(save_path="branch_analysis_plots.png")
        
        # Export to CSV
        analyzer.export_analysis_to_csv("branch_analysis_data.csv")
        
    else:
        print(f"Pickle file not found: {pickle_file}")
        print("Please run the modified trainer first to generate branch analysis data.")


if __name__ == "__main__":
    main()
