import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class BranchAnalyzer:
    """
    Utility class to analyze PIDNet P and I branch gradients and determine which branch is most crucial
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
        Analyze the importance of P and I branches based on gradient magnitudes and losses.
        
        Returns:
            dict: Analysis results for P and I branches only
        """
        if self.data is None:
            return None
            
        # Unpack data: (patch, IoU, branch_losses, branch_gradients)
        patch, iou, branch_losses, branch_gradients = self.data
        
        analysis = {}
        
        # Focus only on P and I branches
        for branch_name in ['P', 'I']:
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
                
                # Calculate gradient consistency (lower std/mean ratio = more consistent)
                grad_consistency = 1.0 / (1.0 + (std_grad_magnitude / avg_grad_magnitude)) if avg_grad_magnitude > 0 else 0
                
                analysis[branch_name] = {
                    'avg_loss': avg_loss,
                    'avg_grad_magnitude': avg_grad_magnitude,
                    'std_grad_magnitude': std_grad_magnitude,
                    'grad_variance': grad_variance,
                    'grad_consistency': grad_consistency,
                    'grad_magnitudes': grad_magnitudes,
                    'losses': branch_losses[branch_name]
                }
        
        return analysis
    
    def plot_branch_comparison(self, save_path=None):
        """
        Create comprehensive plots comparing P and I branches only.
        
        Args:
            save_path (str, optional): Path to save the plots
        """
        analysis = self.analyze_branch_importance()
        if analysis is None:
            print("No analysis data available")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PIDNet P vs I Branch Analysis for Adversarial Patch Attack', fontsize=16)
        
        branches = list(analysis.keys())
        if len(branches) < 2:
            print("Need data for both P and I branches")
            return
            
        # Plot 1: Average gradient magnitudes comparison
        avg_grads = [analysis[branch]['avg_grad_magnitude'] for branch in branches]
        std_grads = [analysis[branch]['std_grad_magnitude'] for branch in branches]
        
        bars = axes[0, 0].bar(branches, avg_grads, yerr=std_grads, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Average Gradient Magnitudes (P vs I)')
        axes[0, 0].set_ylabel('Gradient Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Color bars based on importance
        colors = ['red' if g == max(avg_grads) else 'blue' for g in avg_grads]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Plot 2: Loss progression over iterations
        for branch in branches:
            if len(analysis[branch]['losses']) > 0:
                axes[0, 1].plot(analysis[branch]['losses'], label=f'{branch} Branch', alpha=0.8, linewidth=2)
        
        axes[0, 1].set_title('Loss Progression Over Iterations')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cross-Entropy Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient magnitude progression
        for branch in branches:
            if len(analysis[branch]['grad_magnitudes']) > 0:
                axes[0, 2].plot(analysis[branch]['grad_magnitudes'], label=f'{branch} Branch', alpha=0.8, linewidth=2)
        
        axes[0, 2].set_title('Gradient Magnitude Progression')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Gradient Magnitude')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Gradient stability (variance)
        grad_variances = [analysis[branch]['grad_variance'] for branch in branches]
        bars = axes[1, 0].bar(branches, grad_variances, alpha=0.7)
        axes[1, 0].set_title('Gradient Stability (Lower = More Stable)')
        axes[1, 0].set_ylabel('Gradient Variance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Color bars based on stability
        colors = ['green' if v == min(grad_variances) else 'orange' for v in grad_variances]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Plot 5: Gradient consistency
        grad_consistencies = [analysis[branch]['grad_consistency'] for branch in branches]
        bars = axes[1, 1].bar(branches, grad_consistencies, alpha=0.7)
        axes[1, 1].set_title('Gradient Consistency (Higher = More Consistent)')
        axes[1, 1].set_ylabel('Consistency Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Color bars based on consistency
        colors = ['green' if c == max(grad_consistencies) else 'orange' for c in grad_consistencies]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Plot 6: Side-by-side comparison of key metrics
        metrics = ['avg_grad_magnitude', 'grad_variance', 'grad_consistency']
        metric_labels = ['Grad Magnitude', 'Grad Variance', 'Grad Consistency']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        p_values = [analysis['P'][metric] for metric in metrics]
        i_values = [analysis['I'][metric] for metric in metrics]
        
        # Normalize values for better comparison
        p_norm = [v/max(p_values) if max(p_values) > 0 else 0 for v in p_values]
        i_norm = [v/max(i_values) if max(i_values) > 0 else 0 for v in i_values]
        
        axes[1, 2].bar(x - width/2, p_norm, width, label='P Branch', alpha=0.7, color='red')
        axes[1, 2].bar(x + width/2, i_norm, width, label='I Branch', alpha=0.7, color='blue')
        axes[1, 2].set_title('Normalized Metrics Comparison')
        axes[1, 2].set_ylabel('Normalized Values')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metric_labels, rotation=45)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
    
    def get_crucial_branch_ranking(self):
        """
        Rank P and I branches by their importance for adversarial attacks.
        
        Returns:
            list: Ranked list of P and I branches from most to least crucial
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
            
            # Score based on gradient consistency (higher = more consistent = more important)
            consistency_score = branch_data['grad_consistency']
            
            # Combined score (weighted average)
            combined_score = 0.5 * grad_score + 0.3 * stability_score + 0.2 * consistency_score
            
            branch_scores[branch_name] = {
                'combined_score': combined_score,
                'grad_score': grad_score,
                'stability_score': stability_score,
                'consistency_score': consistency_score,
                'avg_loss': branch_data['avg_loss']
            }
        
        # Sort by combined score
        ranked_branches = sorted(branch_scores.items(), 
                               key=lambda x: x[1]['combined_score'], 
                               reverse=True)
        
        return ranked_branches
    
    def statistical_significance_test(self):
        """
        Perform statistical significance test between P and I branches.
        
        Returns:
            dict: Statistical test results
        """
        analysis = self.analyze_branch_importance()
        if analysis is None or len(analysis) < 2:
            return None
            
        from scipy import stats
        
        # Get gradient magnitudes for both branches
        p_grads = analysis['P']['grad_magnitudes']
        i_grads = analysis['I']['grad_magnitudes']
        
        if len(p_grads) < 2 or len(i_grads) < 2:
            return None
            
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(p_grads, i_grads)
        
        # Perform Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(p_grads, i_grads, alternative='two-sided')
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(p_grads) - 1) * np.var(p_grads) + (len(i_grads) - 1) * np.var(i_grads)) / (len(p_grads) + len(i_grads) - 2))
        cohens_d = (np.mean(p_grads) - np.mean(i_grads)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_test': {'statistic': t_stat, 'p_value': p_value},
            'mann_whitney': {'statistic': u_stat, 'p_value': u_p_value},
            'cohens_d': cohens_d,
            'p_branch_mean': np.mean(p_grads),
            'i_branch_mean': np.mean(i_grads),
            'p_branch_std': np.std(p_grads),
            'i_branch_std': np.std(i_grads)
        }
    
    def print_detailed_analysis(self):
        """Print a detailed analysis report focusing on P vs I branches."""
        analysis = self.analyze_branch_importance()
        if analysis is None:
            print("No analysis data available")
            return
            
        print("=" * 70)
        print("PIDNet P vs I Branch Analysis Report")
        print("=" * 70)
        
        # Print summary statistics
        for branch_name, branch_data in analysis.items():
            print(f"\n{branch_name} Branch Analysis:")
            print(f"  Average Loss: {branch_data['avg_loss']:.4f}")
            print(f"  Average Gradient Magnitude: {branch_data['avg_grad_magnitude']:.4f}")
            print(f"  Gradient Standard Deviation: {branch_data['std_grad_magnitude']:.4f}")
            print(f"  Gradient Variance: {branch_data['grad_variance']:.4f}")
            print(f"  Gradient Consistency: {branch_data['grad_consistency']:.4f}")
        
        # Print ranking
        print("\n" + "=" * 50)
        print("Branch Importance Ranking (P vs I)")
        print("=" * 50)
        
        ranked_branches = self.get_crucial_branch_ranking()
        for i, (branch_name, scores) in enumerate(ranked_branches, 1):
            print(f"{i}. {branch_name} Branch")
            print(f"   Combined Score: {scores['combined_score']:.4f}")
            print(f"   Gradient Score: {scores['grad_score']:.4f}")
            print(f"   Stability Score: {scores['stability_score']:.4f}")
            print(f"   Consistency Score: {scores['consistency_score']:.4f}")
            print(f"   Average Loss: {scores['avg_loss']:.4f}")
        
        # Print statistical significance test results
        print("\n" + "=" * 50)
        print("Statistical Significance Analysis")
        print("=" * 50)
        
        stats_results = self.statistical_significance_test()
        if stats_results:
            print(f"T-Test: t-statistic = {stats_results['t_test']['statistic']:.4f}, p-value = {stats_results['t_test']['p_value']:.4f}")
            print(f"Mann-Whitney U: U-statistic = {stats_results['mann_whitney']['statistic']:.4f}, p-value = {stats_results['mann_whitney']['p_value']:.4f}")
            print(f"Effect Size (Cohen's d): {stats_results['cohens_d']:.4f}")
            print(f"P Branch: Mean = {stats_results['p_branch_mean']:.4f}, Std = {stats_results['p_branch_std']:.4f}")
            print(f"I Branch: Mean = {stats_results['i_branch_mean']:.4f}, Std = {stats_results['i_branch_std']:.4f}")
            
            # Interpret significance
            alpha = 0.05
            if stats_results['t_test']['p_value'] < alpha:
                print(f"✓ Statistically significant difference between P and I branches (p < {alpha})")
            else:
                print(f"✗ No statistically significant difference between P and I branches (p >= {alpha})")
            
            # Interpret effect size
            if abs(stats_results['cohens_d']) >= 0.8:
                effect_size = "large"
            elif abs(stats_results['cohens_d']) >= 0.5:
                effect_size = "medium"
            else:
                effect_size = "small"
            print(f"Effect size is considered {effect_size} (|d| = {abs(stats_results['cohens_d']):.4f})")
        
        # Print recommendations
        print("\n" + "=" * 50)
        print("Recommendations")
        print("=" * 50)
        
        most_crucial = ranked_branches[0][0]
        print(f"• The {most_crucial} branch appears to be most crucial for adversarial attacks")
        print(f"• Focus on optimizing the patch to maximize impact on the {most_crucial} branch")
        print(f"• Consider using branch-specific loss functions to target the {most_crucial} branch")
        
        if len(ranked_branches) > 1:
            second_crucial = ranked_branches[1][0]
            print(f"• The {second_crucial} branch is the second most important")
            print(f"• Consider multi-branch targeting strategies for P and I branches")
        
        if stats_results and stats_results['t_test']['p_value'] < 0.05:
            print(f"• Statistical analysis confirms significant difference between P and I branches")
            print(f"• The {most_crucial} branch dominance is statistically validated")
    
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
    """Example usage of the BranchAnalyzer for P vs I analysis."""
    
    # Example usage - replace with your actual pickle file path
    pickle_file = "path/to/your/branch_analysis.pickle"
    
    if Path(pickle_file).exists():
        analyzer = BranchAnalyzer(pickle_file)
        
        # Print detailed analysis
        analyzer.print_detailed_analysis()
        
        # Create plots
        analyzer.plot_branch_comparison(save_path="p_vs_i_branch_analysis.png")
        
        # Export to CSV
        analyzer.export_analysis_to_csv("p_vs_i_branch_analysis_data.csv")
        
    else:
        print(f"Pickle file not found: {pickle_file}")
        print("Please run the modified trainer first to generate branch analysis data.")


if __name__ == "__main__":
    main()
