"""
STEP 5 & 6: EVALUATION MODULE & COMPLETE PIPELINE
===================================================

This module:
1. Evaluates all models with multiple metrics
2. Compares performance
3. Generates visualizations
4. Creates inference system for top-10 recommendations

Author: Your Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

class Evaluator:
    """
    Comprehensive evaluation of all models
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.results = {}
    
    def calculate_rmse(self, actual, predicted):
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(actual, predicted))
    
    def calculate_mae(self, actual, predicted):
        """Mean Absolute Error"""
        return mean_absolute_error(actual, predicted)
    
    def evaluate_model(self, model_name, predictions_df, k=10):
        """
        Comprehensive model evaluation
        
        Args:
            model_name: Name of model
            predictions_df: DataFrame with predictions
            k: For top-k metrics
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*70}")
        
        # Rating metrics
        rmse = self.calculate_rmse(
            predictions_df['actual_rating'],
            predictions_df['predicted_rating']
        )
        mae = self.calculate_mae(
            predictions_df['actual_rating'],
            predictions_df['predicted_rating']
        )
        
        metrics = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
        }
        
        # Helpfulness prediction (if available)
        if 'predicted_helpfulness' in predictions_df.columns:
            help_rmse = self.calculate_rmse(
                predictions_df['actual_helpfulness'],
                predictions_df['predicted_helpfulness']
            )
            metrics['Helpfulness_RMSE'] = help_rmse
            
            print(f"  Rating RMSE: {rmse:.4f}")
            print(f"  Rating MAE: {mae:.4f}")
            print(f"  Helpfulness RMSE: {help_rmse:.4f}")
        else:
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
        
        self.results[model_name] = metrics
        return metrics
    
    def compare_models(self):
        """Create comparison table"""
        if not self.results:
            print("No models evaluated yet")
            return None
        
        comparison_df = pd.DataFrame(self.results).T
        
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}")
        print(comparison_df.to_string())
        print(f"{'='*70}")
        
        return comparison_df
    
    def plot_comparison(self, save_path='./results/model_comparison.png'):
        """Plot model comparison"""
        if not self.results:
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        comparison_df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        models = comparison_df.index.tolist()
        rmse_values = comparison_df['RMSE'].values
        mae_values = comparison_df['MAE'].values
        
        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(models)]
        
        # RMSE
        axes[0].bar(models, rmse_values, color=colors, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('RMSE', fontsize=13, fontweight='bold')
        axes[0].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticklabels(models, rotation=15, ha='right')
        
        for i, v in enumerate(rmse_values):
            axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=10)
        
        # MAE
        axes[1].bar(models, mae_values, color=colors, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('MAE', fontsize=13, fontweight='bold')
        axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_xticklabels(models, rotation=15, ha='right')
        
        for i, v in enumerate(mae_values):
            axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom',
                        fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to: {save_path}")
        plt.show()


class RecommendationInference:
    """
    Inference system for making recommendations
    
    Usage:
        rec = RecommendationInference(model, loader)
        top_10 = rec.get_top_recommendations(user_id, category, n=10)
    """
    
    def __init__(self, model, data_loader, device='cpu'):
        """
        Initialize inference system
        
        Args:
            model: Trained model
            data_loader: Data loader with mappings
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.loader = data_loader
        self.device = device
        self.model.eval()
    
    def get_top_recommendations(self, user_id, category, n=10):
        """
        Get top-N product recommendations for a user in a category
        
        Args:
            user_id: User ID (original ID, not index)
            category: Category name (e.g., "Books", "Music")
            n: Number of recommendations
        
        Returns:
            DataFrame with top-N products with predicted ratings and helpfulness
        """
        # Convert to indices
        if user_id not in self.loader.user_to_idx:
            print(f"✗ User ID {user_id} not found")
            return pd.DataFrame()
        
        if category not in self.loader.category_to_idx:
            print(f"✗ Category '{category}' not found")
            return pd.DataFrame()
        
        user_idx = self.loader.user_to_idx[user_id]
        category_idx = self.loader.category_to_idx[category]
        
        # Get all items in category
        category_items = self.loader.get_category_items(category_idx)
        
        if not category_items:
            print(f"No items found in category '{category}'")
            return pd.DataFrame()
        
        # Get recommendations
        import torch
        
        recommendations = []
        
        with torch.no_grad():
            for item_idx in category_items:
                # Get social neighbors
                social_neighbors = [self.loader.get_user_social_neighbors(user_idx)]
                
                # Get item neighbors (user's history)
                user_items = self.loader.get_user_items(user_idx)
                item_neighbors = [[item[0] for item in user_items]]
                
                # Prepare tensors
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                item_tensor = torch.LongTensor([item_idx]).to(self.device)
                category_tensor = torch.LongTensor([category_idx]).to(self.device)
                
                # Predict
                if hasattr(self.model, 'model'):
                    # If wrapped in trainer
                    pred_rating, pred_helpful = self.model.model(
                        user_tensor, item_tensor, category_tensor,
                        social_neighbors, item_neighbors
                    )
                else:
                    # Direct model
                    pred_rating, pred_helpful = self.model(
                        user_tensor, item_tensor, category_tensor,
                        social_neighbors, item_neighbors
                    )
                
                pred_rating = torch.clamp(pred_rating, 1.0, 5.0).item()
                pred_helpful = torch.clamp(pred_helpful, 0.0, 1.0).item()
                
                # Get original product ID
                product_id = self.loader.idx_to_item[item_idx]
                
                recommendations.append({
                    'product_id': product_id,
                    'product_idx': item_idx,
                    'predicted_rating': pred_rating,
                    'predicted_helpfulness': pred_helpful,
                    'rank': 0  # Will be set after sorting
                })
        
        # Sort by rating (descending), then by helpfulness
        recommendations = sorted(
            recommendations,
            key=lambda x: (-x['predicted_rating'], -x['predicted_helpfulness'])
        )
        
        # Add ranks
        for i, rec in enumerate(recommendations[:n], 1):
            rec['rank'] = i
        
        return pd.DataFrame(recommendations[:n])
    
    def print_recommendations(self, user_id, category, n=10):
        """Pretty print recommendations"""
        recs = self.get_top_recommendations(user_id, category, n)
        
        if recs.empty:
            return
        
        print(f"\n{'='*80}")
        print(f"TOP-{n} RECOMMENDATIONS FOR USER {user_id} - CATEGORY: {category}")
        print(f"{'='*80}")
        print(f"\n{'Rank':<6} {'Product ID':<15} {'Rating':<10} {'Helpfulness':<15}")
        print(f"{'-'*80}")
        
        for _, row in recs.iterrows():
            print(f"{int(row['rank']):<6} {int(row['product_id']):<15} "
                  f"{row['predicted_rating']:<10.2f} {row['predicted_helpfulness']:<15.2f}")
        
        print(f"\n✓ Recommendations complete!")


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

class ExperimentPipeline:
    """
    Complete experimental pipeline
    """
    
    def __init__(self, data_dir='./data', results_dir='./results', models_dir='./models'):
        """Initialize pipeline"""
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.models_dir = models_dir
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        self.loader = None
        self.train_df = None
        self.test_df = None
        
        self.baseline_trainer = None
        self.diffnet_trainer = None
        self.multihead_trainer = None
        
        self.evaluator = None
    
    def run_complete_pipeline(self):
        """Run all steps"""
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                CIAO DATASET - COMPLETE RECOMMENDATION PIPELINE             ║
║                                                                            ║
║                         Multi-Head DiffNet++ System                        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
        """)
        
        print("\n[STEP 1] LOADING DATA")
        from ciao_data_loader_v2 import CiaoDataLoaderV2
        
        self.loader = CiaoDataLoaderV2(data_dir=self.data_dir)
        self.loader.load_ratings()
        self.loader.load_trust_network()
        self.loader.create_mappings()
        self.loader.build_social_graph()
        self.loader.enrich_ratings_with_indices()
        self.loader.identify_cold_start_users()
        self.loader.print_statistics()
        self.loader.save_mappings(self.models_dir)
        
        # Load pre-split data
        self.train_df, self.test_df = self.loader.split_data(
            train_path='ciao_train.csv',
            test_path='ciao_test.csv'
        )
        
        n_users = len(self.loader.user_to_idx)
        n_items = len(self.loader.item_to_idx)
        n_categories = len(self.loader.category_to_idx)
        
        print(f"\n[STEP 2-4] TRAINING MODELS")
        
        # Import model trainers
        from baseline_mf_v2 import BaselineMFTrainer
        from diffnet_v2 import DiffNetTrainer
        from multihead_diffnetpp_v2 import MultiHeadDiffNetTrainer
        
        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
        
        # Baseline MF
        print(f"\n--- Training Baseline MF ---")
        self.baseline_trainer = BaselineMFTrainer(device=device)
        self.baseline_trainer.train(self.train_df, n_users, n_items, n_categories)
        self.baseline_trainer.save_model(self.models_dir)
        
        # DiffNet
        print(f"\n--- Training DiffNet ---")
        self.diffnet_trainer = DiffNetTrainer(device=device)
        self.diffnet_trainer.train(
            self.train_df, self.loader.trust_network,
            self.loader.user_to_idx, n_users, n_items, n_categories
        )
        self.diffnet_trainer.save_model(self.models_dir)
        
        # Multi-Head DiffNet++
        print(f"\n--- Training Multi-Head DiffNet++ ---")
        self.multihead_trainer = MultiHeadDiffNetTrainer(device=device)
        self.multihead_trainer.train(
            self.train_df, self.loader.trust_network,
            self.loader.user_to_idx, n_users, n_items, n_categories
        )
        self.multihead_trainer.save_model(self.models_dir)
        
        print(f"\n[STEP 5] EVALUATION")
        
        # Make predictions
        print(f"\n--- Making Predictions ---")
        baseline_preds = self.baseline_trainer.predict_batch(self.test_df)
        diffnet_preds = self.diffnet_trainer.predict_batch(self.test_df, self.loader.user_to_idx)
        multihead_preds = self.multihead_trainer.predict_batch(self.test_df)
        
        # Evaluate
        self.evaluator = Evaluator()
        self.evaluator.evaluate_model('Baseline MF', baseline_preds)
        self.evaluator.evaluate_model('DiffNet', diffnet_preds)
        self.evaluator.evaluate_model('Multi-Head DiffNet++', multihead_preds)
        
        # Compare
        comparison = self.evaluator.compare_models()
        comparison.to_csv(f'{self.results_dir}/model_comparison.csv')
        
        # Plot
        self.evaluator.plot_comparison(f'{self.results_dir}/model_comparison.png')
        
        print(f"\n[STEP 6] INFERENCE EXAMPLES")
        
        # Create inference system
        inference = RecommendationInference(
            self.multihead_trainer, self.loader, device=device
        )
        
        # Get some sample users
        sample_users = self.train_df['user_idx'].unique()[:5]
        sample_categories = list(self.loader.category_to_idx.keys())[:3]
        
        for user_idx in sample_users:
            user_id = self.loader.idx_to_user[user_idx]
            for category in sample_categories:
                inference.print_recommendations(user_id, category, n=10)
        
        print(f"\n{'='*80}")
        print("✓ PIPELINE COMPLETE!")
        print(f"{'='*80}")
        print(f"Results saved to: {self.results_dir}/")
        print(f"Models saved to: {self.models_dir}/")


if __name__ == "__main__":
    pipeline = ExperimentPipeline(
        data_dir='./data',
        results_dir='./results',
        models_dir='./models'
    )
    pipeline.run_complete_pipeline()