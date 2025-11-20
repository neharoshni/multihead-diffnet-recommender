"""
QUICK START SCRIPT - EVALUATION FIX
====================================

✓ FIXED: Added user_to_idx argument to diffnet_trainer.predict_batch()

Usage:
    python quick_start.py --mode train
    python quick_start.py --mode inference

Author: Your Team
Date: November 2025
"""

import os
import sys
import torch
import argparse
import pickle
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': './data',
    'models_dir': './models',
    'results_dir': './results',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_epochs': 20,
    'n_factors': 64,
    'n_layers': 2,
    'n_heads': 4,
    'batch_size': 256,
    'learning_rate': 0.001,
}

print(f"\n{'='*80}")
print("CONFIGURATION")
print(f"{'='*80}")
print(f"Device: {CONFIG['device']}")
print(f"Data Directory: {CONFIG['data_dir']}")
print(f"Models Directory: {CONFIG['models_dir']}")
print(f"Results Directory: {CONFIG['results_dir']}")
print(f"{'='*80}\n")


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def setup_directories():
    """Create necessary directories"""
    print("[SETUP] Creating directories...")
    
    for directory in [CONFIG['data_dir'], CONFIG['models_dir'], CONFIG['results_dir']]:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}/")


def verify_data():
    """Verify data files exist"""
    print("\n[SETUP] Verifying data files...")
    
    required_files = [
        'ratings.csv',
        'trustnetwork.csv',
        'ciao_train.csv',
        'ciao_test.csv'
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(CONFIG['data_dir'], filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            missing_files.append(filename)
    
    return len(missing_files) == 0


def load_data():
    """Load and preprocess data"""
    print("\n[STEP 1] Loading data...")
    
    try:
        from ciao_data_loader_v2 import CiaoDataLoaderFinal as CiaoDataLoaderV2
        
        loader = CiaoDataLoaderV2(data_dir=CONFIG['data_dir'])
        loader.load_ratings()
        loader.load_trust_network()
        loader.create_mappings()
        loader.build_social_graph()
        loader.enrich_ratings_with_indices()
        loader.build_product_category_dict()  # ✓ CRITICAL
        loader.identify_cold_start_users()
        loader.print_statistics()
        loader.save_mappings(CONFIG['models_dir'])
        
        # Load pre-split data
        train_df, test_df = loader.split_data(
            train_path=os.path.join(CONFIG['data_dir'], 'ciao_train.csv'),
            test_path=os.path.join(CONFIG['data_dir'], 'ciao_test.csv')
        )
        
        return loader, train_df, test_df
    
    except Exception as e:
        print(f"\n✗ Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def train_models(loader, train_df, test_df):
    """Train all three models"""
    print(f"\n[STEP 2-4] Training models...")
    
    try:
        from baseline_mf_v2 import BaselineMFTrainer
        from diffnet_v2 import DiffNetTrainer
        from multihead_diffnetpp_v2 import MultiHeadDiffNetTrainer
        
        n_users = len(loader.user_to_idx)
        n_products = len(loader.product_name_to_idx)
        n_categories = len(loader.category_to_idx)
        
        device = CONFIG['device']
        
        print(f"\nDataset sizes:")
        print(f"  - Users: {n_users}")
        print(f"  - Products: {n_products}")
        print(f"  - Categories: {n_categories}")
        
        # ===== Baseline MF =====
        print(f"\n{'='*70}")
        print("Training Baseline MF")
        print(f"{'='*70}")
        baseline_trainer = BaselineMFTrainer(
            n_factors=CONFIG['n_factors'],
            n_epochs=CONFIG['n_epochs'],
            batch_size=CONFIG['batch_size'],
            device=device
        )
        baseline_trainer.train(train_df, n_users, n_products, n_categories)
        baseline_trainer.save_model(CONFIG['models_dir'])
        
        # ===== DiffNet =====
        print(f"\n{'='*70}")
        print("Training DiffNet")
        print(f"{'='*70}")
        diffnet_trainer = DiffNetTrainer(
            n_factors=CONFIG['n_factors'],
            n_layers=CONFIG['n_layers'],
            n_epochs=CONFIG['n_epochs'],
            batch_size=CONFIG['batch_size'],
            device=device
        )
        diffnet_trainer.train(
            train_df,
            loader.trust_network,
            loader.user_to_idx,
            n_users, n_products, n_categories
        )
        diffnet_trainer.save_model(CONFIG['models_dir'])
        
        # ===== Multi-Head DiffNet++ =====
        print(f"\n{'='*70}")
        print("Training Multi-Head DiffNet++ (YOUR CONTRIBUTION!)")
        print(f"{'='*70}")
        multihead_trainer = MultiHeadDiffNetTrainer(
            n_factors=CONFIG['n_factors'],
            n_layers=CONFIG['n_layers'],
            n_heads=CONFIG['n_heads'],
            n_epochs=CONFIG['n_epochs'],
            batch_size=CONFIG['batch_size'],
            device=device
        )
        multihead_trainer.train(
            train_df,
            loader.trust_network,
            loader.user_to_idx,
            n_users, n_products, n_categories
        )
        multihead_trainer.save_model(CONFIG['models_dir'])
        
        return baseline_trainer, diffnet_trainer, multihead_trainer, loader
    
    except Exception as e:
        print(f"\n✗ Error training models: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate_models(baseline_trainer, diffnet_trainer, multihead_trainer, test_df, loader):
    """Evaluate all models - FIXED VERSION"""
    print(f"\n[STEP 5] Evaluating models...")
    
    try:
        from evaluation import Evaluator
        
        # Make predictions
        print("\n--- Making predictions ---")
        
        # Baseline MF
        baseline_preds = baseline_trainer.predict_batch(test_df)
        
        # DiffNet - ✓ FIXED: Added user_to_idx argument
        diffnet_preds = diffnet_trainer.predict_batch(test_df, loader.user_to_idx)
        
        # Multi-Head DiffNet++
        multihead_preds = multihead_trainer.predict_batch(test_df)
        
        # Evaluate
        evaluator = Evaluator()
        evaluator.evaluate_model('Baseline MF', baseline_preds)
        evaluator.evaluate_model('DiffNet', diffnet_preds)
        evaluator.evaluate_model('Multi-Head DiffNet++', multihead_preds)
        
        # Compare
        comparison = evaluator.compare_models()
        comparison.to_csv(f'{CONFIG["results_dir"]}/model_comparison.csv')
        
        # Plot
        try:
            evaluator.plot_comparison(f'{CONFIG["results_dir"]}/model_comparison.png')
        except:
            print("  (Plotting skipped - matplotlib might not be available)")
        
        print(f"\n✓ Evaluation complete!")
        
        return evaluator, multihead_preds
    
    except Exception as e:
        print(f"\n✗ Error evaluating models: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_inference_server(loader, multihead_trainer):
    """Start interactive inference server"""
    print(f"\n[INFERENCE SERVER] Starting...")
    print(f"{'='*80}")
    
    try:
        # Prepare reference data
        print("\nLoading reference data...")
        idx_to_product = {v: k for k, v in loader.product_name_to_idx.items()}
        idx_to_category = {v: k for k, v in loader.category_to_idx.items()}
        
        print("\nAvailable categories:")
        for idx, (cat_id, cat_name) in enumerate(loader.category_to_idx.items()):
            print(f"  {cat_id}: {cat_name}")
        
        print(f"\n{'='*80}")
        
        while True:
            print("\n[COMMAND] Options:")
            print("  1. Predict rating for user-product pair")
            print("  2. Get top recommendations for user in category")
            print("  3. Show available users")
            print("  4. Show available products")
            print("  5. Show available categories")
            print("  6. Exit")
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                # Single rating prediction
                try:
                    user_idx = int(input("Enter user index: "))
                    product_idx = int(input("Enter product index: "))
                    
                    if user_idx in multihead_trainer.user_to_idx.values():
                        # Create fake test row
                        import pandas as pd
                        test_row = pd.DataFrame({
                            'user_idx': [user_idx],
                            'product_idx': [product_idx],
                            'category_idx': [0],  # Default category
                            'rating': [3.0],  # Dummy rating
                            'helpfulness': [0.5]  # Dummy helpfulness
                        })
                        
                        pred_df = multihead_trainer.predict_batch(test_row)
                        pred = pred_df.iloc[0]['predicted_rating']
                        
                        product_name = idx_to_product.get(product_idx, f"Product {product_idx}")
                        print(f"\n{'='*80}")
                        print(f"PREDICTION FOR USER {user_idx}")
                        print(f"{'='*80}")
                        print(f"Product: {product_name} (ID: {product_idx})")
                        print(f"Predicted Rating: {pred:.2f} / 5.0")
                        print(f"{'='*80}")
                    else:
                        print("✗ User not found in training data")
                
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
            
            elif choice == '2':
                # Top recommendations
                try:
                    user_idx = int(input("Enter user index: "))
                    category_idx = int(input("Enter category index: "))
                    n_recs = int(input("Number of recommendations (default 10): ") or "10")
                    
                    if user_idx in multihead_trainer.user_to_idx.values():
                        # Get products in this category
                        products_in_category = [
                            prod_idx for prod_idx, cat_idx in loader.product_category_dict.items()
                            if cat_idx == category_idx
                        ]

                        print(f"\n[DEBUG] Found {len(products_in_category)} products in category {category_idx}")

                        if len(products_in_category) == 0:
                            print(f"✗ No products found in category {category_idx}")
                            print(f"[DEBUG] Available categories in product_category_dict:")
                            unique_cats = set(loader.product_category_dict.values())
                            print(f"  Categories: {sorted(unique_cats)}")
                            continue

                        print(f"✓ Generating predictions for {len(products_in_category)} products...")
                        
                        # Create test rows for all products in category
                        test_rows = []
                        for prod_idx in products_in_category:
                            test_rows.append({
                                'user_idx': user_idx,
                                'product_idx': prod_idx,
                                'category_idx': category_idx,
                                'rating': 3.0,
                                'helpfulness': 0.5
                            })
                        
                        import pandas as pd
                        test_df = pd.DataFrame(test_rows)
                        preds = multihead_trainer.predict_batch(test_df)
                        
                        # Sort by predicted rating
                        preds_sorted = preds.sort_values('predicted_rating', ascending=False)
                        top_n = preds_sorted.head(n_recs)
                        
                        print(f"\n{'='*80}")
                        print(f"TOP-{len(top_n)} RECOMMENDATIONS FOR USER {user_idx} (Category {category_idx})")
                        print(f"{'='*80}\n")
                        
                        for idx, row in top_n.iterrows():
                            prod_idx = int(row['product_idx'])
                            rating = row['predicted_rating']
                            product_name = idx_to_product.get(prod_idx, f"Product {prod_idx}")
                            print(f"{idx+1}. {product_name} (ID: {prod_idx}) - Rating: {rating:.2f}/5.0")
                        
                        print(f"\n{'='*80}")
                    else:
                        print("✗ User not found")
                
                except Exception as e:
                    print(f"✗ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == '3':
                print("\nSample available users:")
                users = sorted(loader.user_to_idx.keys())[:20]
                for user in users:
                    user_idx = loader.user_to_idx[user]
                    print(f"  {user} (Index: {user_idx})")
                print(f"  ... and {len(loader.user_to_idx) - 20} more")
                print(f"  Total: {len(loader.user_to_idx)} users")
            
            elif choice == '4':
                print("\nSample available products:")
                products = sorted(loader.product_name_to_idx.keys())[:20]
                for product in products:
                    prod_idx = loader.product_name_to_idx[product]
                    print(f"  {product} (Index: {prod_idx})")
                print(f"  ... and {len(loader.product_name_to_idx) - 20} more")
                print(f"  Total: {len(loader.product_name_to_idx)} products")
            
            elif choice == '5':
                print("\nAvailable categories:")
                for cat, idx in sorted(loader.category_to_idx.items()):
                    print(f"  {idx}: {cat}")
            
            elif choice == '6':
                print("\n✓ Exiting inference server")
                break
            
            else:
                print("✗ Invalid choice")
    
    except Exception as e:
        print(f"\n✗ Error in inference server: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def train_mode():
    """Complete training pipeline"""
    print(f"\n{'='*80}")
    print("STARTING TRAINING PIPELINE")
    print(f"{'='*80}")
    
    # Setup
    setup_directories()
    
    if not verify_data():
        print("\n✗ Missing required data files")
        sys.exit(1)
    
    # Load data
    loader, train_df, test_df = load_data()
    
    # Train models
    baseline_trainer, diffnet_trainer, multihead_trainer, loader = train_models(
        loader, train_df, test_df
    )
    
    # Evaluate - ✓ FIXED: Added loader argument
    evaluator, predictions = evaluate_models(
        baseline_trainer, diffnet_trainer, multihead_trainer,
        test_df, loader  # ← ADDED THIS
    )
    
    print(f"\n{'='*80}")
    print("✓ TRAINING PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {CONFIG['results_dir']}/")
    print(f"Models saved to: {CONFIG['models_dir']}/")


def inference_mode():
    """Inference server mode"""
    print(f"\n{'='*80}")
    print("STARTING INFERENCE SERVER")
    print(f"{'='*80}")
    
    # Setup
    setup_directories()
    
    if not verify_data():
        print("\n✗ Missing required data files")
        sys.exit(1)
    
    # Load data
    loader, _, _ = load_data()
    
    # Load trained model
    print("\n[Loading Trained Model]")
    from multihead_diffnetpp_v2 import MultiHeadDiffNetTrainer
    
    trainer = MultiHeadDiffNetTrainer(device=CONFIG['device'])
    
    try:
        trainer.load_model(
            CONFIG['models_dir'],
            n_users=len(loader.user_to_idx),
            n_products=len(loader.product_name_to_idx),
            n_categories=len(loader.category_to_idx)
        )
    except FileNotFoundError as e:
        print(f"\n✗ Model not found. Please train the model first using:")
        print(f"   python quick_start.py --mode train")
        sys.exit(1)
    
    # Start server
    run_inference_server(loader, trainer)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Ciao Dataset - Multi-Head DiffNet++ Implementation'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'inference'],
        default='train',
        help='Run mode: train (default) or inference'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'train':
            train_mode()
        elif args.mode == 'inference':
            inference_mode()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()