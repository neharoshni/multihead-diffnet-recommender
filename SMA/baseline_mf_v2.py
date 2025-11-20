"""
BASELINE MATRIX FACTORIZATION - FIXED
======================================

Updated to use product_idx instead of item_idx

Author: Your Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class BaselineMFWithCategory(nn.Module):
    """Matrix Factorization with Category Support"""
    
    def __init__(self, n_users, n_products, n_categories, n_factors=64):
        """
        Args:
            n_users: Number of users
            n_products: Number of products (changed from items)
            n_categories: Number of categories
            n_factors: Embedding dimension
        """
        super(BaselineMFWithCategory, self).__init__()
        
        self.n_users = n_users
        self.n_products = n_products
        self.n_categories = n_categories
        self.n_factors = n_factors
        
        # Embeddings
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.product_embeddings = nn.Embedding(n_products, n_factors)  # CHANGED: item → product
        self.category_embeddings = nn.Embedding(n_categories, n_factors)
        
        # Biases
        self.user_bias = nn.Embedding(n_users, 1)
        self.product_bias = nn.Embedding(n_products, 1)  # CHANGED: item → product
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.product_embeddings.weight, std=0.01)  # CHANGED
        nn.init.normal_(self.category_embeddings.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0)
        nn.init.constant_(self.product_bias.weight, 0)  # CHANGED
    
    def forward(self, user_idx, product_idx, category_idx):  # CHANGED: item_idx → product_idx
        """Forward pass"""
        # Get embeddings
        user_embed = self.user_embeddings(user_idx)
        product_embed = self.product_embeddings(product_idx)  # CHANGED
        category_embed = self.category_embeddings(category_idx)
        
        # Combine
        combined_product_embed = product_embed + 0.3 * category_embed
        
        # Get biases
        u_bias = self.user_bias(user_idx).squeeze()
        p_bias = self.product_bias(product_idx).squeeze()  # CHANGED
        
        # Predict
        interaction = (user_embed * combined_product_embed).sum(dim=1)
        predictions = self.global_bias + u_bias + p_bias + interaction
        
        return predictions


class BaselineMFTrainer:
    """Trainer for Baseline Matrix Factorization"""
    
    def __init__(self, n_factors=64, n_epochs=20, learning_rate=0.005,
                 reg_weight=0.02, batch_size=256, device='cpu'):
        """Initialize trainer"""
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def train(self, train_df, n_users, n_products, n_categories):  # CHANGED: n_items → n_products
        """Train the model"""
        print(f"\n{'='*70}")
        print("TRAINING BASELINE MATRIX FACTORIZATION")
        print(f"{'='*70}")
        
        # Initialize model
        self.model = BaselineMFWithCategory(
            n_users, n_products, n_categories,  # CHANGED
            n_factors=self.n_factors
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.reg_weight
        )
        self.criterion = nn.MSELoss()
        
        print(f"\n[Training Setup]")
        print(f"  - Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - Training samples: {len(train_df)}")
        
        # Extract data - CHANGED: item_idx → product_idx
        user_indices = train_df['user_idx'].values
        product_indices = train_df['product_idx'].values
        category_indices = train_df['category_idx'].values
        ratings = train_df['rating'].values.astype(np.float32)
        
        print(f"\n[Training Progress]")
        
        for epoch in range(self.n_epochs):
            self.model.train()
            
            perm = np.random.permutation(len(user_indices))
            epoch_loss = 0.0
            n_batches = (len(user_indices) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(user_indices))
                batch_perm = perm[start_idx:end_idx]
                
                batch_users = user_indices[batch_perm]
                batch_products = product_indices[batch_perm]  # CHANGED
                batch_categories = category_indices[batch_perm]
                batch_ratings = ratings[batch_perm]
                
                user_tensor = torch.LongTensor(batch_users).to(self.device)
                product_tensor = torch.LongTensor(batch_products).to(self.device)  # CHANGED
                category_tensor = torch.LongTensor(batch_categories).to(self.device)
                rating_tensor = torch.FloatTensor(batch_ratings).to(self.device)
                
                predictions = self.model(user_tensor, product_tensor, category_tensor)  # CHANGED
                loss = self.criterion(predictions, rating_tensor)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_users)
            
            rmse = np.sqrt(epoch_loss / len(user_indices))
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.n_epochs - 1:
                print(f"  Epoch {epoch+1:3d}/{self.n_epochs} | RMSE: {rmse:.4f}")
        
        print(f"\n✓ Training complete!")
    
    def predict_batch(self, test_df):
        """Make predictions"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            user_indices = test_df['user_idx'].values
            product_indices = test_df['product_idx'].values  # CHANGED
            category_indices = test_df['category_idx'].values
            
            for i in tqdm(range(0, len(test_df), self.batch_size), 
                         desc="Predicting (Baseline MF)"):
                
                end_idx = min(i + self.batch_size, len(test_df))
                batch_users = user_indices[i:end_idx]
                batch_products = product_indices[i:end_idx]  # CHANGED
                batch_categories = category_indices[i:end_idx]
                
                user_tensor = torch.LongTensor(batch_users).to(self.device)
                product_tensor = torch.LongTensor(batch_products).to(self.device)  # CHANGED
                category_tensor = torch.LongTensor(batch_categories).to(self.device)
                
                batch_pred = self.model(user_tensor, product_tensor, category_tensor)  # CHANGED
                batch_pred = torch.clamp(batch_pred, 1.0, 5.0).cpu().numpy()
                
                for j, pred in enumerate(batch_pred):
                    predictions.append({
                        'user_idx': batch_users[j],
                        'product_idx': batch_products[j],  # CHANGED
                        'actual_rating': test_df.iloc[i+j]['rating'],
                        'predicted_rating': pred,
                        'actual_helpfulness': test_df.iloc[i+j]['helpfulness'],
                    })
        
        return pd.DataFrame(predictions)
    
    def save_model(self, save_dir='./models'):
        """Save model"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = f'{save_dir}/baseline_mf_model.pth'
        torch.save(self.model.state_dict(), model_path)
        print(f"\n✓ Model saved to {model_path}")


if __name__ == "__main__":
    print("\nBaseline MF module (FIXED) loaded successfully!")