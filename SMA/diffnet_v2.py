"""
DIFFNET - FIXED VERSION
=======================

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

class DiffNetModel(nn.Module):
    """DiffNet: Neural Influence Diffusion"""
    
    def __init__(self, n_users, n_products, n_categories, n_factors=64, n_layers=2):
        """
        Args:
            n_users: Number of users
            n_products: Number of products (changed from items)
            n_categories: Number of categories
            n_factors: Embedding dimension
            n_layers: Number of diffusion layers
        """
        super(DiffNetModel, self).__init__()
        
        self.n_users = n_users
        self.n_products = n_products
        self.n_categories = n_categories
        self.n_factors = n_factors
        self.n_layers = n_layers
        
        # Embeddings
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.product_embeddings = nn.Embedding(n_products, n_factors)  # CHANGED: item → product
        self.category_embeddings = nn.Embedding(n_categories, n_factors)
        
        # Biases
        self.user_bias = nn.Embedding(n_users, 1)
        self.product_bias = nn.Embedding(n_products, 1)  # CHANGED
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Diffusion layers
        self.diffusion_layers = nn.ModuleList([
            nn.Linear(n_factors, n_factors) for _ in range(n_layers)
        ])
        
        self.activation = nn.ReLU()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.product_embeddings.weight, std=0.01)  # CHANGED
        nn.init.normal_(self.category_embeddings.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0)
        nn.init.constant_(self.product_bias.weight, 0)  # CHANGED
        
        for layer in self.diffusion_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def social_diffusion(self, user_idx, social_neighbors_list):
        """Social influence diffusion"""
        batch_size = user_idx.size(0)
        user_embeds = self.user_embeddings(user_idx)
        
        for layer_idx in range(self.n_layers):
            diffused_embeds = []
            
            for i in range(batch_size):
                neighbors = social_neighbors_list[i]
                
                if len(neighbors) == 0:
                    diffused_embeds.append(user_embeds[i])
                else:
                    neighbor_indices = torch.LongTensor(neighbors).to(user_embeds.device)
                    neighbor_embeds = self.user_embeddings(neighbor_indices)
                    neighbor_mean = neighbor_embeds.mean(dim=0)
                    
                    combined = user_embeds[i] + neighbor_mean
                    transformed = self.diffusion_layers[layer_idx](combined)
                    diffused = self.activation(transformed)
                    
                    diffused_embeds.append(diffused)
            
            user_embeds = torch.stack(diffused_embeds)
        
        return user_embeds
    
    def forward(self, user_idx, product_idx, category_idx, social_neighbors_list):  # CHANGED: item_idx → product_idx
        """Forward pass"""
        # Get diffused user embeddings
        user_embeds = self.social_diffusion(user_idx, social_neighbors_list)
        
        # Get product embeddings (CHANGED: item → product)
        product_embed = self.product_embeddings(product_idx)
        category_embed = self.category_embeddings(category_idx)
        
        # Combine
        combined_product_embed = product_embed + 0.3 * category_embed
        
        # Get biases
        u_bias = self.user_bias(user_idx).squeeze()
        p_bias = self.product_bias(product_idx).squeeze()  # CHANGED
        
        # Predict
        interaction = (user_embeds * combined_product_embed).sum(dim=1)
        predictions = self.global_bias + u_bias + p_bias + interaction
        
        return predictions


class DiffNetTrainer:
    """Trainer for DiffNet"""
    
    def __init__(self, n_factors=64, n_layers=2, n_epochs=20,
                 learning_rate=0.001, reg_weight=0.01,
                 batch_size=256, device='cpu'):
        """Initialize trainer"""
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg_weight = reg_weight
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        self.model = None
        self.social_graph = None
        
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║          DIFFNET TRAINER (Neural Influence Diffusion)          ║
║                                                                ║
║  Configuration:                                                ║
║    - Embedding dimension: {n_factors}                               ║
║    - Diffusion layers: {n_layers}                                 ║
║    - Training epochs: {n_epochs}                                   ║
║    - Learning rate: {learning_rate}                           ║
║    - Regularization: {reg_weight}                              ║
║    - Device: {device}                                      ║
╚════════════════════════════════════════════════════════════════╝
        """)
    
    def build_social_graph(self, trust_df, user_to_idx):
        """Build social graph"""
        print(f"\n[Building Social Graph]")
        
        self.social_graph = {}
        
        for user_idx in range(len(user_to_idx)):
            self.social_graph[user_idx] = []
        
        for _, row in trust_df.iterrows():
            user_id = row['user_id']
            friend_id = row['friend_id']
            
            if user_id in user_to_idx and friend_id in user_to_idx:
                u_idx = user_to_idx[user_id]
                f_idx = user_to_idx[friend_id]
                self.social_graph[u_idx].append(f_idx)
        
        print(f"  ✓ Social graph built")
        print(f"  - Total users: {len(self.social_graph)}")
        
        friends_per_user = [len(v) for v in self.social_graph.values()]
        print(f"  - Average friends per user: {np.mean(friends_per_user):.2f}")
        print(f"  - Max friends: {max(friends_per_user)}")
    
    def get_user_neighbors(self, user_idx):
        """Get friends of user"""
        if self.social_graph is None or user_idx not in self.social_graph:
            return []
        return self.social_graph[user_idx]
    
    def train(self, train_df, trust_df, user_to_idx, n_users, n_products, n_categories):  # CHANGED: n_items → n_products
        """Train DiffNet"""
        print(f"\n{'='*70}")
        print("TRAINING DIFFNET (Neural Influence Diffusion)")
        print(f"{'='*70}")
        
        # Build social graph
        self.build_social_graph(trust_df, user_to_idx)
        
        # Initialize model
        self.model = DiffNetModel(
            n_users, n_products, n_categories,  # CHANGED
            n_factors=self.n_factors,
            n_layers=self.n_layers
        ).to(self.device)
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.reg_weight
        )
        criterion = nn.MSELoss()
        
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
                
                # Get social neighbors
                social_neighbors = [self.get_user_neighbors(u) for u in batch_users]
                
                user_tensor = torch.LongTensor(batch_users).to(self.device)
                product_tensor = torch.LongTensor(batch_products).to(self.device)  # CHANGED
                category_tensor = torch.LongTensor(batch_categories).to(self.device)
                rating_tensor = torch.FloatTensor(batch_ratings).to(self.device)
                
                predictions = self.model(
                    user_tensor, product_tensor, category_tensor, social_neighbors  # CHANGED
                )
                
                loss = criterion(predictions, rating_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_users)
            
            rmse = np.sqrt(epoch_loss / len(user_indices))
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.n_epochs - 1:
                print(f"  Epoch {epoch+1:3d}/{self.n_epochs} | RMSE: {rmse:.4f}")
        
        print(f"\n✓ Training complete!")
    
    def predict_batch(self, test_df, user_to_idx):
        """Make predictions"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            user_indices = test_df['user_idx'].values
            product_indices = test_df['product_idx'].values  # CHANGED
            category_indices = test_df['category_idx'].values
            
            for i in tqdm(range(0, len(test_df), self.batch_size),
                         desc="Predicting (DiffNet)"):
                
                end_idx = min(i + self.batch_size, len(test_df))
                batch_users = user_indices[i:end_idx]
                batch_products = product_indices[i:end_idx]  # CHANGED
                batch_categories = category_indices[i:end_idx]
                
                social_neighbors = [self.get_user_neighbors(u) for u in batch_users]
                
                user_tensor = torch.LongTensor(batch_users).to(self.device)
                product_tensor = torch.LongTensor(batch_products).to(self.device)  # CHANGED
                category_tensor = torch.LongTensor(batch_categories).to(self.device)
                
                batch_pred = self.model(
                    user_tensor, product_tensor, category_tensor, social_neighbors  # CHANGED
                )
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
        model_path = f'{save_dir}/diffnet_model.pth'
        torch.save(self.model.state_dict(), model_path)
        print(f"\n✓ Model saved to {model_path}")


if __name__ == "__main__":
    print("\nDiffNet module (FIXED) loaded successfully!")