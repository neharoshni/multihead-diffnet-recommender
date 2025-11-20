"""
FIXED DATA LOADER - WITH product_category_dict
===============================================

✓ Added: build_product_category_dict() method
✓ Fixed: save/load mappings to include product_category_dict
✓ Complete: All dictionaries now properly created

Author: Your Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import os
import pickle

class CiaoDataLoaderFinal:
    """Fixed data loader for Ciao dataset - NOW WITH product_category_dict"""
    
    def __init__(self, data_dir='./data'):
        """Initialize data loader"""
        self.data_dir = data_dir
        
        # Core data
        self.ratings = None
        self.trust_network = None
        self.social_graph = None
        
        # Mappings
        self.user_to_idx = {}
        self.product_name_to_idx = {}
        self.category_to_idx = {}
        self.product_category_dict = {}  # ✓ NEW: Product → Category mapping
        
        self.idx_to_user = {}
        self.idx_to_product_name = {}
        self.idx_to_category = {}
        
        # Statistics
        self.stats = {}
        
        print(f"""
╔════════════════════════════════════════════════════════════════════╗
║          CIAO DATASET LOADER - FIXED VERSION                       ║
║          ✓ product_category_dict included                          ║
╚════════════════════════════════════════════════════════════════════╝
        """)
    
    def convert_helpfulness_to_numeric(self, value):
        """Convert helpfulness from text to numeric (0-1)"""
        if isinstance(value, (int, float)):
            return float(value)
        
        if not isinstance(value, str):
            return 0.5
        
        value = str(value).lower().strip()
        
        helpfulness_map = {
            'very helpful': 1.0,
            'very unhelpful': 0.1,
            'helpful': 0.7,
            'unhelpful': 0.3,
            'somewhat helpful': 0.6,
            '1': 1.0,
            '0': 0.0,
        }
        
        for key, score in helpfulness_map.items():
            if key in value:
                return score
        
        try:
            return float(value)
        except:
            return 0.5
    
    def load_ratings(self, filename='ratings.csv'):
        """Load ratings from CSV"""
        filepath = os.path.join(self.data_dir, filename)
        
        print(f"\n[STEP 1.1] Loading ratings from: {filepath}")
        
        try:
            self.ratings = pd.read_csv(filepath, on_bad_lines='skip')
            
            print(f"\nCSV Columns: {list(self.ratings.columns)}")
            print(f"First row:\n{self.ratings.iloc[0]}\n")
            
            needed_cols = ['user', 'product', 'category', 'rating', 'helpfulness']
            available_cols = [c for c in needed_cols if c in self.ratings.columns]
            
            print(f"Using columns: {available_cols}")
            self.ratings = self.ratings[available_cols].copy()
            
            # Data cleaning
            self.ratings['user'] = self.ratings['user'].astype(int)
            self.ratings['product'] = self.ratings['product'].astype(str)
            self.ratings['rating'] = self.ratings['rating'].astype(float)
            
            print(f"Converting helpfulness to numeric...")
            self.ratings['helpfulness'] = self.ratings['helpfulness'].apply(
                self.convert_helpfulness_to_numeric
            )
            
            # Filter valid ratings
            before = len(self.ratings)
            self.ratings = self.ratings[(self.ratings['rating'] >= 1) & 
                                       (self.ratings['rating'] <= 5)]
            after = len(self.ratings)
            print(f"Filtered ratings: {before} → {after} (removed {before-after})")
            
            # Remove duplicates
            self.ratings = self.ratings.drop_duplicates(
                subset=['user', 'product'], 
                keep='first'
            ).reset_index(drop=True)
            
            print(f"\n✓ Loaded {len(self.ratings)} unique ratings")
            print(f"  - Users: {self.ratings['user'].nunique()}")
            print(f"  - Products: {self.ratings['product'].nunique()}")
            print(f"  - Categories: {self.ratings['category'].nunique()}")
            print(f"  - Rating range: {self.ratings['rating'].min():.1f} - {self.ratings['rating'].max():.1f}")
            
            return self.ratings
            
        except Exception as e:
            print(f"✗ Error loading ratings: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_trust_network(self, filename='trustnetwork.csv'):
        """Load social trust network"""
        filepath = os.path.join(self.data_dir, filename)
        
        print(f"\n[STEP 1.2] Loading trust network from: {filepath}")
        
        try:
            trust_raw = pd.read_csv(filepath)
            
            print(f"Trust CSV Columns: {list(trust_raw.columns)}")
            print(f"First few rows:\n{trust_raw.head(3)}\n")
            
            trust_pairs = []
            
            for _, row in trust_raw.iterrows():
                user_id = int(row['userid'])
                friends = row['friends']
                
                if isinstance(friends, str):
                    if ',' in friends:
                        friend_list = [int(f.strip()) for f in friends.split(',') if f.strip()]
                    elif ' ' in friends:
                        friend_list = [int(f.strip()) for f in friends.split() if f.strip()]
                    else:
                        friend_list = [int(friends)]
                else:
                    friend_list = [int(friends)]
                
                for friend_id in friend_list:
                    trust_pairs.append({
                        'user_id': user_id,
                        'friend_id': friend_id
                    })
            
            self.trust_network = pd.DataFrame(trust_pairs)
            
            self.trust_network = self.trust_network[
                self.trust_network['user_id'] != self.trust_network['friend_id']
            ].drop_duplicates()
            
            print(f"✓ Loaded {len(self.trust_network)} trust relationships")
            print(f"  - Unique users: {self.trust_network['user_id'].nunique()}")
            
            return self.trust_network
            
        except Exception as e:
            print(f"✗ Error loading trust network: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_mappings(self):
        """Create ID mappings"""
        print(f"\n[STEP 1.3] Creating ID mappings...")
        
        try:
            users = sorted(self.ratings['user'].unique())
            self.user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
            self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
            
            product_names = sorted(self.ratings['product'].unique())
            self.product_name_to_idx = {name: idx for idx, name in enumerate(product_names)}
            self.idx_to_product_name = {idx: name for name, idx in self.product_name_to_idx.items()}
            
            categories = sorted(self.ratings['category'].unique())
            self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
            self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
            
            print(f"✓ Created mappings:")
            print(f"  - Users: {len(self.user_to_idx)}")
            print(f"  - Products: {len(self.product_name_to_idx)}")
            print(f"  - Categories: {len(self.category_to_idx)}")
            
            return self.user_to_idx, self.product_name_to_idx, self.category_to_idx
        
        except Exception as e:
            print(f"✗ Error creating mappings: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def build_social_graph(self):
        """Build directed graph"""
        print(f"\n[STEP 1.4] Building social graph...")
        
        try:
            self.social_graph = nx.DiGraph()
            
            for _, row in self.trust_network.iterrows():
                user_id = row['user_id']
                friend_id = row['friend_id']
                
                if user_id in self.user_to_idx and friend_id in self.user_to_idx:
                    u_idx = self.user_to_idx[user_id]
                    f_idx = self.user_to_idx[friend_id]
                    self.social_graph.add_edge(u_idx, f_idx, weight=1.0)
            
            print(f"✓ Social graph built:")
            print(f"  - Nodes: {self.social_graph.number_of_nodes()}")
            print(f"  - Edges: {self.social_graph.number_of_edges()}")
            
            return self.social_graph
        
        except Exception as e:
            print(f"✗ Error building social graph: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def enrich_ratings_with_indices(self):
        """Add index columns"""
        print(f"\n[STEP 1.5] Enriching ratings with indices...")
        
        self.ratings['user_idx'] = self.ratings['user'].map(self.user_to_idx)
        self.ratings['product_idx'] = self.ratings['product'].map(self.product_name_to_idx)
        self.ratings['category_idx'] = self.ratings['category'].map(self.category_to_idx)
        
        before = len(self.ratings)
        self.ratings = self.ratings.dropna(subset=['user_idx', 'product_idx', 'category_idx'])
        after = len(self.ratings)
        
        self.ratings['user_idx'] = self.ratings['user_idx'].astype(int)
        self.ratings['product_idx'] = self.ratings['product_idx'].astype(int)
        self.ratings['category_idx'] = self.ratings['category_idx'].astype(int)
        
        print(f"✓ Ratings enriched with indices")
        print(f"  - Rows before: {before}")
        print(f"  - Rows after: {after}")
        
        return self.ratings
    
    def build_product_category_dict(self):
        """
        BUILD PRODUCT-CATEGORY DICTIONARY
        
        ✓ THIS IS THE FIX FOR THE ERROR!
        Maps each product index to its category index
        """
        print(f"\n[STEP 1.6] Building product-category dictionary...")
        
        if self.ratings is None:
            raise ValueError("Load ratings first!")
        
        self.product_category_dict = {}
        
        # For each rating row, map product → category
        for _, row in self.ratings.iterrows():
            product_idx = row['product_idx']
            category_idx = row['category_idx']
            
            # Only store once (first occurrence)
            if product_idx not in self.product_category_dict:
                self.product_category_dict[product_idx] = category_idx
        
        print(f"✓ Built product_category_dict with {len(self.product_category_dict)} products")
        print(f"  Sample mappings:")
        for i, (prod_idx, cat_idx) in enumerate(list(self.product_category_dict.items())[:5]):
            prod_name = self.idx_to_product_name.get(prod_idx, f"Product {prod_idx}")[:30]
            cat_name = self.idx_to_category.get(cat_idx, f"Category {cat_idx}")
            print(f"    Product {prod_idx} ({prod_name}) → Category {cat_idx} ({cat_name})")
        
        return self.product_category_dict
    
    def identify_cold_start_users(self, threshold=5):
        """Identify cold-start users"""
        print(f"\n[STEP 1.7] Identifying cold-start users (threshold ≤ {threshold} ratings)...")
        
        user_counts = self.ratings.groupby('user').size()
        cold_start_user_ids = user_counts[user_counts <= threshold].index.tolist()
        
        cold_start_indices = [self.user_to_idx[uid] for uid in cold_start_user_ids 
                             if uid in self.user_to_idx]
        
        print(f"✓ Cold-start analysis:")
        print(f"  - Cold-start users: {len(cold_start_indices)}")
        print(f"  - Total users: {len(self.user_to_idx)}")
        print(f"  - Percentage: {len(cold_start_indices) / len(self.user_to_idx) * 100:.2f}%")
        
        return cold_start_indices
    
    def split_data(self, train_path=None, test_path=None, test_size=0.2, random_state=42):
        """Split or load pre-split data"""
        print(f"\n[STEP 1.8] Splitting data...")
        
        if train_path and test_path:
            print(f"Loading pre-split data...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            print(f"Converting helpfulness to numeric (pre-split data)...")
            train_df['helpfulness'] = train_df['helpfulness'].apply(
                self.convert_helpfulness_to_numeric
            )
            test_df['helpfulness'] = test_df['helpfulness'].apply(
                self.convert_helpfulness_to_numeric
            )
        else:
            print(f"Creating new split...")
            train_df, test_df = train_test_split(
                self.ratings,
                test_size=test_size,
                random_state=random_state
            )
        
        processed_dfs = []
        
        for df_name, df in [('train', train_df), ('test', test_df)]:
            df = df.copy()
            
            df['user_idx'] = df['user'].map(self.user_to_idx)
            df['product_idx'] = df['product'].map(self.product_name_to_idx)
            df['category_idx'] = df['category'].map(self.category_to_idx)
            
            before_len = len(df)
            
            df = df.dropna(subset=['user_idx', 'product_idx', 'category_idx', 'helpfulness', 'rating'])
            after_len = len(df)
            
            if after_len < before_len:
                print(f"  Dropped {before_len - after_len} rows with NaN values from {df_name} set")
            
            df['user_idx'] = df['user_idx'].astype(int)
            df['product_idx'] = df['product_idx'].astype(int)
            df['category_idx'] = df['category_idx'].astype(int)
            
            processed_dfs.append(df)
        
        train_df, test_df = processed_dfs
        
        print(f"✓ Data split:")
        print(f"  - Training: {len(train_df)} ratings")
        print(f"  - Test: {len(test_df)} ratings")
        
        return train_df, test_df
    
    def get_statistics(self):
        """Get dataset statistics"""
        if self.ratings is None or self.trust_network is None:
            return {}
        
        stats = {
            'n_users': len(self.user_to_idx),
            'n_products': len(self.product_name_to_idx),
            'n_categories': len(self.category_to_idx),
            'n_ratings': len(self.ratings),
            'n_social_links': len(self.trust_network),
            'avg_rating': self.ratings['rating'].mean(),
            'avg_helpfulness': self.ratings['helpfulness'].mean(),
            'sparsity': self._calculate_sparsity(),
        }
        
        self.stats = stats
        return stats
    
    def _calculate_sparsity(self):
        """Calculate sparsity"""
        if self.ratings is None:
            return 1.0
        
        n_users = len(self.user_to_idx)
        n_products = len(self.product_name_to_idx)
        n_ratings = len(self.ratings)
        
        return 1 - (n_ratings / (n_users * n_products))
    
    def print_statistics(self):
        """Pretty print statistics"""
        stats = self.get_statistics()
        
        print(f"\n{'='*70}")
        print("DATASET STATISTICS")
        print(f"{'='*70}")
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")
        
        print(f"{'='*70}")
    
    def save_mappings(self, save_dir='./models'):
        """Save all mappings including product_category_dict"""
        os.makedirs(save_dir, exist_ok=True)
        
        mappings = {
            'user_to_idx': self.user_to_idx,
            'product_name_to_idx': self.product_name_to_idx,
            'category_to_idx': self.category_to_idx,
            'product_category_dict': self.product_category_dict,  # ✓ NOW INCLUDED
            'idx_to_user': self.idx_to_user,
            'idx_to_product_name': self.idx_to_product_name,
            'idx_to_category': self.idx_to_category,
            'stats': self.stats
        }
        
        with open(f'{save_dir}/mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        print(f"\n✓ Mappings saved to {save_dir}/mappings.pkl")
    
    def load_mappings(self, save_dir='./models'):
        """Load all mappings including product_category_dict"""
        with open(f'{save_dir}/mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        self.user_to_idx = mappings['user_to_idx']
        self.product_name_to_idx = mappings['product_name_to_idx']
        self.category_to_idx = mappings['category_to_idx']
        self.product_category_dict = mappings.get('product_category_dict', {})  # ✓ NOW LOADED
        self.idx_to_user = mappings['idx_to_user']
        self.idx_to_product_name = mappings['idx_to_product_name']
        self.idx_to_category = mappings['idx_to_category']
        self.stats = mappings['stats']
        
        print(f"\n✓ Mappings loaded from {save_dir}/mappings.pkl")


if __name__ == "__main__":
    loader = CiaoDataLoaderFinal(data_dir='./data')
    loader.load_ratings()
    loader.load_trust_network()
    loader.create_mappings()
    loader.build_social_graph()
    loader.enrich_ratings_with_indices()
    loader.build_product_category_dict()  # ✓ NOW CALLED
    loader.identify_cold_start_users()
    loader.print_statistics()
    loader.save_mappings()
    print("\n✓ Fixed loader works correctly!")