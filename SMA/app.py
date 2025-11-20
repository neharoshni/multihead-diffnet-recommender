"""
STREAMLIT FRONTEND - Multi-Head DiffNet++ Recommendation System
================================================================

Complete UI for:
- Model predictions
- Top-N recommendations
- Performance metrics
- Data exploration

Usage:
    streamlit run app.py

Author: Your Team
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Multi-Head DiffNet++ Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONFIG = {
    'models_dir': './models',
    'results_dir': './results',
    'data_dir': './data',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# LOAD RESOURCES
# ============================================================================

@st.cache_resource
def load_trained_model():
    """Load trained Multi-Head DiffNet++ model"""
    try:
        from multihead_diffnetpp_v2 import MultiHeadDiffNetTrainer
        from ciao_data_loader_v2 import CiaoDataLoaderFinal
        
        # Load data loader with mappings
        loader = CiaoDataLoaderFinal(data_dir=CONFIG['data_dir'])
        loader.load_mappings(CONFIG['models_dir'])
        
        # Load trained model
        trainer = MultiHeadDiffNetTrainer(device=CONFIG['device'])
        trainer.load_model(
            CONFIG['models_dir'],
            n_users=len(loader.user_to_idx),
            n_products=len(loader.product_name_to_idx),
            n_categories=len(loader.category_to_idx)
        )
        
        return trainer, loader
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

@st.cache_data
def load_model_metrics():
    """Load model comparison metrics"""
    try:
        metrics_path = os.path.join(CONFIG['results_dir'], 'model_comparison.csv')
        if os.path.exists(metrics_path):
            return pd.read_csv(metrics_path)
        else:
            st.warning("Model comparison metrics not found. Please train models first.")
            return None
    except Exception as e:
        st.warning(f"Could not load metrics: {str(e)}")
        return None

@st.cache_data
def load_dataset_stats(_loader):
    """Load dataset statistics"""
    return _loader.stats

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_single_rating(trainer, loader, user_idx, product_idx):
    """Predict rating for single user-product pair"""
    try:
        # Get actual category for this product
        if product_idx in loader.product_category_dict:
            category_idx = loader.product_category_dict[product_idx]
        else:
            category_idx = 0  # Default
        
        # Create test dataframe
        test_df = pd.DataFrame({
            'user_idx': [user_idx],
            'product_idx': [product_idx],
            'category_idx': [category_idx],
            'rating': [3.0],
            'helpfulness': [0.5]
        })
        
        # Predict
        predictions = trainer.predict_batch(test_df)
        
        return {
            'predicted_rating': predictions.iloc[0]['predicted_rating'],
            'predicted_helpfulness': predictions.iloc[0]['predicted_helpfulness'],
            'category_idx': category_idx,
            'category_name': loader.idx_to_category.get(category_idx, f"Category {category_idx}")
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_top_recommendations(trainer, loader, user_idx, category_idx, n_recs=10):
    """Get top-N recommendations for user in category"""
    try:
        # Get products in category
        products_in_category = [
            prod_idx for prod_idx, cat_idx in loader.product_category_dict.items()
            if cat_idx == category_idx
        ]
        
        if not products_in_category:
            return None
        
        # Create test rows
        test_rows = []
        for prod_idx in products_in_category:
            test_rows.append({
                'user_idx': user_idx,
                'product_idx': prod_idx,
                'category_idx': category_idx,
                'rating': 3.0,
                'helpfulness': 0.5
            })
        
        test_df = pd.DataFrame(test_rows)
        predictions = trainer.predict_batch(test_df)
        
        # Sort by rating
        predictions_sorted = predictions.sort_values('predicted_rating', ascending=False)
        top_n = predictions_sorted.head(n_recs)
        
        # Add product names
        top_n['product_name'] = top_n['product_idx'].map(
            lambda x: loader.idx_to_product_name.get(x, f"Product {x}")
        )
        
        return top_n
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return None

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar(loader):
    """Render sidebar with navigation"""
    st.sidebar.title("🎯 Multi-Head DiffNet++")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predictions", "📊 Model Performance", "📈 Data Explorer"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.metric("Users", f"{len(loader.user_to_idx):,}")
    st.sidebar.metric("Products", f"{len(loader.product_name_to_idx):,}")
    st.sidebar.metric("Categories", f"{len(loader.category_to_idx)}")
    
    return page

def render_home_page(loader, stats):
    """Render home page"""
    st.title("🎯 Multi-Head DiffNet++ Recommendation System")
    st.markdown("### *Socially-Aware Product Recommendations with Multi-Head Attention*")
    st.markdown("---")
    
    # Hero section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 👥 Social Influence\nLeverages trust network for better recommendations")
    
    with col2:
        st.success("### 🎯 Multi-Head Attention\n4 attention heads capture diverse relationships")
    
    with col3:
        st.warning("### 📊 Dual Output\nPredicts both rating and helpfulness")
    
    st.markdown("---")
    
    # Dataset statistics
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Ratings", f"{stats['n_ratings']:,}")
    
    with col2:
        st.metric("Social Links", f"{stats['n_social_links']:,}")
    
    with col3:
        st.metric("Avg Rating", f"{stats['avg_rating']:.2f}/5.0")
    
    with col4:
        st.metric("Sparsity", f"{stats['sparsity']*100:.2f}%")
    
    st.markdown("---")
    
    # Category distribution
    st.subheader("📚 Categories Available")
    
    categories_df = pd.DataFrame([
        {"Category": cat, "Index": idx}
        for cat, idx in sorted(loader.category_to_idx.items(), key=lambda x: x[1])
    ])
    
    # Display in columns
    n_cols = 3
    cols = st.columns(n_cols)
    
    for i, row in categories_df.iterrows():
        col_idx = i % n_cols
        with cols[col_idx]:
            st.markdown(f"**{row['Index']}**: {row['Category']}")
    
    st.markdown("---")
    
    # Model architecture
    st.subheader("🧠 Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Key Features:**
        - 64-dimensional embeddings
        - 2-layer diffusion process
        - 4 attention heads per layer
        - Dual prediction heads
        - 1M+ trainable parameters
        """)
    
    with col2:
        st.markdown("""
        **Innovation:**
        - Social diffusion (friend influence)
        - Interest diffusion (product similarity)
        - Multi-head attention mechanism
        - Helpfulness prediction
        - Category-aware embeddings
        """)

def render_predictions_page(trainer, loader):
    """Render predictions page"""
    st.title("🔮 Make Predictions")
    st.markdown("---")
    
    # Two tabs: Single prediction and Recommendations
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "📋 Top Recommendations"])
    
    # ===== TAB 1: Single Prediction =====
    with tab1:
        st.subheader("Predict Rating for User-Product Pair")
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_idx = st.number_input(
                "Enter User Index",
                min_value=0,
                max_value=len(loader.user_to_idx)-1,
                value=14,
                help="User index from the training set"
            )
            
            # Show user info
            if user_idx in loader.idx_to_user:
                user_id = loader.idx_to_user[user_idx]
                st.info(f"**User ID:** {user_id}")
                
                # Show user's social connections
                if user_idx in trainer.social_graph:
                    n_friends = len(trainer.social_graph[user_idx])
                    st.caption(f"👥 {n_friends} friends in network")
        
        with col2:
            product_idx = st.number_input(
                "Enter Product Index",
                min_value=0,
                max_value=len(loader.product_name_to_idx)-1,
                value=7554,
                help="Product index from the catalog"
            )
            
            # Show product info
            if product_idx in loader.idx_to_product_name:
                product_name = loader.idx_to_product_name[product_idx]
                st.info(f"**Product:** {product_name}")
        
        if st.button("🔮 Predict Rating", type="primary"):
            with st.spinner("Generating prediction..."):
                result = predict_single_rating(trainer, loader, user_idx, product_idx)
                
                if result:
                    st.success("Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Predicted Rating",
                            f"{result['predicted_rating']:.2f}/5.0",
                            delta=f"{result['predicted_rating']-3.0:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Helpfulness Score",
                            f"{result['predicted_helpfulness']:.2%}"
                        )
                    
                    with col3:
                        st.info(f"**Category**\n{result['category_name']}")
                    
                    # Visual gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['predicted_rating'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Predicted Rating"},
                        delta={'reference': 3.0},
                        gauge={
                            'axis': {'range': [1, 5]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [1, 2], 'color': "lightgray"},
                                {'range': [2, 3], 'color': "gray"},
                                {'range': [3, 4], 'color': "lightblue"},
                                {'range': [4, 5], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 4.5
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 2: Recommendations =====
    with tab2:
        st.subheader("Get Top-N Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_idx_rec = st.number_input(
                "User Index",
                min_value=0,
                max_value=len(loader.user_to_idx)-1,
                value=14,
                key="user_rec"
            )
        
        with col2:
            category_idx = st.selectbox(
                "Category",
                options=sorted(loader.category_to_idx.values()),
                format_func=lambda x: f"{x}: {loader.idx_to_category.get(x, f'Category {x}')}"
            )
        
        with col3:
            n_recs = st.slider(
                "Number of Recommendations",
                min_value=5,
                max_value=50,
                value=10,
                step=5
            )
        
        if st.button("📋 Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                recommendations = get_top_recommendations(
                    trainer, loader, user_idx_rec, category_idx, n_recs
                )
                
                if recommendations is not None and not recommendations.empty:
                    st.success(f"Found {len(recommendations)} recommendations!")
                    
                    # Display as table
                    display_df = recommendations[['product_name', 'predicted_rating', 'predicted_helpfulness']].copy()
                    display_df.columns = ['Product', 'Rating', 'Helpfulness']
                    display_df['Rating'] = display_df['Rating'].apply(lambda x: f"{x:.2f}/5.0")
                    display_df['Helpfulness'] = display_df['Helpfulness'].apply(lambda x: f"{x:.1%}")
                    display_df.index = range(1, len(display_df)+1)
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Bar chart
                    fig = px.bar(
                        recommendations.head(15),
                        x='predicted_rating',
                        y='product_name',
                        orientation='h',
                        title=f"Top {min(15, len(recommendations))} Products by Rating",
                        labels={'predicted_rating': 'Predicted Rating', 'product_name': 'Product'},
                        color='predicted_rating',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No products found in this category.")

def render_performance_page(metrics_df):
    """Render model performance page"""
    st.title("📊 Model Performance Metrics")
    st.markdown("---")
    
    if metrics_df is None:
        st.warning("No metrics available. Please train models first.")
        return
    
    # Model comparison
    st.subheader("🏆 Model Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    for i, (idx, row) in enumerate(metrics_df.iterrows()):
        col = [col1, col2, col3][i % 3]
        with col:
            st.metric(
                row['Model'],
                f"RMSE: {row['RMSE']:.4f}",
                delta=f"MAE: {row['MAE']:.4f}"
            )
    
    st.markdown("---")
    
    # Bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rmse = px.bar(
            metrics_df,
            x='Model',
            y='RMSE',
            title='Root Mean Squared Error (RMSE)',
            color='RMSE',
            color_continuous_scale='Reds_r'
        )
        fig_rmse.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col2:
        fig_mae = px.bar(
            metrics_df,
            x='Model',
            y='MAE',
            title='Mean Absolute Error (MAE)',
            color='MAE',
            color_continuous_scale='Blues_r'
        )
        fig_mae.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # Helpfulness if available
    if 'Helpfulness_RMSE' in metrics_df.columns:
        st.markdown("---")
        st.subheader("🎯 Helpfulness Prediction")
        
        helpfulness_data = metrics_df[metrics_df['Helpfulness_RMSE'].notna()]
        
        if not helpfulness_data.empty:
            fig_help = px.bar(
                helpfulness_data,
                x='Model',
                y='Helpfulness_RMSE',
                title='Helpfulness Prediction Error (RMSE)',
                color='Helpfulness_RMSE',
                color_continuous_scale='Greens_r'
            )
            fig_help.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_help, use_container_width=True)
    
    # Detailed table
    st.markdown("---")
    st.subheader("📋 Detailed Metrics")
    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)

def render_data_explorer(loader):
    """Render data explorer page"""
    st.title("📈 Data Explorer")
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["👥 Users", "📦 Products", "📚 Categories"])
    
    # ===== TAB 1: Users =====
    with tab1:
        st.subheader("User Directory")
        
        n_users_show = st.slider("Number of users to display", 10, 100, 20)
        
        users_data = []
        for user_id, user_idx in sorted(loader.user_to_idx.items())[:n_users_show]:
            n_friends = len(trainer.social_graph.get(user_idx, []))
            n_products = len(trainer.user_product_dict.get(user_idx, []))
            
            users_data.append({
                'User ID': user_id,
                'User Index': user_idx,
                'Friends': n_friends,
                'Products Rated': n_products
            })
        
        users_df = pd.DataFrame(users_data)
        st.dataframe(users_df, use_container_width=True, hide_index=True)
        
        # Distribution
        fig = px.histogram(
            users_df,
            x='Friends',
            title='Distribution of Friends per User',
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 2: Products =====
    with tab2:
        st.subheader("Product Catalog")
        
        # Category filter
        selected_category = st.selectbox(
            "Filter by Category",
            options=['All'] + sorted(loader.category_to_idx.keys())
        )
        
        n_products_show = st.slider("Number of products to display", 10, 100, 20, key="prod_slider")
        
        products_data = []
        for prod_name, prod_idx in sorted(loader.product_name_to_idx.items())[:n_products_show]:
            cat_idx = loader.product_category_dict.get(prod_idx, 0)
            cat_name = loader.idx_to_category.get(cat_idx, 'Unknown')
            
            if selected_category == 'All' or cat_name == selected_category:
                products_data.append({
                    'Product': prod_name,
                    'Index': prod_idx,
                    'Category': cat_name
                })
        
        products_df = pd.DataFrame(products_data)
        st.dataframe(products_df, use_container_width=True, hide_index=True)
    
    # ===== TAB 3: Categories =====
    with tab3:
        st.subheader("Category Overview")
        
        categories_data = []
        for cat_name, cat_idx in sorted(loader.category_to_idx.items(), key=lambda x: x[1]):
            n_products = sum(1 for cat in loader.product_category_dict.values() if cat == cat_idx)
            
            categories_data.append({
                'Index': cat_idx,
                'Category': cat_name,
                'Products': n_products
            })
        
        categories_df = pd.DataFrame(categories_data)
        st.dataframe(categories_df, use_container_width=True, hide_index=True)
        
        # Bar chart
        fig = px.bar(
            categories_df.sort_values('Products', ascending=False),
            x='Category',
            y='Products',
            title='Products per Category',
            color='Products',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Load resources
    trainer, loader = load_trained_model()
    metrics_df = load_model_metrics()
    stats = load_dataset_stats(loader)
    
    # Render sidebar and get selected page
    page = render_sidebar(loader)
    
    # Render selected page
    if page == "🏠 Home":
        render_home_page(loader, stats)
    
    elif page == "🔮 Predictions":
        render_predictions_page(trainer, loader)
    
    elif page == "📊 Model Performance":
        render_performance_page(metrics_df)
    
    elif page == "📈 Data Explorer":
        render_data_explorer(loader)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Multi-Head DiffNet++ Recommendation System | "
        "Powered by PyTorch & Streamlit | "
        "© 2025 Your Team"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()