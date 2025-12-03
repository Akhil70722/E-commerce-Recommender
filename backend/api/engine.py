import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from .models import Product, UserInteraction, User

class RecommendationEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')

    def get_user_item_matrix(self):
        """Create user-item interaction matrix"""
        interactions = UserInteraction.objects.select_related('user', 'product').all()
        
        data = []
        for interaction in interactions:
            weight = self._get_interaction_weight(interaction.interaction_type, interaction.rating)
            data.append({
                'user_id': interaction.user.id,
                'product_id': interaction.product.id,
                'weight': weight
            })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        matrix = df.pivot_table(
            index='user_id',
            columns='product_id',
            values='weight',
            aggfunc='sum',
            fill_value=0
        )
        return matrix

    def _get_interaction_weight(self, interaction_type, rating=None):
        """Assign weights to different interaction types"""
        weights = {
            'view': 1,
            'cart': 2,
            'purchase': 5,
            'rating': rating * 2 if rating else 1
        }
        return weights.get(interaction_type, 1)

    def collaborative_filtering(self, user_id, n_recommendations=10):
        """User-based collaborative filtering"""
        matrix = self.get_user_item_matrix()
        
        if matrix.empty or user_id not in matrix.index:
            return []
        
        # Calculate user similarity
        user_similarities = cosine_similarity([matrix.loc[user_id]], matrix)
        user_similarities = user_similarities[0]
        
        # Get similar users
        similar_users_idx = np.argsort(user_similarities)[::-1][1:6]  # Top 5 similar users
        similar_users = matrix.index[similar_users_idx]
        
        # Get products liked by similar users
        user_products = set(matrix.columns[matrix.loc[user_id] > 0])
        recommendations = {}
        
        for similar_user_id in similar_users:
            similar_user_products = matrix.loc[similar_user_id]
            similarity_score = user_similarities[matrix.index.get_loc(similar_user_id)]
            
            for product_id in similar_user_products[similar_user_products > 0].index:
                if product_id not in user_products:
                    if product_id not in recommendations:
                        recommendations[product_id] = 0
                    recommendations[product_id] += similar_user_products[product_id] * similarity_score
        
        # Sort by score and return top N
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [product_id for product_id, score in sorted_recommendations[:n_recommendations]]

    def content_based_filtering(self, user_id, n_recommendations=10):
        """Content-based filtering using product features"""
        # Get user's interacted products
        user_interactions = UserInteraction.objects.filter(user_id=user_id).select_related('product')
        
        if not user_interactions.exists():
            return []
        
        # Get all products
        all_products = Product.objects.all()
        
        # Create feature vectors from product descriptions and tags
        product_features = []
        product_ids = []
        
        for product in all_products:
            features = f"{product.description} {' '.join(product.get_tags_list())} {product.category.name}"
            product_features.append(features)
            product_ids.append(product.id)
        
        # Vectorize
        try:
            feature_vectors = self.vectorizer.fit_transform(product_features)
        except:
            return []
        
        # Get user's preferred products
        user_product_ids = [interaction.product_id for interaction in user_interactions]
        user_product_indices = [product_ids.index(pid) for pid in user_product_ids if pid in product_ids]
        
        if not user_product_indices:
            return []
        
        # Average user's product vectors (convert to dense NumPy array)
        # feature_vectors is a sparse matrix; slice the rows the user interacted with
        user_vectors = feature_vectors[user_product_indices].toarray()
        # Compute mean vector and ensure it's a 2D NumPy array for sklearn
        user_profile = np.asarray(user_vectors.mean(axis=0)).reshape(1, -1)
        
        # Convert all product vectors to a dense array for cosine_similarity
        feature_array = feature_vectors.toarray()
        
        # Calculate similarity with all products
        similarities = cosine_similarity(user_profile, feature_array)[0]
        
        # Exclude already interacted products
        for idx, product_id in enumerate(product_ids):
            if product_id in user_product_ids:
                similarities[idx] = -1
        
        # Get top N recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        recommendations = [product_ids[idx] for idx in top_indices if similarities[idx] > 0]
        
        return recommendations

    def hybrid_recommendation(self, user_id, n_recommendations=10):
        """Combine collaborative and content-based filtering"""
        cf_recommendations = self.collaborative_filtering(user_id, n_recommendations * 2)
        cb_recommendations = self.content_based_filtering(user_id, n_recommendations * 2)
        
        # Combine and deduplicate
        all_recommendations = {}
        
        # Add collaborative filtering results (weight: 0.6)
        for idx, product_id in enumerate(cf_recommendations):
            score = 0.6 * (1 - idx / len(cf_recommendations)) if cf_recommendations else 0
            if product_id not in all_recommendations:
                all_recommendations[product_id] = 0
            all_recommendations[product_id] += score
        
        # Add content-based results (weight: 0.4)
        for idx, product_id in enumerate(cb_recommendations):
            score = 0.4 * (1 - idx / len(cb_recommendations)) if cb_recommendations else 0
            if product_id not in all_recommendations:
                all_recommendations[product_id] = 0
            all_recommendations[product_id] += score
        
        # Sort by combined score
        sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        return [(product_id, score) for product_id, score in sorted_recommendations[:n_recommendations]]

