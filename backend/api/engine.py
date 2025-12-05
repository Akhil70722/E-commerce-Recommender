import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from django.conf import settings
from django.core.cache import cache
from django.db.models import Count, Avg, Q
from .models import (
    Product, UserInteraction, User, BrowsingHistory, 
    SearchHistory, Wishlist
)


class RecommendationEngine:
    """
    Advanced Real-Time Recommendation Engine with Enhanced Accuracy
    
    Features:
    - User-based and Item-based Collaborative Filtering
    - Enhanced Content-Based Filtering
    - Matrix Factorization (SVD)
    - Diversity and Popularity Bias Correction
    - Advanced Time Decay
    - Category Diversity
    - Price Range Matching
    - Cold-Start Handling
    - Sophisticated Hybrid Combination
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=300,  # Increased for better feature extraction
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better matching
            min_df=1,
            max_df=0.95  # Ignore very common words
        )
        self.svd_model = None
        self.model_cache_key = 'recommendation_model_cache'
        self.last_training_time = None
        
    def train_model(self, force_retrain=False):
        """Train the recommendation model with current data"""
        cache_key = 'model_training_lock'
        
        if not force_retrain:
            last_training = cache.get('last_model_training')
            if last_training:
                time_diff = datetime.now() - last_training
                if time_diff < timedelta(hours=1):
                    return True
        
        if cache.get(cache_key):
            return False
        
        try:
            cache.set(cache_key, True, timeout=300)
            
            print("Training recommendation model...")
            
            interactions = UserInteraction.objects.select_related('user', 'product').all()
            browsing_data = BrowsingHistory.objects.select_related('user', 'product').all()
            wishlist_data = Wishlist.objects.select_related('user', 'product').all()
            
            if not interactions.exists():
                print("No interactions found. Model training skipped.")
                return False
            
            matrix = self._build_comprehensive_matrix(interactions, browsing_data, wishlist_data)
            
            if matrix.empty or matrix.shape[0] < 2:
                print("Insufficient data for model training.")
                return False
            
            # Train SVD with optimal components
            n_components = min(50, max(10, matrix.shape[0] - 1), max(10, matrix.shape[1] - 1))
            if n_components > 0:
                self.svd_model = TruncatedSVD(n_components=n_components, random_state=42, n_iter=10)
                self.svd_model.fit(matrix.values)
                
                model_data = {
                    'svd_model': self.svd_model,
                    'matrix': matrix,
                    'trained_at': datetime.now().isoformat()
                }
                cache.set(self.model_cache_key, model_data, timeout=3600)
            
            self._train_content_model()
            cache.set('last_model_training', datetime.now(), timeout=3600)
            self.last_training_time = datetime.now()
            
            print("Model training completed successfully.")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
        finally:
            cache.delete(cache_key)
    
    def _build_comprehensive_matrix(self, interactions, browsing_data, wishlist_data):
        """Build comprehensive user-item interaction matrix with enhanced weighting"""
        data = []
        
        for interaction in interactions:
            weight = self._get_interaction_weight(
                interaction.interaction_type, 
                interaction.rating,
                interaction.timestamp
            )
            data.append({
                'user_id': interaction.user.id,
                'product_id': interaction.product.id,
                'weight': weight
            })
        
        for browsing in browsing_data:
            time_weight = self._calculate_time_weight(browsing.timestamp, browsing.time_spent)
            data.append({
                'user_id': browsing.user.id,
                'product_id': browsing.product.id,
                'weight': time_weight
            })
        
        for wishlist_item in wishlist_data:
            data.append({
                'user_id': wishlist_item.user.id,
                'product_id': wishlist_item.product.id,
                'weight': 3.5
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
    
    def _calculate_time_weight(self, timestamp, time_spent):
        """Enhanced time weight calculation with exponential decay"""
        days_ago = (datetime.now(timestamp.tzinfo) - timestamp).days
        
        # Exponential decay: e^(-λt) where λ = 1/30
        recency_factor = np.exp(-days_ago / 30.0)
        recency_factor = max(0.1, recency_factor)  # Minimum 10%
        
        # Time spent factor with logarithmic scaling
        time_factor = min(1.0, np.log1p(time_spent) / np.log1p(300))  # Log scale for time
        
        return 0.6 * recency_factor * time_factor
    
    def _get_interaction_weight(self, interaction_type, rating=None, timestamp=None):
        """Enhanced interaction weights with time decay"""
        base_weights = {
            'view': 1.0,
            'cart': 3.0,  # Increased from 2.5
            'purchase': 6.0,  # Increased from 5.0
            'rating': rating * 2.5 if rating else 1.0  # Increased multiplier
        }
        base_weight = base_weights.get(interaction_type, 1.0)
        
        if timestamp:
            days_ago = (datetime.now(timestamp.tzinfo) - timestamp).days
            # Exponential decay with longer tail
            recency_factor = np.exp(-days_ago / 60.0)
            recency_factor = max(0.2, recency_factor)  # Minimum 20%
            return base_weight * recency_factor
        
        return base_weight
    
    def _train_content_model(self):
        """Train content-based model with enhanced features"""
        products = Product.objects.all()
        
        if not products.exists():
            return
        
        product_features = []
        for product in products:
            # Enhanced feature extraction
            price_range = self._get_price_range(product.price)
            features = (
                f"{product.name} {product.description} "
                f"{' '.join(product.get_tags_list())} {product.category.name} "
                f"price_{price_range}"
            )
            product_features.append(features)
        
        try:
            self.vectorizer.fit(product_features)
            cache.set('content_vectorizer', self.vectorizer, timeout=3600)
        except Exception as e:
            print(f"Error training content model: {e}")
    
    def _get_price_range(self, price):
        """Categorize price into ranges for better matching"""
        price_float = float(price)
        if price_float < 25:
            return "budget"
        elif price_float < 100:
            return "mid_range"
        elif price_float < 500:
            return "premium"
        else:
            return "luxury"
    
    def get_user_item_matrix(self, use_cache=True):
        """Get user-item matrix with caching"""
        if use_cache:
            cached_model = cache.get(self.model_cache_key)
            if cached_model and 'matrix' in cached_model:
                return cached_model['matrix']
        
        interactions = UserInteraction.objects.select_related('user', 'product').all()
        browsing_data = BrowsingHistory.objects.select_related('user', 'product').all()
        wishlist_data = Wishlist.objects.select_related('user', 'product').all()
        
        return self._build_comprehensive_matrix(interactions, browsing_data, wishlist_data)
    
    def _get_user_demographic_similarity(self, user_id, other_user_id):
        """Enhanced demographic similarity calculation"""
        try:
            user = User.objects.get(id=user_id)
            other_user = User.objects.get(id=other_user_id)
            
            similarity = 0.0
            factors = 0
            weights = []
            
            if user.age and other_user.age:
                age_diff = abs(user.age - other_user.age)
                age_similarity = np.exp(-age_diff / 15.0)  # Exponential decay
                similarity += age_similarity * 0.4  # Weighted
                weights.append(0.4)
                factors += 1
            
            if user.gender and other_user.gender:
                if user.gender.lower() == other_user.gender.lower():
                    similarity += 1.0 * 0.3
                    weights.append(0.3)
                factors += 1
            
            if user.location and other_user.location:
                if user.location.lower() == other_user.location.lower():
                    similarity += 1.0 * 0.3
                    weights.append(0.3)
                factors += 1
            
            total_weight = sum(weights) if weights else 1.0
            return similarity / total_weight if total_weight > 0 else 0.0
        except User.DoesNotExist:
            return 0.0
    
    def _get_popularity_bias_correction(self, product_id):
        """Correct for popularity bias - boost less popular items slightly"""
        try:
            # Get interaction count for product
            interaction_count = UserInteraction.objects.filter(product_id=product_id).count()
            browsing_count = BrowsingHistory.objects.filter(product_id=product_id).count()
            total_interactions = interaction_count + browsing_count
            
            # Get average interactions per product
            avg_interactions = UserInteraction.objects.aggregate(
                avg=Avg('product__interactions__count')
            )['avg'] or 1
            
            # If product is less popular, give small boost
            if total_interactions < avg_interactions * 0.5:
                return 1.1  # 10% boost for less popular items
            elif total_interactions > avg_interactions * 2:
                return 0.95  # 5% penalty for very popular items
            
            return 1.0
        except:
            return 1.0
    
    def collaborative_filtering(self, user_id, n_recommendations=10, use_svd=True):
        """Enhanced collaborative filtering with item-based support"""
        matrix = self.get_user_item_matrix()
        
        if matrix.empty or user_id not in matrix.index:
            return []
        
        # User-based collaborative filtering
        if use_svd and self.svd_model:
            try:
                cached_model = cache.get(self.model_cache_key)
                if cached_model and 'svd_model' in cached_model:
                    svd_model = cached_model['svd_model']
                    matrix_transformed = svd_model.transform(matrix.values)
                    user_index = matrix.index.get_loc(user_id)
                    user_vector = matrix_transformed[user_index:user_index+1]
                    similarities = cosine_similarity(user_vector, matrix_transformed)[0]
                else:
                    similarities = cosine_similarity([matrix.loc[user_id]], matrix.values)[0]
            except Exception as e:
                print(f"SVD transformation failed: {e}")
                similarities = cosine_similarity([matrix.loc[user_id]], matrix.values)[0]
        else:
            similarities = cosine_similarity([matrix.loc[user_id]], matrix.values)[0]
        
        # Enhanced demographic similarity
        for idx, other_user_id in enumerate(matrix.index):
            if other_user_id != user_id:
                demo_similarity = self._get_user_demographic_similarity(user_id, other_user_id)
                similarities[idx] = 0.8 * similarities[idx] + 0.2 * demo_similarity
        
        # Get top similar users (increased from 10 to 15)
        similar_users_idx = np.argsort(similarities)[::-1][1:min(16, len(similarities))]
        similar_users = matrix.index[similar_users_idx]
        
        user_products = set(matrix.columns[matrix.loc[user_id] > 0])
        recommendations = {}
        
        for similar_user_id in similar_users:
            similar_user_products = matrix.loc[similar_user_id]
            similarity_score = similarities[matrix.index.get_loc(similar_user_id)]
            
            # Only consider users with similarity > 0.1
            if similarity_score < 0.1:
                continue
            
            for product_id in similar_user_products[similar_user_products > 0].index:
                if product_id not in user_products:
                    if product_id not in recommendations:
                        recommendations[product_id] = 0
                    
                    # Enhanced scoring with popularity correction
                    product_score = similar_user_products[product_id] * similarity_score
                    popularity_correction = self._get_popularity_bias_correction(product_id)
                    recommendations[product_id] += product_score * popularity_correction
        
        # Item-based collaborative filtering (complement to user-based)
        item_based_recs = self._item_based_collaborative_filtering(user_id, matrix, user_products, n_recommendations // 2)
        
        # Merge item-based recommendations
        for product_id, score in item_based_recs:
            if product_id not in recommendations:
                recommendations[product_id] = 0
            recommendations[product_id] += score * 0.3  # 30% weight for item-based
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [product_id for product_id, score in sorted_recommendations[:n_recommendations]]
    
    def _item_based_collaborative_filtering(self, user_id, matrix, user_products, n_recommendations=5):
        """Item-based collaborative filtering - finds similar items to user's items"""
        if user_id not in matrix.index or len(user_products) == 0:
            return []
        
        # Get user's product vector
        user_vector = matrix.loc[user_id]
        user_product_ids = [pid for pid in user_products if pid in matrix.columns]
        
        if not user_product_ids:
            return []
        
        # Calculate item-item similarity matrix (transpose)
        item_matrix = matrix.T  # Items as rows, users as columns
        
        recommendations = {}
        for product_id in user_product_ids:
            if product_id not in item_matrix.index:
                continue
            
            # Find items similar to this product
            product_vector = item_matrix.loc[product_id:product_id]
            item_similarities = cosine_similarity(product_vector, item_matrix)[0]
            
            # Get top similar items
            similar_items_idx = np.argsort(item_similarities)[::-1][1:min(6, len(item_similarities))]
            
            for idx in similar_items_idx:
                similar_product_id = item_matrix.index[idx]
                if similar_product_id not in user_products and item_similarities[idx] > 0.2:
                    if similar_product_id not in recommendations:
                        recommendations[similar_product_id] = 0
                    recommendations[similar_product_id] += item_similarities[idx] * user_vector[product_id]
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]
    
    def content_based_filtering(self, user_id, n_recommendations=10):
        """Enhanced content-based filtering with price range matching"""
        thirty_days_ago = datetime.now() - timedelta(days=30)
        
        user_interactions = UserInteraction.objects.filter(
            user_id=user_id,
            timestamp__gte=thirty_days_ago
        ).select_related('product')
        
        user_browsing = BrowsingHistory.objects.filter(
            user_id=user_id,
            timestamp__gte=thirty_days_ago
        ).select_related('product')
        
        user_searches = SearchHistory.objects.filter(
            user_id=user_id,
            timestamp__gte=thirty_days_ago
        ).order_by('-timestamp')[:10]
        
        user_wishlist = Wishlist.objects.filter(user_id=user_id).select_related('product')
        
        # Get user's price preferences
        user_prices = []
        for interaction in user_interactions:
            user_prices.append(float(interaction.product.price))
        for browsing in user_browsing:
            user_prices.append(float(browsing.product.price))
        
        avg_user_price = np.mean(user_prices) if user_prices else None
        price_std = np.std(user_prices) if len(user_prices) > 1 else None
        
        user_product_ids = set()
        user_products = []
        product_weights = {}
        
        for interaction in user_interactions:
            if interaction.product_id not in user_product_ids:
                user_product_ids.add(interaction.product_id)
                user_products.append(interaction.product)
                weight = self._get_interaction_weight(
                    interaction.interaction_type,
                    interaction.rating,
                    interaction.timestamp
                )
                product_weights[interaction.product_id] = weight
        
        for browsing in user_browsing:
            if browsing.product_id not in user_product_ids:
                user_product_ids.add(browsing.product_id)
                user_products.append(browsing.product)
                product_weights[browsing.product_id] = self._calculate_time_weight(
                    browsing.timestamp, browsing.time_spent
                )
        
        for wishlist_item in user_wishlist:
            if wishlist_item.product_id not in user_product_ids:
                user_product_ids.add(wishlist_item.product_id)
                user_products.append(wishlist_item.product)
                product_weights[wishlist_item.product_id] = 3.5
        
        if not user_products:
            return []
        
        vectorizer = cache.get('content_vectorizer')
        if not vectorizer:
            self._train_content_model()
            vectorizer = self.vectorizer
        
        all_products = Product.objects.all()
        product_features = []
        product_ids = []
        search_queries = [search.query.lower() for search in user_searches]
        
        for product in all_products:
            price_range = self._get_price_range(product.price)
            features = (
                f"{product.name} {product.description} "
                f"{' '.join(product.get_tags_list())} {product.category.name} "
                f"price_{price_range}"
            )
            
            # Enhanced search query boost
            for query in search_queries:
                if query in product.name.lower() or query in product.description.lower():
                    features += f" {query} {query} {query} {query}"  # Quadruple boost
            
            product_features.append(features)
            product_ids.append(product.id)
        
        try:
            feature_vectors = vectorizer.transform(product_features)
        except:
            return []
        
        user_product_indices = [product_ids.index(p.id) for p in user_products if p.id in product_ids]
        
        if not user_product_indices:
            return []
        
        # Weighted average with normalization
        user_vectors = feature_vectors[user_product_indices].toarray()
        weights = np.array([product_weights.get(p.id, 1.0) for p in user_products if p.id in product_ids])
        weights = weights / (weights.sum() + 1e-10)  # Normalize with epsilon
        
        user_profile = np.average(user_vectors, axis=0, weights=weights).reshape(1, -1)
        feature_array = feature_vectors.toarray()
        similarities = cosine_similarity(user_profile, feature_array)[0]
        
        # Apply price range matching boost
        if avg_user_price and price_std:
            for idx, product_id in enumerate(product_ids):
                if product_id not in user_product_ids:
                    product = all_products[idx]
                    product_price = float(product.price)
                    
                    # Boost if price is within user's range
                    if abs(product_price - avg_user_price) <= price_std * 1.5:
                        similarities[idx] *= 1.15  # 15% boost
                    elif abs(product_price - avg_user_price) <= price_std * 2.5:
                        similarities[idx] *= 1.05  # 5% boost
        
        # Exclude already interacted products
        for idx, product_id in enumerate(product_ids):
            if product_id in user_product_ids:
                similarities[idx] = -1
        
        # Get top N with minimum similarity threshold
        min_similarity = 0.1  # Minimum similarity threshold
        top_indices = np.argsort(similarities)[::-1]
        recommendations = [
            product_ids[idx] for idx in top_indices 
            if similarities[idx] > min_similarity
        ][:n_recommendations]
        
        return recommendations
    
    def get_cross_sell_recommendations(self, user_id, n_recommendations=4):
        """Enhanced cross-sell with confidence scoring"""
        from django.db.models import Count
        
        ninety_days_ago = datetime.now() - timedelta(days=90)
        purchases = UserInteraction.objects.filter(
            user_id=user_id,
            interaction_type='purchase',
            timestamp__gte=ninety_days_ago
        ).select_related('product')
        
        if not purchases.exists():
            return []
        
        purchased_product_ids = [p.product_id for p in purchases]
        
        # Find users who bought same products with confidence scoring
        co_purchased_users = UserInteraction.objects.filter(
            interaction_type='purchase',
            product_id__in=purchased_product_ids,
            timestamp__gte=ninety_days_ago
        ).exclude(user_id=user_id).values('user_id').annotate(
            count=Count('user_id')
        ).filter(count__gte=1).order_by('-count')[:30]  # Increased from 20
        
        user_ids = [item['user_id'] for item in co_purchased_users]
        
        if not user_ids:
            return []
        
        # Get products with confidence scores
        recommendations = UserInteraction.objects.filter(
            user_id__in=user_ids,
            interaction_type='purchase',
            timestamp__gte=ninety_days_ago
        ).exclude(
            product_id__in=purchased_product_ids
        ).values('product_id').annotate(
            count=Count('product_id')
        ).order_by('-count')[:n_recommendations * 2]  # Get more for filtering
        
        # Filter by minimum co-occurrence (at least 2 users)
        filtered_recs = [rec['product_id'] for rec in recommendations if rec['count'] >= 2]
        
        return filtered_recs[:n_recommendations]
    
    def _ensure_diversity(self, recommendations, max_per_category=3):
        """Ensure category diversity in recommendations"""
        if not recommendations:
            return recommendations
        
        # Get product categories
        product_ids = [rec[0] if isinstance(rec, tuple) else rec for rec in recommendations]
        products = Product.objects.filter(id__in=product_ids).select_related('category')
        
        product_category_map = {p.id: p.category.name for p in products}
        
        diverse_recs = []
        category_counts = {}
        
        for rec in recommendations:
            product_id = rec[0] if isinstance(rec, tuple) else rec
            category = product_category_map.get(product_id, 'Unknown')
            
            if category_counts.get(category, 0) < max_per_category:
                diverse_recs.append(rec)
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Fill remaining slots with any category if needed
        if len(diverse_recs) < len(recommendations):
            for rec in recommendations:
                if rec not in diverse_recs:
                    diverse_recs.append(rec)
                    if len(diverse_recs) >= len(recommendations):
                        break
        
        return diverse_recs
    
    def hybrid_recommendation(self, user_id, n_recommendations=20, real_time=True):
        """
        Enhanced hybrid recommendation with improved accuracy
        """
        if real_time:
            try:
                self.train_model(force_retrain=False)
            except:
                pass
        
        # Get recommendations from all methods
        cf_recommendations = self.collaborative_filtering(user_id, n_recommendations * 3, use_svd=True)
        cb_recommendations = self.content_based_filtering(user_id, n_recommendations * 3)
        cross_sell = self.get_cross_sell_recommendations(user_id, n_recommendations)
        
        # Get recent search queries
        seven_days_ago = datetime.now() - timedelta(days=7)
        user_searches = SearchHistory.objects.filter(
            user_id=user_id,
            timestamp__gte=seven_days_ago
        ).order_by('-timestamp')[:5]
        search_queries = [search.query.lower() for search in user_searches]
        
        # Enhanced scoring with normalization
        all_recommendations = {}
        max_cf_score = len(cf_recommendations) if cf_recommendations else 1
        max_cb_score = len(cb_recommendations) if cb_recommendations else 1
        max_cs_score = len(cross_sell) if cross_sell else 1
        
        # Collaborative filtering (45% weight - increased)
        for idx, product_id in enumerate(cf_recommendations):
            # Normalized position score
            normalized_score = 1 - (idx / max_cf_score)
            score = 0.45 * normalized_score
            if product_id not in all_recommendations:
                all_recommendations[product_id] = 0
            all_recommendations[product_id] += score
        
        # Content-based (38% weight - increased)
        for idx, product_id in enumerate(cb_recommendations):
            normalized_score = 1 - (idx / max_cb_score)
            score = 0.38 * normalized_score
            if product_id not in all_recommendations:
                all_recommendations[product_id] = 0
            all_recommendations[product_id] += score
        
        # Cross-sell (12% weight)
        for idx, product_id in enumerate(cross_sell):
            normalized_score = 1 - (idx / max_cs_score)
            score = 0.12 * normalized_score
            if product_id not in all_recommendations:
                all_recommendations[product_id] = 0
            all_recommendations[product_id] += score
        
        # Search boost (5% weight - but can be significant)
        if search_queries:
            matching_products = Product.objects.filter(
                id__in=list(all_recommendations.keys())
            )
            for product in matching_products:
                for query in search_queries:
                    if query in product.name.lower() or query in product.description.lower():
                        all_recommendations[product.id] += 0.20  # Significant boost
                        break
        
        # Apply popularity bias correction to final scores
        for product_id in all_recommendations:
            popularity_correction = self._get_popularity_bias_correction(product_id)
            all_recommendations[product_id] *= popularity_correction
        
        # Sort by score
        sorted_recommendations = sorted(
            all_recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Ensure diversity
        diverse_recs = self._ensure_diversity(
            sorted_recommendations[:n_recommendations * 2],  # Get more for diversity
            max_per_category=4
        )
        
        # Normalize final scores to 0-1 range
        if diverse_recs:
            max_score = max(score for _, score in diverse_recs)
            if max_score > 0:
                normalized_recs = [
                    (product_id, score / max_score) 
                    for product_id, score in diverse_recs[:n_recommendations]
                ]
                return normalized_recs
        
        return [(product_id, score) for product_id, score in sorted_recommendations[:n_recommendations]]
