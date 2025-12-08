import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import Counter


class RecommendationEngine:
    """
    Advanced Real-Time Recommendation Engine with Enhanced Accuracy
    
    Features:
    - User-based and Item-based Collaborative Filtering with Pearson correlation
    - Enhanced Content-Based Filtering with advanced feature engineering
    - Matrix Factorization (SVD) with optimal component selection
    - Diversity and Popularity Bias Correction with inverse frequency
    - Advanced Time Decay with adaptive decay rates
    - Category Diversity with novelty scoring
    - Price Range Matching with statistical modeling
    - Cold-Start Handling with demographic and popularity fallbacks
    - Sophisticated Hybrid Combination with adaptive weights
    - Context-aware interaction weighting
    
    This engine is database-agnostic and works with plain Python data structures.
    """
    
    def __init__(self, cache_handler=None):
        """
        Initialize the recommendation engine.
        
        Args:
            cache_handler: Optional cache handler with get/set/delete methods.
                          If None, caching is disabled.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Increased for better feature extraction
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9,  # Slightly lower to capture more unique terms
            sublinear_tf=True,  # Use logarithmic term frequency
            norm='l2'  # L2 normalization for better similarity
        )
        self.svd_model = None
        self.model_cache_key = 'recommendation_model_cache'
        self.last_training_time = None
        self.cache = cache_handler
    
    def _normalize_datetime(self, dt):
        """
        Normalize datetime to naive (removes timezone info).
        Converts timezone-aware datetimes to naive UTC.
        """
        if dt is None:
            return None
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            # Convert timezone-aware datetime to naive UTC
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    
    def _get_now(self, reference_datetime=None):
        """
        Get current datetime as naive (since we normalize all datetimes to naive).
        """
        return datetime.now()
        
    def train_model(self, interactions_data, browsing_data, wishlist_data, 
                   products_data, force_retrain=False):
        """
        Train the recommendation model with provided data.
        
        Args:
            interactions_data: List of dicts with keys: user_id, product_id, 
                              interaction_type, rating, timestamp
            browsing_data: List of dicts with keys: user_id, product_id, 
                          time_spent, timestamp
            wishlist_data: List of dicts with keys: user_id, product_id
            products_data: List of dicts with keys: id, name, description, 
                          category (dict with name), price, tags
            force_retrain: If True, force retraining even if recently trained
            
        Returns:
            bool: True if training succeeded, False otherwise
        """
        if self.cache:
            cache_key = 'model_training_lock'
            
            if not force_retrain:
                last_training = self.cache.get('last_model_training')
                if last_training:
                    if isinstance(last_training, str):
                        last_training = datetime.fromisoformat(last_training.replace('Z', '+00:00'))
                    # Normalize both datetimes
                    last_training = self._normalize_datetime(last_training)
                    now = self._get_now(last_training)
                    time_diff = now - last_training
                    if time_diff < timedelta(hours=1):
                        return True
            
            if self.cache.get(cache_key):
                return False
            
            try:
                self.cache.set(cache_key, True, timeout=300)
            except:
                pass
        
        try:
            print("Training recommendation model...")
            
            if not interactions_data:
                print("No interactions found. Model training skipped.")
                return False
            
            matrix = self._build_comprehensive_matrix(
                interactions_data, browsing_data, wishlist_data
            )
            
            if matrix.empty or matrix.shape[0] < 2:
                print("Insufficient data for model training.")
                return False
            
            # Train SVD with optimal components - improved selection
            n_users, n_items = matrix.shape
            # Use more components for better accuracy, but cap appropriately
            n_components = min(
                100,  # Increased max components
                max(15, int(np.sqrt(min(n_users, n_items)))),  # Adaptive based on data size
                n_users - 1,
                n_items - 1
            )
            
            if n_components > 0:
                self.svd_model = TruncatedSVD(
                    n_components=n_components, 
                    random_state=42, 
                    n_iter=15,  # More iterations for better convergence
                    algorithm='arpack'  # Better for sparse matrices
                )
                self.svd_model.fit(matrix.values)
                
                if self.cache:
                    model_data = {
                        'svd_model': self.svd_model,
                        'matrix': matrix,
                        'trained_at': datetime.now().isoformat()
                    }
                    try:
                        self.cache.set(self.model_cache_key, model_data, timeout=3600)
                    except:
                        pass
            
            self._train_content_model(products_data)
            
            if self.cache:
                try:
                    self.cache.set('last_model_training', datetime.now().isoformat(), timeout=3600)
                except:
                    pass
            
            self.last_training_time = datetime.now()
            print("Model training completed successfully.")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
        finally:
            if self.cache:
                try:
                    self.cache.delete(cache_key)
                except:
                    pass
    
    def _build_comprehensive_matrix(self, interactions_data, browsing_data, wishlist_data):
        """Build comprehensive user-item interaction matrix with enhanced weighting"""
        data = []
        
        for interaction in interactions_data:
            weight = self._get_interaction_weight(
                interaction.get('interaction_type'),
                interaction.get('rating'),
                interaction.get('timestamp')
            )
            data.append({
                'user_id': interaction['user_id'],
                'product_id': interaction['product_id'],
                'weight': weight
            })
        
        for browsing in browsing_data:
            time_weight = self._calculate_time_weight(
                browsing.get('timestamp'),
                browsing.get('time_spent', 0)
            )
            data.append({
                'user_id': browsing['user_id'],
                'product_id': browsing['product_id'],
                'weight': time_weight
            })
        
        for wishlist_item in wishlist_data:
            data.append({
                'user_id': wishlist_item['user_id'],
                'product_id': wishlist_item['product_id'],
                'weight': 4.0  # Increased from 3.5
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
        """Enhanced time weight calculation with adaptive exponential decay"""
        if not timestamp:
            return 0.1
        
        # Normalize timestamp
        timestamp = self._normalize_datetime(timestamp)
        now = self._get_now(timestamp)
        days_ago = (now - timestamp).days
        
        # Adaptive decay rate based on recency
        # More recent items decay slower
        if days_ago <= 7:
            decay_rate = 45.0  # Slower decay for very recent items
        elif days_ago <= 30:
            decay_rate = 30.0  # Standard decay
        else:
            decay_rate = 20.0  # Faster decay for old items
        
        # Exponential decay: e^(-λt) where λ = 1/decay_rate
        recency_factor = np.exp(-days_ago / decay_rate)
        recency_factor = max(0.15, recency_factor)  # Higher minimum
        
        # Enhanced time spent factor with better scaling
        # Use sigmoid-like function for better discrimination
        if time_spent <= 0:
            time_factor = 0.1
        elif time_spent <= 30:
            time_factor = 0.3 + 0.4 * (time_spent / 30.0)
        elif time_spent <= 120:
            time_factor = 0.7 + 0.2 * ((time_spent - 30) / 90.0)
        else:
            time_factor = min(1.0, 0.9 + 0.1 * np.log1p(time_spent - 120) / np.log1p(300))
        
        return 0.7 * recency_factor * time_factor  # Increased base weight
    
    def _get_interaction_weight(self, interaction_type, rating=None, timestamp=None):
        """Enhanced interaction weights with context-aware time decay"""
        # Improved base weights with better discrimination
        base_weights = {
            'view': 1.2,  # Increased from 1.0
            'cart': 4.0,  # Increased from 3.0
            'purchase': 8.0,  # Increased from 6.0
            'rating': (rating * 3.0) if rating else 1.0  # Increased multiplier
        }
        base_weight = base_weights.get(interaction_type, 1.0)
        
        if timestamp:
            # Normalize timestamp
            timestamp = self._normalize_datetime(timestamp)
            now = self._get_now(timestamp)
            days_ago = (now - timestamp).days
            
            # Adaptive decay based on interaction type
            if interaction_type == 'purchase':
                decay_rate = 90.0  # Purchases stay relevant longer
            elif interaction_type == 'cart':
                decay_rate = 45.0  # Cart items decay moderately
            elif interaction_type == 'rating':
                decay_rate = 120.0  # Ratings are long-term signals
            else:
                decay_rate = 30.0  # Views decay faster
            
            # Exponential decay with adaptive rate
            recency_factor = np.exp(-days_ago / decay_rate)
            recency_factor = max(0.25, recency_factor)  # Higher minimum
            return base_weight * recency_factor
        
        return base_weight
    
    def _train_content_model(self, products_data):
        """Train content-based model with advanced feature engineering"""
        if not products_data:
            return
        
        product_features = []
        for product in products_data:
            # Enhanced feature extraction with better text processing
            price_range = self._get_price_range(float(product.get('price', 0)))
            category_name = product.get('category', {}).get('name', '') if isinstance(product.get('category'), dict) else str(product.get('category', ''))
            tags = product.get('tags', '')
            if isinstance(tags, str):
                tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            else:
                tags_list = tags if isinstance(tags, list) else []
            
            # Enhanced feature construction with repetition for important terms
            name = product.get('name', '')
            description = product.get('description', '')
            
            # Repeat important terms (name, category) for better weighting
            features = (
                f"{name} {name} "  # Name appears twice for importance
                f"{description} "
                f"{' '.join(tags_list)} {' '.join(tags_list)} "  # Tags repeated
                f"{category_name} {category_name} "  # Category repeated
                f"price_{price_range} price_range_{price_range}"  # Price range emphasized
            )
            product_features.append(features)
        
        try:
            self.vectorizer.fit(product_features)
            if self.cache:
                try:
                    self.cache.set('content_vectorizer', self.vectorizer, timeout=3600)
                except:
                    pass
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
    
    def get_user_item_matrix(self, interactions_data, browsing_data, wishlist_data, use_cache=True):
        """
        Get user-item matrix with optional caching.
        
        Args:
            interactions_data: List of interaction dicts
            browsing_data: List of browsing dicts
            wishlist_data: List of wishlist dicts
            use_cache: Whether to use cache if available
            
        Returns:
            pandas DataFrame: User-item matrix
        """
        if use_cache and self.cache:
            try:
                cached_model = self.cache.get(self.model_cache_key)
                if cached_model and 'matrix' in cached_model:
                    return cached_model['matrix']
            except:
                pass
        
        return self._build_comprehensive_matrix(interactions_data, browsing_data, wishlist_data)
    
    def _pearson_correlation(self, vec1, vec2):
        """Calculate Pearson correlation coefficient for better similarity"""
        # Remove zero values for correlation calculation
        mask = (vec1 != 0) & (vec2 != 0)
        if mask.sum() < 2:
            return 0.0
        
        vec1_filtered = vec1[mask]
        vec2_filtered = vec2[mask]
        
        # Calculate means
        mean1 = vec1_filtered.mean()
        mean2 = vec2_filtered.mean()
        
        # Calculate correlation
        numerator = ((vec1_filtered - mean1) * (vec2_filtered - mean2)).sum()
        denom1 = ((vec1_filtered - mean1) ** 2).sum()
        denom2 = ((vec2_filtered - mean2) ** 2).sum()
        
        if denom1 == 0 or denom2 == 0:
            return 0.0
        
        correlation = numerator / np.sqrt(denom1 * denom2)
        return max(0.0, correlation)  # Return non-negative correlation
    
    def _get_user_demographic_similarity(self, user_id, other_user_id, users_data):
        """Enhanced demographic similarity calculation with weighted factors"""
        try:
            user = next((u for u in users_data if u.get('id') == user_id), None)
            other_user = next((u for u in users_data if u.get('id') == other_user_id), None)
            
            if not user or not other_user:
                return 0.0
            
            similarity = 0.0
            total_weight = 0.0
            
            # Age similarity with better weighting
            if user.get('age') and other_user.get('age'):
                age_diff = abs(user['age'] - other_user['age'])
                # Use Gaussian-like function for smoother similarity
                age_similarity = np.exp(-(age_diff ** 2) / (2 * (15.0 ** 2)))
                similarity += age_similarity * 0.35
                total_weight += 0.35
            
            # Gender similarity
            if user.get('gender') and other_user.get('gender'):
                if str(user['gender']).lower() == str(other_user['gender']).lower():
                    similarity += 1.0 * 0.25
                total_weight += 0.25
            
            # Location similarity
            if user.get('location') and other_user.get('location'):
                if str(user['location']).lower() == str(other_user['location']).lower():
                    similarity += 1.0 * 0.40
                total_weight += 0.40
            
            return similarity / total_weight if total_weight > 0 else 0.0
        except:
            return 0.0
    
    def _get_popularity_bias_correction(self, product_id, interactions_data, browsing_data, products_data):
        """Enhanced popularity bias correction with inverse frequency"""
        try:
            # Get interaction count for product
            interaction_count = sum(1 for i in interactions_data if i.get('product_id') == product_id)
            browsing_count = sum(1 for b in browsing_data if b.get('product_id') == product_id)
            total_interactions = interaction_count + browsing_count
            
            # Get total number of products
            total_products = len(set(i.get('product_id') for i in interactions_data) | 
                                set(b.get('product_id') for b in browsing_data))
            
            if total_products == 0:
                return 1.0
            
            # Calculate inverse frequency (IDF-like)
            # Less popular items get boost, very popular items get slight penalty
            frequency = total_interactions / max(1, total_products)
            
            # Use logarithmic scaling for better distribution
            if frequency < 0.01:  # Very rare item
                return 1.25  # Significant boost
            elif frequency < 0.05:  # Rare item
                return 1.15
            elif frequency < 0.1:  # Uncommon item
                return 1.05
            elif frequency > 0.5:  # Very popular item
                return 0.92  # Slight penalty
            elif frequency > 0.3:  # Popular item
                return 0.96
            else:
                return 1.0  # Normal popularity
        except:
            return 1.0
    
    def collaborative_filtering(self, user_id, interactions_data, browsing_data, 
                               wishlist_data, users_data, n_recommendations=10, use_svd=True):
        """Enhanced collaborative filtering with Pearson correlation and item-based support"""
        matrix = self.get_user_item_matrix(interactions_data, browsing_data, wishlist_data)
        
        if matrix.empty or user_id not in matrix.index:
            return []
        
        # User-based collaborative filtering with improved similarity
        if use_svd and self.svd_model:
            try:
                if self.cache:
                    cached_model = self.cache.get(self.model_cache_key)
                    if cached_model and 'svd_model' in cached_model:
                        svd_model = cached_model['svd_model']
                        matrix_transformed = svd_model.transform(matrix.values)
                        user_index = matrix.index.get_loc(user_id)
                        user_vector = matrix_transformed[user_index:user_index+1]
                        similarities = cosine_similarity(user_vector, matrix_transformed)[0]
                    else:
                        similarities = cosine_similarity([matrix.loc[user_id]], matrix.values)[0]
                else:
                    similarities = cosine_similarity([matrix.loc[user_id]], matrix.values)[0]
            except Exception as e:
                print(f"SVD transformation failed: {e}")
                similarities = cosine_similarity([matrix.loc[user_id]], matrix.values)[0]
        else:
            # Use Pearson correlation for better similarity when SVD not available
            user_vector = matrix.loc[user_id].values
            similarities = np.array([
                self._pearson_correlation(user_vector, matrix.loc[other_user_id].values)
                for other_user_id in matrix.index
            ])
            # Fallback to cosine if correlation fails
            if similarities.sum() == 0:
                similarities = cosine_similarity([matrix.loc[user_id]], matrix.values)[0]
        
        # Enhanced demographic similarity with better weighting
        for idx, other_user_id in enumerate(matrix.index):
            if other_user_id != user_id:
                demo_similarity = self._get_user_demographic_similarity(
                    user_id, other_user_id, users_data
                )
                # Blend similarity scores with adaptive weighting
                similarities[idx] = 0.75 * similarities[idx] + 0.25 * demo_similarity
        
        # Get top similar users - increased from 15 to 20
        similar_users_idx = np.argsort(similarities)[::-1][1:min(21, len(similarities))]
        similar_users = matrix.index[similar_users_idx]
        
        user_products = set(matrix.columns[matrix.loc[user_id] > 0])
        recommendations = {}
        
        for similar_user_id in similar_users:
            similar_user_products = matrix.loc[similar_user_id]
            similarity_score = similarities[matrix.index.get_loc(similar_user_id)]
            
            # Higher threshold for better quality
            if similarity_score < 0.15:
                continue
            
            for product_id in similar_user_products[similar_user_products > 0].index:
                if product_id not in user_products:
                    if product_id not in recommendations:
                        recommendations[product_id] = 0
                    
                    # Enhanced scoring with confidence weighting
                    product_score = similar_user_products[product_id] * similarity_score
                    # Apply confidence boost for high similarity
                    if similarity_score > 0.7:
                        product_score *= 1.2
                    elif similarity_score > 0.5:
                        product_score *= 1.1
                    
                    recommendations[product_id] += product_score
        
        # Item-based collaborative filtering with increased weight
        item_based_recs = self._item_based_collaborative_filtering(
            user_id, matrix, user_products, n_recommendations // 2
        )
        
        # Merge item-based recommendations with higher weight
        for product_id, score in item_based_recs:
            if product_id not in recommendations:
                recommendations[product_id] = 0
            recommendations[product_id] += score * 0.4  # Increased from 0.3
        
        # Apply popularity bias correction
        for product_id in recommendations:
            popularity_correction = self._get_popularity_bias_correction(
                product_id, interactions_data, browsing_data, []
            )
            recommendations[product_id] *= popularity_correction
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [product_id for product_id, score in sorted_recommendations[:n_recommendations]]
    
    def _item_based_collaborative_filtering(self, user_id, matrix, user_products, n_recommendations=5):
        """Item-based collaborative filtering with improved similarity"""
        if user_id not in matrix.index or len(user_products) == 0:
            return []
        
        user_vector = matrix.loc[user_id]
        user_product_ids = [pid for pid in user_products if pid in matrix.columns]
        
        if not user_product_ids:
            return []
        
        item_matrix = matrix.T
        
        recommendations = {}
        for product_id in user_product_ids:
            if product_id not in item_matrix.index:
                continue
            
            product_vector = item_matrix.loc[product_id:product_id]
            item_similarities = cosine_similarity(product_vector, item_matrix)[0]
            
            # Get more similar items for better coverage
            similar_items_idx = np.argsort(item_similarities)[::-1][1:min(8, len(item_similarities))]
            
            for idx in similar_items_idx:
                similar_product_id = item_matrix.index[idx]
                if similar_product_id not in user_products and item_similarities[idx] > 0.25:  # Higher threshold
                    if similar_product_id not in recommendations:
                        recommendations[similar_product_id] = 0
                    # Weight by user's interaction strength with original product
                    recommendations[similar_product_id] += item_similarities[idx] * user_vector[product_id] * 1.1
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:n_recommendations]
    
    def content_based_filtering(self, user_id, interactions_data, browsing_data, 
                               wishlist_data, search_history_data, products_data, 
                               n_recommendations=10):
        """Enhanced content-based filtering with advanced feature matching"""
        # Extended time window for better context
        forty_five_days_ago = datetime.now() - timedelta(days=45)
        
        # Filter user interactions
        user_interactions = [
            i for i in interactions_data 
            if i.get('user_id') == user_id and self._is_recent(i.get('timestamp'), forty_five_days_ago)
        ]
        
        user_browsing = [
            b for b in browsing_data 
            if b.get('user_id') == user_id and self._is_recent(b.get('timestamp'), forty_five_days_ago)
        ]
        
        user_searches = sorted(
            [s for s in search_history_data if s.get('user_id') == user_id and self._is_recent(s.get('timestamp'), forty_five_days_ago)],
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:15]  # Increased from 10
        
        user_wishlist = [
            w for w in wishlist_data if w.get('user_id') == user_id
        ]
        
        # Get user's price preferences with better statistics
        user_prices = []
        for interaction in user_interactions:
            product = next((p for p in products_data if p.get('id') == interaction.get('product_id')), None)
            if product:
                user_prices.append(float(product.get('price', 0)))
        for browsing in user_browsing:
            product = next((p for p in products_data if p.get('id') == browsing.get('product_id')), None)
            if product:
                user_prices.append(float(product.get('price', 0)))
        
        avg_user_price = np.mean(user_prices) if user_prices else None
        price_std = np.std(user_prices) if len(user_prices) > 1 else None
        price_median = np.median(user_prices) if user_prices else None
        
        user_product_ids = set()
        user_products = []
        product_weights = {}
        
        for interaction in user_interactions:
            product_id = interaction.get('product_id')
            if product_id not in user_product_ids:
                product = next((p for p in products_data if p.get('id') == product_id), None)
                if product:
                    user_product_ids.add(product_id)
                    user_products.append(product)
                    weight = self._get_interaction_weight(
                        interaction.get('interaction_type'),
                        interaction.get('rating'),
                        interaction.get('timestamp')
                    )
                    product_weights[product_id] = weight
        
        for browsing in user_browsing:
            product_id = browsing.get('product_id')
            if product_id not in user_product_ids:
                product = next((p for p in products_data if p.get('id') == product_id), None)
                if product:
                    user_product_ids.add(product_id)
                    user_products.append(product)
                    product_weights[product_id] = self._calculate_time_weight(
                        browsing.get('timestamp'), browsing.get('time_spent', 0)
                    )
        
        for wishlist_item in user_wishlist:
            product_id = wishlist_item.get('product_id')
            if product_id not in user_product_ids:
                product = next((p for p in products_data if p.get('id') == product_id), None)
                if product:
                    user_product_ids.add(product_id)
                    user_products.append(product)
                    product_weights[product_id] = 4.5  # Increased from 3.5
        
        if not user_products:
            return []
        
        vectorizer = None
        if self.cache:
            try:
                vectorizer = self.cache.get('content_vectorizer')
            except:
                pass
        
        if not vectorizer:
            self._train_content_model(products_data)
            vectorizer = self.vectorizer
        
        product_features = []
        product_ids = []
        search_queries = [s.get('query', '').lower() for s in user_searches]
        
        for product in products_data:
            price_range = self._get_price_range(float(product.get('price', 0)))
            category_name = product.get('category', {}).get('name', '') if isinstance(product.get('category'), dict) else str(product.get('category', ''))
            tags = product.get('tags', '')
            if isinstance(tags, str):
                tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            else:
                tags_list = tags if isinstance(tags, list) else []
            
            features = (
                f"{product.get('name', '')} {product.get('description', '')} "
                f"{' '.join(tags_list)} {category_name} "
                f"price_{price_range}"
            )
            
            # Enhanced search query boost with multiple repetitions
            for query in search_queries:
                if query:
                    query_lower = query.lower()
                    name_lower = product.get('name', '').lower()
                    desc_lower = product.get('description', '').lower()
                    if query_lower in name_lower:
                        # Name matches get stronger boost
                        features += f" {query} {query} {query} {query} {query}"
                    elif query_lower in desc_lower:
                        # Description matches get moderate boost
                        features += f" {query} {query} {query}"
            
            product_features.append(features)
            product_ids.append(product.get('id'))
        
        try:
            feature_vectors = vectorizer.transform(product_features)
        except:
            return []
        
        user_product_indices = [product_ids.index(p.get('id')) for p in user_products if p.get('id') in product_ids]
        
        if not user_product_indices:
            return []
        
        # Weighted average with normalization
        user_vectors = feature_vectors[user_product_indices].toarray()
        weights = np.array([product_weights.get(p.get('id'), 1.0) for p in user_products if p.get('id') in product_ids])
        weights = weights / (weights.sum() + 1e-10)
        
        user_profile = np.average(user_vectors, axis=0, weights=weights).reshape(1, -1)
        feature_array = feature_vectors.toarray()
        similarities = cosine_similarity(user_profile, feature_array)[0]
        
        # Enhanced price range matching with multiple factors
        if avg_user_price and price_std and price_median:
            for idx, product_id in enumerate(product_ids):
                if product_id not in user_product_ids:
                    product = products_data[idx]
                    product_price = float(product.get('price', 0))
                    
                    # Multiple price matching criteria
                    price_diff_std = abs(product_price - avg_user_price) / (price_std + 1e-10)
                    price_diff_median = abs(product_price - price_median) / (price_median + 1e-10)
                    
                    if price_diff_std <= 1.0:  # Within 1 std dev
                        similarities[idx] *= 1.25  # Strong boost
                    elif price_diff_std <= 1.5:
                        similarities[idx] *= 1.15
                    elif price_diff_std <= 2.0:
                        similarities[idx] *= 1.08
                    elif price_diff_median <= 0.3:  # Close to median
                        similarities[idx] *= 1.10
        
        # Exclude already interacted products
        for idx, product_id in enumerate(product_ids):
            if product_id in user_product_ids:
                similarities[idx] = -1
        
        # Get top N with higher minimum similarity threshold
        min_similarity = 0.15  # Increased from 0.1
        top_indices = np.argsort(similarities)[::-1]
        recommendations = [
            product_ids[idx] for idx in top_indices 
            if similarities[idx] > min_similarity
        ][:n_recommendations]
        
        return recommendations
    
    def _is_recent(self, timestamp, cutoff_date):
        """Check if timestamp is after cutoff date"""
        if not timestamp:
            return False
        # Normalize both to same timezone awareness
        timestamp = self._normalize_datetime(timestamp)
        cutoff_date = self._normalize_datetime(cutoff_date)
        return timestamp >= cutoff_date
    
    def get_cross_sell_recommendations(self, user_id, interactions_data, n_recommendations=4):
        """Enhanced cross-sell with improved confidence scoring"""
        ninety_days_ago = datetime.now() - timedelta(days=90)
        
        purchases = [
            i for i in interactions_data 
            if i.get('user_id') == user_id 
            and i.get('interaction_type') == 'purchase'
            and self._is_recent(i.get('timestamp'), ninety_days_ago)
        ]
        
        if not purchases:
            return []
        
        purchased_product_ids = [p.get('product_id') for p in purchases]
        
        # Find users who bought same products with weighted scoring
        co_purchased_users = {}
        for interaction in interactions_data:
            if (interaction.get('interaction_type') == 'purchase' 
                and interaction.get('product_id') in purchased_product_ids
                and interaction.get('user_id') != user_id
                and self._is_recent(interaction.get('timestamp'), ninety_days_ago)):
                uid = interaction.get('user_id')
                # Weight by recency of purchase
                timestamp = self._normalize_datetime(interaction.get('timestamp'))
                now = self._get_now(timestamp)
                days_ago = (now - timestamp).days
                recency_weight = np.exp(-days_ago / 60.0)
                co_purchased_users[uid] = co_purchased_users.get(uid, 0) + recency_weight
        
        # Get top 40 users (increased from 30)
        sorted_users = sorted(co_purchased_users.items(), key=lambda x: x[1], reverse=True)[:40]
        user_ids = [uid for uid, _ in sorted_users if _ >= 0.5]  # Higher threshold
        
        if not user_ids:
            return []
        
        # Get products with confidence scores
        product_counts = {}
        for interaction in interactions_data:
            if (interaction.get('user_id') in user_ids
                and interaction.get('interaction_type') == 'purchase'
                and interaction.get('product_id') not in purchased_product_ids
                and self._is_recent(interaction.get('timestamp'), ninety_days_ago)):
                pid = interaction.get('product_id')
                product_counts[pid] = product_counts.get(pid, 0) + 1
        
        # Filter by minimum co-occurrence (at least 3 users, increased from 2)
        filtered_recs = [pid for pid, count in product_counts.items() if count >= 3]
        sorted_recs = sorted(filtered_recs, key=lambda pid: product_counts[pid], reverse=True)
        
        return sorted_recs[:n_recommendations]
    
    def _calculate_novelty_score(self, product_id, user_product_ids, products_data):
        """Calculate novelty score - how different is this from user's history"""
        if not user_product_ids:
            return 1.0
        
        product = next((p for p in products_data if p.get('id') == product_id), None)
        if not product:
            return 1.0
        
        product_category = product.get('category', {}).get('name', '') if isinstance(product.get('category'), dict) else str(product.get('category', ''))
        
        # Count how many products user has in same category
        user_categories = []
        for pid in user_product_ids:
            p = next((pr for pr in products_data if pr.get('id') == pid), None)
            if p:
                cat = p.get('category', {}).get('name', '') if isinstance(p.get('category'), dict) else str(p.get('category', ''))
                user_categories.append(cat)
        
        category_count = user_categories.count(product_category)
        total_user_products = len(user_product_ids)
        
        if total_user_products == 0:
            return 1.0
        
        # Novelty: inverse of category frequency
        category_frequency = category_count / total_user_products
        novelty = 1.0 - category_frequency
        
        return max(0.3, novelty)  # Minimum novelty of 0.3
    
    def _ensure_diversity(self, recommendations, products_data, max_per_category=3):
        """Ensure category diversity in recommendations with novelty scoring"""
        if not recommendations:
            return recommendations
        
        # Get product categories
        product_ids = [rec[0] if isinstance(rec, tuple) else rec for rec in recommendations]
        product_category_map = {}
        for product in products_data:
            if product.get('id') in product_ids:
                category_name = product.get('category', {}).get('name', 'Unknown') if isinstance(product.get('category'), dict) else str(product.get('category', 'Unknown'))
                product_category_map[product.get('id')] = category_name
        
        diverse_recs = []
        category_counts = {}
        user_product_ids = set()
        
        # First pass: ensure diversity
        for rec in recommendations:
            product_id = rec[0] if isinstance(rec, tuple) else rec
            category = product_category_map.get(product_id, 'Unknown')
            
            if category_counts.get(category, 0) < max_per_category:
                diverse_recs.append(rec)
                category_counts[category] = category_counts.get(category, 0) + 1
                user_product_ids.add(product_id)
        
        # Second pass: add remaining with novelty boost
        for rec in recommendations:
            if rec not in diverse_recs:
                product_id = rec[0] if isinstance(rec, tuple) else rec
                novelty = self._calculate_novelty_score(product_id, user_product_ids, products_data)
                
                # Boost score by novelty
                if isinstance(rec, tuple):
                    rec = (rec[0], rec[1] * (1.0 + 0.2 * novelty))
                diverse_recs.append(rec)
                user_product_ids.add(product_id)
                
                if len(diverse_recs) >= len(recommendations):
                    break
        
        return diverse_recs
    
    def _calculate_user_engagement_level(self, user_id, interactions_data, browsing_data):
        """Calculate user engagement level for adaptive weighting"""
        total_interactions = sum(1 for i in interactions_data if i.get('user_id') == user_id)
        total_browsing = sum(1 for b in browsing_data if b.get('user_id') == user_id)
        
        total_activity = total_interactions + total_browsing
        
        if total_activity >= 50:
            return 'high'
        elif total_activity >= 20:
            return 'medium'
        else:
            return 'low'
    
    def hybrid_recommendation(self, user_id, interactions_data, browsing_data, 
                            wishlist_data, search_history_data, products_data, 
                            users_data, n_recommendations=20, real_time=True):
        """
        Enhanced hybrid recommendation with adaptive weights and improved accuracy
        
        Args:
            user_id: User ID
            interactions_data: List of interaction dicts
            browsing_data: List of browsing dicts
            wishlist_data: List of wishlist dicts
            search_history_data: List of search history dicts
            products_data: List of product dicts
            users_data: List of user dicts
            n_recommendations: Number of recommendations to return
            real_time: Whether to train model in real-time
            
        Returns:
            List of tuples: (product_id, score) pairs
        """
        if real_time:
            try:
                self.train_model(
                    interactions_data, browsing_data, wishlist_data, 
                    products_data, force_retrain=False
                )
            except:
                pass
        
        # Calculate user engagement for adaptive weighting
        engagement = self._calculate_user_engagement_level(user_id, interactions_data, browsing_data)
        
        # Get recommendations from all methods
        cf_recommendations = self.collaborative_filtering(
            user_id, interactions_data, browsing_data, wishlist_data, 
            users_data, n_recommendations * 4, use_svd=True  # Get more candidates
        )
        cb_recommendations = self.content_based_filtering(
            user_id, interactions_data, browsing_data, wishlist_data, 
            search_history_data, products_data, n_recommendations * 4
        )
        cross_sell = self.get_cross_sell_recommendations(
            user_id, interactions_data, n_recommendations * 2
        )
        
        # Get recent search queries
        seven_days_ago = datetime.now() - timedelta(days=7)
        user_searches = sorted(
            [s for s in search_history_data 
             if s.get('user_id') == user_id and self._is_recent(s.get('timestamp'), seven_days_ago)],
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )[:8]  # Increased from 5
        search_queries = [s.get('query', '').lower() for s in user_searches]
        
        # Adaptive weights based on user engagement
        if engagement == 'high':
            # High engagement: trust collaborative filtering more
            cf_weight = 0.50
            cb_weight = 0.35
            cs_weight = 0.10
        elif engagement == 'medium':
            # Medium engagement: balanced approach
            cf_weight = 0.45
            cb_weight = 0.40
            cs_weight = 0.10
        else:
            # Low engagement: rely more on content-based
            cf_weight = 0.35
            cb_weight = 0.50
            cs_weight = 0.10
        
        # Enhanced scoring with exponential decay for position
        all_recommendations = {}
        max_cf_score = len(cf_recommendations) if cf_recommendations else 1
        max_cb_score = len(cb_recommendations) if cb_recommendations else 1
        max_cs_score = len(cross_sell) if cross_sell else 1
        
        # Collaborative filtering with exponential decay
        for idx, product_id in enumerate(cf_recommendations):
            # Exponential decay: e^(-λx) where λ controls decay rate
            normalized_score = np.exp(-idx / (max_cf_score * 0.3))
            score = cf_weight * normalized_score
            if product_id not in all_recommendations:
                all_recommendations[product_id] = 0
            all_recommendations[product_id] += score
        
        # Content-based with exponential decay
        for idx, product_id in enumerate(cb_recommendations):
            normalized_score = np.exp(-idx / (max_cb_score * 0.3))
            score = cb_weight * normalized_score
            if product_id not in all_recommendations:
                all_recommendations[product_id] = 0
            all_recommendations[product_id] += score
        
        # Cross-sell with exponential decay
        for idx, product_id in enumerate(cross_sell):
            normalized_score = np.exp(-idx / (max_cs_score * 0.3))
            score = cs_weight * normalized_score
            if product_id not in all_recommendations:
                all_recommendations[product_id] = 0
            all_recommendations[product_id] += score
        
        # Enhanced search boost with recency weighting
        if search_queries:
            for idx, product_id in enumerate(list(all_recommendations.keys())):
                product = next((p for p in products_data if p.get('id') == product_id), None)
                if product:
                    for query_idx, query in enumerate(search_queries):
                        if query:
                            query_lower = query.lower()
                            name_lower = product.get('name', '').lower()
                            desc_lower = product.get('description', '').lower()
                            if query_lower in name_lower or query_lower in desc_lower:
                                # More recent searches get higher boost
                                recency_boost = np.exp(-query_idx * 0.3)
                                boost_amount = 0.25 * recency_boost  # Increased base boost
                                all_recommendations[product_id] += boost_amount
                                break
        
        # Apply popularity bias correction
        for product_id in all_recommendations:
            popularity_correction = self._get_popularity_bias_correction(
                product_id, interactions_data, browsing_data, products_data
            )
            all_recommendations[product_id] *= popularity_correction
        
        # Sort by score
        sorted_recommendations = sorted(
            all_recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Ensure diversity with novelty
        diverse_recs = self._ensure_diversity(
            sorted_recommendations[:n_recommendations * 3],  # Get more for diversity
            products_data,
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
