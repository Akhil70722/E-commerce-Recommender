"""
CSV Data Loader - Loads data from CSV files instead of database
"""
import os
import csv
import pandas as pd
from pathlib import Path


class CSVDataLoader:
    """Load data from CSV files"""
    
    def __init__(self, csv_folder_path=None):
        """
        Initialize CSV loader.
        
        Args:
            csv_folder_path: Path to CSV folder. If None, uses default location.
        """
        if csv_folder_path is None:
            # Default to dataset folder in project root
            base_dir = Path(__file__).resolve().parent.parent.parent
            csv_folder_path = base_dir / 'dataset'
        
        self.csv_folder = Path(csv_folder_path)
        if not self.csv_folder.exists():
            raise FileNotFoundError(f"CSV folder not found: {csv_folder_path}")
    
    def _safe_int(self, value, default=0):
        """Safely convert value to int, handling None and empty strings"""
        if value is None or value == '':
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float, handling None and empty strings"""
        if value is None or value == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def load_categories(self):
        """Load categories from CSV"""
        csv_path = self.csv_folder / 'categories.csv'
        if not csv_path.exists():
            return []
        
        categories = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                categories.append({
                    'id': self._safe_int(row.get('id'), 0),
                    'name': row.get('name', ''),
                    'description': row.get('description', '')
                })
        return categories
    
    def load_products(self):
        """Load products from CSV"""
        csv_path = self.csv_folder / 'products.csv'
        if not csv_path.exists():
            return []
        
        products = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append({
                    'id': self._safe_int(row.get('id'), 0),
                    'name': row.get('name', ''),
                    'description': row.get('description', ''),
                    'category': {'id': self._safe_int(row.get('category_id'), 0), 'name': ''},  # Will be filled later
                    'price': self._safe_float(row.get('price'), 0.0),
                    'tags': row.get('tags', ''),
                    'image_url': row.get('image_url', ''),
                    'stock': self._safe_int(row.get('stock'), 0)
                })
        
        # Fill category names
        categories = {cat['id']: cat['name'] for cat in self.load_categories()}
        for product in products:
            category_id = product['category']['id']
            product['category']['name'] = categories.get(category_id, 'Unknown')
        
        return products
    
    def load_users(self):
        """Load users from CSV"""
        csv_path = self.csv_folder / 'users.csv'
        if not csv_path.exists():
            return []
        
        users = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                age_value = row.get('age', '').strip()
                users.append({
                    'id': self._safe_int(row.get('id'), 0),
                    'name': row.get('name', ''),
                    'email': row.get('email', ''),
                    'age': self._safe_int(age_value) if age_value else None,
                    'gender': row.get('gender', ''),
                    'location': row.get('location', '')
                })
        return users
    
    def load_interactions(self):
        """Load user interactions from CSV"""
        csv_path = self.csv_folder / 'interactions.csv'
        if not csv_path.exists():
            return []
        
        interactions = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rating_value = row.get('rating', '').strip()
                interactions.append({
                    'user_id': self._safe_int(row.get('user_id'), 0),
                    'product_id': self._safe_int(row.get('product_id'), 0),
                    'interaction_type': row.get('interaction_type', ''),
                    'rating': self._safe_int(rating_value) if rating_value else None,
                    'timestamp': row.get('timestamp', '')
                })
        return interactions
    
    def load_browsing_history(self):
        """Load browsing history from CSV"""
        csv_path = self.csv_folder / 'browsing_history.csv'
        if not csv_path.exists():
            return []
        
        browsing = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                browsing.append({
                    'user_id': self._safe_int(row.get('user_id'), 0),
                    'product_id': self._safe_int(row.get('product_id'), 0),
                    'time_spent': self._safe_int(row.get('time_spent'), 0),
                    'timestamp': row.get('timestamp', '')
                })
        return browsing
    
    def load_search_history(self):
        """Load search history from CSV"""
        csv_path = self.csv_folder / 'search_history.csv'
        if not csv_path.exists():
            return []
        
        searches = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                searches.append({
                    'user_id': self._safe_int(row.get('user_id'), 0),
                    'query': row.get('query', ''),
                    'results_count': self._safe_int(row.get('results_count'), 0),
                    'timestamp': row.get('timestamp', '')
                })
        return searches
    
    def load_wishlist(self):
        """Load wishlist from CSV"""
        csv_path = self.csv_folder / 'wishlist.csv'
        if not csv_path.exists():
            return []
        
        wishlist = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wishlist.append({
                    'user_id': self._safe_int(row.get('user_id'), 0),
                    'product_id': self._safe_int(row.get('product_id'), 0)
                })
        return wishlist
    
    def get_all_data(self):
        """Get all data as a dictionary"""
        return {
            'categories': self.load_categories(),
            'products': self.load_products(),
            'users': self.load_users(),
            'interactions': self.load_interactions(),
            'browsing_history': self.load_browsing_history(),
            'search_history': self.load_search_history(),
            'wishlist': self.load_wishlist()
        }
