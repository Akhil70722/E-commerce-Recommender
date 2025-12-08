"""
Management command to seed the database with sample data for testing recommendations.
Usage: python manage.py seed_data
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
import random
from api.models import Category, Product, User, UserInteraction, BrowsingHistory, SearchHistory, Wishlist


class Command(BaseCommand):
    help = 'Seeds the database with sample data for testing recommendations'

    def add_arguments(self, parser):
        parser.add_argument(
            '--users',
            type=int,
            default=20,
            help='Number of users to create (default: 20)'
        )
        parser.add_argument(
            '--products',
            type=int,
            default=50,
            help='Number of products to create (default: 50)'
        )
        parser.add_argument(
            '--interactions',
            type=int,
            default=200,
            help='Number of interactions to create (default: 200)'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting data seeding...'))
        
        num_users = options['users']
        num_products = options['products']
        num_interactions = options['interactions']
        
        # Create Categories
        self.stdout.write('Creating categories...')
        categories_data = [
            {'name': 'Electronics', 'description': 'Electronic devices and gadgets'},
            {'name': 'Clothing', 'description': 'Apparel and fashion items'},
            {'name': 'Books', 'description': 'Books and literature'},
            {'name': 'Home & Garden', 'description': 'Home improvement and garden supplies'},
            {'name': 'Sports & Outdoors', 'description': 'Sports equipment and outdoor gear'},
            {'name': 'Toys & Games', 'description': 'Toys and games for all ages'},
            {'name': 'Beauty & Personal Care', 'description': 'Beauty and personal care products'},
            {'name': 'Food & Beverages', 'description': 'Food items and beverages'},
        ]
        
        categories = []
        for cat_data in categories_data:
            category, created = Category.objects.get_or_create(
                name=cat_data['name'],
                defaults={'description': cat_data['description']}
            )
            categories.append(category)
            if created:
                self.stdout.write(f'  Created category: {category.name}')
        
        # Create Products
        self.stdout.write(f'Creating {num_products} products...')
        products = []
        
        product_templates = [
            # Electronics
            {'name': 'Wireless Headphones', 'description': 'High-quality wireless headphones with noise cancellation', 'category': 'Electronics', 'price_range': (50, 200), 'tags': 'audio, wireless, headphones, bluetooth'},
            {'name': 'Smartphone', 'description': 'Latest smartphone with advanced features', 'category': 'Electronics', 'price_range': (300, 1000), 'tags': 'mobile, phone, smartphone, tech'},
            {'name': 'Laptop', 'description': 'High-performance laptop for work and gaming', 'category': 'Electronics', 'price_range': (500, 2000), 'tags': 'computer, laptop, tech, productivity'},
            {'name': 'Smart Watch', 'description': 'Fitness tracking smartwatch with health monitoring', 'category': 'Electronics', 'price_range': (100, 400), 'tags': 'wearable, fitness, smartwatch, health'},
            {'name': 'Bluetooth Speaker', 'description': 'Portable Bluetooth speaker with excellent sound quality', 'category': 'Electronics', 'price_range': (30, 150), 'tags': 'audio, speaker, bluetooth, portable'},
            
            # Clothing
            {'name': 'Cotton T-Shirt', 'description': 'Comfortable cotton t-shirt in various colors', 'category': 'Clothing', 'price_range': (15, 40), 'tags': 'clothing, t-shirt, casual, cotton'},
            {'name': 'Jeans', 'description': 'Classic denim jeans with perfect fit', 'category': 'Clothing', 'price_range': (40, 100), 'tags': 'clothing, jeans, denim, casual'},
            {'name': 'Running Shoes', 'description': 'Comfortable running shoes for athletes', 'category': 'Clothing', 'price_range': (60, 150), 'tags': 'shoes, running, sports, athletic'},
            {'name': 'Winter Jacket', 'description': 'Warm winter jacket for cold weather', 'category': 'Clothing', 'price_range': (80, 200), 'tags': 'clothing, jacket, winter, warm'},
            {'name': 'Sunglasses', 'description': 'Stylish sunglasses with UV protection', 'category': 'Clothing', 'price_range': (20, 100), 'tags': 'accessories, sunglasses, style, UV'},
            
            # Books
            {'name': 'Science Fiction Novel', 'description': 'Engaging science fiction novel with compelling story', 'category': 'Books', 'price_range': (10, 25), 'tags': 'book, fiction, sci-fi, novel'},
            {'name': 'Cookbook', 'description': 'Comprehensive cookbook with delicious recipes', 'category': 'Books', 'price_range': (15, 35), 'tags': 'book, cooking, recipes, food'},
            {'name': 'History Book', 'description': 'Fascinating history book covering important events', 'category': 'Books', 'price_range': (12, 30), 'tags': 'book, history, education, non-fiction'},
            {'name': 'Self-Help Guide', 'description': 'Motivational self-help guide for personal growth', 'category': 'Books', 'price_range': (10, 25), 'tags': 'book, self-help, motivation, personal development'},
            
            # Home & Garden
            {'name': 'Coffee Maker', 'description': 'Automatic coffee maker for perfect brew', 'category': 'Home & Garden', 'price_range': (40, 150), 'tags': 'home, kitchen, coffee, appliance'},
            {'name': 'Garden Tools Set', 'description': 'Complete set of garden tools for gardening', 'category': 'Home & Garden', 'price_range': (30, 80), 'tags': 'garden, tools, outdoor, gardening'},
            {'name': 'Bedding Set', 'description': 'Comfortable bedding set for bedroom', 'category': 'Home & Garden', 'price_range': (50, 150), 'tags': 'home, bedroom, bedding, comfort'},
            
            # Sports & Outdoors
            {'name': 'Yoga Mat', 'description': 'Premium yoga mat for exercise and meditation', 'category': 'Sports & Outdoors', 'price_range': (20, 60), 'tags': 'sports, yoga, fitness, exercise'},
            {'name': 'Bicycle', 'description': 'Mountain bicycle for outdoor adventures', 'category': 'Sports & Outdoors', 'price_range': (200, 800), 'tags': 'sports, bicycle, outdoor, cycling'},
            {'name': 'Tennis Racket', 'description': 'Professional tennis racket for players', 'category': 'Sports & Outdoors', 'price_range': (50, 200), 'tags': 'sports, tennis, racket, athletic'},
        ]
        
        for i in range(num_products):
            template = random.choice(product_templates)
            category = next((c for c in categories if c.name == template['category']), categories[0])
            
            price = random.uniform(*template['price_range'])
            
            product = Product.objects.create(
                name=f"{template['name']} #{i+1}",
                description=template['description'],
                category=category,
                price=price,
                tags=template['tags'],
                stock=random.randint(10, 100),
                image_url=''
            )
            products.append(product)
        
        self.stdout.write(self.style.SUCCESS(f'  Created {len(products)} products'))
        
        # Create Users
        self.stdout.write(f'Creating {num_users} users...')
        users = []
        first_names = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Emily', 'Chris', 'Lisa', 'Tom', 'Anna']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Wilson', 'Moore']
        
        for i in range(num_users):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            user = User.objects.create(
                name=f"{first_name} {last_name}",
                email=f"{first_name.lower()}.{last_name.lower()}{i}@example.com",
                age=random.randint(18, 65),
                gender=random.choice(['Male', 'Female', 'Other', '']),
                location=random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose', ''])
            )
            users.append(user)
        
        self.stdout.write(self.style.SUCCESS(f'  Created {len(users)} users'))
        
        # Create Interactions
        self.stdout.write(f'Creating {num_interactions} interactions...')
        interaction_types = ['view', 'cart', 'purchase', 'rating']
        
        for i in range(num_interactions):
            user = random.choice(users)
            product = random.choice(products)
            
            # Generate timestamp within last 90 days
            days_ago = random.randint(0, 90)
            timestamp = timezone.now() - timedelta(days=days_ago)
            
            interaction_type = random.choice(interaction_types)
            rating = None
            if interaction_type == 'rating':
                rating = random.randint(1, 5)
            
            UserInteraction.objects.create(
                user=user,
                product=product,
                interaction_type=interaction_type,
                rating=rating,
                timestamp=timestamp
            )
        
        self.stdout.write(self.style.SUCCESS(f'  Created {num_interactions} interactions'))
        
        # Create Browsing History
        self.stdout.write('Creating browsing history...')
        num_browsing = num_interactions // 2
        
        for i in range(num_browsing):
            user = random.choice(users)
            product = random.choice(products)
            days_ago = random.randint(0, 30)
            timestamp = timezone.now() - timedelta(days=days_ago)
            time_spent = random.randint(10, 300)
            
            BrowsingHistory.objects.create(
                user=user,
                product=product,
                time_spent=time_spent,
                timestamp=timestamp
            )
        
        self.stdout.write(self.style.SUCCESS(f'  Created {num_browsing} browsing records'))
        
        # Create Search History
        self.stdout.write('Creating search history...')
        search_queries = ['headphones', 'laptop', 'books', 'clothing', 'shoes', 'electronics', 'sports', 'home', 'garden', 'beauty']
        
        for i in range(num_interactions // 3):
            user = random.choice(users)
            query = random.choice(search_queries)
            days_ago = random.randint(0, 14)
            timestamp = timezone.now() - timedelta(days=days_ago)
            results_count = random.randint(5, 50)
            
            SearchHistory.objects.create(
                user=user,
                query=query,
                results_count=results_count,
                timestamp=timestamp
            )
        
        self.stdout.write(self.style.SUCCESS(f'  Created search history records'))
        
        # Create Wishlist items
        self.stdout.write('Creating wishlist items...')
        num_wishlist = num_users * 2
        
        for i in range(num_wishlist):
            user = random.choice(users)
            product = random.choice(products)
            
            # Avoid duplicates
            if not Wishlist.objects.filter(user=user, product=product).exists():
                Wishlist.objects.create(
                    user=user,
                    product=product
                )
        
        self.stdout.write(self.style.SUCCESS(f'  Created wishlist items'))
        
        self.stdout.write(self.style.SUCCESS('\n✅ Data seeding completed successfully!'))
        self.stdout.write(self.style.SUCCESS(f'\nSummary:'))
        self.stdout.write(f'  - Categories: {Category.objects.count()}')
        self.stdout.write(f'  - Products: {Product.objects.count()}')
        self.stdout.write(f'  - Users: {User.objects.count()}')
        self.stdout.write(f'  - Interactions: {UserInteraction.objects.count()}')
        self.stdout.write(f'  - Browsing History: {BrowsingHistory.objects.count()}')
        self.stdout.write(f'  - Search History: {SearchHistory.objects.count()}')
        self.stdout.write(f'  - Wishlist Items: {Wishlist.objects.count()}')
        self.stdout.write(self.style.SUCCESS('\nYou can now test recommendations by calling:'))
        self.stdout.write(self.style.SUCCESS('  GET /api/recommendations/user/{user_id}/'))
