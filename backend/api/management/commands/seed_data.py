from django.core.management.base import BaseCommand
from api.models import Category, Product, User, UserInteraction
import random

class Command(BaseCommand):
    help = 'Seed the database with sample data for testing'

    def handle(self, *args, **options):
        self.stdout.write('Seeding database with sample data...')
        
        # Create Categories
        categories_data = [
            {'name': 'Electronics', 'description': 'Electronic devices and gadgets'},
            {'name': 'Clothing', 'description': 'Apparel and fashion items'},
            {'name': 'Books', 'description': 'Books and literature'},
            {'name': 'Home & Garden', 'description': 'Home improvement and garden supplies'},
            {'name': 'Sports', 'description': 'Sports equipment and accessories'},
        ]
        
        categories = []
        for cat_data in categories_data:
            category, created = Category.objects.get_or_create(
                name=cat_data['name'],
                defaults={'description': cat_data['description']}
            )
            categories.append(category)
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created category: {category.name}'))
        
        # Create Products
        products_data = [
            {'name': 'Wireless Headphones', 'description': 'High-quality wireless headphones with noise cancellation', 'category': 'Electronics', 'price': 99.99, 'tags': 'audio, wireless, headphones, bluetooth'},
            {'name': 'Smartphone', 'description': 'Latest smartphone with advanced features', 'category': 'Electronics', 'price': 599.99, 'tags': 'mobile, smartphone, technology'},
            {'name': 'Laptop', 'description': 'High-performance laptop for work and gaming', 'category': 'Electronics', 'price': 1299.99, 'tags': 'computer, laptop, technology'},
            {'name': 'T-Shirt', 'description': 'Comfortable cotton t-shirt', 'category': 'Clothing', 'price': 19.99, 'tags': 'clothing, casual, cotton'},
            {'name': 'Jeans', 'description': 'Classic denim jeans', 'category': 'Clothing', 'price': 49.99, 'tags': 'clothing, denim, casual'},
            {'name': 'Python Programming Book', 'description': 'Comprehensive guide to Python programming', 'category': 'Books', 'price': 39.99, 'tags': 'programming, python, education'},
            {'name': 'Django Web Development', 'description': 'Learn Django framework for web development', 'category': 'Books', 'price': 44.99, 'tags': 'programming, django, web'},
            {'name': 'Garden Tools Set', 'description': 'Complete set of gardening tools', 'category': 'Home & Garden', 'price': 79.99, 'tags': 'garden, tools, outdoor'},
            {'name': 'Yoga Mat', 'description': 'Premium yoga mat for exercise', 'category': 'Sports', 'price': 29.99, 'tags': 'fitness, yoga, exercise'},
            {'name': 'Running Shoes', 'description': 'Comfortable running shoes for athletes', 'category': 'Sports', 'price': 89.99, 'tags': 'sports, running, shoes'},
        ]
        
        products = []
        for prod_data in products_data:
            category = next(cat for cat in categories if cat.name == prod_data['category'])
            product, created = Product.objects.get_or_create(
                name=prod_data['name'],
                defaults={
                    'description': prod_data['description'],
                    'category': category,
                    'price': prod_data['price'],
                    'tags': prod_data['tags'],
                    'stock': random.randint(10, 100)
                }
            )
            products.append(product)
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created product: {product.name}'))
        
        # Create Users
        users_data = [
            {'name': 'John Doe', 'email': 'john@example.com'},
            {'name': 'Jane Smith', 'email': 'jane@example.com'},
            {'name': 'Bob Johnson', 'email': 'bob@example.com'},
        ]
        
        users = []
        for user_data in users_data:
            user, created = User.objects.get_or_create(
                email=user_data['email'],
                defaults={'name': user_data['name']}
            )
            users.append(user)
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created user: {user.email}'))
        
        # Create Interactions
        interaction_types = ['view', 'cart', 'purchase', 'rating']
        
        for user in users:
            # Each user interacts with 3-5 random products
            user_products = random.sample(products, random.randint(3, min(5, len(products))))
            
            for product in user_products:
                interaction_type = random.choice(interaction_types)
                rating = random.randint(3, 5) if interaction_type == 'rating' else None
                
                UserInteraction.objects.get_or_create(
                    user=user,
                    product=product,
                    interaction_type=interaction_type,
                    defaults={'rating': rating}
                )
        
        self.stdout.write(self.style.SUCCESS(f'Created interactions for {len(users)} users'))
        self.stdout.write(self.style.SUCCESS('Database seeding completed!'))
        self.stdout.write(self.style.WARNING('\nYou can now test recommendations by calling:'))
        self.stdout.write(self.style.WARNING('GET /api/recommendations/user/1/'))

