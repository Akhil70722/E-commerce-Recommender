# E-commerce Product Recommender

An open-source Django-based recommendation system that combines hybrid recommendation algorithms with Groq API-powered LLM explanations to provide personalized product recommendations with intelligent explanations.

## 🚀 Features

- **Hybrid Recommendation Engine**: Combines collaborative filtering and content-based filtering for accurate recommendations
- **LLM-Powered Explanations**: Uses Groq API (Llama 2, Mixtral, or Gemma) to generate personalized explanations for each recommendation
- **RESTful API**: Complete Django REST Framework API for products, users, interactions, and recommendations
- **PostgreSQL Database**: Robust database solution for production-ready applications
- **Admin Interface**: Django admin panel for easy data management

## 🛠️ Tech Stack

- **Backend**: Django 4.2 (Single App Architecture)
- **Database**: PostgreSQL 12+
- **LLM**: Groq API (Llama 2, Mixtral, or Gemma models)
- **Recommendation Engine**: Hybrid (Collaborative + Content-based filtering)
- **Data Processing**: pandas, scikit-learn, numpy
- **API**: JSON responses (no serializers)

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 12+ (or use Docker)
- Groq API Key (free at https://console.groq.com/)

## 🔧 Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd assignment
```

### Step 2: Install PostgreSQL

**Option A: Local Installation**
- Download from https://www.postgresql.org/download/
- Create database: `CREATE DATABASE ecommerce_recommender;`

**Option B: Docker (Recommended)**
```bash
docker-compose up -d
```

**Option C: Cloud Database**
- Use Supabase, ElephantSQL, or Railway for free PostgreSQL hosting

### Step 3: Get Groq API Key

1. Sign up at https://console.groq.com/
2. Navigate to API Keys section
3. Create a new API key
4. Copy the key (starts with `gsk_`)

### Step 4: Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Django Settings
SECRET_KEY=django-insecure-your-secret-key-change-this-in-production
DEBUG=True

# PostgreSQL Database Configuration
DB_NAME=ecommerce_recommender
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

# Groq API Configuration (Required)
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama2-70b-4096
# Alternative models: mixtral-8x7b-32768, llama2-70b-4096, gemma-7b-it
```

### Step 6: Run Database Migrations

```bash
cd backend
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### Step 7: Start the Development Server

```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000/`

## 📚 API Endpoints

### Categories
- `GET /api/categories/` - List all categories
- `GET /api/categories/{id}/` - Get category details
- `POST /api/categories/` - Create a category
- `PUT /api/categories/{id}/` - Update a category
- `DELETE /api/categories/{id}/` - Delete a category

### Products
- `GET /api/products/` - List all products (supports ?category={id}&search={term})
- `GET /api/products/{id}/` - Get product details
- `POST /api/products/` - Create a product
- `PUT /api/products/{id}/` - Update a product
- `DELETE /api/products/{id}/` - Delete a product

### Users
- `GET /api/users/` - List all users
- `GET /api/users/{id}/` - Get user details
- `POST /api/users/` - Create a user
- `PUT /api/users/{id}/` - Update a user
- `DELETE /api/users/{id}/` - Delete a user

### Interactions
- `GET /api/interactions/` - List all interactions (supports ?user_id={id}&product_id={id})
- `POST /api/interactions/` - Record a user interaction (view, purchase, cart, rating)

### Recommendations
- `GET /api/recommendations/user/{user_id}/` - Get personalized recommendations with LLM explanations

## 📖 Usage Example

### 1. Create a User
```bash
curl -X POST http://localhost:8000/api/users/ \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "email": "john@example.com"}'
```

### 2. Create Products
```bash
# First create a category
curl -X POST http://localhost:8000/api/products/categories/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Electronics", "description": "Electronic products"}'

# Then create a product
curl -X POST http://localhost:8000/api/products/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Wireless Headphones",
    "description": "High-quality wireless headphones",
    "category_id": 1,
    "price": "99.99",
    "tags": "audio, wireless, headphones",
    "stock": 50
  }'
```

### 3. Record User Interactions
```bash
curl -X POST http://localhost:8000/api/interactions/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "product_id": 1,
    "interaction_type": "view"
  }'
```

### 4. Get Recommendations
```bash
curl http://localhost:8000/api/recommendations/user/1/
```

**Response:**
```json
{
  "user_id": 1,
  "user_email": "john@example.com",
  "recommendations": [
    {
      "product_id": 2,
      "product_name": "Bluetooth Speaker",
      "category": "Electronics",
      "price": 49.99,
      "description": "Portable Bluetooth speaker",
      "image_url": "",
      "score": 0.85,
      "explanation": "Based on your recent interest in audio products and your view of wireless headphones, we recommend this Bluetooth speaker as it complements your audio setup and matches your preference for wireless technology."
    }
  ]
}
```

## 🎯 Recommendation Algorithm

The system uses a **hybrid approach** combining:

1. **Collaborative Filtering**: Finds users with similar preferences and recommends products they liked
2. **Content-Based Filtering**: Analyzes product features (description, tags, category) to match user preferences

The final score combines both approaches with weights:
- Collaborative filtering: 60%
- Content-based filtering: 40%

## 🤖 LLM Integration

The system uses **Groq API** to generate personalized explanations for each recommendation. The LLM receives:
- User's interaction history
- Product details (name, category, description, tags)
- Context about user preferences

And generates a 2-3 sentence explanation of why the product is recommended.

## 📁 Project Structure

```
ecommerce-recommender/
├── backend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   ├── recommender/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── ...
│   └── api/                    # Single app containing everything
│       ├── models.py           # All models (Category, Product, User, UserInteraction, Recommendation)
│       ├── views.py            # All API views (JSON responses, no serializers)
│       ├── urls.py             # All URL patterns
│       ├── admin.py            # Django admin configuration
│       ├── engine.py           # Recommendation algorithm
│       ├── llm_service.py      # Groq API integration
│       └── management/commands/
│           └── seed_data.py    # Sample data seeding
├── docker-compose.yml
└── README.md
```

## 🧪 Testing

To test the recommendation system:

1. Create multiple users
2. Create products in different categories
3. Record various interactions (views, purchases, ratings)
4. Request recommendations for a user
5. Verify explanations are generated

## 🐛 Troubleshooting

### PostgreSQL Connection Issues
- Ensure PostgreSQL is running: `sudo systemctl status postgresql` (Linux)
- Verify database credentials in `.env`
- Check if database exists: `psql -U postgres -l`

### Groq API Issues
- Verify `GROQ_API_KEY` is set in `.env`
- Check API key is valid at https://console.groq.com/
- Ensure you have API credits/quota

### Import Errors
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version: `python --version` (should be 3.9+)

## 📝 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SECRET_KEY` | Django secret key | Yes |
| `DEBUG` | Debug mode | Yes |
| `DB_NAME` | PostgreSQL database name | Yes |
| `DB_USER` | PostgreSQL username | Yes |
| `DB_PASSWORD` | PostgreSQL password | Yes |
| `DB_HOST` | PostgreSQL host | Yes |
| `DB_PORT` | PostgreSQL port | Yes |
| `GROQ_API_KEY` | Groq API key | Yes |
| `GROQ_MODEL` | Groq model name | No (default: llama2-70b-4096) |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🎓 Assignment Requirements

This project fulfills the following requirements:

✅ **Backend API** for recommendations  
✅ **Database** for products & user interactions (PostgreSQL)  
✅ **LLM** for explanation text (Groq API)  
✅ **Recommendation accuracy** (Hybrid algorithm)  
✅ **LLM explanation quality** (Groq API with context)  
✅ **Code design** (Clean Django architecture)  
✅ **Documentation** (Comprehensive README)

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

**Note**: Make sure to set your `GROQ_API_KEY` in the `.env` file before running the server. The recommendation system requires this for generating explanations.

