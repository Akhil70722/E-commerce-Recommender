# Quick Setup Guide

## Prerequisites Check

- [ ] Python 3.9+ installed (`python --version`)
- [ ] PostgreSQL installed or Docker available
- [ ] Groq API account created (https://console.groq.com/)

## Step-by-Step Setup

### 1. Setup PostgreSQL Database

**Option A: Using Docker (Easiest)**
```bash
docker-compose up -d
```
The database will be created automatically.

**Option B: Automated Database Creation (PowerShell)**
```powershell
# Run the database creation script
.\create_database.ps1
```

**Option C: Manual Database Creation**
```powershell
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE ecommerce_recommender;

# Exit
\q
```

Or using command line:
```powershell
# Set password (replace 'your_password' with actual password)
$env:PGPASSWORD='your_password'

# Create database
psql -U postgres -h localhost -c "CREATE DATABASE ecommerce_recommender;"
```

### 2. Get Groq API Key

1. Go to https://console.groq.com/
2. Sign up or log in
3. Navigate to "API Keys"
4. Click "Create API Key"
5. Copy the key (starts with `gsk_`)

### 3. Setup Python Environment

**Option A: Automated Setup (PowerShell - Recommended for Windows)**
```powershell
# Navigate to project directory
cd assignment

# Run the setup script
.\setup.ps1
```

**Option B: Manual Setup**
```bash
# Navigate to project directory
cd assignment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the environment template and create your `.env` file:

```bash
# Copy the template
cp backend/env.template backend/.env

# Or on Windows:
copy backend\env.template backend\.env
```

Then edit `backend/.env` and update the following:
- `SECRET_KEY`: Change to a random string
- `DB_PASSWORD`: Your PostgreSQL password
- `GROQ_API_KEY`: Your actual Groq API key (starts with `gsk_`)

**Important**: Replace `your-groq-api-key-here` with your actual Groq API key from https://console.groq.com/

### 5. Run Migrations

```bash
# Make sure you're in the backend directory
cd backend

# Create database tables
python manage.py makemigrations
python manage.py migrate

# Create admin user (optional, for Django admin panel)
python manage.py createsuperuser
```

### 6. Seed Sample Data (Optional but Recommended)

```bash
python manage.py seed_data
```

This will create:
- 5 categories
- 10 products
- 3 users
- Sample interactions

### 7. Start the Server

```bash
python manage.py runserver
```

The API will be available at: `http://localhost:8000/`

## Testing the API

### 1. Test Products Endpoint
```bash
curl http://localhost:8000/api/products/
```

### 2. Test Users Endpoint
```bash
curl http://localhost:8000/api/users/
```

### 3. Test Recommendations (After seeding data)
```bash
curl http://localhost:8000/api/recommendations/user/1/
```

## Access Django Admin

1. Go to: http://localhost:8000/admin/
2. Login with the superuser credentials you created
3. Manage products, users, interactions, and recommendations

## Troubleshooting

### "ModuleNotFoundError: No module named 'psycopg2'"
```bash
pip install psycopg2-binary
```

### "django.db.utils.OperationalError: could not connect to server"
- Check if PostgreSQL is running
- Verify database credentials in `.env`
- If using Docker: `docker-compose ps` to check container status

### "GROQ_API_KEY is required"
- Make sure `.env` file exists in `backend/` directory
- Verify the key is correct (starts with `gsk_`)
- Restart the Django server after adding the key

### "No recommendations available"
- Run `python manage.py seed_data` to create sample data
- Make sure users have interactions recorded
- Check that products exist in the database

## Next Steps

1. Explore the API endpoints using the Django REST Framework browsable API
2. Test recommendations with different users
3. Check the generated explanations from Groq API
4. Customize the recommendation algorithm if needed

## Production Deployment

Before deploying to production:

1. Change `DEBUG=False` in `.env`
2. Generate a secure `SECRET_KEY`
3. Use a production PostgreSQL database
4. Set up proper CORS settings
5. Use environment variables for all sensitive data
6. Set up proper logging and monitoring

