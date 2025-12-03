# Quick Fix: Create Database

## The Error
```
FATAL: database "ecommerce_recommender" does not exist
```

## Solution Options

### Option 1: Using Docker (Recommended if using Docker)

If you're using Docker for PostgreSQL:

```powershell
# Start Docker container (creates database automatically)
docker-compose up -d

# Verify it's running
docker ps
```

### Option 2: Manual Database Creation

**Step 1: Find your PostgreSQL password**

Check your `backend/.env` file for `DB_PASSWORD` value.

**Step 2: Create database using psql**

```powershell
# Replace 'your_password' with your actual PostgreSQL password from .env
$env:PGPASSWORD='your_password'
psql -U postgres -h localhost -c "CREATE DATABASE ecommerce_recommender;"
```

**Or connect interactively:**

```powershell
psql -U postgres -h localhost
```

Then in psql prompt:
```sql
CREATE DATABASE ecommerce_recommender;
\q
```

### Option 3: Using pgAdmin (GUI)

1. Open pgAdmin
2. Connect to your PostgreSQL server
3. Right-click on "Databases" → "Create" → "Database"
4. Name: `ecommerce_recommender`
5. Click "Save"

### Option 4: Check if PostgreSQL is Running

```powershell
# Check PostgreSQL service status
Get-Service postgresql*

# Or check if port 5432 is listening
netstat -an | Select-String "5432"
```

## After Creating Database

Once the database is created, run:

```powershell
cd backend
python manage.py migrate
python manage.py createsuperuser
python manage.py seed_data
```

## Common Issues

### Issue: "password authentication failed"
- **Solution**: Check your `DB_PASSWORD` in `backend/.env` file
- Make sure it matches your PostgreSQL password

### Issue: "connection refused"
- **Solution**: PostgreSQL service might not be running
- Start PostgreSQL service or use Docker

### Issue: "psql: command not found"
- **Solution**: PostgreSQL client tools not in PATH
- Use pgAdmin GUI or add PostgreSQL bin to PATH

