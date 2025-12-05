# PowerShell Setup Script for E-commerce Product Recommender
# Run this script in PowerShell: .\setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "E-commerce Product Recommender Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.9+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
Write-Host ""
Write-Host "Setting up virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Set-Location backend
pip install -r requirements.txt
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# Check for .env file
Write-Host ""
Write-Host "Checking environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✓ .env file exists" -ForegroundColor Green
} else {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item "env.template" ".env"
    Write-Host "✓ .env file created from template" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠ IMPORTANT: Please edit .env file and set:" -ForegroundColor Yellow
    Write-Host "  - SECRET_KEY (generate a random string)" -ForegroundColor Yellow
    Write-Host "  - DB_PASSWORD (your PostgreSQL password)" -ForegroundColor Yellow
    Write-Host "  - GROQ_API_KEY (your Groq API key from https://console.groq.com/)" -ForegroundColor Yellow
    Write-Host ""
}

# Check PostgreSQL connection
Write-Host ""
Write-Host "Checking PostgreSQL connection..." -ForegroundColor Yellow
$dbName = (Get-Content .env | Select-String "DB_NAME").ToString().Split("=")[1].Trim()
$dbUser = (Get-Content .env | Select-String "DB_USER").ToString().Split("=")[1].Trim()
$dbHost = (Get-Content .env | Select-String "DB_HOST").ToString().Split("=")[1].Trim()
$dbPort = (Get-Content .env | Select-String "DB_PORT").ToString().Split("=")[1].Trim()

Write-Host "Database configuration:" -ForegroundColor Cyan
Write-Host "  Name: $dbName" -ForegroundColor White
Write-Host "  User: $dbUser" -ForegroundColor White
Write-Host "  Host: $dbHost" -ForegroundColor White
Write-Host "  Port: $dbPort" -ForegroundColor White

# Run migrations
Write-Host ""
Write-Host "Running database migrations..." -ForegroundColor Yellow
python manage.py makemigrations
python manage.py migrate
Write-Host "✓ Migrations completed" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Make sure PostgreSQL is running" -ForegroundColor White
Write-Host "2. Edit .env file with your credentials" -ForegroundColor White
Write-Host "3. Create superuser: python manage.py createsuperuser" -ForegroundColor White
Write-Host "4. Add products and users via Django admin or API" -ForegroundColor White
Write-Host "5. Start server: python manage.py runserver" -ForegroundColor White
Write-Host ""
Set-Location ..

