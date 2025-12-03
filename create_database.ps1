# PowerShell script to create PostgreSQL database
# Run this script: .\create_database.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Creating PostgreSQL Database" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Read database configuration from .env file
$envPath = "backend\.env"
if (-not (Test-Path $envPath)) {
    Write-Host "ERROR: .env file not found at $envPath" -ForegroundColor Red
    Write-Host "Please create .env file first by copying env.template" -ForegroundColor Yellow
    exit 1
}

Write-Host "Reading database configuration from .env file..." -ForegroundColor Yellow

# Read .env file and extract database settings
$envContent = Get-Content $envPath
$dbConfig = @{}

foreach ($line in $envContent) {
    if ($line -match '^\s*([^#][^=]+)=(.*)$') {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()
        $dbConfig[$key] = $value
    }
}

$dbName = $dbConfig['DB_NAME']
$dbUser = $dbConfig['DB_USER']
$dbPassword = $dbConfig['DB_PASSWORD']
$dbHost = $dbConfig['DB_HOST']
$dbPort = $dbConfig['DB_PORT']

if (-not $dbName) {
    Write-Host "ERROR: DB_NAME not found in .env file" -ForegroundColor Red
    exit 1
}

Write-Host "Database Configuration:" -ForegroundColor Cyan
Write-Host "  Name: $dbName" -ForegroundColor White
Write-Host "  User: $dbUser" -ForegroundColor White
Write-Host "  Host: $dbHost" -ForegroundColor White
Write-Host "  Port: $dbPort" -ForegroundColor White
Write-Host ""

# Check if psql is available
Write-Host "Checking for PostgreSQL client (psql)..." -ForegroundColor Yellow
$psqlPath = Get-Command psql -ErrorAction SilentlyContinue

if (-not $psqlPath) {
    Write-Host "ERROR: psql command not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install PostgreSQL or add it to your PATH." -ForegroundColor Yellow
    Write-Host "You can also create the database manually:" -ForegroundColor Yellow
    Write-Host "  1. Open pgAdmin or psql" -ForegroundColor White
    Write-Host "  2. Connect to PostgreSQL server" -ForegroundColor White
    Write-Host "  3. Run: CREATE DATABASE $dbName;" -ForegroundColor White
    Write-Host ""
    Write-Host "Or use Docker:" -ForegroundColor Yellow
    Write-Host "  docker-compose up -d" -ForegroundColor White
    exit 1
}

Write-Host "✓ psql found" -ForegroundColor Green
Write-Host ""

# Set PGPASSWORD environment variable for psql
$env:PGPASSWORD = $dbPassword

Write-Host "Creating database '$dbName'..." -ForegroundColor Yellow

# Create database using psql
$createDbQuery = "CREATE DATABASE $dbName;"
$psqlCommand = "psql -h $dbHost -p $dbPort -U $dbUser -d postgres -c `"$createDbQuery`""

try {
    $result = Invoke-Expression $psqlCommand 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Database '$dbName' created successfully!" -ForegroundColor Green
    } else {
        # Check if database already exists
        if ($result -match "already exists") {
            Write-Host "✓ Database '$dbName' already exists" -ForegroundColor Yellow
        } else {
            Write-Host "ERROR: Failed to create database" -ForegroundColor Red
            Write-Host $result -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "ERROR: Failed to execute psql command" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual steps:" -ForegroundColor Yellow
    Write-Host "  1. Connect to PostgreSQL: psql -U $dbUser -h $dbHost -p $dbPort" -ForegroundColor White
    Write-Host "  2. Run: CREATE DATABASE $dbName;" -ForegroundColor White
    exit 1
} finally {
    # Clear password from environment
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Database Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run migrations: python manage.py migrate" -ForegroundColor White
Write-Host "2. Create superuser: python manage.py createsuperuser" -ForegroundColor White
Write-Host "3. Seed data: python manage.py seed_data" -ForegroundColor White
Write-Host ""

