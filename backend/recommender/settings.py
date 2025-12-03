"""
Django settings for recommender project.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
# Try multiple possible locations for .env file
env_paths = [
    BASE_DIR / '.env',  # backend/.env
    BASE_DIR.parent / '.env',  # root/.env
    Path('.env'),  # current directory
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        env_loaded = True
        break

# If no .env file found, try loading without explicit path (default behavior)
if not env_loaded:
    load_dotenv(override=True)

SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-change-this-in-production')
DEBUG = os.getenv('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'recommender.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'recommender.wsgi.application'

# PostgreSQL Database Configuration
# All database settings must be provided via environment variables

# Get database configuration from environment variables
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')

# Validate that all required database environment variables are set
required_db_vars = {
    'DB_NAME': db_name,
    'DB_USER': db_user,
    'DB_PASSWORD': db_password,
    'DB_HOST': db_host,
    'DB_PORT': db_port,
}

missing_vars = [var for var, value in required_db_vars.items() if not value or (isinstance(value, str) and value.strip() == '')]
if missing_vars:
    # Check if .env file exists
    env_file_path = BASE_DIR / '.env'
    env_template_path = BASE_DIR / 'env.template'
    
    error_msg = f"Missing required database environment variables: {', '.join(missing_vars)}.\n\n"
    
    if not env_file_path.exists():
        error_msg += f"ERROR: .env file not found at: {env_file_path}\n"
        if env_template_path.exists():
            error_msg += f"SOLUTION: Copy {env_template_path} to {env_file_path} and update the values.\n"
        else:
            error_msg += f"SOLUTION: Create a .env file at {env_file_path} with the following variables:\n"
            error_msg += "  DB_NAME=your_database_name\n"
            error_msg += "  DB_USER=your_database_user\n"
            error_msg += "  DB_PASSWORD=your_database_password\n"
            error_msg += "  DB_HOST=your_database_host\n"
            error_msg += "  DB_PORT=your_database_port\n"
    else:
        error_msg += f"ERROR: .env file exists at {env_file_path} but is missing required variables.\n"
        error_msg += f"SOLUTION: Add the missing variables to your .env file.\n"
    
    raise ValueError(error_msg)

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': db_name.strip() if db_name and isinstance(db_name, str) else db_name,
        'USER': db_user.strip() if db_user and isinstance(db_user, str) else db_user,
        'PASSWORD': db_password.strip() if db_password and isinstance(db_password, str) else db_password,
        'HOST': db_host.strip() if db_host and isinstance(db_host, str) else db_host,
        'PORT': db_port.strip() if db_port and isinstance(db_port, str) else db_port,
        'OPTIONS': {
            'connect_timeout': 10,
        },
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

CORS_ALLOW_ALL_ORIGINS = True  # For development only

