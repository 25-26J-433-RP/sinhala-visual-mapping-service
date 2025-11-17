"""
Configuration settings for Sinhala Mind Map API
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration."""
    
    # Flask settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max request size
    
    # External API settings
    EXTERNAL_API_TIMEOUT = int(os.getenv('EXTERNAL_API_TIMEOUT', 10))
    
    # Mind map generation settings
    MAX_NODES = int(os.getenv('MAX_NODES', 100))
    MAX_LEVELS = int(os.getenv('MAX_LEVELS', 4))
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config(env='default'):
    """Get configuration based on environment."""
    return config.get(env, config['default'])
