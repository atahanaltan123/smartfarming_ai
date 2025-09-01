import os

class Config:
    # Claude API yapılandırması
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '')
    
    # API endpoint
    CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
    
    # Model ayarları - Güncel model adı
    CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
    
    # Uygulama ayarları
    APP_NAME = "AgriSoilTech Akıllı Tarım"
    DEBUG = True
    
    # Dosya yükleme ayarları
    UPLOAD_FOLDER = "uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    @classmethod
    def get_claude_api_key(cls):
        """API anahtarını güvenli şekilde döndürür"""
        return cls.CLAUDE_API_KEY
