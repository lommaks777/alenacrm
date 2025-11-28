import os
import base64
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration management for VoiceStock Bot"""
    
    # Telegram Configuration
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    ALLOWED_USER_IDS = os.getenv("ALLOWED_USER_IDS")  # Comma-separated list of user IDs
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Google Sheets Configuration
    GOOGLE_SHEETS_CREDENTIALS_BASE64 = os.getenv("GOOGLE_SHEETS_CREDENTIALS_BASE64")
    GOOGLE_SHEET_KEY = os.getenv("GOOGLE_SHEET_KEY")
    
    # Timezone Configuration
    TIMEZONE = os.getenv("TIMEZONE", "Europe/Moscow")
    
    @classmethod
    def validate(cls):
        """Validate that all required configuration is present"""
        missing = []
        
        if not cls.BOT_TOKEN:
            missing.append("BOT_TOKEN")
        if not cls.ALLOWED_USER_IDS:
            missing.append("ALLOWED_USER_IDS")
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.GOOGLE_SHEETS_CREDENTIALS_BASE64:
            missing.append("GOOGLE_SHEETS_CREDENTIALS_BASE64")
        if not cls.GOOGLE_SHEET_KEY:
            missing.append("GOOGLE_SHEET_KEY")
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        # Validate ALLOWED_USER_IDS are numeric
        try:
            user_ids = [id.strip() for id in cls.ALLOWED_USER_IDS.split(',')]
            for user_id in user_ids:
                int(user_id)  # Validate each ID is numeric
        except ValueError:
            raise ValueError("ALLOWED_USER_IDS must be comma-separated integers")
        
        logger.info("Configuration validated successfully")
    
    @classmethod
    def get_google_credentials(cls):
        """Decode and return Google Sheets credentials as dict"""
        try:
            decoded = base64.b64decode(cls.GOOGLE_SHEETS_CREDENTIALS_BASE64)
            credentials = json.loads(decoded)
            logger.info("Google credentials decoded successfully")
            return credentials
        except Exception as e:
            logger.error(f"Failed to decode Google credentials: {e}")
            raise ValueError(f"Invalid GOOGLE_SHEETS_CREDENTIALS_BASE64: {e}")
    
    @classmethod
    def get_allowed_user_ids(cls):
        """Get allowed user IDs as list of integers"""
        user_ids = [id.strip() for id in cls.ALLOWED_USER_IDS.split(',')]
        return [int(user_id) for user_id in user_ids]
    
    @classmethod
    def is_user_allowed(cls, user_id: int) -> bool:
        """Check if user ID is in allowed list"""
        return user_id in cls.get_allowed_user_ids()


# Validate configuration on module import
try:
    Config.validate()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    raise
