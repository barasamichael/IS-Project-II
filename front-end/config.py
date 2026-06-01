import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    # Flask security configuration options
    SECRET_KEY = (
        os.environ.get("SECRET_KEY")
        or "E6GlTcYgNJNgtCKue5xMJgVtT8tfdOORVqRcIjhl9qvfbHgtvz"
    )
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024

    # SQLAlchemy configuration options
    SSL_REDIRECT = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True
    SLOW_DB_QUERY_TIME = 0.5

    # Application configuration options
    ORGANIZATION_NAME = os.environ.get("ORGANIZATION_NAME") or "SettleBot"

    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)

    # File upload configuration options
    USERS_PROFILE_UPLOAD_PATH = os.path.join(
        basedir, "app/static/uploads/profile_pictures/"
    )
    UPLOAD_EXTENSIONS = [".jpg", ".gif", ".jpeg", ".png", ".webp"]
    DELETED_ACCOUNTS_FILE = os.path.join(basedir + "/deleted_accounts.json")

    # Zoho Mail API — OAuth credentials
    ZOHO_CLIENT_ID = os.environ.get("ZOHO_CLIENT_ID")
    ZOHO_CLIENT_SECRET = os.environ.get("ZOHO_CLIENT_SECRET")
    ZOHO_REFRESH_TOKEN = os.environ.get("ZOHO_REFRESH_TOKEN")
    ZOHO_ACCOUNT_ID = os.environ.get("ZOHO_ACCOUNT_ID")

    # Zoho Mail API — sending identity
    ZOHO_FROM_ADDRESS = os.environ.get("ZOHO_FROM_ADDRESS")

    # Zoho Mail API — base URLs (override for non-EU data centres)
    ZOHO_ACCOUNTS_BASE = os.environ.get(
        "ZOHO_ACCOUNTS_BASE", "https://accounts.zoho.com"
    )
    ZOHO_MAIL_BASE = os.environ.get("ZOHO_MAIL_BASE", "https://mail.zoho.com")
    ZOHO_HTTP_TIMEOUT = int(os.environ.get("ZOHO_HTTP_TIMEOUT", "20"))

    # Administrator notifications
    ADMINISTRATOR_SENDER = "Administrator <mkuu@jisortublow.co.ke>"
    ADMINISTRATOR_MAIL = os.environ.get(
        "ADMINISTRATOR_MAIL", "mkuu@jisortublow.co.ke"
    )

    # RAG API Configuration
    SETTLEBOT_API_URL = os.environ.get(
        "SETTLEBOT_API_URL", "http://127.0.0.1:8000/")
    SETTLEBOT_API_KEY = os.environ.get(
        "SETTLEBOT_API_KEY", "your_secure_random_key_here")

    # Cache configuration
    CACHE_TYPE = "SimpleCache"
    CACHE_DEFAULT_TIMEOUT = 300

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DEVELOPMENT_DATABASE_URL"
    ) or "sqlite:///" + os.path.join(basedir, "development.db")
    MINIFY_HTML = False
    MINIFY_CSS = False
    MINIFY_JS = False


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("TEST_DATABASE_URL") or "sqlite:///:memory:"
    )
    SECRET_KEY = "chdbvbjnhmkldfhgcavbxntkuymluv"

    WTF_CSRF_ENABLED = False
    CACHE_TYPE = "NullCache"


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    DB_NAME = os.environ.get("DB_NAME") or "strathquery"
    DB_USERNAME = os.environ.get("DB_USERNAME") or "strathquery"
    DB_HOST = os.environ.get("DB_HOST") or "localhost"
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("PRODUCTION_DATABASE_URL")
        or f"mysql+mysqlconnector://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    )

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Log to stderr
        import logging
        from logging import StreamHandler

        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        # Enable SSL in production
        cls.SSL_REDIRECT = True


config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
