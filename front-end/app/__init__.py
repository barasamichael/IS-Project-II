import flask
import flask_login
import flask_moment
import flask_mailman
import flask_sqlalchemy
from flask_minify import minify
from flask_caching import Cache
from flask_wtf import CSRFProtect
from config import config

# set endpoint for the login page
login_manager = flask_login.LoginManager()
login_manager.login_view = "authentication.user_login"
login_manager.refresh_view = "authentication.reauthenticate"
login_manager.needs_refresh_message = (
    "To protect your account, please login afresh to access this page."
)
login_manager.needs_refresh_message_category = "info"


@login_manager.user_loader
def load_user(user_id):
    from .models import User

    return User.query.get(int(user_id))


csrf = CSRFProtect()
mail = flask_mailman.Mail()
moment = flask_moment.Moment()
db = flask_sqlalchemy.SQLAlchemy()
cache = Cache()


def create_app(config_name="default"):
    """
    Initialize and configure the Flask application.

    :param config_name: str - The name of the configuration class defined in
        config.py.

    :return app: Flask - The configured Flask application instance.
    """
    app = flask.Flask(__name__)
    app.config.from_object(config[config_name])

    # Initialize extensions
    minify(app)
    db.init_app(app)
    csrf.init_app(app)
    cache.init_app(app)
    mail.init_app(app)
    moment.init_app(app)
    login_manager.init_app(app)

    # Enable SSL redirection if configured
    if app.config["SSL_REDIRECT"]:
        from flask_sslify import SSLify

        SSLify(app)

    # Register blueprints for different parts of the application

    from .main import main as main_blueprint
    from .accounts import accounts as accounts_blueprint
    from .administration import administration as administration_blueprint
    from .authentication import authentication as authentication_blueprint

    app.register_blueprint(main_blueprint)
    app.register_blueprint(accounts_blueprint, url_prefix="/account")
    app.register_blueprint(
        administration_blueprint, url_prefix="/administration"
    )
    app.register_blueprint(
        authentication_blueprint, url_prefix="/authentication"
    )

    return app
