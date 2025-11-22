from flask import Blueprint
from flask import current_app

authentication = Blueprint("authentication", __name__)
from . import views, errors


@authentication.app_context_processor
def global_variables():
    """
    Provide global variables for templates within the 'authentication' blueprint.

    :return: A dictionary containing global variables to inject into templates.
    :rtype: dict

    :params: None
    """
    return dict(
        app_name=current_app.config["ORGANIZATION_NAME"],
    )
