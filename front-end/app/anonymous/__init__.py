from flask import Blueprint
from flask import current_app

anonymous = Blueprint("anonymous", __name__)
from . import views, errors


@anonymous.app_context_processor
def global_variables():
    """
    Provide global variables for templates within the 'anonymous' blueprint.

    :params: None
    :return: A dictionary containing global variables to inject into templates.
    :rtype: dict
    """
    return dict(
        app_name=current_app.config["ORGANIZATION_NAME"],
    )
