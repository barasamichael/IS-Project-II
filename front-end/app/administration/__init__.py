from flask import Blueprint
from flask import current_app

administration = Blueprint("administration", __name__)

from . import views, errors, forms


@administration.app_context_processor
def global_variables():
    """
    Provide global variables for templates within the 'administration' blueprint.

    :params: None
    :return: A dictionary containing global variables to inject into templates.
    :rtype: dict
    """
    return dict(
        app_name=current_app.config["ORGANIZATION_NAME"],
    )
