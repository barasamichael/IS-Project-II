from flask import Blueprint
from flask import current_app

main = Blueprint("main", __name__)
from . import views, errors


@main.app_context_processor
def global_variables():
    """
    Provide global variables for templates within the 'main' blueprint.

    :return: A dictionary containing global variables to inject into templates.
    :rtype: dict
    """
    return dict(
        app_name=current_app.config.get("ORGANIZATION_NAME", "WildTrack"),
    )
