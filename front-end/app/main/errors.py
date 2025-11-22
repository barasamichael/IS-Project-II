from flask import render_template
from flask import request
from flask import jsonify

from . import main

@main.app_errorhandler(403)
def forbidden(e):
    """
    Handle a 403 Forbidden error.

    :param e: The exception triggering the error handler.
    :type e: Exception

    :return: A rendered HTML template for a 403 Forbidden error.
    :rtype: tuple
    """
    if request.accept_mimetypes.accept_json and \
            not request.accept_mimetypes.accept_html:
                return jsonify({"error": "forbidden"}), 403

    return render_template('main/403.html'), 403


@main.app_errorhandler(404)
def page_not_found(e):
    """
    Handle a 404 Not Found error.

    :param e: The exception triggering the error handler.
    :type e: Exception

    :return: A rendered HTML template for a 404 Not Found error.
    :rtype: tuple
    """
    if request.accept_mimetypes.accept_json and \
            not request.accept_mimetypes.accept_html:
                return jsonify({"error": "not found"}), 404

    return render_template('main/404.html'), 404


@main.app_errorhandler(500)
def internal_server_error(e):
    """
    Handle a 500 Internal Server Error.

    :param e: The exception triggering the error handler.
    :type e: Exception

    :return: A rendered HTML template for a 500 Internal Server Error.
    :rtype: tuple
    """
    if request.accept_mimetypes.accept_json and \
            not request.accept_mimetypes.accept_html:
                return jsonify({"error": "internal server error"}), 500

    return render_template('main/500.html'), 500
