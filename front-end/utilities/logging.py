import logging


# Configure logging settings
def configure_logging(app):
    """
    Configure logging for the Flask application.

    This function sets up logging for the Flask application by creating a custom
        formatter, adding a FileHandler for writing logs to a file, and a StreamHandler    for displaying logs in the console.

    :param app (Flask): The Flask application instance.
    """
    # Create the formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # Create a file handler
    file_handler = logging.FileHandler("application.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add the handlers to the app logger
    app.logger.addHandler(file_handler)

    # Set the log level for the app logger
    app.logger.setLevel(logging.DEBUG)
