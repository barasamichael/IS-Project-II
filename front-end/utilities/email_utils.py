from threading import Thread
from flask import current_app
from flask import render_template
from flask_mailman import EmailMultiAlternatives


def send_async_email(app, message):
    """
    Asynchronously sends an email using Flask-Mailman within the given Flask
        app context.

    Parameters:
        - app: Flask app object
        - msg: Message object containing email details
    """
    with app.app_context():
        message.send()


def send_email(to, subject, template, cc=None, bcc=None, **kwargs):
    """
    Asynchronously send an email using Flask-Mailman with support for HTML templates only.

    Parameters:
        - to: Email recipient(s) (string or list of strings)
        - subject: Email subject
        - template: Base name of the email template (without the file extension)
        - cc: Carbon copy recipients (optional, string or list of strings)
        - bcc: Blind carbon copy recipients (optional, string or list of strings)
        - **kwargs: Additional keyword arguments to pass to the email template

    Returns:
        - Thread object representing the asynchronous email sending process
    """
    app = current_app._get_current_object()

    rendered_html = render_template(template + ".html", **kwargs)

    # Create the EmailMultiAlternatives message
    message = EmailMultiAlternatives(subject=subject, body=rendered_html, to=to)

    # Attach HTML version of the email
    message.attach_alternative(rendered_html, "text/html")

    # Add CC and BCC if provided
    if cc:
        if isinstance(cc, str):
            cc = [cc]
        message.cc = cc

    if bcc:
        if isinstance(bcc, str):
            bcc = [bcc]
        message.bcc = bcc

    # Send the email asynchronously
    thread = Thread(target=send_async_email, args=[app, message])
    thread.start()
