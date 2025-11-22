import flask
from flask_login import current_user

from . import authentication
from .forms import LoginForm
from .forms import PasswordResetForm
from .forms import UserRegistrationForm

from ..models import User
from utilities.securities import load_deleted_accounts


@authentication.route("/register", methods=["GET", "POST"])
def user_registration():
    """
    Handle user registration process.

    :return: Rendered registration template or redirect to login page
    """
    # Limit functionality to anonymous users
    if current_user.is_authenticated:
        return flask.redirect(flask.url_for("main.index"))

    # Initialize the registration form
    form = UserRegistrationForm()

    if form.validate_on_submit():
        details = {
            "fullName": form.fullName.data,
            "username": form.username.data,
            "emailAddress": form.emailAddress.data.lower(),
            "password": form.password.data,
        }
        User.create(details)

        flask.flash(f"{form.fullName.data} registered successfully.", "success")
        return flask.redirect(flask.url_for("authentication.user_login"))

    return flask.render_template(
        "authentication/user_registration.html", form=form
    )


@authentication.route("/sign-in", methods=["GET", "POST"])
def user_login():
    """
    Handle user login process.

    :return: Rendered login template or redirect to profile page
    """
    # Limit functionality to Anonymous Users
    if current_user.is_authenticated:
        return flask.redirect(flask.url_for("accounts.profile"))

    form = LoginForm()

    if form.validate_on_submit():
        details = {
            "password": form.password.data,
            "remember_me": form.remember_me.data,
        }

        # Find user with given email address
        user = User.query.filter_by(
            emailAddress=form.emailAddress.data.lower()
        ).first()

        # Check if user is active
        if not user.isActive:
            flask.flash(
                "Sorry. Your account is currently deactivated."
                + " Please reach our help desk for assistance."
            )
            return flask.redirect(flask.url_for("authentication.user_login"))

        # Check if the email is in the deleted accounts list
        deleted_accounts = load_deleted_accounts()
        if form.emailAddress.data.lower() in deleted_accounts:
            flask.flash("This account is no longer active.", "error")
            return flask.redirect(flask.url_for("authentication.user_login"))

        # Login user if found
        if user:
            success, message = user.login(details)

            if success:
                next_page = flask.request.args.get("next")
                if not next_page or not next_page.startswith("/"):
                    next_page = flask.url_for("accounts.profile")

                flask.flash(f"Hello {current_user.fullName}. Welcome back!")
                return flask.redirect(next_page)

        # Notify user of invalid credentials
        flask.flash(
            "You provided invalid credentials. Please try again.", "warning"
        )

    return flask.render_template("authentication/user_login.html", form=form)


@authentication.route("/sign-out")
def user_logout():
    """
    Handle user logout process.

    :return: Redirect to login page
    """
    current_user.logout()
    flask.flash("You have been logged out successfully.")
    return flask.redirect(flask.url_for("authentication.user_login"))


@authentication.route("/password-reset/request", methods=["GET", "POST"])
def password_reset_request():
    """
    Handle password reset request process.

    :return: Rendered password reset request template or redirect
    """
    if flask.request.method == "POST":
        # Retrieve user
        email_address = flask.request.form["email"]
        user = User.query.filter_by(emailAddress=email_address).first()

        # Check if user exists
        if user:
            # Send password reset email
            user.sendPasswordResetEmail()

            # Flash success message
            flask.flash("Password reset email sent successfully", "success")
            return flask.redirect(
                flask.url_for("authentication.password_reset_request")
            )

        # Flash error message
        flask.flash("The provided email address is invalid", "failure")

    return flask.render_template("authentication/password_reset_request.html")


@authentication.route("/password-reset/<token>", methods=["GET", "POST"])
def password_reset(token):
    """
    Handle password reset process.

    :param token: Password reset token
    :return: Rendered password reset template or redirect
    """
    # Functionality limited to stranded users
    if not current_user.is_anonymous:
        flask.flash(
            "Reset password is for users that have forgotten their "
            + "passwords. To update your password, simply go to profile page",
            "warning",
        )
        return flask.redirect(flask.url_for("accounts.profile"))

    # Handle form rendering and submission
    form = PasswordResetForm()
    if form.validate_on_submit():
        # Reset user's password
        success, message = User.resetPassword(token, form.password.data)

        # Handle successful reset
        if success:
            flask.flash("Password updated successfully", "success")
            return flask.redirect(flask.url_for("authentication.user_login"))

        # Flash failure message
        flask.flash(
            "The link you used is either expired or corrupted", "warning"
        )

    return flask.render_template(
        "authentication/password_reset.html", form=form
    )


@authentication.route("/reauthenticate")
def reauthenticate():
    """
    Handle reauthentication process.

    :return: Redirect to logout
    """
    return flask.redirect(flask.url_for("authentication.user_logout"))
