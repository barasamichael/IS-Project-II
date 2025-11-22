from wtforms import EmailField
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms import SubmitField
from wtforms import BooleanField
from wtforms import PasswordField
from wtforms import ValidationError

from wtforms.validators import Email
from wtforms.validators import Length
from wtforms.validators import Regexp
from wtforms.validators import EqualTo
from wtforms.validators import DataRequired

from ..models import User


class ProfileUpdateForm(FlaskForm):
    """
    Form for updating user profile details.
    """

    fullName = StringField(
        "Full Name",
        validators=[
            DataRequired(message="Full name is required"),
            Length(
                min=2,
                max=110,
                message="Full name must be between 2 and 110 characters",
            ),
        ],
    )
    username = StringField(
        "Username",
        validators=[
            DataRequired(message="Username is required"),
            Length(
                min=3,
                max=100,
                message="Username must be between 3 and 100 characters",
            ),
            Regexp(
                r"^[A-Za-z0-9_]+$",
                message="Username can only contain letters, numbers, and underscores",
            ),
        ],
    )
    emailAddress = EmailField(
        "Email Address",
        validators=[
            DataRequired(message="Email address is required"),
            Email(message="Please enter a valid email address"),
            Length(max=120, message="Email cannot exceed 120 characters"),
        ],
    )
    allowDataUsage = BooleanField(
        "Allow StrathQuery Team to use my conversations to improve performance"
    )
    submit = SubmitField("Update Profile")

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop("user", None)
        super(ProfileUpdateForm, self).__init__(*args, **kwargs)

    def validate_username(self, field):
        """Validate username is unique unless it's unchanged."""
        if self.user and field.data != self.user.username:
            user = User.query.filter_by(username=field.data).first()
            if user:
                raise ValidationError(
                    "This username is already in use. Please choose another."
                )

    def validate_emailAddress(self, field):
        """Validate email is unique unless it's unchanged."""
        if self.user and field.data.lower() != self.user.emailAddress.lower():
            user = User.query.filter_by(emailAddress=field.data.lower()).first()
            if user:
                raise ValidationError(
                    "This email address is already in use. Please use another."
                )


class PasswordChangeForm(FlaskForm):
    """
    Form for changing user password.
    """

    currentPassword = PasswordField(
        "Current Password",
        validators=[DataRequired(message="Please enter your current password")],
    )
    newPassword = PasswordField(
        "New Password",
        validators=[
            DataRequired(message="Please enter a new password"),
            Length(
                min=8, message="Password must be at least 8 characters long"
            ),
            Regexp(
                r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])",
                message="Password must include at least one lowercase letter, one uppercase letter, and one number",
            ),
        ],
    )
    confirmPassword = PasswordField(
        "Confirm New Password",
        validators=[
            DataRequired(message="Please confirm your new password"),
            EqualTo("newPassword", message="Passwords must match"),
        ],
    )
    submit = SubmitField("Change Password")
