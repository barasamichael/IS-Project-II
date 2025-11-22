from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms import SubmitField
from wtforms import BooleanField
from wtforms import PasswordField
from wtforms import ValidationError

from wtforms.validators import Email
from wtforms.validators import Length
from wtforms.validators import EqualTo
from wtforms.validators import DataRequired

from ..models import User


class LoginForm(FlaskForm):
    emailAddress = StringField(
        "Enter your email address",
        validators=[DataRequired(), Length(1, 128), Email()],
        render_kw={"placeholder": "Enter email address here"},
    )
    password = PasswordField(
        "Enter your password",
        validators=[DataRequired()],
        render_kw={"placeholder": "Enter password here"},
    )
    remember_me = BooleanField("Keep me signed in")
    submit = SubmitField("Sign In")


class PasswordResetForm(FlaskForm):
    password = PasswordField(
        "Enter your Password",
        validators=[
            DataRequired(),
        ],
        render_kw={
            "autocomplete": "new-password",
        },
    )
    confirmPassword = PasswordField(
        "Confirm your password",
        validators=[DataRequired(), EqualTo("password")],
        render_kw={"autocomplete": "new-password"},
    )
    submit = SubmitField("Submit")


class UserRegistrationForm(FlaskForm):
    fullName = StringField(
        "Enter your full name", validators=[DataRequired(), Length(1, 110)]
    )
    username = StringField(
        "Enter your username", validators=[DataRequired(), Length(1, 100)]
    )
    emailAddress = StringField(
        "Enter your email address",
        validators=[DataRequired(), Length(1, 120), Email()],
    )
    password = PasswordField(
        "Enter your Password",
        validators=[
            DataRequired(),
        ],
        render_kw={
            "autocomplete": "new-password",
        },
    )
    confirmPassword = PasswordField(
        "Confirm your password",
        validators=[DataRequired(), EqualTo("password")],
        render_kw={
            "autocomplete": "new-password",
        },
    )
    consent = BooleanField(
        "I agree to the Terms and Conditions",
        validators=[DataRequired()],
    )
    submit = SubmitField("Create Account")

    def validate_emailAddress(self, field):
        if User.query.filter_by(emailAddress=field.data.lower()).first():
            raise ValidationError("Email address already registered.")

    def validate_username(self, field):
        if User.query.filter_by(username=field.data.lower()).first():
            raise ValidationError("Username already registered.")
