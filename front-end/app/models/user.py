from datetime import UTC
from datetime import datetime

import flask
from flask_login import UserMixin
from flask_login import login_user
from flask_login import logout_user
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer

from app import db
from utilities.email_utils import send_email
from utilities.securities import get_gravatar_hash


class User(UserMixin, db.Model):
    """Model representing a User."""

    __tablename__ = "user"

    userId = db.Column(db.Integer, autoincrement=True, primary_key=True)
    fullName = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False, index=True)
    emailAddress = db.Column(
        db.String(100), unique=True, nullable=False, index=True
    )
    passwordHash = db.Column(db.String(255))
    role = db.Column(db.String(20), nullable=False, default="student")
    dateCreated = db.Column(db.DateTime, default=db.func.current_timestamp())
    lastLogin = db.Column(db.DateTime, nullable=True)
    isActive = db.Column(db.Boolean, default=True)
    allowDataUsage = db.Column(db.Boolean, default=False)
    lastUpdated = db.Column(
        db.DateTime,
        default=db.func.current_timestamp(),
        onupdate=db.func.current_timestamp(),
    )
    avatarHash = db.Column(db.String(255))

    # Relationships
    conversations = db.relationship(
        "Conversation",
        back_populates="user",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """String representation of the User."""
        return f"User(userId={self.userId}, fullName='{self.fullName}')"

    @property
    def password(self) -> AttributeError:
        """Prevent password from being accessed."""
        raise AttributeError("password is not a readable attribute")

    @password.setter
    def password(self, password: str) -> None:
        """Hash the user's password."""
        self.passwordHash = generate_password_hash(password)

    def get_id(self) -> int:
        """Override get_id method for Flask-Login."""
        return self.userId

    def getGravatar(self, size=100, default="identicon", rating="g"):
        """
        Generates a Gravatar URL for the user based on their email address.

        :param size: int - The size of the Gravatar image
        :param default: str - The default image to be displayed if no Gravatar is found
        :param rating: str - The content rating for the image
        :return: str - The Gravatar URL
        """
        url = "https://secure.gravatar.com/avatar"

        # Generate avatar hash if it does not exist
        if not self.avatarHash:
            self.avatarHash = get_gravatar_hash(self.emailAddress)
            db.session.commit()

        hash = self.avatarHash
        return f"{url}/{hash}?s={size}&d={default}&r={rating}"

    def verifyPassword(self, password: str) -> bool:
        """
        Verify password against stored hash.

        :param password: str - The password to verify
        :return: bool - True if the password matches, False otherwise
        """
        return check_password_hash(self.passwordHash, password)

    def generateResetToken(self) -> str:
        """
        Generate a password reset token.

        :return: str - The generated token
        """
        serializer = Serializer(flask.current_app.config["SECRET_KEY"])
        return serializer.dumps({"userId": self.userId}, salt="password-reset")

    @classmethod
    def verifyResetToken(cls, token: str, expiration=3600):
        """
        Verify the reset token.

        :param token: str - The token to verify
        :param expiration: int - Expiration time in seconds
        :return: User or None - The user if token is valid, None otherwise
        """
        serializer = Serializer(flask.current_app.config["SECRET_KEY"])
        try:
            data = serializer.loads(
                token, salt="password-reset", max_age=expiration
            )
            user_id = data.get("userId")
            if not user_id:
                return None
            return User.query.get(user_id)

        except Exception:
            return None

    def login(self, details: dict) -> tuple:
        """
        Log in a user.

        :param details: dict - Login details including password and remember_me
        :return: tuple - (success_flag, message)
        """
        if not self.verifyPassword(details.get("password")):
            return (False, "Invalid password")

        if not self.isActive:
            return (False, "Account is inactive")

        login_user(self, remember=details.get("remember_me", False))
        self.lastLogin = None
        db.session.commit()

        # Store role in flask.session
        flask.session["user_role"] = self.role

        return (True, "Login successful")

    def logout(self) -> tuple:
        """
        Log out a user.

        :return: tuple - (success_flag, message)
        """
        self.lastLogin = datetime.now(UTC)
        logout_user()

        # Clear role from flask.session
        flask.session.pop("user_role", None)

        return (True, "Logout successful")

    def update(self, details: dict) -> tuple:
        """
        Update user details.

        :param details: dict - New user details
        :return: tuple - (success_flag, message)
        """
        try:
            for key, value in details.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            db.session.commit()
            return (True, "User details updated successfully")
        except Exception as e:
            db.session.rollback()
            return (False, f"Failed to update user details: {str(e)}")

    def updatePassword(self, current_password: str, new_password: str) -> tuple:
        """
        Update user password.

        :param current_password: str - Current password
        :param new_password: str - New password
        :return: tuple - (success_flag, message)
        """
        if not self.verifyPassword(current_password):
            return (False, "Current password is incorrect")

        try:
            self.password = new_password
            db.session.commit()
            return (True, "Password updated successfully")
        except Exception as e:
            db.session.rollback()
            return (False, f"Failed to update password: {str(e)}")

    def delete(self) -> tuple:
        """
        Delete user account.

        :return: tuple - (success_flag, message)
        """
        try:
            db.session.delete(self)
            db.session.commit()
            return (True, "User account deleted successfully")
        except Exception as e:
            db.session.rollback()
            return (False, f"Failed to delete user account: {str(e)}")

    def sendPasswordResetEmail(self) -> tuple:
        """
        Send password reset email to user.

        :return: tuple - (success_flag, message)
        """
        try:
            token = self.generateResetToken()
            reset_url = flask.url_for(
                "authentication.password_reset",
                token=token,
                _scheme="https",
                _external=True,
            )
            send_email(
                to=[self.emailAddress],
                subject="Password Reset",
                template="email/password_reset",
                user=self,
                reset_url=reset_url,
            )

            return (True, "Password reset email sent successfully")
        except Exception as e:
            return (False, f"Failed to send password reset email: {str(e)}")

    @classmethod
    def resetPassword(cls, token: str, new_password: str) -> tuple:
        """
        Reset password using token.

        :param token: str - Password reset token
        :param new_password: str - New password
        :return: tuple - (success_flag, message)
        """
        user = cls.verifyResetToken(token)
        if not user:
            return (False, "Invalid or expired token")

        try:
            user.password = new_password
            db.session.commit()
            return (True, "Password reset successfully")
        except Exception as e:
            db.session.rollback()
            return (False, f"Failed to reset password: {str(e)}")

    @classmethod
    def create(cls, details: dict) -> "User":
        """
        Register a new user account.

        :param details: dict - User details
        :return: User - The created user
        """
        try:
            user = cls(
                fullName=details.get("fullName"),
                username=details.get("username").lower(),
                emailAddress=details.get("emailAddress").lower(),
                role=details.get("role", "student"),
                password=details.get("password"),
                allowDataUsage=details.get("allowDataUsage", False),
            )

            # Generate avatar hash
            if user.emailAddress:
                user.avatarHash = get_gravatar_hash(user.emailAddress)

            db.session.add(user)
            db.session.flush()

            user.lastLogin = user.dateCreated
            db.session.commit()

            # Send welcome email
            send_email(
                to=[user.emailAddress],
                subject="Welcome to StrathQuery",
                template="email/welcome",
                user=user,
            )

            return user
        except Exception as e:
            db.session.rollback()
            raise ValueError(f"Failed to create user: {str(e)}")

    def getDetails(self) -> dict:
        """
        Get user details as a dictionary.

        :return: dict - User details
        """
        return {
            "userId": self.userId,
            "fullName": self.fullName,
            "username": self.username,
            "emailAddress": self.emailAddress,
            "avatarUrl": self.getGravatar(size=250),
            "role": self.role,
            "dateCreated": self.dateCreated.isoformat()
            if self.dateCreated
            else None,
            "lastLogin": self.lastLogin.isoformat() if self.lastLogin else None,
            "isActive": self.isActive,
            "allowDataUsage": self.allowDataUsage,
        }
