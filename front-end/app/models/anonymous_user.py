from app import db
from datetime import datetime
from datetime import timedelta


class AnonymousSession(db.Model):
    """Model for tracking anonymous user sessions."""

    __tablename__ = "anonymous_session"

    sessionId = db.Column(db.String(64), primary_key=True)
    sessionToken = db.Column(db.String(128), unique=True,
                             nullable=False, index=True)
    ipAddress = db.Column(db.String(45))
    userAgent = db.Column(db.String(255))
    dateCreated = db.Column(db.DateTime, default=db.func.current_timestamp())
    lastActivity = db.Column(
        db.DateTime,
        default=db.func.current_timestamp(),
        onupdate=db.func.current_timestamp()
    )
    expiresAt = db.Column(db.DateTime, nullable=False)
    isActive = db.Column(db.Boolean, default=True)

    # Relationships
    conversations = db.relationship(
        "AnonymousConversation",
        back_populates="session",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"AnonymousSession(sessionId={self.sessionId})"

    @classmethod
    def create(cls, session_duration_hours=24):
        """
        Create a new anonymous session.

        :param session_duration_hours: Session validity duration
        :return: AnonymousSession
        """
        import secrets
        from flask import request

        try:
            session_id = secrets.token_urlsafe(32)
            session_token = secrets.token_urlsafe(64)

            session = cls(
                sessionId=session_id,
                sessionToken=session_token,
                ipAddress=request.remote_addr if request else None,
                userAgent=request.headers.get(
                    'User-Agent')[:255] if request else None,
                expiresAt=datetime.utcnow() + timedelta(hours=session_duration_hours)
            )

            db.session.add(session)
            db.session.commit()

            return session

        except Exception as e:
            db.session.rollback()
            raise ValueError(f"Failed to create anonymous session: {str(e)}")

    @classmethod
    def get_by_token(cls, token):
        """Get session by token if valid."""
        session = cls.query.filter_by(
            sessionToken=token, isActive=True).first()

        if session and session.is_expired():
            session.deactivate()
            return None

        return session

    def is_expired(self):
        """Check if session has expired."""
        return datetime.utcnow() > self.expiresAt

    def refresh_activity(self):
        """Update last activity timestamp."""
        self.lastActivity = datetime.utcnow()
        db.session.commit()

    def deactivate(self):
        """Deactivate this session."""
        self.isActive = False
        db.session.commit()

    def extend_expiration(self, hours=24):
        """Extend session expiration."""
        self.expiresAt = datetime.utcnow() + timedelta(hours=hours)
        db.session.commit()


class AnonymousConversation(db.Model):
    """Model for anonymous user conversations."""

    __tablename__ = "anonymous_conversation"

    conversationId = db.Column(
        db.Integer, primary_key=True, autoincrement=True)
    sessionId = db.Column(
        db.String(64),
        db.ForeignKey("anonymous_session.sessionId", ondelete="CASCADE"),
        nullable=False
    )
    title = db.Column(db.String(100), nullable=False)
    dateCreated = db.Column(db.DateTime, default=db.func.current_timestamp())
    lastUpdated = db.Column(
        db.DateTime,
        default=db.func.current_timestamp(),
        onupdate=db.func.current_timestamp()
    )

    # Relationships
    session = db.relationship(
        "AnonymousSession", back_populates="conversations")
    messages = db.relationship(
        "AnonymousMessage",
        back_populates="conversation",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"AnonymousConversation(conversationId={self.conversationId})"

    @classmethod
    def create(cls, details):
        """Create new anonymous conversation."""
        try:
            conversation = cls(
                sessionId=details.get("sessionId"),
                title=details.get("title", "Anonymous Chat")
            )

            db.session.add(conversation)
            db.session.commit()

            return conversation

        except Exception as e:
            db.session.rollback()
            raise ValueError(f"Failed to create conversation: {str(e)}")

    def getDetails(self):
        """Get conversation details."""
        return {
            "conversationId": self.conversationId,
            "sessionId": self.sessionId,
            "title": self.title,
            "dateCreated": self.dateCreated.isoformat() if self.dateCreated else None,
            "lastUpdated": self.lastUpdated.isoformat() if self.lastUpdated else None,
            "messageCount": self.messages.count()
        }


class AnonymousMessage(db.Model):
    """Model for messages in anonymous conversations."""

    __tablename__ = "anonymous_message"

    messageId = db.Column(db.Integer, primary_key=True, autoincrement=True)
    conversationId = db.Column(
        db.Integer,
        db.ForeignKey("anonymous_conversation.conversationId",
                      ondelete="CASCADE"),
        nullable=False
    )
    isUserMessage = db.Column(db.Boolean, nullable=False)
    content = db.Column(db.Text, nullable=False)
    intentType = db.Column(db.String(50))
    topic = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    tokenCount = db.Column(db.Integer)
    dateCreated = db.Column(db.DateTime, default=db.func.current_timestamp())

    # Relationships
    conversation = db.relationship(
        "AnonymousConversation", back_populates="messages")

    def __repr__(self):
        return f"AnonymousMessage(messageId={self.messageId})"

    @classmethod
    def create(cls, details):
        """Create new anonymous message."""
        try:
            message = cls(
                conversationId=details.get("conversationId"),
                isUserMessage=details.get("isUserMessage"),
                content=details.get("content"),
                intentType=details.get("intentType"),
                topic=details.get("topic"),
                confidence=details.get("confidence"),
                tokenCount=details.get("tokenCount")
            )

            db.session.add(message)
            db.session.commit()

            return message

        except Exception as e:
            db.session.rollback()
            raise ValueError(f"Failed to create message: {str(e)}")

    def getDetails(self):
        """Get message details."""
        return {
            "messageId": self.messageId,
            "conversationId": self.conversationId,
            "isUserMessage": self.isUserMessage,
            "content": self.content,
            "intentType": self.intentType,
            "topic": self.topic,
            "confidence": self.confidence,
            "tokenCount": self.tokenCount,
            "dateCreated": self.dateCreated.isoformat() if self.dateCreated else None
        }
