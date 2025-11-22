from app import db


class Message(db.Model):
    """Model representing a message in a conversation."""

    __tablename__ = "message"

    messageId = db.Column(db.Integer, autoincrement=True, primary_key=True)
    conversationId = db.Column(
        db.Integer,
        db.ForeignKey("conversation.conversationId", ondelete="CASCADE"),
        nullable=False,
    )
    isUserMessage = db.Column(db.Boolean, nullable=False)
    content = db.Column(db.Text, nullable=False)
    intentType = db.Column(db.String(50))
    topic = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    tokenCount = db.Column(db.Integer)
    dateCreated = db.Column(db.DateTime, default=db.func.current_timestamp())
    lastUpdated = db.Column(
        db.DateTime,
        default=db.func.current_timestamp(),
        onupdate=db.func.current_timestamp(),
    )

    # Indexes for optimized admin queries
    __table_args__ = (
        db.Index("idx_message_intent_is_user", isUserMessage, intentType),
        db.Index("idx_message_topic_is_user", isUserMessage, topic),
        db.Index("idx_message_date_created", dateCreated),
    )

    # Relationships
    conversation = db.relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        """String representation of the Message."""
        return f"Message(messageId={self.messageId}, conversationId={self.conversationId})"

    @classmethod
    def create(cls, details: dict) -> "Message":
        """
        Create a new message.

        :param details: dict - Message details
        :return: Message - The created message
        """
        try:
            message = cls(
                conversationId=details.get("conversationId"),
                isUserMessage=details.get("isUserMessage"),
                content=details.get("content"),
                intentType=details.get("intentType"),
                topic=details.get("topic"),
                confidence=details.get("confidence"),
                tokenCount=details.get("tokenCount"),
            )

            db.session.add(message)
            db.session.commit()

            return message
        except Exception as e:
            db.session.rollback()
            raise ValueError(f"Failed to create message: {str(e)}")

    def update(self, details: dict) -> tuple:
        """
        Update message details.

        :param details: dict - New message details
        :return: tuple - (success_flag, message)
        """
        try:
            for key, value in details.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            db.session.commit()
            return (True, "Message updated successfully")

        except Exception as e:
            db.session.rollback()
            return (False, f"Failed to update message: {str(e)}")

    def delete(self) -> tuple:
        """
        Delete this message.

        :return: tuple - (success_flag, message)
        """
        try:
            db.session.delete(self)
            db.session.commit()
            return (True, "Message deleted successfully")

        except Exception as e:
            db.session.rollback()
            return (False, f"Failed to delete message: {str(e)}")

    def getDetails(self) -> dict:
        """
        Retrieve message details.

        :return: dict - Message details
        """
        return {
            "messageId": self.messageId,
            "conversationId": self.conversationId,
            "isUserMessage": self.isUserMessage,
            "content": self.content,
            "intentType": self.intentType,
            "topic": self.topic,
            "confidence": self.confidence,
            "tokenCount": self.tokenCount,
            "dateCreated": self.dateCreated.isoformat()
            if self.dateCreated
            else None,
            "lastUpdated": self.lastUpdated.isoformat()
            if self.lastUpdated
            else None,
        }
