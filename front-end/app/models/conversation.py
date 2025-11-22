from app import db


class Conversation(db.Model):
    """Model representing a conversation."""

    __tablename__ = "conversation"

    conversationId = db.Column(db.Integer, primary_key=True, autoincrement=True)
    userId = db.Column(
        db.Integer,
        db.ForeignKey("user.userId", ondelete="CASCADE"),
        nullable=False,
    )
    title = db.Column(db.String(100), nullable=False)
    dateCreated = db.Column(db.DateTime, default=db.func.current_timestamp())
    lastUpdated = db.Column(
        db.DateTime,
        default=db.func.current_timestamp(),
        onupdate=db.func.current_timestamp(),
    )

    # Relationships
    user = db.relationship("User", back_populates="conversations")
    messages = db.relationship(
        "Message",
        back_populates="conversation",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """String representation of the Conversation."""
        return f"Conversation(conversationId={self.conversationId}, title='{self.title}')"

    @classmethod
    def create(cls, details: dict) -> "Conversation":
        """
        Create a new conversation.

        :param details: dict - Conversation details
        :return: Conversation - The created conversation
        """
        try:
            conversation = cls(
                userId=details.get("userId"),
                title=details.get("title", "New Conversation"),
            )

            db.session.add(conversation)
            db.session.commit()

            return conversation

        except Exception as e:
            db.session.rollback()
            raise ValueError(f"Failed to create conversation: {str(e)}")

    def update(self, details: dict) -> tuple:
        """
        Update conversation details.

        :param details: dict - New conversation details
        :return: tuple - (success_flag, message)
        """
        try:
            for key, value in details.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            db.session.commit()
            return (True, "Conversation updated successfully")
        except Exception as e:
            db.session.rollback()
            return (False, f"Failed to update conversation: {str(e)}")

    def delete(self) -> tuple:
        """
        Delete this conversation.

        :return: tuple - (success_flag, message)
        """
        try:
            db.session.delete(self)
            db.session.commit()
            return (True, "Conversation deleted successfully")

        except Exception as e:
            db.session.rollback()
            return (False, f"Failed to delete conversation: {str(e)}")

    def getDetails(self) -> dict:
        """
        Get conversation details as a dictionary.

        :return: dict - Conversation details
        """
        return {
            "conversationId": self.conversationId,
            "userId": self.userId,
            "title": self.title,
            "dateCreated": self.dateCreated.isoformat()
            if self.dateCreated
            else None,
            "lastUpdated": self.lastUpdated.isoformat()
            if self.lastUpdated
            else None,
            "messageCount": self.messages.count(),
        }
