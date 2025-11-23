import datetime
import requests

import flask
from flask_login import current_user
from flask_login import login_required

from app import db, csrf
from . import accounts
from ..models import Message
from ..models import Conversation
from .forms import ProfileUpdateForm
from .forms import PasswordChangeForm


@accounts.route("/profile")
@login_required
def profile():
    """
    Display user profile page.

    :return: Rendered profile template
    """
    return flask.render_template("accounts/profile.html")


@accounts.route("/update-profile", methods=["GET", "POST"])
@login_required
def update_profile():
    """
    Handle user profile update process.

    :return: Rendered update profile template or flask.redirect
    """
    form = ProfileUpdateForm(obj=current_user, user=current_user)

    if form.validate_on_submit():
        # Update user details
        details = {
            "fullName": form.fullName.data,
            "username": form.username.data,
            "emailAddress": form.emailAddress.data.lower(),
            "allowDataUsage": form.allowDataUsage.data,
        }

        # Update user profile
        success, message = current_user.update(details)

        if success:
            flask.flash("Profile updated successfully", "success")
        else:
            flask.flash(message, "error")

        return flask.redirect(flask.url_for("accounts.profile"))

    return flask.render_template("accounts/update_profile.html", form=form)


@accounts.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    """
    Handle user password change process.

    :return: Rendered change password template or flask.redirect
    """
    form = PasswordChangeForm()

    if form.validate_on_submit():
        success, message = current_user.updatePassword(
            form.currentPassword.data, form.newPassword.data
        )

        if success:
            flask.flash("Password changed successfully", "success")
            return flask.redirect(flask.url_for("accounts.profile"))
        else:
            flask.flash(message, "error")

    return flask.render_template("accounts/change_password.html", form=form)


@accounts.route("/delete-account", methods=["POST"])
@login_required
def delete_account():
    """
    Handle user account deletion.

    :return: Redirect to login page
    """
    success, message = current_user.delete()

    if success:
        flask.flash("Your account has been deleted", "success")
        return flask.redirect(flask.url_for("authentication.user_login"))

    flask.flash(message, "error")
    return flask.redirect(flask.url_for("accounts.profile"))


@accounts.route("/conversations")
@login_required
def user_conversations():
    """
    Display user's conversation history.

    :return: Rendered conversations template
    """
    conversations = (
        Conversation.query#.filter_by(userId=current_user.userId)
        .order_by(Conversation.lastUpdated.desc())
        .all()
    )
    return flask.render_template(
        "accounts/conversations.html", conversations=conversations
    )


@accounts.route("/conversation/<int:conversation_id>")
@login_required
def view_conversation(conversation_id):
    """
    Display a specific conversation.

    :param conversation_id: ID of the conversation to view
    :return: Rendered conversation template
    """
    conversation = Conversation.query.filter_by(
        conversationId=conversation_id, #userId=current_user.userId
    ).first_or_404()

    messages = (
        Message.query.filter_by(conversationId=conversation.conversationId)
        .order_by(Message.dateCreated)
        .all()
    )

    conversations = (
        Conversation.query#.filter_by(userId=current_user.userId)
        .order_by(Conversation.lastUpdated.desc())
        .limit(30)
    )

    return flask.render_template(
        "accounts/conversation_detail.html",
        conversation=conversation,
        conversations=conversations,
        messages=messages,
    )


@accounts.route("/create-conversation", methods=["POST"])
@login_required
def create_conversation():
    """
    Create a new conversation.

    :return: Redirect to the new conversation
    """
    title = flask.request.form.get(
        "title",
        f"Settlement Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )

    try:
        conversation = Conversation.create(
            {"userId": current_user.userId, "title": title}
        )

        return flask.redirect(
            flask.url_for(
                "accounts.view_conversation",
                conversation_id=conversation.conversationId,
            )
        )
    except Exception as e:
        flask.flash(f"Error creating conversation: {str(e)}", "error")
        return flask.redirect(flask.url_for("accounts.user_conversations"))


@accounts.route("/api/conversation/<int:conversation_id>/message", methods=["POST"])
@csrf.exempt
@login_required
def add_message(conversation_id):
    """
    API endpoint to add a message to a conversation and get a response.

    :param conversation_id: ID of the conversation
    :return: JSON response
    """
    conversation = Conversation.query.filter_by(
        conversationId=conversation_id#, userId=current_user.userId
    ).first_or_404()

    # Get message content from request
    data = flask.request.get_json()
    if not data or "content" not in data:
        return flask.jsonify({"error": "No message content provided"}), 400

    message_content = data["content"].strip()
    if not message_content:
        return flask.jsonify({"error": "Message content cannot be empty"}), 400

    # Extract additional options
    use_context = data.get("include_context", True)
    language_detection = data.get("language_detection", True)
    show_sources = data.get("show_sources", False)

    try:
        # Get conversation history for context
        conversation_context = {}
        recent_messages = (
            Message.query.filter_by(conversationId=conversation_id)
            .order_by(Message.dateCreated.desc())
            .limit(10)
            .all()
        )

        if recent_messages:
            conversation_context = {
                "previous_messages": [
                    {
                        "content": msg.content,
                        "is_user_message": msg.isUserMessage,
                        "intent_type": msg.intentType,
                        "topic": msg.topic,
                        "timestamp": msg.dateCreated.isoformat()
                        if msg.dateCreated
                        else None,
                    }
                    # Last 5 messages
                    for msg in reversed(recent_messages[-5:])
                ],
                "conversation_topic": recent_messages[0].topic
                if recent_messages[0].topic
                else None,
                "user_preferences": {
                    "language": "auto",
                    "detail_level": "comprehensive",
                },
            }

        # Save user message to database first
        user_message = Message.create(
            {
                "conversationId": conversation_id,
                "isUserMessage": True,
                "content": message_content,
                "tokenCount": len(message_content.split()),
            }
        )

        # Prepare request for SettleBot API
        api_url = flask.current_app.config.get(
            "SETTLEBOT_API_URL", "http://localhost:8000"
        )
        api_key = flask.current_app.config.get("SETTLEBOT_API_KEY")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

        # Enhanced request payload for SettleBot API
        request_payload = {
            "query": message_content,
            "top_k": 15,
            "include_context": use_context,
            "language_detection": language_detection,
            "conversation_context": conversation_context,
            "user_preferences": {
                "response_style": "comprehensive_empathetic",
                "include_safety_protocols": "true" if True else "false",
                "include_cost_information": "true" if True else "false",
                "show_sources": "true" if show_sources else "false",
            },
        }
        response = requests.post(
            f"{api_url}/query",
            json=request_payload,
            headers=headers,
            timeout=None,
        )
        if response.status_code == 401:
            return flask.jsonify({"error": "API authentication failed"}), 500
        elif response.status_code == 422:
            error_detail = response.json().get("detail", "Validation error")
            return (
                flask.jsonify({"error": f"Invalid request: {error_detail}"}),
                400,
            )
        elif response.status_code != 200:
            return (
                flask.jsonify(
                    {"error": f"SettleBot API error: {response.status_code}"}
                ),
                500,
            )

        # Process SettleBot API response
        settlebot_response = response.json()

        # Extract comprehensive response data
        response_text = settlebot_response.get(
            "response", "I apologize, but I couldn't generate a response."
        )
        intent_type = settlebot_response.get("intent_type", "general_query")
        topic = settlebot_response.get("topic", "general")
        confidence = settlebot_response.get("confidence", 0.0)

        # Language information
        language_info = settlebot_response.get("language_info", {})
        detected_language = language_info.get("detected_language", "english")
        translation_needed = language_info.get("translation_needed", False)

        # Settlement-specific information
        empathy_applied = settlebot_response.get("empathy_applied", False)
        safety_protocols_added = settlebot_response.get(
            "safety_protocols_added", False
        )
        crisis_level = settlebot_response.get("crisis_level", "none")
        emotional_state = settlebot_response.get("emotional_state")
        web_search_used = settlebot_response.get("web_search_used", False)

        # Token usage and context information
        token_usage = settlebot_response.get("token_usage", {})
        token_count = token_usage.get(
            "total_tokens", len(response_text.split())
        )
        retrieved_chunks = settlebot_response.get("retrieved_chunks", [])

        # Save assistant message to database with enhanced metadata
        assistant_message = Message.create(
            {
                "conversationId": conversation_id,
                "isUserMessage": False,
                "content": response_text,
                "intentType": intent_type,
                "topic": topic,
                "confidence": confidence,
                "tokenCount": token_count,
            }
        )

        # Update conversation timestamp and title if it's a new conversation
        if (
            conversation.title.startswith("Conversation ")
            or conversation.title == "New Conversation"
        ):
            # Generate a better title based on the intent and topic
            title_mapping = {
                "housing_inquiry": f"Housing in {topic.title()}",
                "transportation": "Transport & Travel",
                "safety_concern": "Safety & Security",
                "university_info": "University Information",
                "immigration_visa": "Visa & Immigration",
                "banking_finance": "Banking & Finance",
                "healthcare": "Healthcare & Medical",
                "cultural_adaptation": "Cultural Guidance",
                "emergency_help": "Emergency Assistance",
            }

            new_title = title_mapping.get(
                intent_type, f"{topic.title()} Assistance"
            )
            conversation.update({"title": new_title})

        conversation.lastUpdated = datetime.datetime.utcnow()
        db.session.commit()

        # Prepare comprehensive response
        response_data = {
            "user_message": user_message.getDetails(),
            "assistant_message": assistant_message.getDetails(),
            "settlement_info": {
                "intent_type": intent_type,
                "topic": topic,
                "confidence": confidence,
                "detected_language": detected_language,
                "translation_needed": translation_needed,
                "empathy_applied": empathy_applied,
                "safety_protocols_added": safety_protocols_added,
                "crisis_level": crisis_level,
                "emotional_state": emotional_state,
                "web_search_used": web_search_used,
            },
            "token_usage": token_usage,
            "current_time": settlebot_response.get("current_time"),
            "conversation_updated": True,
        }

        # Add context information if requested
        if show_sources and retrieved_chunks:
            response_data["retrieved_chunks"] = retrieved_chunks[:5]

        return flask.jsonify(response_data)

    except requests.exceptions.Timeout:
        return (
            flask.jsonify(
                {
                    "error": "SettleBot service timeout"
                }
            ),
            504,
        )
    except requests.exceptions.ConnectionError:
        return (
            flask.jsonify(
                {
                    "error": "Cannot connect to SettleBot service"
                }
            ),
            503,
        )
    except requests.exceptions.RequestException as e:
        return (
            flask.jsonify({"error": f"SettleBot service error: {str(e)}"}),
            500,
        )
    except Exception as e:
        db.session.rollback()
        flask.current_app.logger.error(f"Error in add_message: {str(e)}")
        return flask.jsonify({"error": f"Server error: {str(e)}"}), 500
