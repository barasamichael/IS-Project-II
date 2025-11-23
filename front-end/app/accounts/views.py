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
        Conversation.query.filter_by(userId=current_user.userId)
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
        conversationId=conversation_id, userId=current_user.userId
    ).first_or_404()

    messages = (
        Message.query.filter_by(conversationId=conversation.conversationId)
        .order_by(Message.dateCreated)
        .all()
    )

    conversations = (
        Conversation.query.filter_by(userId=current_user.userId)
        .order_by(Conversation.lastUpdated.desc())
        .limit(30)
    )

    return flask.render_template(
        "accounts/conversation_detail.html",
        conversation=conversation,
        conversations=conversations,
        messages=messages,
    )


@accounts.route("/settlement-assistant", methods=["GET", "POST"])
@login_required
def settlement_assistant():
    """
    Enhanced settlement assistant interface using the new SettleBot API.

    :return: Rendered settlement assistant template
    """
    response_data = None
    query = None
    api_response_details = {}
    error_message = None

    if flask.request.method == "POST":
        query = flask.request.form.get("query", "").strip()

        if not query:
            flask.flash("Please enter a settlement question", "warning")
            return flask.render_template("accounts/settlement_assistant.html")

        api_url = flask.current_app.config.get(
            "SETTLEBOT_API_URL", "http://localhost:8000"
        )
        api_key = flask.current_app.config.get("SETTLEBOT_API_KEY")

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else "",
            }

            # Enhanced request payload for SettleBot API
            request_payload = {
                "query": query,
                "top_k": 15,
                "include_context": True,
                "language_detection": True,
                "user_preferences": {
                    "response_style": "comprehensive",
                    "include_safety_info": True,
                    "include_cost_info": True,
                },
            }

            response = requests.post(
                f"{api_url}/query",
                json=request_payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                api_response = response.json()

                # Extract main response
                response_data = api_response.get("response", "")

                # Extract detailed API response information
                api_response_details = {
                    "intent_type": api_response.get("intent_type", "Unknown"),
                    "topic": api_response.get("topic", "Unknown"),
                    "confidence": api_response.get("confidence", 0.0),
                    "language_info": api_response.get("language_info", {}),
                    "empathy_applied": api_response.get(
                        "empathy_applied", False
                    ),
                    "safety_protocols_added": api_response.get(
                        "safety_protocols_added", False
                    ),
                    "crisis_level": api_response.get("crisis_level", "none"),
                    "emotional_state": api_response.get("emotional_state"),
                    "web_search_used": api_response.get(
                        "web_search_used", False
                    ),
                    "current_time": api_response.get("current_time"),
                    "token_usage": api_response.get("token_usage", {}),
                    "retrieved_chunks": api_response.get("retrieved_chunks", [])
                    if flask.request.form.get("show_context")
                    else [],
                }

            elif response.status_code == 401:
                error_message = "API authentication failed. Please check your API key configuration."
            elif response.status_code == 422:
                error_message = f"Invalid request: {response.json().get('detail', 'Validation error')}"
            elif response.status_code >= 500:
                error_message = (
                    "SettleBot API server error. Please try again later."
                )
            else:
                error_message = (
                    f"API error ({response.status_code}): {response.text[:200]}"
                )

        except requests.exceptions.Timeout:
            error_message = "Request timeout - the SettleBot service is taking too long to respond"
        except requests.exceptions.ConnectionError:
            error_message = (
                "Cannot connect to SettleBot service - please try again later"
            )
        except requests.exceptions.RequestIOError as e:
            error_message = f"Request error: {str(e)}"
        except IOError as e:
            error_message = f"Unexpected error: {str(e)}"

    return flask.render_template(
        "accounts/settlement_assistant.html",
        response_data=response_data,
        query=query,
        api_response_details=api_response_details,
        error_message=error_message,
    )


@accounts.route("/api-system-status")
@login_required
def api_system_status():
    """
    Display SettleBot API system status and health information.

    :return: Rendered system status template
    """
    if current_user.role not in ["admin", "developer"]:
        flask.flash(
            "You do not have permissions to access this page", "warning"
        )
        return flask.redirect(flask.url_for("accounts.profile"))

    api_url = flask.current_app.config.get(
        "SETTLEBOT_API_URL", "http://localhost:8000"
    )
    api_key = flask.current_app.config.get("SETTLEBOT_API_KEY")

    system_status = None
    health_status = None
    error_message = None

    try:
        headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}

        # Get health status (no auth required)
        health_response = requests.get(f"{api_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_status = health_response.json()

        # Get detailed system status (requires auth)
        if api_key:
            status_response = requests.get(
                f"{api_url}/system/status", headers=headers, timeout=15
            )
            if status_response.status_code == 200:
                system_status = status_response.json()
            elif status_response.status_code == 401:
                error_message = "Authentication required for detailed status"
        else:
            error_message = "API key required for detailed system status"

    except requests.exceptions.RequestIOError as e:
        error_message = f"Could not connect to SettleBot API: {str(e)}"
    except IOError as e:
        error_message = f"Error getting system status: {str(e)}"

    return flask.render_template(
        "accounts/api_system_status.html",
        health_status=health_status,
        system_status=system_status,
        error_message=error_message,
        api_url=api_url,
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
    except IOError as e:
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
        conversationId=conversation_id, userId=current_user.userId
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
            timeout=45,
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
    except requests.exceptions.RequestIOError as e:
        return (
            flask.jsonify({"error": f"SettleBot service error: {str(e)}"}),
            500,
        )
    except IOError as e:
        db.session.rollback()
        flask.current_app.logger.error(f"Error in add_message: {str(e)}")
        return flask.jsonify({"error": f"Server error: {str(e)}"}), 500


@accounts.route("/api/conversation/new", methods=["POST"])
@csrf.exempt
@login_required
def api_new_conversation():
    """
    API endpoint to create a new conversation.

    :return: JSON response with conversation details
    """
    try:
        data = flask.request.get_json()
        title = data.get(
            "title",
            f"Settlement Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )

        conversation = Conversation.create(
            {"userId": current_user.userId, "title": title}
        )

        return flask.jsonify(conversation.getDetails())
    except IOError as e:
        return (
            flask.jsonify(
                {"error": f"Failed to create conversation: {str(e)}"}
            ),
            500,
        )


@accounts.route("/api/conversations/<int:conversation_id>", methods=["DELETE"])
@csrf.exempt
@login_required
def api_delete_conversation(conversation_id):
    """
    API endpoint to delete a conversation.

    :param conversation_id: ID of the conversation to delete
    :return: JSON response indicating success or failure
    """
    conversation = Conversation.query.filter_by(
        conversationId=conversation_id, userId=current_user.userId
    ).first_or_404()

    try:
        success, message = conversation.delete()

        if success:
            return flask.jsonify({"success": True, "message": message})
        else:
            return flask.jsonify({"success": False, "error": message}), 400
    except IOError as e:
        return flask.jsonify({"success": False, "error": str(e)}), 500


@accounts.route("/api/conversations/<int:conversation_id>", methods=["PUT"])
@csrf.exempt
@login_required
def api_update_conversation(conversation_id):
    """
    API endpoint to update a conversation.

    :param conversation_id: ID of the conversation to update
    :return: JSON response with updated conversation details
    """
    conversation = Conversation.query.filter_by(
        conversationId=conversation_id, userId=current_user.userId
    ).first_or_404()

    data = flask.request.get_json()
    if not data:
        return flask.jsonify({"error": "No update data provided"}), 400

    try:
        success, message = conversation.update(data)

        if success:
            return flask.jsonify(conversation.getDetails())
        else:
            return flask.jsonify({"error": message}), 400
    except IOError as e:
        return flask.jsonify({"error": str(e)}), 500


@accounts.route("/api/statistics")
@login_required
def api_statistics():
    """
    API endpoint to get usage statistics for the current user.

    :return: JSON response with statistics
    """
    try:
        # Query total conversations for the user
        total_conversations = Conversation.query.filter_by(
            userId=current_user.userId
        ).count()

        # Query total messages with enhanced statistics
        total_messages = 0
        user_messages = 0
        assistant_messages = 0
        intent_distribution = {}
        topic_distribution = {}

        conversations = Conversation.query.filter_by(
            userId=current_user.userId
        ).all()
        for conversation in conversations:
            messages = Message.query.filter_by(
                conversationId=conversation.conversationId
            ).all()
            total_messages += len(messages)

            for message in messages:
                if message.isUserMessage:
                    user_messages += 1
                else:
                    assistant_messages += 1

                    # Count intent types and topics for assistant messages
                    if message.intentType:
                        intent_distribution[message.intentType] = (
                            intent_distribution.get(message.intentType, 0) + 1
                        )
                    if message.topic:
                        topic_distribution[message.topic] = (
                            topic_distribution.get(message.topic, 0) + 1
                        )

        # Calculate total tokens
        total_tokens = (
            Message.query.filter(
                Message.conversationId.in_(
                    [c.conversationId for c in conversations]
                )
            )
            .with_entities(db.func.sum(Message.tokenCount))
            .scalar()
            or 0
        )

        # Calculate average confidence for assistant messages
        avg_confidence = (
            Message.query.filter(
                Message.conversationId.in_(
                    [c.conversationId for c in conversations]
                ),
                Message.isUserMessage == False,
                Message.confidence.isnot(None),
            )
            .with_entities(db.func.avg(Message.confidence))
            .scalar()
            or 0.0
        )

        return flask.jsonify(
            {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "total_tokens": total_tokens,
                "avg_confidence": round(float(avg_confidence), 3),
                "intent_distribution": intent_distribution,
                "topic_distribution": topic_distribution,
                "most_common_intent": max(
                    intent_distribution.items(), key=lambda x: x[1]
                )[0]
                if intent_distribution
                else None,
                "most_common_topic": max(
                    topic_distribution.items(), key=lambda x: x[1]
                )[0]
                if topic_distribution
                else None,
            }
        )
    except IOError as e:
        return flask.jsonify({"error": str(e)}), 500


@accounts.route("/api/conversations")
@login_required
def api_get_conversations():
    """
    API endpoint to get all conversations for the current user with enhanced details.

    :return: JSON list of conversations
    """
    try:
        conversations = (
            Conversation.query.filter_by(userId=current_user.userId)
            .order_by(Conversation.lastUpdated.desc())
            .all()
        )

        conversation_list = []
        for conversation in conversations:
            conv_details = conversation.getDetails()

            # Add additional metadata
            first_message = (
                Message.query.filter_by(
                    conversationId=conversation.conversationId
                )
                .order_by(Message.dateCreated.asc())
                .first()
            )

            if first_message:
                conv_details["firstMessage"] = {
                    "content": first_message.content[:100] + "..."
                    if len(first_message.content) > 100
                    else first_message.content,
                    "isUserMessage": first_message.isUserMessage,
                    "dateCreated": first_message.dateCreated.isoformat()
                    if first_message.dateCreated
                    else None,
                }
                conv_details["firstTopic"] = first_message.topic
                conv_details["firstIntent"] = first_message.intentType
            else:
                conv_details["firstMessage"] = None
                conv_details["firstTopic"] = None
                conv_details["firstIntent"] = None

            conversation_list.append(conv_details)

        return flask.jsonify(conversation_list)
    except IOError as e:
        return flask.jsonify({"error": str(e)}), 500


@accounts.route("/api/conversations/<int:conversation_id>/messages")
@login_required
def api_get_conversation_messages(conversation_id):
    """
    API endpoint to get all messages for a specific conversation with enhanced details.

    :param conversation_id: ID of the conversation
    :return: JSON list of messages
    """
    try:
        # Verify conversation exists and user has access
        conversation = Conversation.query.filter_by(
            conversationId=conversation_id, userId=current_user.userId
        ).first_or_404()

        messages = (
            Message.query.filter_by(conversationId=conversation.conversationId)
            .order_by(Message.dateCreated)
            .all()
        )

        message_list = []
        for message in messages:
            msg_details = message.getDetails()

            # Add formatting hints for the frontend
            if not message.isUserMessage:
                msg_details["hasSettlementInfo"] = bool(
                    message.intentType and message.intentType != "general_query"
                )
                msg_details["isHighConfidence"] = (
                    message.confidence and message.confidence > 0.8
                )
                msg_details["topicCategory"] = message.topic

            message_list.append(msg_details)

        return flask.jsonify(
            {
                "conversation": conversation.getDetails(),
                "messages": message_list,
                "totalMessages": len(messages),
            }
        )
    except IOError as e:
        return flask.jsonify({"error": str(e)}), 500


@accounts.route("/api/settlebot-health")
@login_required
def api_settlebot_health():
    """
    API endpoint to check SettleBot service health.

    :return: JSON response with health status
    """
    api_url = flask.current_app.config.get(
        "SETTLEBOT_API_URL", "http://localhost:8000"
    )

    try:
        # Check basic health endpoint
        health_response = requests.get(f"{api_url}/health", timeout=5)

        if health_response.status_code == 200:
            health_data = health_response.json()

            return flask.jsonify(
                {
                    "status": "healthy",
                    "api_url": api_url,
                    "health_data": health_data,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }
            )
        else:
            return (
                flask.jsonify(
                    {
                        "status": "unhealthy",
                        "error": f"Health check failed with status {health_response.status_code}",
                        "api_url": api_url,
                    }
                ),
                500,
            )

    except requests.exceptions.ConnectionError:
        return (
            flask.jsonify(
                {
                    "status": "unreachable",
                    "error": "Cannot connect to SettleBot service",
                    "api_url": api_url,
                }
            ),
            503,
        )
    except IOError as e:
        return (
            flask.jsonify(
                {"status": "error", "error": str(e), "api_url": api_url}
            ),
            500,
        )


@accounts.route("/api/search", methods=["POST"])
@csrf.exempt
@login_required
def api_search():
    """
    API endpoint to search the SettleBot knowledge base.

    :return: JSON response with search results
    """
    data = flask.request.get_json()
    if not data or "query" not in data:
        return flask.jsonify({"error": "No search query provided"}), 400

    search_query = data["query"].strip()
    if not search_query:
        return flask.jsonify({"error": "Search query cannot be empty"}), 400

    api_url = flask.current_app.config.get(
        "SETTLEBOT_API_URL", "http://localhost:8000"
    )
    api_key = flask.current_app.config.get("SETTLEBOT_API_KEY")

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

        search_payload = {
            "query": search_query,
            "top_k": data.get("top_k", 10),
            "topic_filter": data.get("topic_filter"),
            "location_filter": data.get("location_filter"),
        }

        response = requests.post(
            f"{api_url}/search",
            json=search_payload,
            headers=headers,
            timeout=15,
        )

        if response.status_code == 200:
            return flask.jsonify(response.json())
        elif response.status_code == 401:
            return flask.jsonify({"error": "API authentication failed"}), 401
        else:
            return (
                flask.jsonify(
                    {"error": f"Search failed: {response.status_code}"}
                ),
                500,
            )

    except requests.exceptions.RequestIOError as e:
        return flask.jsonify({"error": f"Search service error: {str(e)}"}), 500

    except IOError as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@accounts.route("/api/intent-analysis", methods=["POST"])
@csrf.exempt
@login_required
def api_intent_analysis():
    """
    API endpoint to analyze query intent using SettleBot.

    :return: JSON response with intent analysis
    """
    data = flask.request.get_json()
    if not data or "query" not in data:
        return flask.jsonify({"error": "No query provided"}), 400

    query = data["query"].strip()
    if not query:
        return flask.jsonify({"error": "Query cannot be empty"}), 400

    api_url = flask.current_app.config.get(
        "SETTLEBOT_API_URL", "http://localhost:8000"
    )
    api_key = flask.current_app.config.get("SETTLEBOT_API_KEY")

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "",
        }

        intent_payload = {
            "query": query,
            "include_semantic_scores": data.get(
                "include_semantic_scores", True
            ),
        }

        response = requests.post(
            f"{api_url}/intent/analyze",
            json=intent_payload,
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            intent_data = response.json()

            # Add user-friendly descriptions
            intent_descriptions = {
                "housing_inquiry": "Housing and accommodation questions",
                "transportation": "Transport and travel information",
                "safety_concern": "Safety and security matters",
                "university_info": "University and academic information",
                "immigration_visa": "Visa and immigration procedures",
                "banking_finance": "Banking and financial services",
                "healthcare": "Healthcare and medical services",
                "cultural_adaptation": "Cultural guidance and adaptation",
                "emergency_help": "Emergency assistance and urgent help",
                "procedural_query": "Step-by-step procedures and processes",
                "cost_inquiry": "Cost and budget information",
                "off_topic": "Non-settlement related query",
            }

            intent_data["intent_description"] = intent_descriptions.get(
                intent_data.get("intent_type", ""), "Unknown intent type"
            )

            return flask.jsonify(intent_data)
        elif response.status_code == 401:
            return flask.jsonify({"error": "API authentication failed"}), 401
        else:
            return (
                flask.jsonify(
                    {"error": f"Intent analysis failed: {response.status_code}"}
                ),
                500,
            )

    except requests.exceptions.RequestIOError as e:
        return (
            flask.jsonify(
                {"error": f"Intent analysis service error: {str(e)}"}
            ),
            500,
        )
    except IOError as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@accounts.route("/api/language-detect", methods=["POST"])
@csrf.exempt
@login_required
def api_language_detect():
    """
    API endpoint to detect language using SettleBot.

    :return: JSON response with language detection results
    """
    data = flask.request.get_json()
    if not data or "text" not in data:
        return flask.jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return flask.jsonify({"error": "Text cannot be empty"}), 400

    api_url = flask.current_app.config.get(
        "SETTLEBOT_API_URL", "http://localhost:8000"
    )
    api_key = flask.current_app.config.get("SETTLEBOT_API_KEY")

    try:
        headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}

        # Use query parameter for language detection endpoint
        response = requests.post(
            f"{api_url}/language/detect",
            params={"text": text},
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            return flask.jsonify(response.json())
        elif response.status_code == 401:
            return flask.jsonify({"error": "API authentication failed"}), 401
        else:
            return (
                flask.jsonify(
                    {
                        "error": f"Language detection failed: {response.status_code}"
                    }
                ),
                500,
            )

    except requests.exceptions.RequestIOError as e:
        return (
            flask.jsonify(
                {"error": f"Language detection service error: {str(e)}"}
            ),
            500,
        )
    except IOError as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@accounts.route("/api/translate", methods=["POST"])
@csrf.exempt
@login_required
def api_translate():
    """
    API endpoint to translate text using SettleBot.

    :return: JSON response with translation results
    """
    data = flask.request.get_json()
    if not data or "text" not in data or "target_language" not in data:
        return (
            flask.jsonify({"error": "Text and target_language are required"}),
            400,
        )

    text = data["text"].strip()
    target_language = data["target_language"].strip()

    if not text or not target_language:
        return (
            flask.jsonify(
                {"error": "Text and target language cannot be empty"}
            ),
            400,
        )

    api_url = flask.current_app.config.get(
        "SETTLEBOT_API_URL", "http://localhost:8000"
    )
    api_key = flask.current_app.config.get("SETTLEBOT_API_KEY")

    try:
        headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}

        # Use query parameters for translation endpoint
        response = requests.post(
            f"{api_url}/language/translate",
            params={"text": text, "target_language": target_language},
            headers=headers,
            timeout=15,
        )

        if response.status_code == 200:
            return flask.jsonify(response.json())
        elif response.status_code == 401:
            return flask.jsonify({"error": "API authentication failed"}), 401
        else:
            return (
                flask.jsonify(
                    {"error": f"Translation failed: {response.status_code}"}
                ),
                500,
            )

    except requests.exceptions.RequestIOError as e:
        return (
            flask.jsonify({"error": f"Translation service error: {str(e)}"}),
            500,
        )
    except IOError as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@accounts.route("/api/supported-languages")
@login_required
def api_supported_languages():
    """
    API endpoint to get supported languages from SettleBot.

    :return: JSON response with supported languages
    """
    api_url = flask.current_app.config.get(
        "SETTLEBOT_API_URL", "http://localhost:8000"
    )
    api_key = flask.current_app.config.get("SETTLEBOT_API_KEY")

    try:
        headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}

        response = requests.get(
            f"{api_url}/language/supported", headers=headers, timeout=10
        )

        if response.status_code == 200:
            return flask.jsonify(response.json())
        elif response.status_code == 401:
            return flask.jsonify({"error": "API authentication failed"}), 401
        else:
            return (
                flask.jsonify(
                    {"error": f"Request failed: {response.status_code}"}
                ),
                500,
            )

    except requests.exceptions.RequestIOError as e:
        return flask.jsonify({"error": f"Service error: {str(e)}"}), 500
    except IOError as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@accounts.route("/api/settlement-topics")
@login_required
def api_settlement_topics():
    """
    API endpoint to get settlement topics and sample queries.

    :return: JSON response with settlement topics
    """
    api_url = flask.current_app.config.get(
        "SETTLEBOT_API_URL", "http://localhost:8000"
    )

    try:
        # Get settlement topics (no auth required for utility endpoints)
        topics_response = requests.get(
            f"{api_url}/utils/settlement-topics", timeout=10
        )

        # Get sample queries (no auth required for utility endpoints)
        samples_response = requests.get(
            f"{api_url}/utils/sample-queries", timeout=10
        )

        result = {}

        if topics_response.status_code == 200:
            result["topics"] = topics_response.json()
        else:
            result[
                "topics_error"
            ] = f"Failed to get topics: {topics_response.status_code}"

        if samples_response.status_code == 200:
            result["sample_queries"] = samples_response.json()
        else:
            result[
                "samples_error"
            ] = f"Failed to get samples: {samples_response.status_code}"

        if not result.get("topics") and not result.get("sample_queries"):
            return (
                flask.jsonify(
                    {"error": "Failed to retrieve settlement information"}
                ),
                500,
            )

        return flask.jsonify(result)

    except requests.exceptions.RequestIOError as e:
        return flask.jsonify({"error": f"Service error: {str(e)}"}), 500
    except IOError as e:
        return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@accounts.route("/api/conversation-export/<int:conversation_id>")
@login_required
def api_export_conversation(conversation_id):
    """
    API endpoint to export a conversation in various formats.

    :param conversation_id: ID of the conversation to export
    :return: JSON response or file download
    """
    # Verify conversation exists and user has access
    conversation = Conversation.query.filter_by(
        conversationId=conversation_id, userId=current_user.userId
    ).first_or_404()

    export_format = flask.request.args.get("format", "json").lower()

    try:
        messages = (
            Message.query.filter_by(conversationId=conversation.conversationId)
            .order_by(Message.dateCreated)
            .all()
        )

        if export_format == "json":
            export_data = {
                "conversation": conversation.getDetails(),
                "user": {
                    "fullName": current_user.fullName,
                    "username": current_user.username,
                    "exportDate": datetime.datetime.utcnow().isoformat(),
                },
                "messages": [message.getDetails() for message in messages],
                "statistics": {
                    "total_messages": len(messages),
                    "user_messages": sum(
                        1 for m in messages if m.isUserMessage
                    ),
                    "assistant_messages": sum(
                        1 for m in messages if not m.isUserMessage
                    ),
                    "total_tokens": sum(m.tokenCount or 0 for m in messages),
                    "topics_discussed": list(
                        set(
                            m.topic
                            for m in messages
                            if m.topic and not m.isUserMessage
                        )
                    ),
                    "intents_processed": list(
                        set(
                            m.intentType
                            for m in messages
                            if m.intentType and not m.isUserMessage
                        )
                    ),
                },
            }
            return flask.jsonify(export_data)

        elif export_format == "txt":
            lines = [
                "SettleBot Conversation Export",
                f"Conversation: {conversation.title}",
                f"User: {current_user.fullName} ({current_user.username})",
                f"Export Date: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"Total Messages: {len(messages)}",
                "=" * 50,
                "",
            ]

            for message in messages:
                timestamp = (
                    message.dateCreated.strftime("%Y-%m-%d %H:%M:%S")
                    if message.dateCreated
                    else "Unknown"
                )
                speaker = "User" if message.isUserMessage else "SettleBot"

                lines.append(f"[{timestamp}] {speaker}:")
                lines.append(message.content)

                if not message.isUserMessage and (
                    message.intentType or message.topic
                ):
                    metadata = []
                    if message.intentType:
                        metadata.append(f"Intent: {message.intentType}")
                    if message.topic:
                        metadata.append(f"Topic: {message.topic}")
                    if message.confidence:
                        metadata.append(
                            f"Confidence: {message.confidence:.2f}")
                    lines.append(f"[{', '.join(metadata)}]")

                lines.append("")

            # Create response with text file
            response = flask.make_response("\n".join(lines))
            response.headers[
                "Content-Disposition"
            ] = f"attachment; filename=conversation_{conversation_id}.txt"
            response.headers["Content-Type"] = "text/plain"
            return response

        else:
            return (
                flask.jsonify(
                    {"error": "Unsupported export format. Use 'json' or 'txt'"}
                ),
                400,
            )

    except IOError as e:
        return flask.jsonify({"error": f"Export failed: {str(e)}"}), 500


@accounts.route("/api/user-insights")
@login_required
def api_user_insights():
    """
    API endpoint to get user-specific insights and recommendations.

    :return: JSON response with user insights
    """
    try:
        # Get user's conversations and messages
        conversations = Conversation.query.filter_by(
            userId=current_user.userId
        ).all()

        if not conversations:
            return flask.jsonify(
                {
                    "message": "No conversation data available",
                    "recommendations": [
                        "Start a conversation to get personalized settlement assistance",
                        "Ask about housing options in Nairobi",
                        "Inquire about transportation and safety",
                    ],
                }
            )

        # Gather statistics
        all_messages = []
        for conv in conversations:
            messages = Message.query.filter_by(
                conversationId=conv.conversationId
            ).all()
            all_messages.extend(messages)

        # Analyze user behavior
        user_messages = [m for m in all_messages if m.isUserMessage]
        assistant_messages = [m for m in all_messages if not m.isUserMessage]

        # Topic analysis
        topics = {}
        intents = {}
        for message in assistant_messages:
            if message.topic:
                topics[message.topic] = topics.get(message.topic, 0) + 1
            if message.intentType:
                intents[message.intentType] = (
                    intents.get(message.intentType, 0) + 1
                )

        # Time-based analysis
        recent_messages = [
            m
            for m in user_messages
            if m.dateCreated
            and (datetime.datetime.utcnow() - m.dateCreated).days <= 7
        ]

        # Generate insights
        insights = {
            "activity_summary": {
                "total_conversations": len(conversations),
                "total_messages": len(all_messages),
                "messages_sent": len(user_messages),
                "responses_received": len(assistant_messages),
                "recent_activity": len(recent_messages),
                "most_discussed_topic": max(topics.items(), key=lambda x: x[1])[
                    0
                ]
                if topics
                else None,
                "most_common_intent": max(intents.items(), key=lambda x: x[1])[
                    0
                ]
                if intents
                else None,
            },
            "topic_distribution": topics,
            "intent_distribution": intents,
            "engagement_pattern": {
                "active_days_last_week": len(
                    set(
                        m.dateCreated.date()
                        for m in recent_messages
                        if m.dateCreated
                    )
                ),
                "avg_message_length": sum(len(m.content) for m in user_messages)
                / len(user_messages)
                if user_messages
                else 0,
                "most_active_hour": None,  # Would need more complex analysis
            },
        }

        # Generate personalized recommendations
        recommendations = []

        # Topic-based recommendations
        if topics:
            top_topic = max(topics.items(), key=lambda x: x[1])[0]
            topic_recommendations = {
                "housing": "Consider exploring more specific housing areas or rental tips",
                "transportation": "Learn about different transport options and cost-saving tips",
                "safety": "Stay updated on current safety guidelines and emergency procedures",
                "finance": "Explore banking options and budgeting strategies",
                "healthcare": "Learn about health insurance and medical facilities",
                "education": "Check out academic support services and university resources",
            }
            recommendations.append(
                topic_recommendations.get(
                    top_topic, "Continue exploring settlement topics"
                )
            )

        # Activity-based recommendations
        if len(recent_messages) < 3:
            recommendations.append(
                "Consider asking more questions to get comprehensive settlement guidance"
            )

        if not topics:
            recommendations.extend(
                [
                    "Try asking about housing options in different Nairobi areas",
                    "Inquire about transportation and getting around the city",
                    "Ask about safety tips and emergency information",
                ]
            )

        # Add general recommendations
        recommendations.extend(
            [
                "Explore different settlement topics to get comprehensive guidance",
                "Ask follow-up questions for more detailed information",
                "Consider saving important conversations for future reference",
            ]
        )

        insights["recommendations"] = recommendations[
            :5
        ]  # Limit to 5 recommendations

        return flask.jsonify(insights)

    except IOError as e:
        return (
            flask.jsonify({"error": f"Failed to generate insights: {str(e)}"}),
            500,
        )


@accounts.route("/api/quick-help")
@login_required
def api_quick_help():
    """
    API endpoint to get quick help suggestions based on common settlement needs.

    :return: JSON response with quick help options
    """
    quick_help_data = {
        "emergency_contacts": {
            "title": "Emergency Contacts",
            "items": [
                {
                    "label": "Universal Emergency",
                    "value": "999",
                    "type": "phone",
                },
                {"label": "Police Emergency", "value": "999", "type": "phone"},
                {
                    "label": "Red Cross Kenya",
                    "value": "0700 395 395",
                    "type": "phone",
                },
                {
                    "label": "AA Kenya (Roadside)",
                    "value": "0700 200 007",
                    "type": "phone",
                },
            ],
        },
        "quick_questions": {
            "title": "Common Questions",
            "items": [
                {
                    "label": "Housing in Westlands",
                    "query": "Where can I find affordable housing in Westlands?",
                },
                {
                    "label": "Airport to City Transport",
                    "query": "How do I get from JKIA airport to the city center safely?",
                },
                {
                    "label": "Opening Bank Account",
                    "query": "How do I open a bank account as an international student?",
                },
                {
                    "label": "Safety in Nairobi",
                    "query": "What areas should I avoid in Nairobi for safety?",
                },
                {
                    "label": "University Admission",
                    "query": "What documents do I need for university admission in Kenya?",
                },
                {
                    "label": "Student Visa Renewal",
                    "query": "How do I renew my student visa in Kenya?",
                },
            ],
        },
        "useful_locations": {
            "title": "Important Locations",
            "items": [
                {
                    "label": "Immigration Office",
                    "value": "Nyayo House, Uhuru Highway",
                    "type": "address",
                },
                {
                    "label": "Nairobi Hospital",
                    "value": "Argwings Kodhek Road, Upper Hill",
                    "type": "address",
                },
                {
                    "label": "University of Nairobi",
                    "value": "University Way, Nairobi",
                    "type": "address",
                },
                {
                    "label": "Strathmore University",
                    "value": "Ole Sangale Road, Madaraka",
                    "type": "address",
                },
            ],
        },
        "settlement_tips": {
            "title": "Quick Settlement Tips",
            "items": [
                {
                    "tip": "Always carry copies of your passport and student permit"
                },
                {
                    "tip": "Use registered taxi services like Uber or Bolt for safety"
                },
                {
                    "tip": "M-Pesa is widely used for payments - get a Safaricom line"
                },
                {
                    "tip": "Join international student groups for support and advice"
                },
                {"tip": "Keep emergency contacts saved in your phone"},
                {"tip": "Learn basic Swahili phrases for better communication"},
            ],
        },
    }

    return flask.jsonify(quick_help_data)


@accounts.route("/api/conversation-summary/<int:conversation_id>")
@login_required
def api_conversation_summary(conversation_id):
    """
    API endpoint to get a summary of a conversation.

    :param conversation_id: ID of the conversation
    :return: JSON response with conversation summary
    """
    # Verify conversation exists and user has access
    conversation = Conversation.query.filter_by(
        conversationId=conversation_id, userId=current_user.userId
    ).first_or_404()

    try:
        messages = (
            Message.query.filter_by(conversationId=conversation.conversationId)
            .order_by(Message.dateCreated)
            .all()
        )

        if not messages:
            return flask.jsonify(
                {
                    "conversation_id": conversation_id,
                    "title": conversation.title,
                    "summary": "No messages in this conversation yet.",
                }
            )

        # Analyze conversation
        user_messages = [m for m in messages if m.isUserMessage]
        assistant_messages = [m for m in messages if not m.isUserMessage]

        # Extract topics and intents
        topics_discussed = list(
            set(m.topic for m in assistant_messages if m.topic)
        )
        intents_processed = list(
            set(m.intentType for m in assistant_messages if m.intentType)
        )

        # Get key insights
        total_tokens = sum(m.tokenCount or 0 for m in messages)
        avg_confidence = (
            sum(m.confidence or 0 for m in assistant_messages)
            / len(assistant_messages)
            if assistant_messages
            else 0
        )

        # Duration
        start_time = (
            messages[0].dateCreated if messages[0].dateCreated else None
        )
        end_time = (
            messages[-1].dateCreated if messages[-1].dateCreated else None
        )
        duration = None

        if start_time and end_time:
            duration_delta = end_time - start_time
            duration = {
                "total_minutes": int(duration_delta.total_seconds() / 60),
                "formatted": str(duration_delta).split(".")[
                    0
                ],  # Remove microseconds
            }

        summary = {
            "conversation_id": conversation_id,
            "title": conversation.title,
            "created": conversation.dateCreated.isoformat()
            if conversation.dateCreated
            else None,
            "last_updated": conversation.lastUpdated.isoformat()
            if conversation.lastUpdated
            else None,
            "duration": duration,
            "statistics": {
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "total_tokens": total_tokens,
                "average_confidence": round(avg_confidence, 3)
                if avg_confidence
                else None,
            },
            "topics_discussed": topics_discussed,
            "intents_processed": intents_processed,
            "settlement_categories": {
                "housing": "housing" in topics_discussed,
                "transportation": "transport" in topics_discussed
                or "transportation" in topics_discussed,
                "safety": "safety" in topics_discussed,
                "finance": "finance" in topics_discussed,
                "education": "education" in topics_discussed
                or "academics" in topics_discussed,
                "healthcare": "health" in topics_discussed
                or "healthcare" in topics_discussed,
                "legal": "legal" in topics_discussed,
                "culture": "culture" in topics_discussed,
            },
        }

        return flask.jsonify(summary)

    except IOError as e:
        return (
            flask.jsonify({"error": f"Failed to generate summary: {str(e)}"}),
            500,
        )


def get_settlebot_api_config():
    """Get SettleBot API configuration."""
    return {
        "url": flask.current_app.config.get(
            "SETTLEBOT_API_URL", "http://localhost:8000"
        ),
        "key": flask.current_app.config.get("SETTLEBOT_API_KEY"),
        "timeout": 30,
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {flask.current_app.config.get('SETTLEBOT_API_KEY')}"
            if flask.current_app.config.get("SETTLEBOT_API_KEY")
            else "",
        },
    }
