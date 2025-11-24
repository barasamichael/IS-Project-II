import json
import requests
from typing import Any
from typing import Dict
from typing import Optional
from datetime import datetime
from datetime import timedelta
from collections import Counter

import flask
from sqlalchemy import desc
from sqlalchemy import func
from flask import current_app
from flask_login import current_user
from flask_login import login_required

from app import db
from . import administration
from ..models import User
from ..models import Message
from ..models import Conversation
from .forms import UserSearchForm
from .forms import APITestingForm
from .forms import SystemActionForm
from .forms import DocumentSearchForm
from .forms import DocumentUploadForm
from .forms import UserBulkActionForm
from .forms import AnalyticsExportForm
from .forms import ConversationSearchForm
from .forms import DocumentBulkActionForm
from .forms import ConversationBulkActionForm
from .forms import MessageAnalyticsFilterForm
from .forms import SettlementInsightsFilterForm


def require_admin_access():
    """
    Check if current user has admin access.

    :return: Redirect to unauthorized page if no access
    """
    if not current_user.is_authenticated or current_user.role not in [
        "admin",
        "developer",
    ]:
        flask.flash("You do not have permissions to access this area", "error")
        return flask.redirect(flask.url_for("main.index"))
    return None


def get_settlebot_api_config() -> Dict[str, str]:
    """
    Get SettleBot API configuration.

    :return: Dictionary with API URL and key
    """
    return {
        "url": current_app.config.get(
            "SETTLEBOT_API_URL", "http://localhost:8000"
        ),
        "key": current_app.config.get("SETTLEBOT_API_KEY", ""),
    }


def make_api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Make request to SettleBot API.

    :param endpoint: API endpoint
    :param method: HTTP method
    :param data: Request data for POST requests
    :param params: Query parameters
    :return: API response data
    """
    api_config = get_settlebot_api_config()
    url = f"{api_config['url']}{endpoint}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_config['key']}"
        if api_config["key"]
        else "",
    }

    try:
        if method.upper() == "GET":
            response = requests.get(
                url, headers=headers, params=params, timeout=30
            )
        else:
            response = requests.post(
                url, json=data, headers=headers, params=params, timeout=60
            )

        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"API error ({response.status_code}): {response.text[:200]}",
            }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}


@administration.before_request
def check_admin_access():
    """Check admin access before each request."""
    return require_admin_access()


@administration.route("/dashboard")
@login_required
def dashboard():
    """
    Display administration dashboard with system overview.

    :return: Rendered dashboard template
    """
    # System health check
    api_health = make_api_request("/health")
    system_status = make_api_request("/system/status")

    # Key metrics
    total_users = User.query.count()
    total_conversations = Conversation.query.count()
    total_messages = Message.query.count()

    # Recent activity (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    new_users = User.query.filter(User.dateCreated >= week_ago).count()
    new_conversations = Conversation.query.filter(
        Conversation.dateCreated >= week_ago
    ).count()

    # Settlement-specific KPIs
    intent_distribution = {}
    topic_distribution = {}

    # Get recent messages for KPIs
    recent_messages = Message.query.filter(
        Message.isUserMessage == False, Message.dateCreated >= week_ago
    ).all()

    for message in recent_messages:
        if message.intentType:
            intent_distribution[message.intentType] = (
                intent_distribution.get(message.intentType, 0) + 1
            )
        if message.topic:
            topic_distribution[message.topic] = (
                topic_distribution.get(message.topic, 0) + 1
            )

    # Document statistics from API
    documents_info = make_api_request("/documents")

    dashboard_data = {
        "system_health": api_health.get("data", {}),
        "system_status": system_status.get("data", {}),
        "key_metrics": {
            "total_users": total_users,
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "total_documents": documents_info.get("data", {}).get("count", 0)
            if documents_info.get("success")
            else 0,
        },
        "recent_activity": {
            "new_users_week": new_users,
            "new_conversations_week": new_conversations,
        },
        "settlement_kpis": {
            "intent_distribution": intent_distribution,
            "topic_distribution": topic_distribution,
            "most_common_intent": max(
                intent_distribution.items(), key=lambda x: x[1]
            )[0]
            if intent_distribution
            else "None",
            "most_common_topic": max(
                topic_distribution.items(), key=lambda x: x[1]
            )[0]
            if topic_distribution
            else "None",
        },
        "api_connected": api_health.get("success", False),
    }

    return flask.render_template(
        "administration/dashboard.html", **dashboard_data
    )


@administration.route("/users")
@login_required
def users():
    """
    Display user management interface.

    :return: Rendered users template
    """
    form = UserSearchForm()
    bulk_form = UserBulkActionForm()

    # Build query
    query = User.query

    # Apply search filters from request args
    search_query = flask.request.args.get("search_query", "")
    role_filter = flask.request.args.get("role_filter", "")
    status_filter = flask.request.args.get("status_filter", "")
    data_usage_filter = flask.request.args.get("data_usage_filter", "")

    if search_query:
        query = query.filter(
            db.or_(
                User.fullName.contains(search_query),
                User.username.contains(search_query),
                User.emailAddress.contains(search_query),
            )
        )

    if role_filter:
        query = query.filter(User.role == role_filter)

    if status_filter == "active":
        query = query.filter(User.isActive == True)
    elif status_filter == "inactive":
        query = query.filter(User.isActive == False)

    if data_usage_filter == "true":
        query = query.filter(User.allowDataUsage == True)
    elif data_usage_filter == "false":
        query = query.filter(User.allowDataUsage == False)

    # Pagination
    page = flask.request.args.get("page", 1, type=int)
    per_page = 20
    users_pagination = query.order_by(User.dateCreated.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    # User statistics
    total_users = User.query.count()
    active_users = User.query.filter(User.isActive == True).count()
    role_stats = (
        db.session.query(User.role, func.count(User.userId))
        .group_by(User.role)
        .all()
    )

    return flask.render_template(
        "administration/users.html",
        users=users_pagination.items,
        pagination=users_pagination,
        form=form,
        bulk_form=bulk_form,
        search_query=search_query,
        role_filter=role_filter,
        status_filter=status_filter,
        data_usage_filter=data_usage_filter,
        total_users=total_users,
        active_users=active_users,
        role_stats=dict(role_stats),
    )


@administration.route("/users/<int:user_id>")
@login_required
def user_details(user_id):
    """
    Display detailed information for a specific user.

    :param user_id: User ID to view
    :return: Rendered user details template
    """
    user = User.query.get_or_404(user_id)

    # Get user's conversations
    conversations = (
        Conversation.query.filter_by(userId=user.userId)
        .order_by(Conversation.lastUpdated.desc())
        .all()
    )

    # Calculate usage statistics
    total_conversations = len(conversations)
    total_messages = 0
    user_messages = 0
    assistant_messages = 0
    intent_distribution = Counter()
    topic_distribution = Counter()
    total_tokens = 0

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
                if message.intentType:
                    intent_distribution[message.intentType] += 1
                if message.topic:
                    topic_distribution[message.topic] += 1

            if message.tokenCount:
                total_tokens += message.tokenCount

    # Settlement-specific data
    preferred_topics = dict(topic_distribution.most_common(5))
    preferred_language = "English"
    crisis_interventions = 0

    # Activity timeline (recent conversations)
    recent_conversations = conversations[:10]

    user_stats = {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
        "total_tokens": total_tokens,
        "avg_messages_per_conversation": round(
            total_messages / total_conversations, 2
        )
        if total_conversations > 0
        else 0,
        "preferred_topics": preferred_topics,
        "most_common_intent": intent_distribution.most_common(1)[0][0]
        if intent_distribution
        else "None",
        "preferred_language": preferred_language,
        "crisis_interventions": crisis_interventions,
        "intent_distribution": dict(intent_distribution),
        "topic_distribution": dict(topic_distribution),
    }

    return flask.render_template(
        "administration/user_details.html",
        user=user,
        conversations=recent_conversations,
        user_stats=user_stats,
    )


@administration.route("/users/bulk-action", methods=["POST"])
@login_required
def user_bulk_action():
    """
    Handle bulk actions on users.

    :return: Redirect with status message
    """
    form = UserBulkActionForm()

    if form.validate_on_submit():
        try:
            selected_ids = [
                int(id.strip())
                for id in form.selected_users.data.split(",")
                if id.strip()
            ]
            action = form.action.data

            if not selected_ids:
                flask.flash("No users selected", "error")
                return flask.redirect(flask.url_for("administration.users"))

            users = User.query.filter(User.userId.in_(selected_ids)).all()

            if action == "activate":
                for user in users:
                    user.isActive = True
                message = f"Activated {len(users)} users"
            elif action == "deactivate":
                for user in users:
                    user.isActive = False
                message = f"Deactivated {len(users)} users"
            elif action.startswith("role_"):
                new_role = action.replace("role_", "")
                for user in users:
                    user.role = new_role
                message = f"Updated role to {new_role} for {len(users)} users"
            elif action == "export":
                # Implement export functionality
                return flask.redirect(
                    flask.url_for(
                        "administration.export_users",
                        user_ids=",".join(map(str, selected_ids)),
                    )
                )

            db.session.commit()
            flask.flash(message, "success")

        except Exception as e:
            db.session.rollback()
            flask.flash(f"Error executing bulk action: {str(e)}", "error")

    return flask.redirect(flask.url_for("administration.users"))


@administration.route("/conversations")
@login_required
def conversations():
    """
    Display conversation management interface.

    :return: Rendered conversations template
    """
    form = ConversationSearchForm()
    bulk_form = ConversationBulkActionForm()

    # Build query with joins
    query = db.session.query(Conversation).join(User)

    # Apply filters from request args
    search_query = flask.request.args.get("search_query", "")
    intent_filter = flask.request.args.get("intent_filter", "")
    topic_filter = flask.request.args.get("topic_filter", "")
    crisis_level_filter = flask.request.args.get("crisis_level_filter", "")
    empathy_applied_filter = flask.request.args.get(
        "empathy_applied_filter", ""
    )
    safety_protocols_filter = flask.request.args.get(
        "safety_protocols_filter", ""
    )

    if search_query:
        query = query.filter(
            db.or_(
                Conversation.title.contains(search_query),
                User.fullName.contains(search_query),
                User.username.contains(search_query),
            )
        )

    # For intent, topic, and settlement-specific filters, we need to join with messages
    if (
        intent_filter
        or topic_filter
        or crisis_level_filter
        or empathy_applied_filter
        or safety_protocols_filter
    ):
        query = query.join(Message)

        if intent_filter:
            query = query.filter(Message.intentType == intent_filter)
        if topic_filter:
            query = query.filter(Message.topic == topic_filter)

    # Pagination
    page = flask.request.args.get("page", 1, type=int)
    per_page = 20
    conversations_pagination = (
        query.distinct()
        .order_by(Conversation.lastUpdated.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    # Conversation analytics
    total_conversations = Conversation.query.count()
    avg_messages = (
        db.session.query(
            func.avg(
                db.session.query(func.count(Message.messageId))
                .filter(Message.conversationId == Conversation.conversationId)
                .scalar_subquery()
            )
        ).scalar()
        or 0
    )

    # Get conversations with additional data
    conversations_data = []
    for conversation in conversations_pagination.items:
        message_count = Message.query.filter_by(
            conversationId=conversation.conversationId
        ).count()
        first_message = (
            Message.query.filter_by(conversationId=conversation.conversationId)
            .order_by(Message.dateCreated.asc())
            .first()
        )

        intent_types = (
            db.session.query(Message.intentType.distinct())
            .filter(
                Message.conversationId == conversation.conversationId,
                Message.intentType.isnot(None),
            )
            .all()
        )

        topics = (
            db.session.query(Message.topic.distinct())
            .filter(
                Message.conversationId == conversation.conversationId,
                Message.topic.isnot(None),
            )
            .all()
        )

        conversations_data.append(
            {
                "conversation": conversation,
                "user": conversation.user,
                "message_count": message_count,
                "first_message": first_message,
                "intent_types": [intent[0] for intent in intent_types],
                "topics": [topic[0] for topic in topics],
            }
        )

    return flask.render_template(
        "administration/conversations.html",
        conversations_data=conversations_data,
        pagination=conversations_pagination,
        form=form,
        bulk_form=bulk_form,
        search_query=search_query,
        total_conversations=total_conversations,
        avg_messages=round(avg_messages, 2),
    )


@administration.route("/conversations/<int:conversation_id>")
@login_required
def conversation_details(conversation_id):
    """
    Display detailed view of a specific conversation.

    :param conversation_id: Conversation ID to view
    :return: Rendered conversation details template
    """
    conversation = Conversation.query.get_or_404(conversation_id)

    # Get all messages for this conversation
    messages = (
        Message.query.filter_by(conversationId=conversation.conversationId)
        .order_by(Message.dateCreated)
        .all()
    )

    # Calculate conversation analytics
    total_messages = len(messages)
    user_messages = [m for m in messages if m.isUserMessage]
    assistant_messages = [m for m in messages if not m.isUserMessage]

    # Intent and topic analysis
    intent_distribution = Counter()
    topic_distribution = Counter()
    confidence_scores = []
    total_tokens = 0

    for message in assistant_messages:
        if message.intentType:
            intent_distribution[message.intentType] += 1
        if message.topic:
            topic_distribution[message.topic] += 1
        if message.confidence:
            confidence_scores.append(message.confidence)
        if message.tokenCount:
            total_tokens += message.tokenCount

    # Settlement-specific insights
    safety_concerns = len(
        [m for m in assistant_messages if m.intentType == "safety_concern"]
    )
    emergency_help = len(
        [m for m in assistant_messages if m.intentType == "emergency_help"]
    )
    cultural_adaptation = len(
        [m for m in assistant_messages if m.topic == "culture"]
    )

    conversation_analytics = {
        "total_messages": total_messages,
        "user_messages": len(user_messages),
        "assistant_messages": len(assistant_messages),
        "total_tokens": total_tokens,
        "avg_confidence": sum(confidence_scores) / len(confidence_scores)
        if confidence_scores
        else 0,
        "intent_distribution": dict(intent_distribution),
        "topic_distribution": dict(topic_distribution),
        "safety_concerns_flagged": safety_concerns,
        "emergency_help_requests": emergency_help,
        "cultural_adaptation_queries": cultural_adaptation,
        "conversation_length": (
            messages[-1].dateCreated - messages[0].dateCreated
        ).total_seconds()
        / 3600
        if messages
        else 0,
    }

    return flask.render_template(
        "administration/conversation_details.html",
        conversation=conversation,
        messages=messages,
        analytics=conversation_analytics,
    )


@administration.route("/conversations/bulk-action", methods=["POST"])
@login_required
def conversation_bulk_action():
    """
    Handle bulk actions on conversations.

    :return: Redirect with status message
    """
    form = ConversationBulkActionForm()

    if form.validate_on_submit():
        try:
            selected_ids = [
                int(id.strip())
                for id in form.selected_conversations.data.split(",")
                if id.strip()
            ]
            action = form.action.data

            if not selected_ids:
                flask.flash("No conversations selected", "error")
                return flask.redirect(
                    flask.url_for("administration.conversations")
                )

            conversations = Conversation.query.filter(
                Conversation.conversationId.in_(selected_ids)
            ).all()

            if action == "export":
                # Implement export functionality
                return flask.redirect(
                    flask.url_for(
                        "administration.export_conversations",
                        conversation_ids=",".join(map(str, selected_ids)),
                    )
                )
            elif action == "delete":
                for conversation in conversations:
                    db.session.delete(conversation)
                message = f"Deleted {len(conversations)} conversations"
                db.session.commit()
                flask.flash(message, "success")

        except Exception as e:
            db.session.rollback()
            flask.flash(f"Error executing bulk action: {str(e)}", "error")

    return flask.redirect(flask.url_for("administration.conversations"))


@administration.route("/messages")
@login_required
def messages():
    """
    Display message analytics dashboard.

    :return: Rendered messages template
    """
    form = MessageAnalyticsFilterForm()

    # Apply filters from request args
    date_range = flask.request.args.get("date_range", "30")
    intent_type = flask.request.args.get("intent_type", "")
    topic_type = flask.request.args.get("topic_type", "")
    min_confidence = flask.request.args.get("min_confidence", type=float)

    # Build base query
    query = Message.query.filter(Message.isUserMessage == False)

    # Date filter
    if date_range != "all":
        days_ago = datetime.utcnow() - timedelta(days=int(date_range))
        query = query.filter(Message.dateCreated >= days_ago)

    # Intent filter
    if intent_type:
        query = query.filter(Message.intentType == intent_type)

    # Topic filter
    if topic_type:
        query = query.filter(Message.topic == topic_type)

    # Confidence filter
    if min_confidence is not None:
        query = query.filter(Message.confidence >= min_confidence)

    messages_data = query.all()

    # Calculate analytics
    total_messages = len(messages_data)

    # Intent distribution
    intent_distribution = Counter(
        m.intentType for m in messages_data if m.intentType
    )

    # Topic distribution
    topic_distribution = Counter(m.topic for m in messages_data if m.topic)

    # Confidence analysis
    confidence_scores = [m.confidence for m in messages_data if m.confidence]
    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores
        else 0
    )

    # Settlement-specific metrics
    crisis_interventions = len(
        [m for m in messages_data if m.intentType == "emergency_help"]
    )
    safety_concerns = len(
        [m for m in messages_data if m.intentType == "safety_concern"]
    )
    cultural_queries = len([m for m in messages_data if m.topic == "culture"])

    # Token usage
    total_tokens = sum(m.tokenCount for m in messages_data if m.tokenCount)

    analytics = {
        "total_messages": total_messages,
        "intent_distribution": dict(intent_distribution.most_common()),
        "topic_distribution": dict(topic_distribution.most_common()),
        "avg_confidence": round(avg_confidence, 3),
        "total_tokens": total_tokens,
        "crisis_interventions": crisis_interventions,
        "safety_concerns": safety_concerns,
        "cultural_queries": cultural_queries,
        "confidence_distribution": {
            "high": len([c for c in confidence_scores if c >= 0.8]),
            "medium": len([c for c in confidence_scores if 0.5 <= c < 0.8]),
            "low": len([c for c in confidence_scores if c < 0.5]),
        },
    }

    return flask.render_template(
        "administration/messages.html",
        form=form,
        analytics=analytics,
        date_range=date_range,
        intent_type=intent_type,
        topic_type=topic_type,
    )


@administration.route("/documents")
@login_required
def documents():
    """
    Display document management interface.

    :return: Rendered documents template
    """
    form = DocumentSearchForm()
    upload_form = DocumentUploadForm()
    bulk_form = DocumentBulkActionForm()

    # Get documents from SettleBot API
    api_params = {}

    # Apply filters from request args
    search_query = flask.request.args.get("search_query", "")
    doc_type_filter = flask.request.args.get("doc_type_filter", "")
    settlement_score_filter = flask.request.args.get(
        "settlement_score_filter", ""
    )

    if doc_type_filter:
        api_params["doc_type"] = doc_type_filter
    if settlement_score_filter:
        api_params["settlement_score_min"] = float(settlement_score_filter)

    documents_response = make_api_request("/documents", params=api_params)

    if documents_response.get("success"):
        documents_data = documents_response["data"]
        documents = documents_data.get("documents", [])

        # Apply search filter locally if needed
        if search_query:
            documents = [
                doc
                for doc in documents
                if search_query.lower() in doc.get("file_name", "").lower()
            ]

        # Document analytics
        total_documents = len(documents)
        doc_type_stats = documents_data.get("doc_type_stats", {})
        total_chunks = documents_data.get("total_chunks", 0)
        avg_settlement_score = documents_data.get("avg_settlement_score", 0)
    else:
        documents = []
        total_documents = 0
        doc_type_stats = {}
        total_chunks = 0
        avg_settlement_score = 0
        flask.flash(
            f"Error loading documents: {documents_response.get('error', 'Unknown error')}",
            "error",
        )

    return flask.render_template(
        "administration/documents.html",
        documents=documents,
        form=form,
        upload_form=upload_form,
        bulk_form=bulk_form,
        total_documents=total_documents,
        doc_type_stats=doc_type_stats,
        total_chunks=total_chunks,
        avg_settlement_score=avg_settlement_score,
        search_query=search_query,
    )


@administration.route("/documents/<doc_id>")
@login_required
def document_details(doc_id):
    """
    Display detailed information for a specific document.

    :param doc_id: Document ID to view
    :return: Rendered document details template
    """
    document_response = make_api_request(f"/documents/{doc_id}")

    if not document_response.get("success"):
        flask.flash(
            f"Error loading document: {document_response.get('error', 'Document not found')}",
            "error",
        )
        return flask.redirect(flask.url_for("administration.documents"))

    document = document_response["data"]

    return flask.render_template(
        "administration/document_details.html",
        document=document,
    )


@administration.route("/documents/upload", methods=["POST"])
@login_required
def upload_document():
    """
    Handle document upload to SettleBot API.

    :return: Redirect with status message
    """
    upload_form = DocumentUploadForm()

    if upload_form.validate_on_submit():
        try:
            # Prepare file for API upload
            file = upload_form.file.data
            api_config = get_settlebot_api_config()

            headers = {
                "Authorization": f"Bearer {api_config['key']}"
                if api_config["key"]
                else "",
            }

            files = {"file": (file.filename, file.stream, file.content_type)}

            response = requests.post(
                f"{api_config['url']}/documents/upload",
                files=files,
                headers=headers,
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                flask.flash(
                    f"Document '{result['file_name']}' uploaded successfully",
                    "success",
                )
            else:
                flask.flash(f"Upload failed: {response.text[:200]}", "error")

        except Exception as e:
            flask.flash(f"Error uploading document: {str(e)}", "error")

    return flask.redirect(flask.url_for("administration.documents"))


@administration.route("/system")
@login_required
def system():
    """
    Display system status and management interface.

    :return: Rendered system template
    """
    # Get comprehensive system status
    health_response = make_api_request("/health")
    status_response = make_api_request("/system/status")
    vector_stats = make_api_request("/vector-db/stats")
    embedding_stats = make_api_request("/embeddings/stats")

    system_action_form = SystemActionForm()

    system_data = {
        "health": health_response.get("data", {})
        if health_response.get("success")
        else {},
        "status": status_response.get("data", {})
        if status_response.get("success")
        else {},
        "vector_stats": vector_stats.get("data", {})
        if vector_stats.get("success")
        else {},
        "embedding_stats": embedding_stats.get("data", {})
        if embedding_stats.get("success")
        else {},
        "api_connected": health_response.get("success", False),
    }

    return flask.render_template(
        "administration/system.html",
        system_data=system_data,
        form=system_action_form,
    )


@administration.route("/system/action", methods=["POST"])
@login_required
def system_action():
    """
    Handle system-level actions.

    :return: Redirect with status message
    """
    form = SystemActionForm()

    if form.validate_on_submit():
        action = form.action.data

        try:
            if action == "clear_cache":
                result = make_api_request("/admin/clear-cache", method="POST")
            elif action == "rebuild_index":
                result = make_api_request(
                    "/vector-db/rebuild-index", method="POST"
                )
            elif action == "optimize_collection":
                result = make_api_request("/vector-db/optimize", method="POST")
            elif action == "generate_embeddings":
                result = make_api_request("/embeddings/generate", method="POST")
            elif action == "health_check":
                result = make_api_request("/health")
            else:
                result = {"success": False, "error": "Unknown action"}

            if result.get("success"):
                flask.flash(
                    f"Action '{action}' completed successfully", "success"
                )
            else:
                flask.flash(
                    f"Action failed: {result.get('error', 'Unknown error')}",
                    "error",
                )

        except Exception as e:
            flask.flash(f"Error executing action: {str(e)}", "error")

    return flask.redirect(flask.url_for("administration.system"))


@administration.route("/api")
@login_required
def api():
    """
    Display API integration management interface.

    :return: Rendered API template
    """
    form = APITestingForm()

    api_config = get_settlebot_api_config()
    test_results = []

    return flask.render_template(
        "administration/api.html",
        form=form,
        api_config=api_config,
        test_results=test_results,
    )


@administration.route("/api/test", methods=["POST"])
@login_required
def api_test():
    """
    Test SettleBot API endpoints.

    :return: Redirect with test results
    """
    form = APITestingForm()

    if form.validate_on_submit():
        endpoint = form.endpoint.data
        method = form.method.data
        request_body = form.request_body.data
        query_params = form.query_params.data

        try:
            # Parse query parameters
            params = {}
            if query_params:
                for param in query_params.split("&"):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params[key] = value

            # Parse request body for POST requests
            data = None
            if method == "POST" and request_body:
                data = json.loads(request_body)

            result = make_api_request(endpoint, method, data, params)

            if result.get("success"):
                flask.flash(f"API test successful for {endpoint}", "success")
            else:
                flask.flash(f"API test failed: {result.get('error')}", "error")

        except json.JSONDecodeError:
            flask.flash("Invalid JSON in request body", "error")
        except Exception as e:
            flask.flash(f"Error testing API: {str(e)}", "error")

    return flask.redirect(flask.url_for("administration.api"))


@administration.route("/analytics")
@login_required
def analytics():
    """
    Display comprehensive analytics and reporting interface.

    :return: Rendered analytics template
    """
    form = AnalyticsExportForm()

    # Calculate date ranges for analytics
    today = datetime.utcnow()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)

    # User engagement metrics
    total_users = User.query.count()
    active_users_week = User.query.filter(User.lastLogin >= week_ago).count()
    active_users_month = User.query.filter(User.lastLogin >= month_ago).count()

    # Conversation metrics
    total_conversations = Conversation.query.count()
    conversations_week = Conversation.query.filter(
        Conversation.dateCreated >= week_ago
    ).count()
    conversations_month = Conversation.query.filter(
        Conversation.dateCreated >= month_ago
    ).count()

    # Message metrics
    total_messages = Message.query.count()
    messages_week = Message.query.filter(
        Message.dateCreated >= week_ago
    ).count()
    messages_month = Message.query.filter(
        Message.dateCreated >= month_ago
    ).count()

    # Intent classification accuracy
    intent_messages = Message.query.filter(
        Message.isUserMessage == False,
        Message.intentType.isnot(None),
        Message.confidence.isnot(None),
    ).all()

    high_confidence = len([m for m in intent_messages if m.confidence >= 0.8])
    medium_confidence = len(
        [m for m in intent_messages if 0.5 <= m.confidence < 0.8]
    )
    low_confidence = len([m for m in intent_messages if m.confidence < 0.5])

    # Settlement-specific insights
    settlement_queries = Message.query.filter(
        Message.isUserMessage == False,
        Message.intentType.in_(
            [
                "housing_inquiry",
                "transportation",
                "safety_concern",
                "university_info",
                "immigration_visa",
                "banking_finance",
                "healthcare",
                "cultural_adaptation",
            ]
        ),
    ).count()

    # Popular locations and universities (would need enhanced tracking)
    popular_locations = {
        "westlands": 45,
        "kilimani": 38,
        "karen": 25,
        "lavington": 22,
        "cbd": 35,
    }  # Placeholder data

    popular_universities = {
        "university_of_nairobi": 65,
        "strathmore_university": 42,
        "jkuat": 28,
        "usiu": 24,
        "kenyatta_university": 31,
    }  # Placeholder data

    analytics_data = {
        "user_engagement": {
            "total_users": total_users,
            "active_users_week": active_users_week,
            "active_users_month": active_users_month,
            "retention_rate": round((active_users_month / total_users * 100), 2)
            if total_users > 0
            else 0,
        },
        "conversation_metrics": {
            "total_conversations": total_conversations,
            "conversations_week": conversations_week,
            "conversations_month": conversations_month,
            "avg_messages_per_conversation": round(
                total_messages / total_conversations, 2
            )
            if total_conversations > 0
            else 0,
        },
        "message_metrics": {
            "total_messages": total_messages,
            "messages_week": messages_week,
            "messages_month": messages_month,
        },
        "intent_accuracy": {
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence,
            "total_classified": len(intent_messages),
            "accuracy_rate": round(
                (high_confidence / len(intent_messages) * 100), 2
            )
            if intent_messages
            else 0,
        },
        "settlement_insights": {
            "settlement_queries": settlement_queries,
            "settlement_percentage": round(
                (settlement_queries / total_messages * 100), 2
            )
            if total_messages > 0
            else 0,
            "popular_locations": popular_locations,
            "popular_universities": popular_universities,
        },
    }

    return flask.render_template(
        "administration/analytics.html",
        analytics=analytics_data,
        form=form,
    )


@administration.route("/settlement")
@login_required
def settlement():
    """
    Display settlement-specific insights dashboard.

    :return: Rendered settlement template
    """
    form = SettlementInsightsFilterForm()

    # Apply filters from request args
    location_filter = flask.request.args.get("location_filter", "")
    university_filter = flask.request.args.get("university_filter", "")
    concern_type = flask.request.args.get("concern_type", "")

    # Base query for settlement-related messages
    query = Message.query.filter(
        Message.isUserMessage == False,
        Message.intentType.in_(
            [
                "housing_inquiry",
                "transportation",
                "safety_concern",
                "university_info",
                "immigration_visa",
                "banking_finance",
                "healthcare",
                "cultural_adaptation",
                "emergency_help",
            ]
        ),
    )

    # Apply filters (would need enhanced message metadata)
    if concern_type:
        if concern_type == "housing_cost":
            query = query.filter(Message.intentType == "housing_inquiry")
        elif concern_type == "safety_concern":
            query = query.filter(Message.intentType == "safety_concern")
        elif concern_type == "transportation_difficulty":
            query = query.filter(Message.intentType == "transportation")
        elif concern_type == "cultural_barrier":
            query = query.filter(Message.topic == "culture")
        elif concern_type == "legal_process":
            query = query.filter(Message.intentType == "immigration_visa")
        elif concern_type == "healthcare_access":
            query = query.filter(Message.intentType == "healthcare")

    settlement_messages = query.all()

    # Geographic trend analysis for Nairobi locations
    location_trends = {
        "westlands": {
            "housing_queries": 45,
            "safety_concerns": 12,
            "transport_queries": 28,
        },
        "kilimani": {
            "housing_queries": 38,
            "safety_concerns": 8,
            "transport_queries": 22,
        },
        "karen": {
            "housing_queries": 25,
            "safety_concerns": 3,
            "transport_queries": 15,
        },
        "lavington": {
            "housing_queries": 22,
            "safety_concerns": 5,
            "transport_queries": 18,
        },
        "cbd": {
            "housing_queries": 35,
            "safety_concerns": 15,
            "transport_queries": 40,
        },
        "eastleigh": {
            "housing_queries": 30,
            "safety_concerns": 20,
            "transport_queries": 35,
        },
    }

    # University and accommodation popularity
    university_data = {
        "university_of_nairobi": {
            "students": 65,
            "housing_needs": 52,
            "avg_budget": 25000,
        },
        "strathmore_university": {
            "students": 42,
            "housing_needs": 35,
            "avg_budget": 35000,
        },
        "jkuat": {"students": 28, "housing_needs": 22, "avg_budget": 20000},
        "usiu": {"students": 24, "housing_needs": 20, "avg_budget": 40000},
        "kenyatta_university": {
            "students": 31,
            "housing_needs": 25,
            "avg_budget": 18000,
        },
    }

    # Crisis intervention analysis
    crisis_data = {
        "emergency_help": Message.query.filter(
            Message.intentType == "emergency_help"
        ).count(),
        "safety_concerns": Message.query.filter(
            Message.intentType == "safety_concern"
        ).count(),
        "high_crisis": 5,  # Would need crisis level tracking
        "medium_crisis": 15,
        "low_crisis": 30,
    }

    # Cultural adaptation monitoring
    cultural_data = {
        "adaptation_queries": Message.query.filter(
            Message.topic == "culture"
        ).count(),
        "language_barriers": 25,  # Would need language difficulty tracking
        "social_integration": 40,
        "academic_adjustment": 35,
    }

    settlement_insights = {
        "location_trends": location_trends,
        "university_data": university_data,
        "crisis_interventions": crisis_data,
        "cultural_adaptation": cultural_data,
        "total_settlement_queries": len(settlement_messages),
        "most_popular_location": max(
            location_trends.keys(),
            key=lambda k: location_trends[k]["housing_queries"],
        ),
        "highest_demand_university": max(
            university_data.keys(), key=lambda k: university_data[k]["students"]
        ),
    }

    return flask.render_template(
        "administration/settlement.html",
        insights=settlement_insights,
        form=form,
        location_filter=location_filter,
        university_filter=university_filter,
    )


@administration.route("/config")
@login_required
def config():
    """
    Display system configuration (read-only).

    :return: Rendered config template
    """
    # Get configuration from SettleBot API
    config_response = make_api_request("/admin/config")
    system_status = make_api_request("/system/status")

    if config_response.get("success"):
        api_config = config_response["data"]
    else:
        api_config = {}
        flask.flash("Unable to load API configuration", "warning")

    if system_status.get("success"):
        system_config = system_status["data"].get("configuration", {})
        settlement_config = system_status["data"].get(
            "settlement_optimization", {}
        )
    else:
        system_config = {}
        settlement_config = {}

    # Local Flask configuration (non-sensitive only)
    flask_config = {
        "SETTLEBOT_API_URL": flask.current_app.config.get(
            "SETTLEBOT_API_URL", "Not configured"
        ),
        "MAIL_SERVER": flask.current_app.config.get(
            "MAIL_SERVER", "Not configured"
        ),
        "MAIL_PORT": flask.current_app.config.get(
            "MAIL_PORT", "Not configured"
        ),
        "DATABASE_URL": "Configured"
        if flask.current_app.config.get("SQLALCHEMY_DATABASE_URI")
        else "Not configured",
        "SECRET_KEY": "Configured"
        if flask.current_app.config.get("SECRET_KEY")
        else "Not configured",
    }

    configuration_data = {
        "api_config": api_config,
        "system_config": system_config,
        "settlement_config": settlement_config,
        "flask_config": flask_config,
    }

    return flask.render_template(
        "administration/config.html",
        config=configuration_data,
    )


@administration.route("/exports")
@login_required
def exports():
    """
    Display export center interface.

    :return: Rendered exports template
    """
    form = AnalyticsExportForm()

    # Export history (would be stored in database)
    export_history = [
        {
            "id": 1,
            "type": "user_analytics",
            "format": "csv",
            "created_date": datetime.utcnow() - timedelta(days=2),
            "status": "completed",
            "file_size": "2.5 MB",
        },
        {
            "id": 2,
            "type": "conversation_analytics",
            "format": "excel",
            "created_date": datetime.utcnow() - timedelta(days=5),
            "status": "completed",
            "file_size": "15.2 MB",
        },
    ]

    # Available export types
    export_types = {
        "user_analytics": "User Analytics - Registration, activity, and engagement data",
        "conversation_analytics": "Conversation Analytics - Discussion patterns and topics",
        "message_analytics": "Message Analytics - Intent classification and confidence scores",
        "document_analytics": "Document Analytics - Upload and processing statistics",
        "settlement_insights": "Settlement Insights - Location trends and university data",
        "system_performance": "System Performance - API response times and error rates",
    }

    return flask.render_template(
        "administration/exports.html",
        form=form,
        export_history=export_history,
        export_types=export_types,
    )


@administration.route("/exports/generate", methods=["POST"])
@login_required
def generate_export():
    """
    Generate data export based on user selection.

    :return: Redirect with status or file download
    """
    form = AnalyticsExportForm()

    if form.validate_on_submit():
        export_type = form.export_type.data
        format_type = form.format.data
        date_range = form.date_range.data
        include_user_data = form.include_user_data.data

        try:
            # Calculate date filter
            if date_range != "all":
                date_filter = datetime.utcnow() - timedelta(
                    days=int(date_range)
                )
            else:
                date_filter = None

            if export_type == "user_analytics":
                data = export_user_analytics(date_filter, include_user_data)
            elif export_type == "conversation_analytics":
                data = export_conversation_analytics(
                    date_filter, include_user_data
                )
            elif export_type == "message_analytics":
                data = export_message_analytics(date_filter)
            elif export_type == "settlement_insights":
                data = export_settlement_insights(date_filter)
            else:
                flask.flash("Export type not yet implemented", "warning")
                return flask.redirect(flask.url_for("administration.exports"))

            # Generate file response
            if format_type == "csv":
                return generate_csv_response(data, export_type)
            elif format_type == "excel":
                return generate_excel_response(data, export_type)
            elif format_type == "json":
                return generate_json_response(data, export_type)

        except Exception as e:
            flask.flash(f"Error generating export: {str(e)}", "error")

    return flask.redirect(flask.url_for("administration.exports"))


def export_user_analytics(date_filter, include_user_data):
    """Export user analytics data."""
    query = User.query
    if date_filter:
        query = query.filter(User.dateCreated >= date_filter)

    users = query.all()

    data = []
    for user in users:
        row = {
            "user_id": user.userId,
            "registration_date": user.dateCreated.isoformat()
            if user.dateCreated
            else None,
            "last_login": user.lastLogin.isoformat()
            if user.lastLogin
            else None,
            "role": user.role,
            "is_active": user.isActive,
            "allow_data_usage": user.allowDataUsage,
            "conversation_count": Conversation.query.filter_by(
                userId=user.userId
            ).count(),
            "message_count": Message.query.join(Conversation)
            .filter(Conversation.userId == user.userId)
            .count(),
        }

        if include_user_data:
            row.update(
                {
                    "full_name": user.fullName,
                    "username": user.username,
                    "email_address": user.emailAddress,
                }
            )

        data.append(row)

    return data


def export_conversation_analytics(date_filter, include_user_data):
    """Export conversation analytics data."""
    query = Conversation.query
    if date_filter:
        query = query.filter(Conversation.dateCreated >= date_filter)

    conversations = query.all()

    data = []
    for conv in conversations:
        message_count = Message.query.filter_by(
            conversationId=conv.conversationId
        ).count()
        last_message = (
            Message.query.filter_by(conversationId=conv.conversationId)
            .order_by(desc(Message.dateCreated))
            .first()
        )

        row = {
            "conversation_id": conv.conversationId,
            "title": conv.title,
            "created_date": conv.dateCreated.isoformat()
            if conv.dateCreated
            else None,
            "last_updated": conv.lastUpdated.isoformat()
            if conv.lastUpdated
            else None,
            "message_count": message_count,
            "last_intent": last_message.intentType if last_message else None,
            "last_topic": last_message.topic if last_message else None,
        }

        if include_user_data:
            row.update(
                {
                    "user_id": conv.userId,
                    "user_name": conv.user.fullName if conv.user else None,
                }
            )

        data.append(row)

    return data


def export_message_analytics(date_filter):
    """Export message analytics data."""
    query = Message.query.filter(Message.isUserMessage == False)
    if date_filter:
        query = query.filter(Message.dateCreated >= date_filter)

    messages = query.all()

    data = []
    for message in messages:
        data.append(
            {
                "message_id": message.messageId,
                "conversation_id": message.conversationId,
                "intent_type": message.intentType,
                "topic": message.topic,
                "confidence": message.confidence,
                "token_count": message.tokenCount,
                "created_date": message.dateCreated.isoformat()
                if message.dateCreated
                else None,
            }
        )

    return data


def export_settlement_insights(date_filter):
    """Export settlement-specific insights."""
    # This would compile settlement-specific data
    return [
        {"location": "westlands", "housing_queries": 45, "safety_concerns": 12},
        {"location": "kilimani", "housing_queries": 38, "safety_concerns": 8},
        {"location": "karen", "housing_queries": 25, "safety_concerns": 3},
    ]


def generate_csv_response(data, export_type):
    """Generate CSV response."""
    import csv
    import io

    output = io.StringIO()
    if data:
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    response = flask.make_response(output.getvalue())
    response.headers[
        "Content-Disposition"
    ] = f"attachment; filename={export_type}_{datetime.now().strftime('%Y%m%d')}.csv"
    response.headers["Content-type"] = "text/csv"

    return response


def generate_excel_response(data, export_type):
    """Generate Excel response."""
    try:
        import pandas as pd
        import io

        df = pd.DataFrame(data)
        output = io.BytesIO()
        df.to_excel(output, index=False, engine="openpyxl")
        output.seek(0)

        response = flask.make_response(output.getvalue())
        response.headers[
            "Content-Disposition"
        ] = f"attachment; filename={export_type}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        response.headers[
            "Content-type"
        ] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        return response
    except ImportError:
        flask.flash(
            "Excel export requires pandas and openpyxl packages", "error"
        )
        return flask.redirect(flask.url_for("administration.exports"))


def generate_json_response(data, export_type):
    """Generate JSON response."""
    response = flask.make_response(json.dumps(data, indent=2, default=str))
    response.headers[
        "Content-Disposition"
    ] = f"attachment; filename={export_type}_{datetime.now().strftime('%Y%m%d')}.json"
    response.headers["Content-type"] = "application/json"

    return response
