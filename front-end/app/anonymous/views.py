import flask
import datetime
import requests

from flask import session
from flask import make_response

from app import db
from app import csrf

from . import anonymous
from ..models.anonymous_user import AnonymousSession
from ..models.anonymous_user import AnonymousMessage
from ..models.anonymous_user import AnonymousConversation


def get_or_create_anonymous_session():
    """Get existing anonymous session or create new one."""
    token = flask.request.cookies.get('anon_session_token')
    
    if token:
        anon_session = AnonymousSession.get_by_token(token)
        if anon_session:
            anon_session.refresh_activity()
            return anon_session
    
    # Create new session
    anon_session = AnonymousSession.create(session_duration_hours=24)
    return anon_session


@anonymous.route('/chat')
def anonymous_chat():
    """Anonymous chat interface - no login required."""
    # Get or create anonymous session
    anon_session = get_or_create_anonymous_session()
    
    # Get or create a conversation for this session
    conversation = AnonymousConversation.query.filter_by(
        sessionId=anon_session.sessionId
    ).order_by(AnonymousConversation.lastUpdated.desc()).first()
    
    if not conversation:
        conversation = AnonymousConversation.create({
            "sessionId": anon_session.sessionId,
            "title": "Quick Chat"
        })
    
    # Get messages for this conversation
    messages = (
        AnonymousMessage.query
        .filter_by(conversationId=conversation.conversationId)
        .order_by(AnonymousMessage.dateCreated)
        .all()
    )
    
    response = make_response(flask.render_template(
        'anonymous/chat.html',
        conversation=conversation,
        messages=messages,
        is_anonymous=True
    ))
    
    # Set session cookie
    response.set_cookie(
        'anon_session_token',
        anon_session.sessionToken,
        max_age=86400,  # 24 hours
        httponly=True,
        secure=flask.current_app.config.get('SESSION_COOKIE_SECURE', False),
        samesite='Lax'
    )
    
    return response


@anonymous.route('/api/message', methods=['POST'])
@csrf.exempt
def anonymous_send_message():
    """Send message in anonymous chat."""
    # Get anonymous session
    token = flask.request.cookies.get('anon_session_token')
    if not token:
        return flask.jsonify({"error": "No session found"}), 401
    
    anon_session = AnonymousSession.get_by_token(token)
    if not anon_session:
        return flask.jsonify({"error": "Invalid or expired session"}), 401
    
    # Get data
    data = flask.request.get_json()
    if not data or "content" not in data:
        return flask.jsonify({"error": "No message content"}), 400
    
    message_content = data["content"].strip()
    if not message_content:
        return flask.jsonify({"error": "Empty message"}), 400
    
    try:
        # Get or create conversation
        conversation = AnonymousConversation.query.filter_by(
            sessionId=anon_session.sessionId
        ).order_by(AnonymousConversation.lastUpdated.desc()).first()
        
        if not conversation:
            conversation = AnonymousConversation.create({
                "sessionId": anon_session.sessionId,
                "title": "Quick Chat"
            })
        
        # Save user message
        user_message = AnonymousMessage.create({
            "conversationId": conversation.conversationId,
            "isUserMessage": True,
            "content": message_content,
            "tokenCount": len(message_content.split())
        })
        
        # Call SettleBot API
        api_url = flask.current_app.config.get(
            "SETTLEBOT_API_URL", "http://localhost:8000"
        )
        api_key = flask.current_app.config.get("SETTLEBOT_API_KEY")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }
        
        request_payload = {
            "query": message_content,
            "top_k": 15,
            "include_context": True,
            "language_detection": True,
            "user_preferences": {
                "response_style": "comprehensive_empathetic",
                "include_safety_protocols": "true",
                "include_cost_information": "true"
            }
        }
        
        response = requests.post(
            f"{api_url}/query",
            json=request_payload,
            headers=headers,
            timeout=None
        )
        
        if response.status_code != 200:
            return flask.jsonify({
                "error": f"API error: {response.status_code}"
            }), 500
        
        # Process response
        settlebot_response = response.json()
        response_text = settlebot_response.get("response", "Sorry, I couldn't generate a response.")
        
        # Save assistant message
        assistant_message = AnonymousMessage.create({
            "conversationId": conversation.conversationId,
            "isUserMessage": False,
            "content": response_text,
            "intentType": settlebot_response.get("intent_type"),
            "topic": settlebot_response.get("topic"),
            "confidence": settlebot_response.get("confidence"),
            "tokenCount": len(response_text.split())
        })
        
        # Update conversation
        conversation.lastUpdated = datetime.datetime.utcnow()
        db.session.commit()
        
        # Refresh session activity
        anon_session.refresh_activity()
        
        return flask.jsonify({
            "user_message": user_message.getDetails(),
            "assistant_message": assistant_message.getDetails(),
            "settlement_info": {
                "intent_type": settlebot_response.get("intent_type"),
                "topic": settlebot_response.get("topic"),
                "confidence": settlebot_response.get("confidence")
            }
        })
        
    except Exception as e:
        db.session.rollback()
        flask.current_app.logger.error(f"Error in anonymous chat: {str(e)}")
        return flask.jsonify({"error": str(e)}), 500


@anonymous.route('/api/new-chat', methods=['POST'])
@csrf.exempt
def anonymous_new_chat():
    """Start a new anonymous chat (clears current conversation)."""
    token = flask.request.cookies.get('anon_session_token')
    if not token:
        return flask.jsonify({"error": "No session found"}), 401
    
    anon_session = AnonymousSession.get_by_token(token)
    if not anon_session:
        return flask.jsonify({"error": "Invalid session"}), 401
    
    try:
        conversation = AnonymousConversation.create({
            "sessionId": anon_session.sessionId,
            "title": "Quick Chat"
        })
        
        return flask.jsonify(conversation.getDetails())
        
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500
