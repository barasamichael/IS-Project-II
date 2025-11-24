from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms import SubmitField
from wtforms import BooleanField
from wtforms import SelectField
from wtforms import TextAreaField
from wtforms import FileField
from wtforms import FloatField

from wtforms.validators import Length
from wtforms.validators import Optional
from wtforms.validators import NumberRange
from wtforms.validators import DataRequired

from flask_wtf.file import FileAllowed


class UserSearchForm(FlaskForm):
    """
    Form for searching and filtering users.
    """

    search_query = StringField(
        "Search Users",
        validators=[
            Optional(),
            Length(
                max=100,
                message="Search query cannot exceed 100 characters",
            ),
        ],
        render_kw={"placeholder": "Search by name, username, or email..."},
    )
    role_filter = SelectField(
        "Filter by Role",
        choices=[
            ("", "All Roles"),
            ("student", "Student"),
            ("admin", "Admin"),
            ("developer", "Developer"),
            ("moderator", "Moderator"),
        ],
        validators=[Optional()],
    )
    status_filter = SelectField(
        "Filter by Status",
        choices=[
            ("", "All Users"),
            ("active", "Active Users"),
            ("inactive", "Inactive Users"),
        ],
        validators=[Optional()],
    )
    data_usage_filter = SelectField(
        "Data Usage Consent",
        choices=[
            ("", "All Users"),
            ("true", "Consented"),
            ("false", "Not Consented"),
        ],
        validators=[Optional()],
    )
    submit = SubmitField("Search & Filter")


class UserBulkActionForm(FlaskForm):
    """
    Form for bulk actions on users.
    """

    action = SelectField(
        "Bulk Action",
        choices=[
            ("", "Select Action"),
            ("activate", "Activate Selected"),
            ("deactivate", "Deactivate Selected"),
            ("export", "Export Selected"),
            ("role_student", "Set Role to Student"),
            ("role_moderator", "Set Role to Moderator"),
        ],
        validators=[DataRequired(message="Please select an action")],
    )
    selected_users = TextAreaField(
        "Selected User IDs",
        validators=[DataRequired(message="No users selected")],
        render_kw={"style": "display:none;"},
    )
    submit = SubmitField("Execute Action")


class ConversationSearchForm(FlaskForm):
    """
    Form for searching and filtering conversations.
    """

    search_query = StringField(
        "Search Conversations",
        validators=[
            Optional(),
            Length(
                max=100,
                message="Search query cannot exceed 100 characters",
            ),
        ],
        render_kw={"placeholder": "Search by title, user, or content..."},
    )
    intent_filter = SelectField(
        "Filter by Intent",
        choices=[
            ("", "All Intents"),
            ("housing_inquiry", "Housing Inquiry"),
            ("university_info", "University Info"),
            ("immigration_visa", "Immigration/Visa"),
            ("transportation", "Transportation"),
            ("safety_concern", "Safety Concern"),
            ("cost_inquiry", "Cost Inquiry"),
            ("banking_finance", "Banking/Finance"),
            ("healthcare", "Healthcare"),
            ("cultural_adaptation", "Cultural Adaptation"),
            ("emergency_help", "Emergency Help"),
            ("procedural_query", "Procedural Query"),
            ("off_topic", "Off Topic"),
        ],
        validators=[Optional()],
    )
    topic_filter = SelectField(
        "Filter by Topic",
        choices=[
            ("", "All Topics"),
            ("housing", "Housing"),
            ("transportation", "Transportation"),
            ("education", "Education"),
            ("legal", "Legal"),
            ("finance", "Finance"),
            ("safety", "Safety"),
            ("healthcare", "Healthcare"),
            ("culture", "Culture"),
            ("general", "General"),
        ],
        validators=[Optional()],
    )
    crisis_level_filter = SelectField(
        "Filter by Crisis Level",
        choices=[
            ("", "All Levels"),
            ("none", "No Crisis"),
            ("low", "Low Crisis"),
            ("medium", "Medium Crisis"),
            ("high", "High Crisis"),
            ("critical", "Critical Crisis"),
        ],
        validators=[Optional()],
    )
    empathy_applied_filter = SelectField(
        "Empathy Applied",
        choices=[
            ("", "All Conversations"),
            ("true", "Empathy Applied"),
            ("false", "No Empathy Applied"),
        ],
        validators=[Optional()],
    )
    safety_protocols_filter = SelectField(
        "Safety Protocols",
        choices=[
            ("", "All Conversations"),
            ("true", "Safety Protocols Triggered"),
            ("false", "No Safety Protocols"),
        ],
        validators=[Optional()],
    )
    submit = SubmitField("Search & Filter")


class ConversationBulkActionForm(FlaskForm):
    """
    Form for bulk actions on conversations.
    """

    action = SelectField(
        "Bulk Action",
        choices=[
            ("", "Select Action"),
            ("export", "Export Selected"),
            ("delete", "Delete Selected"),
        ],
        validators=[DataRequired(message="Please select an action")],
    )
    selected_conversations = TextAreaField(
        "Selected Conversation IDs",
        validators=[DataRequired(message="No conversations selected")],
        render_kw={"style": "display:none;"},
    )
    submit = SubmitField("Execute Action")


class MessageAnalyticsFilterForm(FlaskForm):
    """
    Form for filtering message analytics.
    """

    date_range = SelectField(
        "Date Range",
        choices=[
            ("7", "Last 7 Days"),
            ("30", "Last 30 Days"),
            ("90", "Last 90 Days"),
            ("365", "Last Year"),
            ("all", "All Time"),
        ],
        default="30",
        validators=[Optional()],
    )
    intent_type = SelectField(
        "Intent Type",
        choices=[
            ("", "All Intents"),
            ("housing_inquiry", "Housing Inquiry"),
            ("university_info", "University Info"),
            ("immigration_visa", "Immigration/Visa"),
            ("transportation", "Transportation"),
            ("safety_concern", "Safety Concern"),
            ("cost_inquiry", "Cost Inquiry"),
            ("banking_finance", "Banking/Finance"),
            ("healthcare", "Healthcare"),
            ("cultural_adaptation", "Cultural Adaptation"),
            ("emergency_help", "Emergency Help"),
            ("procedural_query", "Procedural Query"),
            ("off_topic", "Off Topic"),
        ],
        validators=[Optional()],
    )
    topic_type = SelectField(
        "Topic",
        choices=[
            ("", "All Topics"),
            ("housing", "Housing"),
            ("transportation", "Transportation"),
            ("education", "Education"),
            ("legal", "Legal"),
            ("finance", "Finance"),
            ("safety", "Safety"),
            ("healthcare", "Healthcare"),
            ("culture", "Culture"),
            ("general", "General"),
        ],
        validators=[Optional()],
    )
    min_confidence = FloatField(
        "Minimum Confidence",
        validators=[
            Optional(),
            NumberRange(
                min=0.0,
                max=1.0,
                message="Confidence must be between 0.0 and 1.0",
            ),
        ],
        render_kw={"step": "0.01", "min": "0", "max": "1"},
    )
    submit = SubmitField("Apply Filters")


class DocumentSearchForm(FlaskForm):
    """
    Form for searching and filtering documents.
    """

    search_query = StringField(
        "Search Documents",
        validators=[
            Optional(),
            Length(
                max=100,
                message="Search query cannot exceed 100 characters",
            ),
        ],
        render_kw={"placeholder": "Search by filename or content..."},
    )
    doc_type_filter = SelectField(
        "Document Type",
        choices=[
            ("", "All Types"),
            ("pdf", "PDF"),
            ("docx", "Word Document"),
            ("txt", "Text File"),
            ("html", "HTML"),
            ("web_url", "Web URL"),
            ("web_sitemap", "Web Sitemap"),
        ],
        validators=[Optional()],
    )
    settlement_score_filter = SelectField(
        "Settlement Score",
        choices=[
            ("", "All Scores"),
            ("0.8", "Excellent (0.8+)"),
            ("0.6", "Good (0.6+)"),
            ("0.4", "Fair (0.4+)"),
            ("0.0", "All Documents"),
        ],
        validators=[Optional()],
    )
    processing_status_filter = SelectField(
        "Processing Status",
        choices=[
            ("", "All Statuses"),
            ("completed", "Completed"),
            ("processing", "Processing"),
            ("failed", "Failed"),
            ("pending", "Pending"),
        ],
        validators=[Optional()],
    )
    submit = SubmitField("Search & Filter")


class DocumentUploadForm(FlaskForm):
    """
    Form for uploading documents.
    """

    file = FileField(
        "Select Document",
        validators=[
            DataRequired(message="Please select a file"),
            FileAllowed(
                ["pdf", "docx", "txt", "html"],
                "Only PDF, DOCX, TXT, and HTML files are allowed",
            ),
        ],
    )
    submit = SubmitField("Upload Document")


class DocumentBulkActionForm(FlaskForm):
    """
    Form for bulk actions on documents.
    """

    action = SelectField(
        "Bulk Action",
        choices=[
            ("", "Select Action"),
            ("reprocess", "Reprocess Selected"),
            ("delete", "Delete Selected"),
            ("export_metadata", "Export Metadata"),
            ("rebuild_embeddings", "Rebuild Embeddings"),
        ],
        validators=[DataRequired(message="Please select an action")],
    )
    selected_documents = TextAreaField(
        "Selected Document IDs",
        validators=[DataRequired(message="No documents selected")],
        render_kw={"style": "display:none;"},
    )
    submit = SubmitField("Execute Action")


class APITestingForm(FlaskForm):
    """
    Form for testing SettleBot API endpoints.
    """

    endpoint = SelectField(
        "API Endpoint",
        choices=[
            ("", "Select Endpoint"),
            ("/health", "Health Check"),
            ("/system/status", "System Status"),
            ("/query", "Query Processing"),
            ("/documents", "List Documents"),
            ("/search", "Search Knowledge Base"),
            ("/intent/analyze", "Analyze Intent"),
            ("/vector-db/stats", "Vector DB Stats"),
            ("/embeddings/stats", "Embeddings Stats"),
        ],
        validators=[DataRequired(message="Please select an endpoint")],
    )
    method = SelectField(
        "HTTP Method",
        choices=[
            ("GET", "GET"),
            ("POST", "POST"),
        ],
        default="GET",
        validators=[DataRequired()],
    )
    request_body = TextAreaField(
        "Request Body (JSON)",
        validators=[Optional()],
        render_kw={
            "rows": "10",
            "placeholder": "Enter JSON request body for POST requests...",
        },
    )
    query_params = StringField(
        "Query Parameters",
        validators=[Optional()],
        render_kw={"placeholder": "e.g., param1=value1&param2=value2"},
    )
    submit = SubmitField("Test Endpoint")


class SystemActionForm(FlaskForm):
    """
    Form for system-level actions.
    """

    action = SelectField(
        "System Action",
        choices=[
            ("", "Select Action"),
            ("clear_cache", "Clear System Cache"),
            ("rebuild_index", "Rebuild Vector Index"),
            ("optimize_collection", "Optimize Collection"),
            ("generate_embeddings", "Generate Embeddings"),
            ("health_check", "Run Health Check"),
        ],
        validators=[DataRequired(message="Please select an action")],
    )
    confirm_action = BooleanField(
        "I understand this action may affect system performance",
        validators=[DataRequired(message="Please confirm the action")],
    )
    submit = SubmitField("Execute Action")


class AnalyticsExportForm(FlaskForm):
    """
    Form for exporting analytics data.
    """

    export_type = SelectField(
        "Export Type",
        choices=[
            ("", "Select Export Type"),
            ("user_analytics", "User Analytics"),
            ("conversation_analytics", "Conversation Analytics"),
            ("message_analytics", "Message Analytics"),
            ("document_analytics", "Document Analytics"),
            ("settlement_insights", "Settlement Insights"),
            ("system_performance", "System Performance"),
        ],
        validators=[DataRequired(message="Please select export type")],
    )
    format = SelectField(
        "Export Format",
        choices=[
            ("csv", "CSV"),
            ("excel", "Excel"),
            ("json", "JSON"),
        ],
        default="csv",
        validators=[DataRequired()],
    )
    date_range = SelectField(
        "Date Range",
        choices=[
            ("7", "Last 7 Days"),
            ("30", "Last 30 Days"),
            ("90", "Last 90 Days"),
            ("365", "Last Year"),
            ("all", "All Time"),
        ],
        default="30",
        validators=[Optional()],
    )
    include_user_data = BooleanField(
        "Include User Identifiable Data",
        default=False,
    )
    submit = SubmitField("Generate Export")


class SettlementInsightsFilterForm(FlaskForm):
    """
    Form for filtering settlement insights.
    """

    location_filter = SelectField(
        "Nairobi Location",
        choices=[
            ("", "All Locations"),
            ("westlands", "Westlands"),
            ("kilimani", "Kilimani"),
            ("karen", "Karen"),
            ("lavington", "Lavington"),
            ("kileleshwa", "Kileleshwa"),
            ("parklands", "Parklands"),
            ("hurlingham", "Hurlingham"),
            ("riverside", "Riverside"),
            ("cbd", "CBD"),
            ("eastleigh", "Eastleigh"),
        ],
        validators=[Optional()],
    )
    university_filter = SelectField(
        "University",
        choices=[
            ("", "All Universities"),
            ("university_of_nairobi", "University of Nairobi"),
            ("strathmore_university", "Strathmore University"),
            ("jkuat", "JKUAT"),
            ("usiu", "USIU"),
            ("kenyatta_university", "Kenyatta University"),
            ("daystar_university", "Daystar University"),
        ],
        validators=[Optional()],
    )
    concern_type = SelectField(
        "Concern Type",
        choices=[
            ("", "All Concerns"),
            ("housing_cost", "Housing Cost"),
            ("safety_concern", "Safety Issues"),
            ("transportation_difficulty", "Transportation"),
            ("cultural_barrier", "Cultural Barriers"),
            ("legal_process", "Legal Processes"),
            ("healthcare_access", "Healthcare Access"),
        ],
        validators=[Optional()],
    )
    crisis_level = SelectField(
        "Crisis Level",
        choices=[
            ("", "All Levels"),
            ("low", "Low"),
            ("medium", "Medium"),
            ("high", "High"),
            ("critical", "Critical"),
        ],
        validators=[Optional()],
    )
    submit = SubmitField("Apply Filters")
