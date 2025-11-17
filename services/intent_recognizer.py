import re
import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intent_recognizer")


class IntentType(str, Enum):
    HOUSING_INQUIRY = "housing_inquiry"
    UNIVERSITY_INFO = "university_info"
    NEIGHBORHOOD_GUIDE = "neighborhood_guide"
    COST_INQUIRY = "cost_inquiry"
    SAFETY_CONCERN = "safety_concern"
    TRANSPORTATION = "transportation"
    BANKING_FINANCE = "banking_finance"
    IMMIGRATION_VISA = "immigration_visa"
    ENTERTAINMENT_SOCIAL = "entertainment_social"
    ACADEMIC_CONVERSION = "academic_conversion"
    EMERGENCY_HELP = "emergency_help"
    CULTURAL_ADAPTATION = "cultural_adaptation"
    SHOPPING_MARKETS = "shopping_markets"
    HEALTHCARE = "healthcare"
    COMPARISON_QUERY = "comparison_query"
    PROCEDURAL_QUERY = "procedural_query"
    REASSURANCE_SEEKING = "reassurance_seeking"
    GENERAL_CHAT = "general_chat"
    CLARIFICATION = "clarification"
    OFF_TOPIC = "off_topic"


class TopicCategory(str, Enum):
    HOUSING = "housing"
    EDUCATION = "education"
    LOCATION = "location"
    FINANCE = "finance"
    SAFETY = "safety"
    TRANSPORT = "transport"
    LEGAL = "legal"
    LIFESTYLE = "lifestyle"
    ACADEMICS = "academics"
    HEALTH = "health"
    CULTURE = "culture"
    SHOPPING = "shopping"
    GENERAL = "general"


class IntentRecognizer:
    """Advanced intent recognition system for international student settlement queries."""

    def __init__(self):
        self.intent_patterns = {
            IntentType.HOUSING_INQUIRY: [
                r"(?:where|how).*(?:find|get|rent).*(?:house|room|apartment|accommodation)",
                r"(?:housing|accommodation|rent|landlord|tenant)",
                r"(?:studio|bedsitter|one bedroom|two bedroom)",
                r"(?:deposit|rent deposit|security deposit)",
                r"(?:utilities|electricity|water|internet).*(?:included|cost)",
                r"(?:furnished|unfurnished).*(?:apartment|room)",
                r"(?:lease|tenancy|rental agreement)",
                r"best.*(?:areas|neighborhoods).*(?:live|stay|accommodation)",
            ],
            IntentType.UNIVERSITY_INFO: [
                r"(?:university|college|campus).*(?:location|address|contact)",
                r"(?:admission|enrollment|registration).*(?:process|requirements)",
                r"(?:academic|semester|term).*(?:calendar|schedule)",
                r"(?:library|cafeteria|facilities).*(?:hours|location)",
                r"(?:student services|international office)",
                r"(?:tuition|fees).*(?:payment|cost)",
                r"(?:dormitory|hostel).*(?:university|campus)",
            ],
            IntentType.NEIGHBORHOOD_GUIDE: [
                r"(?:safe|unsafe).*(?:area|neighborhood|place)",
                r"(?:best|good|recommended).*(?:area|neighborhood|location).*(?:students|international)",
                r"(?:avoid|dangerous).*(?:area|place|neighborhood)",
                r"(?:Westlands|Kilimani|Karen|Lavington|Kileleshwa|Parklands)",
                r"(?:close to|near).*(?:university|campus|school)",
                r"(?:residential|commercial).*area",
            ],
            IntentType.COST_INQUIRY: [
                r"(?:how much|cost|price|expensive|cheap)",
                r"(?:budget|afford|money).*(?:need|required)",
                r"(?:monthly|weekly|daily).*(?:cost|expense)",
                r"(?:cost of living|living expenses)",
                r"(?:comparison|compare).*(?:price|cost)",
                r"(?:average|typical).*(?:cost|price|rent)",
            ],
            IntentType.SAFETY_CONCERN: [
                r"(?:safe|safety|secure|security)",
                r"(?:crime|theft|robbery|dangerous)",
                r"(?:police|emergency).*(?:number|contact)",
                r"(?:avoid|stay away).*(?:time|night|area)",
                r"(?:walk|travel).*(?:alone|night|dark)",
                r"(?:security|guard|gated)",
            ],
            IntentType.TRANSPORTATION: [
                r"(?:transport|transportation|travel|commute)",
                r"(?:matatu|bus|taxi|uber|bolt)",
                r"(?:public transport|boda boda|motorcycle)",
                r"(?:fare|cost).*(?:transport|travel|taxi)",
                r"(?:route|direction).*(?:to|from)",
                r"(?:driving|license|car).*(?:Kenya|Nairobi)",
                r"(?:train|railway|sgr)",
            ],
            IntentType.BANKING_FINANCE: [
                r"(?:bank|banking).*(?:account|open|create)",
                r"(?:ATM|cash|withdraw|deposit)",
                r"(?:mpesa|mobile money|payment)",
                r"(?:currency|exchange|forex|shilling)",
                r"(?:credit card|debit card|visa|mastercard)",
                r"(?:transfer|send).*money",
                r"(?:loan|credit|finance)",
            ],
            IntentType.IMMIGRATION_VISA: [
                r"(?:visa|permit|passport|immigration)",
                r"(?:student visa|study permit)",
                r"(?:work permit|employment)",
                r"(?:renewal|extend|extension)",
                r"(?:immigration office|DCI|department)",
                r"(?:requirements|documents).*(?:visa|permit)",
                r"(?:overstay|expired|violation)",
            ],
            IntentType.ENTERTAINMENT_SOCIAL: [
                r"(?:entertainment|fun|social|nightlife)",
                r"(?:club|bar|restaurant|cafe)",
                r"(?:meet|friends|social).*(?:people|students)",
                r"(?:events|activities|things to do)",
                r"(?:cinema|movie|theater|concert)",
                r"(?:sports|gym|fitness|recreation)",
                r"(?:shopping|mall|market)",
            ],
            IntentType.ACADEMIC_CONVERSION: [
                r"(?:grade|GPA|marks).*(?:conversion|equivalent)",
                r"(?:credit|unit).*(?:transfer|recognition)",
                r"(?:transcript|certificate).*(?:evaluation|verification)",
                r"(?:education|academic).*(?:system|structure)",
                r"(?:semester|term).*(?:system|calendar)",
            ],
            IntentType.EMERGENCY_HELP: [
                r"(?:emergency|urgent|help|crisis)",
                r"(?:hospital|medical|health).*(?:emergency|urgent)",
                r"(?:police|fire|ambulance)",
                r"(?:lost|stolen|missing).*(?:passport|document)",
                r"(?:accident|injury|sick)",
                r"(?:embassy|consulate|diplomatic)",
            ],
            IntentType.CULTURAL_ADAPTATION: [
                r"(?:culture|cultural|custom|tradition)",
                r"(?:language|speak|communication)",
                r"(?:etiquette|manners|behavior|conduct)",
                r"(?:religion|religious|worship|church|mosque)",
                r"(?:food|eat|diet|halal|vegetarian)",
                r"(?:dress|clothing|appropriate|formal)",
                r"(?:greeting|introduction|social).*(?:norm|custom)",
            ],
            IntentType.SHOPPING_MARKETS: [
                r"(?:shop|shopping|buy|purchase)",
                r"(?:supermarket|grocery|store|mall)",
                r"(?:market|mama mboga|kiosk)",
                r"(?:electronics|clothes|books).*(?:buy|shop)",
                r"(?:cheap|affordable|discount).*(?:shop|buy)",
                r"(?:online|delivery|e-commerce)",
            ],
            IntentType.HEALTHCARE: [
                r"(?:hospital|clinic|medical|health|doctor)",
                r"(?:insurance|medical cover|health insurance)",
                r"(?:medicine|pharmacy|drug|prescription)",
                r"(?:vaccination|immunization|medical check)",
                r"(?:mental health|counseling|therapy)",
                r"(?:dental|dentist|eye|optician)",
            ],
            IntentType.COMPARISON_QUERY: [
                r"(?:compare|comparison|versus|vs)",
                r"(?:better|best|prefer|choose)",
                r"(?:difference|similar|same)",
                r"(?:option|choice|alternative)",
                r"(?:advantage|disadvantage|pros|cons)",
            ],
            IntentType.PROCEDURAL_QUERY: [
                r"(?:how to|how do|how can|steps to)",
                r"(?:process|procedure|method|way)",
                r"(?:apply|register|enroll|sign up)",
                r"(?:requirements|documents|needed|necessary)",
                r"(?:timeline|duration|time|takes)",
            ],
            IntentType.REASSURANCE_SEEKING: [
                r"(?:worried|scared|anxious|nervous)",
                r"(?:normal|common|typical|usual)",
                r"(?:others|other students|everyone)",
                r"(?:experience|faced|deal with)",
                r"(?:support|help|advice|guidance)",
                r"(?:alone|isolated|homesick)",
            ],
            IntentType.GENERAL_CHAT: [
                r"^(?:hi|hello|hey|greetings)",
                r"(?:how are you|what's up|how's it going)",
                r"(?:thank|thanks|appreciate)",
                r"(?:good|great|excellent|wonderful)",
                r"(?:bye|goodbye|see you|farewell)",
            ],
            IntentType.CLARIFICATION: [
                r"(?:clarify|explain|elaborate|specify)",
                r"(?:what do you mean|meaning|definition)",
                r"(?:confused|understand|clear|unclear)",
                r"(?:example|instance|case|sample)",
                r"(?:more information|details|specific)",
            ],
        }

        self.topic_keywords = {
            TopicCategory.HOUSING: [
                "accommodation",
                "housing",
                "rent",
                "apartment",
                "room",
                "bedsitter",
                "studio",
                "landlord",
                "tenant",
                "deposit",
                "utilities",
                "furnished",
                "lease",
                "rental",
                "dormitory",
                "hostel",
                "residential",
            ],
            TopicCategory.EDUCATION: [
                "university",
                "college",
                "campus",
                "school",
                "admission",
                "enrollment",
                "registration",
                "academic",
                "semester",
                "term",
                "library",
                "facilities",
                "tuition",
                "fees",
                "student services",
                "international office",
            ],
            TopicCategory.LOCATION: [
                "area",
                "neighborhood",
                "location",
                "place",
                "region",
                "district",
                "Westlands",
                "Kilimani",
                "Karen",
                "Lavington",
                "Kileleshwa",
                "Parklands",
                "CBD",
                "town",
                "city",
                "suburb",
                "residential",
                "commercial",
            ],
            TopicCategory.FINANCE: [
                "money",
                "cost",
                "price",
                "budget",
                "expensive",
                "cheap",
                "afford",
                "bank",
                "banking",
                "account",
                "ATM",
                "mpesa",
                "currency",
                "exchange",
                "payment",
                "transfer",
                "loan",
                "credit",
                "finance",
                "shilling",
            ],
            TopicCategory.SAFETY: [
                "safe",
                "safety",
                "secure",
                "security",
                "crime",
                "theft",
                "robbery",
                "dangerous",
                "police",
                "emergency",
                "guard",
                "gated",
                "avoid",
                "risk",
            ],
            TopicCategory.TRANSPORT: [
                "transport",
                "transportation",
                "travel",
                "commute",
                "matatu",
                "bus",
                "taxi",
                "uber",
                "bolt",
                "boda",
                "motorcycle",
                "public",
                "fare",
                "route",
            ],
            TopicCategory.LEGAL: [
                "visa",
                "permit",
                "passport",
                "immigration",
                "legal",
                "law",
                "rights",
                "requirements",
                "documents",
                "renewal",
                "extension",
                "embassy",
                "consulate",
            ],
            TopicCategory.LIFESTYLE: [
                "entertainment",
                "fun",
                "social",
                "nightlife",
                "club",
                "bar",
                "restaurant",
                "events",
                "activities",
                "cinema",
                "sports",
                "gym",
                "recreation",
                "culture",
            ],
            TopicCategory.ACADEMICS: [
                "grade",
                "GPA",
                "marks",
                "credit",
                "unit",
                "transcript",
                "certificate",
                "conversion",
                "equivalent",
                "evaluation",
                "recognition",
                "academic system",
            ],
            TopicCategory.HEALTH: [
                "hospital",
                "clinic",
                "medical",
                "health",
                "doctor",
                "insurance",
                "medicine",
                "pharmacy",
                "vaccination",
                "mental health",
                "dental",
            ],
            TopicCategory.CULTURE: [
                "culture",
                "cultural",
                "custom",
                "tradition",
                "language",
                "etiquette",
                "religion",
                "food",
                "dress",
                "behavior",
                "norm",
                "adaptation",
            ],
            TopicCategory.SHOPPING: [
                "shop",
                "shopping",
                "buy",
                "purchase",
                "supermarket",
                "grocery",
                "market",
                "mall",
                "store",
                "electronics",
                "clothes",
                "online",
                "delivery",
            ],
        }

        # Compile patterns for efficiency
        self.compiled_intent_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_intent_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def recognize_intent(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recognize intent with enhanced pattern matching and context awareness.

        Args:
            query: User query
            context: Optional conversation context

        Returns:
            Intent information with confidence scores
        """
        query = query.strip().lower()

        # Check for explicit context indicators first
        if context and context.get("has_context"):
            if any(
                indicator in query
                for indicator in ["clarify", "explain more", "what about"]
            ):
                return {
                    "intent_type": IntentType.CLARIFICATION,
                    "confidence": 0.8,
                    "topic": self._determine_topic(query),
                    "requires_context": True,
                }

        # Score all intents
        intent_scores = {}
        for intent, patterns in self.compiled_intent_patterns.items():
            score = 0
            matches = 0

            for pattern in patterns:
                if pattern.search(query):
                    matches += 1
                    # Weight patterns by specificity
                    pattern_length = len(pattern.pattern)
                    score += min(pattern_length / 50, 2.0)  # Cap contribution

            if matches > 0:
                # Normalize by number of patterns
                intent_scores[intent] = score / len(patterns)

        # Handle special cases
        if self._is_emergency_query(query):
            return {
                "intent_type": IntentType.EMERGENCY_HELP,
                "confidence": 0.95,
                "topic": TopicCategory.HEALTH,
                "urgency": "high",
            }

        if self._is_reassurance_seeking(query):
            base_intent = (
                max(intent_scores, key=intent_scores.get)
                if intent_scores
                else IntentType.REASSURANCE_SEEKING
            )
            return {
                "intent_type": IntentType.REASSURANCE_SEEKING,
                "confidence": 0.8,
                "topic": self._determine_topic(query),
                "underlying_intent": base_intent,
            }

        # Determine best intent
        if not intent_scores:
            # Check if off-topic
            if self._is_off_topic(query):
                return {
                    "intent_type": IntentType.OFF_TOPIC,
                    "confidence": 0.7,
                    "topic": TopicCategory.GENERAL,
                    "reason": "Query outside settlement domain",
                }

            # Default to general chat for greetings or basic queries
            return {
                "intent_type": IntentType.GENERAL_CHAT,
                "confidence": 0.5,
                "topic": TopicCategory.GENERAL,
            }

        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]

        # Boost confidence for multi-pattern matches
        if confidence > 0.5:
            confidence = min(confidence * 1.2, 0.95)

        # Determine context requirements
        requires_context = best_intent in [
            IntentType.CLARIFICATION,
            IntentType.COMPARISON_QUERY,
            IntentType.REASSURANCE_SEEKING,
        ]

        return {
            "intent_type": best_intent,
            "confidence": confidence,
            "topic": self._determine_topic(query),
            "requires_context": requires_context,
            "pattern_matches": len(
                [s for s in intent_scores.values() if s > 0]
            ),
        }

    def _determine_topic(self, query: str) -> TopicCategory:
        """Determine the topic category of a query with weighted scoring."""
        topic_scores = {}
        query_words = set(re.findall(r"\b\w+\b", query.lower()))

        for topic, keywords in self.topic_keywords.items():
            # Exact matches
            exact_matches = sum(
                1 for keyword in keywords if keyword in query.lower()
            )

            # Word-level matches
            word_matches = sum(
                1 for keyword in keywords if keyword in query_words
            )

            # Combined score with weighting
            score = exact_matches * 1.5 + word_matches

            if score > 0:
                topic_scores[topic] = score

        if not topic_scores:
            return TopicCategory.GENERAL

        return max(topic_scores, key=topic_scores.get)

    def _is_emergency_query(self, query: str) -> bool:
        """Detect emergency-related queries."""
        emergency_indicators = [
            "emergency",
            "urgent",
            "crisis",
            "help me",
            "stuck",
            "lost",
            "stolen passport",
            "medical emergency",
            "accident",
            "injured",
            "can't breathe",
            "chest pain",
            "bleeding",
            "unconscious",
        ]
        return any(indicator in query for indicator in emergency_indicators)

    def _is_reassurance_seeking(self, query: str) -> bool:
        """Detect queries seeking reassurance or emotional support."""
        reassurance_patterns = [
            r"(?:am i|is it).*(?:normal|okay|fine|common)",
            r"(?:worried|scared|anxious|nervous|afraid)",
            r"(?:everyone|others|other students).*(?:experience|face|deal)",
            r"(?:feel|feeling).*(?:alone|isolated|homesick|overwhelmed)",
            r"(?:what if|worried about|concerned about)",
        ]
        return any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in reassurance_patterns
        )

    def _is_off_topic(self, query: str) -> bool:
        """Determine if query is off-topic for international student settlement."""
        off_topic_domains = [
            "weather forecast",
            "sports scores",
            "celebrity news",
            "movie reviews",
            "stock market",
            "cryptocurrency",
            "gaming",
            "recipes",
            "jokes",
            "space exploration",
            "quantum physics",
            "artificial intelligence",
            "programming",
            "software development",
            "technical support",
        ]

        # Check for explicit off-topic domains
        for domain in off_topic_domains:
            if domain in query:
                return True

        # Check if query has any settlement-related keywords
        settlement_keywords = [
            "nairobi",
            "kenya",
            "student",
            "university",
            "accommodation",
            "visa",
            "housing",
            "rent",
            "transport",
            "safety",
            "cost",
            "bank",
            "hospital",
            "culture",
            "language",
            "food",
            "shopping",
            "immigration",
            "permit",
        ]

        has_settlement_context = any(
            keyword in query for keyword in settlement_keywords
        )

        # If no settlement context and contains non-relevant topics, likely off-topic
        if not has_settlement_context and any(
            word in query
            for word in [
                "recipe",
                "cooking",
                "weather tomorrow",
                "latest news",
                "stock price",
                "bitcoin",
                "ethereum",
                "game",
                "movie",
                "song",
            ]
        ):
            return True

        return False

    def get_intent_suggestions(self, partial_query: str) -> List[str]:
        """Get suggested intents for autocomplete or guidance."""
        if len(partial_query) < 3:
            return []

        suggestions = []
        query_lower = partial_query.lower()

        # Common settlement-related suggestions
        if "hous" in query_lower:
            suggestions.extend(
                [
                    "housing near university",
                    "cheap housing in Nairobi",
                    "safe neighborhoods for students",
                    "furnished apartments",
                ]
            )
        elif "cost" in query_lower or "price" in query_lower:
            suggestions.extend(
                [
                    "cost of living in Nairobi",
                    "rent prices in Westlands",
                    "transportation costs",
                    "food expenses",
                ]
            )
        elif "safe" in query_lower:
            suggestions.extend(
                [
                    "safe areas in Nairobi",
                    "safety tips for students",
                    "avoiding crime in Kenya",
                    "emergency contacts",
                ]
            )
        elif "transport" in query_lower:
            suggestions.extend(
                [
                    "public transport in Nairobi",
                    "matatu routes",
                    "taxi apps in Kenya",
                    "transportation costs",
                ]
            )

        return suggestions[:4]  # Return top 4 suggestions
