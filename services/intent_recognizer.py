import os
import json
import time
import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intent_recognizer")


class IntentType(str, Enum):
    """Intent types for settlement assistance queries."""

    HOUSING_INQUIRY = "housing_inquiry"
    UNIVERSITY_INFO = "university_info"
    IMMIGRATION_VISA = "immigration_visa"
    TRANSPORTATION = "transportation"
    SAFETY_CONCERN = "safety_concern"
    COST_INQUIRY = "cost_inquiry"
    BANKING_FINANCE = "banking_finance"
    HEALTHCARE = "healthcare"
    CULTURAL_ADAPTATION = "cultural_adaptation"
    EMERGENCY_HELP = "emergency_help"
    PROCEDURAL_QUERY = "procedural_query"
    OFF_TOPIC = "off_topic"


class TopicType(str, Enum):
    """Topic categories for settlement content."""

    HOUSING = "housing"
    ACADEMICS = "academics"
    LEGAL = "legal"
    TRANSPORT = "transport"
    SAFETY = "safety"
    FINANCE = "finance"
    HEALTH = "health"
    CULTURE = "culture"
    EMERGENCY = "emergency"
    PROCEDURES = "procedures"
    OFF_TOPIC = "off_topic"


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent_type: IntentType
    topic: TopicType
    confidence: float
    semantic_scores: Dict[str, float]
    off_topic_indicators: List[str]
    settlement_relevance: float


class IntentRecognizer:
    """
    Semantic embedding-based intent recognizer with pure semantic classification.
    Uses OpenAI embeddings for semantic similarity matching without aggressive filtering.
    """

    def __init__(self, cache_dir: str = ".embeddings_cache"):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "intent_embeddings.npz"
        self.metadata_file = self.cache_dir / "intent_metadata.json"

        # Define intent patterns with semantic examples
        self.intent_patterns = self._initialize_intent_patterns()

        # Load or compute embeddings
        self.pattern_embeddings = self._load_or_compute_embeddings()

        # Off-topic detection thresholds
        self.off_topic_threshold = 0.40
        self.confidence_threshold = 0.75

        # Settlement relevance keywords (weighted)
        self.settlement_keywords = {
            "high_relevance": [
                "nairobi",
                "kenya",
                "international student",
                "university",
                "campus",
                "accommodation",
                "visa",
                "immigration",
                "settlement",
                "studying abroad",
            ],
            "medium_relevance": [
                "student",
                "housing",
                "rent",
                "transport",
                "safety",
                "bank",
                "hospital",
                "culture",
                "cost",
                "budget",
                "emergency",
            ],
            "location_specific": [
                "westlands",
                "kilimani",
                "karen",
                "lavington",
                "kileleshwa",
                "parklands",
                "hurlingham",
                "cbd",
                "eastleigh",
                "kasarani",
                "juja",
                "strathmore",
                "jkuat",
                "usiu",
                "kenyatta",
            ],
        }

        logger.info(
            "Semantic Intent Recognizer initialized with embedding-based classification"
        )

    def _initialize_intent_patterns(self) -> Dict[IntentType, Dict]:
        """Initialize intent patterns with semantic examples for embedding similarity."""
        return {
            IntentType.HOUSING_INQUIRY: {
                "examples": [
                    "Where can I find student accommodation in Nairobi?",
                    "I need help finding a place to stay near the university",
                    "What areas are good for international students to live?",
                    "How much does student housing cost in Westlands?",
                    "I'm looking for a shared apartment near campus",
                    "Can you recommend safe neighborhoods for students?",
                    "What should I look for when renting an apartment?",
                    "Where is the best place to live as a student in Nairobi?",
                    "Is Kilimani a good area for students to rent?",
                    "What's the average rent for a bedsitter in Karen?",
                    "Are there hostels available near University of Nairobi?",
                    "How do I find roommates to share accommodation costs?",
                    "What utilities are typically included in student housing?",
                    "Can you help me understand rental agreements in Kenya?",
                    "What's the difference between on-campus and off-campus housing?",
                    "Are furnished apartments more expensive than unfurnished ones?",
                    "How far should I live from campus for easy commuting?",
                    "What questions should I ask landlords before renting?",
                    "Is it better to live alone or share with other students?",
                    "What are the most affordable areas for student housing?",
                    "How do I avoid rental scams in Nairobi?",
                    "What deposit amount is normal for student accommodation?",
                    "Where should I look for accommodation near JKUAT?",
                    "What housing options are available near Strathmore University?",
                ],
                "keywords": [
                    "accommodation",
                    "housing",
                    "apartment",
                    "room",
                    "rent",
                    "stay",
                    "residence",
                    "neighborhood",
                    "area",
                    "live",
                    "place",
                ],
                "topic": TopicType.HOUSING,
            },
            IntentType.UNIVERSITY_INFO: {
                "examples": [
                    "Tell me about University of Nairobi admission process",
                    "What documents do I need for university registration?",
                    "How do I apply to Strathmore University?",
                    "When does the academic year start in Kenya?",
                    "What courses are available for international students?",
                    "How do I transfer credits from my home country?",
                    "What are the graduation requirements?",
                    "Where is the international student office?",
                    "Does Strathmore offer computer science programs?",
                    "What engineering courses are available at JKUAT?",
                    "Can you tell me about medical programs at University of Nairobi?",
                    "What sports teams does Strathmore have?",
                    "Are there student clubs and societies I can join?",
                    "What's the campus life like at Kenyatta University?",
                    "Does USIU have a business school?",
                    "What are the library facilities like at these universities?",
                    "How do I get my academic transcripts evaluated?",
                    "What student support services are available?",
                    "Are there research opportunities for undergraduate students?",
                    "What's the student-to-faculty ratio at these universities?",
                    "Do universities provide career counseling services?",
                    "What extracurricular activities are offered on campus?",
                    "How do I contact academic advisors at my university?",
                    "What is computer science like at Strathmore?",
                    "Does Strathmore have a football team?",
                ],
                "keywords": [
                    "university",
                    "college",
                    "admission",
                    "registration",
                    "academic",
                    "courses",
                    "campus",
                    "semester",
                    "graduation",
                    "credits",
                ],
                "topic": TopicType.ACADEMICS,
            },
            IntentType.IMMIGRATION_VISA: {
                "examples": [
                    "How do I apply for a student visa to Kenya?",
                    "What documents are required for visa application?",
                    "How long does visa processing take?",
                    "Can I extend my student visa?",
                    "Where is the immigration office in Nairobi?",
                    "What happens if my visa expires?",
                    "Do I need a work permit as a student?",
                    "How do I get a residence permit?",
                    "What's the visa fee for international students?",
                    "Can I travel outside Kenya and return on my student visa?",
                    "What are the requirements for visa renewal?",
                    "Do I need to register with immigration after arrival?",
                    "How do I prove my enrollment for visa purposes?",
                    "What if my visa application gets rejected?",
                    "Can my family visit me on a tourist visa?",
                    "What documents do I need to carry while in Kenya?",
                    "How do I change from tourist to student visa status?",
                    "Are there any visa restrictions for students?",
                    "What's the process for getting an exit permit?",
                    "How do I report a lost passport to immigration?",
                    "Can I work part-time with a student visa?",
                    "What are the consequences of overstaying my visa?",
                    "Where is Nyayo House and what services do they offer?",
                    "What are the immigration requirements for studying in Kenya?",
                ],
                "keywords": [
                    "visa",
                    "immigration",
                    "permit",
                    "passport",
                    "documents",
                    "embassy",
                    "consulate",
                    "application",
                    "renewal",
                    "extension",
                ],
                "topic": TopicType.LEGAL,
            },
            IntentType.TRANSPORTATION: {
                "examples": [
                    "How do I get around Nairobi using public transport?",
                    "What is a matatu and how do I use it?",
                    "Is Uber safe in Nairobi?",
                    "How much does transport cost in the city?",
                    "What's the best way to get to the university?",
                    "Are there student discounts for transportation?",
                    "How do I get from the airport to the city?",
                    "What time does public transport stop running?",
                    "Which matatu route goes to Westlands from the city center?",
                    "How do I use the BRT system in Nairobi?",
                    "What's the difference between matatus and buses?",
                    "Is it safe to take matatus at night?",
                    "How do I pay for transport using M-Pesa?",
                    "What are the main matatu termini in Nairobi?",
                    "Can I get a monthly transport pass as a student?",
                    "How do I get to JKUAT from the city center?",
                    "What's the cheapest way to travel around Nairobi?",
                    "Are there shuttle services to universities?",
                    "How do I book a taxi using mobile apps?",
                    "What should I know about boda boda safety?",
                    "Which areas are well connected by public transport?",
                    "How reliable is public transportation in Nairobi?",
                    "What's the best route from Westlands to University of Nairobi?",
                    "How do I get from Kilimani to Strathmore University?",
                ],
                "keywords": [
                    "transport",
                    "matatu",
                    "bus",
                    "uber",
                    "taxi",
                    "travel",
                    "commute",
                    "route",
                    "fare",
                    "airport",
                ],
                "topic": TopicType.TRANSPORT,
            },
            IntentType.SAFETY_CONCERN: {
                "examples": [
                    "Is Nairobi safe for international students?",
                    "What areas should I avoid in the city?",
                    "How do I stay safe while walking at night?",
                    "What should I do if I feel unsafe?",
                    "Are there any safety precautions I should take?",
                    "What neighborhoods have high crime rates?",
                    "How do I report a crime in Kenya?",
                    "Is it safe to use public transport at night?",
                    "What safety measures should I take in my accommodation?",
                    "How do I protect my belongings from theft?",
                    "Is it safe to walk alone around university campuses?",
                    "What should I do if someone approaches me suspiciously?",
                    "Are there areas where students shouldn't go?",
                    "How can I stay safe when using ATMs?",
                    "What are common scams targeting international students?",
                    "Is it safe to carry cash or should I use cards?",
                    "How do I know if a taxi or matatu is safe?",
                    "What emergency contacts should I have saved?",
                    "Are there safe places to study late at night?",
                    "How do I stay safe during student social events?",
                    "What precautions should I take when meeting people online?",
                    "How do I ensure my accommodation is secure?",
                    "Is Westlands safe for students to walk around?",
                    "What should I know about safety in Kilimani?",
                ],
                "keywords": [
                    "safe",
                    "safety",
                    "security",
                    "crime",
                    "dangerous",
                    "avoid",
                    "precautions",
                    "police",
                    "emergency",
                    "risk",
                ],
                "topic": TopicType.SAFETY,
            },
            IntentType.COST_INQUIRY: {
                "examples": [
                    "How much money do I need per month in Nairobi?",
                    "What's the cost of living for students?",
                    "How much should I budget for food and transport?",
                    "What are typical prices for student accommodation?",
                    "How much does internet cost in Kenya?",
                    "What's the price of utilities like electricity and water?",
                    "How much spending money should I bring?",
                    "What currency is used in Kenya and what's the exchange rate?",
                    "How much do groceries cost for a student?",
                    "What's the price range for meals at restaurants?",
                    "How much should I budget for textbooks and supplies?",
                    "What are the costs associated with university registration?",
                    "How much does mobile phone service cost?",
                    "What's the cost of laundry services?",
                    "How much should I expect to spend on entertainment?",
                    "What are the prices for gym memberships?",
                    "How much does health insurance cost for students?",
                    "What's the cost of traveling within Kenya during holidays?",
                    "How much do personal care items cost?",
                    "What's the price difference between shopping at markets vs supermarkets?",
                    "How much should I budget for clothing in Nairobi?",
                    "What are typical utility deposit amounts for apartments?",
                    "How much does it cost to live in Westlands?",
                    "What's the average cost of student meals at university?",
                ],
                "keywords": [
                    "cost",
                    "price",
                    "budget",
                    "money",
                    "expensive",
                    "cheap",
                    "afford",
                    "currency",
                    "exchange",
                    "ksh",
                    "shilling",
                ],
                "topic": TopicType.FINANCE,
            },
            IntentType.BANKING_FINANCE: {
                "examples": [
                    "How do I open a bank account in Kenya?",
                    "What is M-Pesa and how do I use it?",
                    "Can I use my international bank card in Nairobi?",
                    "What documents do I need to open an account?",
                    "How do I transfer money from home?",
                    "What are the banking fees for international students?",
                    "Which banks are best for students?",
                    "How do I pay school fees from my bank account?",
                    "What's the difference between different mobile money services?",
                    "How do I withdraw money from ATMs safely?",
                    "Can I receive money transfers from abroad?",
                    "What are the limits for M-Pesa transactions?",
                    "How do I set up automatic bill payments?",
                    "What should I do if my bank card gets stolen?",
                    "How do I check my account balance using mobile banking?",
                    "What are the requirements for student bank accounts?",
                    "How do I convert foreign currency to Kenyan shillings?",
                    "Can I get a loan as an international student?",
                    "What credit cards are available for students?",
                    "How do I send money back to my home country?",
                    "What are the banking hours in Kenya?",
                    "How do I report fraudulent transactions?",
                    "Which banks have branches near universities?",
                    "How do I pay for accommodation using mobile money?",
                ],
                "keywords": [
                    "bank",
                    "account",
                    "mpesa",
                    "money",
                    "transfer",
                    "payment",
                    "card",
                    "atm",
                    "fees",
                    "finance",
                ],
                "topic": TopicType.FINANCE,
            },
            IntentType.HEALTHCARE: {
                "examples": [
                    "Where can I find medical care in Nairobi?",
                    "Do I need health insurance as an international student?",
                    "Which hospitals accept international insurance?",
                    "How do I find a doctor who speaks English?",
                    "What vaccinations do I need for Kenya?",
                    "Where can I buy prescription medication?",
                    "How much does medical care cost?",
                    "What should I do in a medical emergency?",
                    "Are there clinics near the universities?",
                    "How do I register with a local doctor?",
                    "What health services are available on campus?",
                    "Where can I get dental care in Nairobi?",
                    "How do I refill my prescription medications?",
                    "What mental health services are available for students?",
                    "Are there 24-hour pharmacies in Nairobi?",
                    "How do I get medical certificates for university?",
                    "What should I do if I get sick and don't have insurance?",
                    "Where are the best hospitals for international patients?",
                    "How do I access contraception and reproductive health services?",
                    "What over-the-counter medications are available?",
                    "How do I find specialists like eye doctors or dermatologists?",
                    "What health screening do I need before starting university?",
                    "Are there student health centers at universities?",
                    "How do I handle medical emergencies as a foreign student?",
                ],
                "keywords": [
                    "health",
                    "medical",
                    "hospital",
                    "doctor",
                    "insurance",
                    "medicine",
                    "clinic",
                    "vaccination",
                    "emergency",
                ],
                "topic": TopicType.HEALTH,
            },
            IntentType.CULTURAL_ADAPTATION: {
                "examples": [
                    "What should I know about Kenyan culture?",
                    "How do I adapt to life in Nairobi?",
                    "What are important customs I should respect?",
                    "How do I make friends with local students?",
                    "What food is popular in Kenya?",
                    "What languages are spoken in Nairobi?",
                    "How should I dress appropriately?",
                    "What social norms should I be aware of?",
                    "How do I greet people in Kenya?",
                    "What are the dining etiquette rules?",
                    "How do I participate in local festivals and celebrations?",
                    "What should I know about religious practices in Kenya?",
                    "How do I show respect to elders and authority figures?",
                    "What are common conversation topics to avoid?",
                    "How do I understand Kenyan humor and social interactions?",
                    "What should I know about dating culture in Kenya?",
                    "How do I handle cultural misunderstandings?",
                    "What gifts are appropriate to give to Kenyan friends?",
                    "How do I learn basic Swahili phrases?",
                    "What cultural events should I attend as a student?",
                    "How do I respect local traditions while maintaining my own identity?",
                    "What should I know about business culture for internships?",
                    "How do I navigate social hierarchies in academic settings?",
                    "What are the most important cultural differences I should understand?",
                ],
                "keywords": [
                    "culture",
                    "customs",
                    "tradition",
                    "language",
                    "food",
                    "friends",
                    "social",
                    "respect",
                    "adapt",
                    "lifestyle",
                ],
                "topic": TopicType.CULTURE,
            },
            IntentType.EMERGENCY_HELP: {
                "examples": [
                    "What are the emergency numbers in Kenya?",
                    "I need urgent help, who should I call?",
                    "What do I do in case of fire or accident?",
                    "How do I contact police in Nairobi?",
                    "I lost my passport, what should I do?",
                    "There's an emergency at my accommodation",
                    "I need immediate medical attention",
                    "How do I reach my embassy in an emergency?",
                    "What should I do if I'm robbed or attacked?",
                    "How do I report a missing person?",
                    "What's the procedure for natural disasters?",
                    "I've been arrested, what are my rights?",
                    "How do I get emergency financial assistance?",
                    "What should I do if my bank cards are stolen?",
                    "I'm stranded and need immediate help",
                    "How do I contact my university in an emergency?",
                    "What should I do if I witness a crime?",
                    "How do I get emergency evacuation assistance?",
                    "I need urgent legal help, who can I contact?",
                    "What should I do if I'm threatened or harassed?",
                    "How do I get emergency accommodation?",
                    "What are the numbers for poison control or crisis hotlines?",
                    "I'm having a mental health crisis, where can I get help?",
                    "How do I contact emergency services if I don't speak the language?",
                ],
                "keywords": [
                    "emergency",
                    "urgent",
                    "help",
                    "police",
                    "fire",
                    "ambulance",
                    "accident",
                    "lost",
                    "stolen",
                    "immediate",
                ],
                "topic": TopicType.EMERGENCY,
            },
            IntentType.PROCEDURAL_QUERY: {
                "examples": [
                    "What steps do I need to take to register for university?",
                    "How do I complete the enrollment process?",
                    "What's the procedure for getting a student ID?",
                    "How do I apply for accommodation on campus?",
                    "What's the process for changing courses?",
                    "How do I submit my academic transcripts?",
                    "What are the steps to withdraw from university?",
                    "How do I request official documents?",
                    "What's the process for registering for classes?",
                    "How do I apply for scholarships or financial aid?",
                    "What steps are needed to get a library card?",
                    "How do I register for student services?",
                    "What's the procedure for appealing academic decisions?",
                    "How do I apply for internship programs?",
                    "What are the steps to transfer between universities?",
                    "How do I register for graduation ceremonies?",
                    "What's the process for getting academic references?",
                    "How do I apply for student exchange programs?",
                    "What steps are needed to defer my studies?",
                    "How do I register as a continuing student?",
                    "What's the procedure for academic leave of absence?",
                    "How do I apply for thesis supervision?",
                    "What are the steps to register for extracurricular activities?",
                    "How do I complete the admission requirements checklist?",
                ],
                "keywords": [
                    "process",
                    "procedure",
                    "steps",
                    "how",
                    "apply",
                    "register",
                    "submit",
                    "complete",
                    "requirements",
                    "documents",
                ],
                "topic": TopicType.PROCEDURES,
            },
        }

    def _load_or_compute_embeddings(self) -> Dict[IntentType, np.ndarray]:
        """Load embeddings from cache or compute them if not available."""
        if self._cache_is_valid():
            logger.info("Loading embeddings from cache")
            return self._load_embeddings_from_cache()
        else:
            logger.info("Computing embeddings (cache not found or invalid)")
            embeddings = self._compute_pattern_embeddings()
            self._save_embeddings_to_cache(embeddings)
            return embeddings

    def _cache_is_valid(self) -> bool:
        """Check if the cache exists and is valid."""
        if not self.cache_file.exists() or not self.metadata_file.exists():
            return False

        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if all intent types are present
            cached_intents = set(metadata.get('intent_types', []))
            current_intents = set(intent.value for intent in IntentType if intent != IntentType.OFF_TOPIC)
            
            return cached_intents == current_intents
        except Exception as e:
            logger.warning(f"Cache validation failed: {str(e)}")
            return False

    def _load_embeddings_from_cache(self) -> Dict[IntentType, np.ndarray]:
        """Load embeddings from cache file."""
        try:
            data = np.load(self.cache_file, allow_pickle=True)
            embeddings = {}
            
            for intent_type in IntentType:
                if intent_type != IntentType.OFF_TOPIC:
                    key = intent_type.value
                    if key in data:
                        embeddings[intent_type] = data[key]
                    else:
                        logger.warning(f"Missing embedding for {intent_type} in cache")
                        embeddings[intent_type] = np.zeros(1536)
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings from cache: {str(e)}")
            return self._compute_pattern_embeddings()

    def _save_embeddings_to_cache(self, embeddings: Dict[IntentType, np.ndarray]):
        """Save embeddings to cache file."""
        try:
            # Prepare data for numpy savez
            save_dict = {intent.value: embedding for intent, embedding in embeddings.items()}
            np.savez(self.cache_file, **save_dict)
            
            # Save metadata
            metadata = {
                'intent_types': [intent.value for intent in embeddings.keys()],
                'embedding_model': 'text-embedding-ada-002',
                'embedding_dim': 1536
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Embeddings cached to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save embeddings to cache: {str(e)}")

    def _compute_pattern_embeddings(self) -> Dict[IntentType, np.ndarray]:
        """Compute embeddings for all intent pattern examples."""
        pattern_embeddings = {}

        for intent_type, pattern_data in self.intent_patterns.items():
            examples = pattern_data["examples"]

            try:
                # Get embeddings for all examples
                response = self.openai_client.embeddings.create(
                    input=examples, model="text-embedding-ada-002"
                )

                embeddings = [item.embedding for item in response.data]
                # Average the embeddings to create a representative vector
                pattern_embeddings[intent_type] = np.mean(embeddings, axis=0)

                logger.info(
                    f"Computed embeddings for {intent_type} with {len(examples)} examples"
                )

            except Exception as e:
                logger.error(
                    f"Failed to compute embeddings for {intent_type}: {str(e)}"
                )
                # Create zero vector as fallback
                pattern_embeddings[intent_type] = np.zeros(1536)

        return pattern_embeddings

    def classify_intent(self, query: str) -> IntentResult:
        """
        Classify intent using pure semantic similarity - no aggressive filtering.

        Args:
            query: User query to classify

        Returns:
            IntentResult with classification details
        """
        try:
            # Step 1: Get query embedding
            query_embedding = self._get_query_embedding(query)
            if query_embedding is None:
                return self._fallback_classification(query)

            # Step 2: Calculate semantic similarities
            similarities = self._calculate_similarities(query_embedding)

            # Step 3: Calculate settlement relevance
            settlement_relevance = self._calculate_settlement_relevance(query)

            # Step 4: Apply semantic threshold for off-topic detection
            max_similarity = max(similarities.values()) if similarities else 0.0

            # Much more permissive - only reject if clearly no semantic match
            if max_similarity < self.off_topic_threshold:
                return IntentResult(
                    intent_type=IntentType.OFF_TOPIC,
                    topic=TopicType.OFF_TOPIC,
                    confidence=0.85,
                    semantic_scores=similarities,
                    off_topic_indicators=[
                        f"Low semantic similarity: {max_similarity:.3f}"
                    ],
                    settlement_relevance=settlement_relevance,
                )

            # Step 5: Get best intent match
            best_intent = max(similarities, key=similarities.get)
            best_confidence = similarities[best_intent]

            # Step 6: Boost confidence if settlement-relevant
            if settlement_relevance > 0.5:
                best_confidence = min(best_confidence + 0.1, 1.0)

            # Step 7: Get topic from intent mapping
            topic = self.intent_patterns[best_intent]["topic"]

            return IntentResult(
                intent_type=best_intent,
                topic=topic,
                confidence=best_confidence,
                semantic_scores=similarities,
                off_topic_indicators=[],
                settlement_relevance=settlement_relevance,
            )

        except Exception as e:
            logger.error(f"Intent classification failed: {str(e)}")
            return self._fallback_classification(query)

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for the query."""
        try:
            start_time = time.perf_counter()
            response = self.openai_client.embeddings.create(
                input=[query], model="text-embedding-ada-002"
            )
            end_time = time.perf_counter()
            print(f"Elapsed time is { end_time - start_time } seconds")
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Failed to get query embedding: {str(e)}")
            return None

    def _calculate_similarities(
        self, query_embedding: np.ndarray
    ) -> Dict[IntentType, float]:
        """Calculate cosine similarities between query and all intent patterns."""
        similarities = {}

        for intent_type, pattern_embedding in self.pattern_embeddings.items():
            if (
                intent_type != IntentType.OFF_TOPIC
            ):  # Skip off-topic for similarity calculation
                try:
                    similarity = cosine_similarity(
                        [query_embedding], [pattern_embedding]
                    )[0][0]
                    similarities[intent_type] = max(
                        0.0, similarity
                    )  # Ensure non-negative
                except Exception as e:
                    logger.warning(
                        f"Similarity calculation failed for {intent_type}: {str(e)}"
                    )
                    similarities[intent_type] = 0.0

        return similarities

    def _calculate_settlement_relevance(self, query: str) -> float:
        """Calculate how relevant the query is to settlement topics."""
        query_lower = query.lower()
        relevance_score = 0.0

        # High relevance keywords (strong indicators)
        for keyword in self.settlement_keywords["high_relevance"]:
            if keyword in query_lower:
                relevance_score += 0.3

        # Medium relevance keywords
        for keyword in self.settlement_keywords["medium_relevance"]:
            if keyword in query_lower:
                relevance_score += 0.2

        # Location-specific keywords
        for keyword in self.settlement_keywords["location_specific"]:
            if keyword in query_lower:
                relevance_score += 0.25

        # International student indicators
        student_indicators = [
            "international",
            "student",
            "studying",
            "study abroad",
            "university",
        ]
        for indicator in student_indicators:
            if indicator in query_lower:
                relevance_score += 0.15

        # Nairobi/Kenya specific boost
        if any(
            location in query_lower
            for location in ["nairobi", "kenya", "kenyan"]
        ):
            relevance_score += 0.2

        return min(relevance_score, 1.0)  # Cap at 1.0

    def _fallback_classification(self, query: str) -> IntentResult:
        """Fallback classification when semantic analysis fails."""
        # Simple keyword-based fallback
        query_lower = query.lower()

        if any(
            word in query_lower
            for word in ["house", "room", "accommodation", "live"]
        ):
            return IntentResult(
                intent_type=IntentType.HOUSING_INQUIRY,
                topic=TopicType.HOUSING,
                confidence=0.6,
                semantic_scores={},
                off_topic_indicators=[],
                settlement_relevance=0.5,
            )
        elif any(
            word in query_lower
            for word in ["university", "college", "academic"]
        ):
            return IntentResult(
                intent_type=IntentType.UNIVERSITY_INFO,
                topic=TopicType.ACADEMICS,
                confidence=0.6,
                semantic_scores={},
                off_topic_indicators=[],
                settlement_relevance=0.5,
            )
        else:
            return IntentResult(
                intent_type=IntentType.OFF_TOPIC,
                topic=TopicType.OFF_TOPIC,
                confidence=0.8,
                semantic_scores={},
                off_topic_indicators=["Semantic analysis failed"],
                settlement_relevance=0.0,
            )

    def get_intent_info(self, query: str) -> Dict[str, Any]:
        """
        Main method for compatibility with existing code.

        Args:
            query: User query to analyze

        Returns:
            Dictionary with intent information
        """
        result = self.classify_intent(query)

        return {
            "intent_type": result.intent_type,
            "topic": result.topic,
            "confidence": result.confidence,
            "settlement_relevance": result.settlement_relevance,
            "semantic_scores": result.semantic_scores,
            "off_topic_indicators": result.off_topic_indicators,
            "classification_method": "semantic_embedding",
            "is_off_topic": result.intent_type == IntentType.OFF_TOPIC,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the intent recognizer."""
        return {
            "total_intents": len(self.intent_patterns),
            "classification_method": "semantic_embedding",
            "off_topic_threshold": self.off_topic_threshold,
            "confidence_threshold": self.confidence_threshold,
            "embedding_model": "text-embedding-ada-002",
            "supported_intents": [intent.value for intent in IntentType],
            "supported_topics": [topic.value for topic in TopicType],
            "cache_enabled": True,
            "cache_location": str(self.cache_file),
            "features": [
                "Semantic similarity matching",
                "Pure semantic classification",
                "Settlement relevance scoring",
                "Training examples",
                "Confidence calibration",
                "Fallback classification",
                "Embedding caching",
            ],
        }

    def validate_patterns(self) -> Dict[str, Any]:
        """Validate that all intent patterns have proper embeddings."""
        validation_results = {
            "valid_patterns": 0,
            "invalid_patterns": 0,
            "pattern_details": {},
        }

        for intent_type, embedding in self.pattern_embeddings.items():
            if embedding is not None and embedding.shape[0] > 0:
                validation_results["valid_patterns"] += 1
                validation_results["pattern_details"][intent_type.value] = {
                    "status": "valid",
                    "embedding_shape": embedding.shape,
                    "example_count": len(
                        self.intent_patterns[intent_type]["examples"]
                    ),
                }
            else:
                validation_results["invalid_patterns"] += 1
                validation_results["pattern_details"][intent_type.value] = {
                    "status": "invalid",
                    "error": "Missing or empty embedding",
                }

        validation_results["overall_health"] = (
            validation_results["invalid_patterns"] == 0
        )
        return validation_results

    def clear_cache(self):
        """Clear the embeddings cache."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")

    def rebuild_cache(self):
        """Force rebuild of the embeddings cache."""
        self.clear_cache()
        self.pattern_embeddings = self._compute_pattern_embeddings()
        self._save_embeddings_to_cache(self.pattern_embeddings)
        logger.info("Cache rebuilt successfully")
