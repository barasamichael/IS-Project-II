import os
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from datetime import datetime

import pytz
from openai import OpenAI

from config.settings import settings
from services.intent_recognizer import IntentType
from services.language_processor import LanguageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_generator")


class ResponseGenerator:
    """Production-grade response generator optimized for settlement queries."""

    def __init__(self):
        self.model = settings.llm.model
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize language processor
        self.language_processor = LanguageProcessor()

    def get_current_nairobi_time(self):
        """Get current date and time in Nairobi (EAT - UTC+3)."""
        nairobi_tz = pytz.timezone("Africa/Nairobi")
        current_time = nairobi_tz.localize(datetime.now())

        formatted_time = current_time.strftime("%A, %B %d, %Y %H:%M")
        time_only = current_time.strftime("%H:%M")

        return formatted_time, time_only

    def generate_response(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        intent_info: Dict[str, Any],
        conversation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate optimized response for settlement queries with multilingual support.

        Args:
            query: User query (may be in multiple languages)
            retrieved_context: Retrieved chunks from vector database
            intent_info: Detected intent information
            conversation_context: Optional conversation context

        Returns:
            Response with language detection and translation
        """
        try:
            # Process language detection and translation
            language_result = self.language_processor.detect_and_process_query(
                query
            )

            # Use English query for RAG processing
            english_query = language_result["english_query"]
            original_language = language_result["detected_language"]
            needs_translation = language_result["needs_translation"]

            # Get current Nairobi time
            current_time_full, current_time = self.get_current_nairobi_time()

            # Generate system instruction based on intent and settlement domain
            system_instruction = self._get_settlement_system_instruction(
                intent_info["intent_type"]
            )

            # Format retrieved context with settlement optimization
            context_text = self._format_settlement_context(
                retrieved_context, intent_info
            )

            # Get response guidelines
            response_guidelines = self._get_settlement_response_guidelines(
                intent_info["intent_type"]
            )

            # Build prompt with settlement context
            messages = [
                {
                    "role": "system",
                    "content": system_instruction
                    + self._get_settlement_context_prompt(),
                },
                {
                    "role": "user",
                    "content": f"""
[CURRENT TIME - NAIROBI, KENYA]
{current_time_full}

[SETTLEMENT CONTEXT]
{context_text}

[USER INTENT]
The user's question relates to {intent_info["intent_type"].value} about {intent_info["topic"].value}.
Query confidence: {intent_info["confidence"]:.2f}

[QUERY]
{english_query}

[RESPONSE GUIDELINES]
{response_guidelines}
""",
                },
            ]

            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            response_content = response.choices[0].message.content

            # Translate response back if needed
            if needs_translation and original_language != "english":
                translated_response = (
                    self.language_processor.translate_response(
                        response_content, original_language
                    )
                )
                # Validate translation quality
                translation_quality = (
                    self.language_processor.validate_translation_quality(
                        response_content, translated_response, original_language
                    )
                )
            else:
                translated_response = response_content
                translation_quality = None

            # Calculate token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            logger.info(
                f"Generated settlement response with {token_usage['total_tokens']} tokens"
            )

            return {
                "response": translated_response,
                "original_response": response_content
                if needs_translation
                else None,
                "intent_type": intent_info["intent_type"],
                "topic": intent_info["topic"],
                "confidence": intent_info["confidence"],
                "language_detected": original_language,
                "translation_needed": needs_translation,
                "translation_quality": translation_quality,
                "token_usage": token_usage,
                "current_time": current_time_full,
                "settlement_optimized": True,
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

            # Return error response in appropriate language
            error_message = "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists."

            if needs_translation and original_language != "english":
                error_message = self.language_processor.translate_response(
                    error_message, original_language
                )

            return {
                "response": error_message,
                "intent_type": intent_info["intent_type"],
                "topic": intent_info["topic"],
                "confidence": intent_info["confidence"],
                "error": str(e),
                "language_detected": original_language,
                "settlement_optimized": True,
            }

    def _get_settlement_system_instruction(
        self, intent_type: IntentType
    ) -> str:
        """Get system instruction optimized for settlement queries."""
        base_instruction = """
You are SettleBot, an expert assistant helping international students navigate settlement in Nairobi, Kenya.
Your expertise covers housing, transportation, education, legal matters, finance, safety, healthcare, and cultural adaptation.

Core Principles:
- Provide ACCURATE, PRACTICAL information specific to Nairobi and Kenya
- Focus on international student needs and perspectives
- Include specific locations, costs, and actionable guidance when available
- Be culturally sensitive and supportive
- Acknowledge the challenges of relocating to a new country
- Provide realistic expectations while being encouraging

ALWAYS ground your responses in the provided context about Nairobi settlement.
When information is missing, clearly state what you don't know rather than making assumptions.
Include specific neighborhood names, transportation options, and cost estimates when available in the context.
"""

        intent_specific_instructions = {
            IntentType.HOUSING_INQUIRY: """
HOUSING FOCUS: Provide specific information about:
- Neighborhood recommendations (Westlands, Kilimani, etc.) with safety and proximity to universities
- Typical rental costs in KSh with deposit requirements
- What to expect in terms of utilities, furniture, and amenities
- Red flags to avoid when renting
- Landlord expectations and tenant rights
Format housing costs clearly and mention any additional fees.
""",
            IntentType.UNIVERSITY_INFO: """
EDUCATION FOCUS: Provide practical guidance about:
- University locations and campus facilities
- Registration processes and international student services
- Academic systems and grading differences
- Student accommodation options
- Campus safety and student support services
Be specific about procedures and timelines when available.
""",
            IntentType.NEIGHBORHOOD_GUIDE: """
LOCATION FOCUS: Provide detailed area information:
- Safety levels at different times of day
- Proximity to universities, markets, and amenities
- Transportation connections (matatu routes, taxi availability)
- Cost of living variations by area
- Expat/international student presence
Include specific street names or landmarks when available in context.
""",
            IntentType.COST_INQUIRY: """
FINANCIAL FOCUS: Provide accurate cost information:
- Always mention currency (KSh - Kenyan Shilling)
- Include ranges rather than exact figures when appropriate
- Mention factors that affect pricing
- Compare costs across different areas/options when relevant
- Include tips for budgeting and saving money
Be transparent about cost variations and additional fees.
""",
            IntentType.SAFETY_CONCERN: """
SAFETY FOCUS: Provide balanced, practical safety guidance:
- Specific areas to be cautious about and safe areas
- Time-of-day considerations
- Common scams targeting international students
- Emergency contacts and procedures
- Cultural norms that affect safety
Balance honesty about risks with reassurance and practical solutions.
""",
            IntentType.TRANSPORTATION: """
TRANSPORT FOCUS: Provide practical mobility information:
- Matatu routes and how the system works
- Taxi/ride-hailing apps (Uber, Bolt, Little Cab)
- Boda boda (motorcycle taxi) safety and usage
- Walking safety and pedestrian infrastructure
- Cost comparisons and payment methods
Include specific route information when available.
""",
            IntentType.BANKING_FINANCE: """
BANKING FOCUS: Provide practical financial guidance:
- Bank account opening requirements for international students
- M-Pesa mobile money system and how to use it
- ATM locations and fees
- Currency exchange options and rates
- International money transfer options
Be specific about documentation requirements and processes.
""",
            IntentType.IMMIGRATION_VISA: """
LEGAL FOCUS: Provide accurate immigration information:
- Student visa requirements and renewal processes
- Immigration office locations and procedures
- Required documentation and timelines
- Compliance requirements and consequences of violations
- Embassy contact information when relevant
Emphasize the importance of staying compliant with regulations.
""",
            IntentType.HEALTHCARE: """
HEALTH FOCUS: Provide practical healthcare guidance:
- Hospital and clinic recommendations near universities
- Health insurance options and requirements
- Pharmacy locations and prescription procedures
- Emergency medical services and contacts
- Preventive healthcare and vaccinations
Include specific facility names and locations when available.
""",
            IntentType.CULTURAL_ADAPTATION: """
CULTURE FOCUS: Provide sensitive cultural guidance:
- Social norms and etiquette expectations
- Language considerations (English, Swahili basics)
- Religious and cultural diversity in Nairobi
- Food culture and dietary accommodation
- Social integration opportunities
Be respectful of both international student backgrounds and Kenyan culture.
""",
            IntentType.EMERGENCY_HELP: """
EMERGENCY FOCUS: Provide immediate, actionable guidance:
- Emergency contact numbers (police: 999, medical: 999)
- Hospital locations for urgent care
- Embassy contact information for passport issues
- Steps to take for specific emergency situations
- Resources for crisis support
Prioritize immediate safety and next steps.
""",
            IntentType.REASSURANCE_SEEKING: """
SUPPORT FOCUS: Provide empathetic, encouraging responses:
- Acknowledge the challenges of international relocation
- Share that adaptation takes time and is normal
- Provide confidence-building information
- Suggest support networks and resources
- Offer practical steps to address concerns
Balance honesty about challenges with optimism and support.
""",
        }

        return base_instruction + intent_specific_instructions.get(
            intent_type, ""
        )

    def _get_settlement_context_prompt(self) -> str:
        """Get settlement-specific context prompt."""
        return """

SETTLEMENT CONTEXT AWARENESS:
- All responses should assume the user is an international student in or planning to move to Nairobi, Kenya
- Prioritize information most relevant to newcomers and international students
- When mentioning costs, always use Kenyan Shilling (KSh) 
- Include location-specific details when available (neighborhoods, landmarks, specific institutions)
- Consider visa requirements and legal status implications when relevant
- Be aware of cultural differences and adaptation challenges
- Provide practical, actionable advice suitable for someone unfamiliar with the local context

CURRENT NAIROBI CONTEXT:
- Nairobi is a large, diverse city with varying safety and cost levels by area
- Major student-friendly areas include Westlands, Kilimani, Lavington, Karen
- Public transport is mainly matatus (shared minibuses) and boda bodas (motorcycle taxis)
- M-Pesa mobile money is widely used for transactions
- English and Swahili are official languages; English is commonly used in universities
- Traffic can be heavy; plan transportation accordingly
"""

    def _format_settlement_context(
        self,
        retrieved_context: List[Dict[str, Any]],
        intent_info: Dict[str, Any],
    ) -> str:
        """Format context with settlement-specific optimization."""
        if not retrieved_context:
            return "No specific settlement information found for this query."

        # Sort by settlement relevance and recency
        sorted_context = sorted(
            retrieved_context,
            key=lambda x: (x.get("settlement_score", 0), x.get("score", 0)),
            reverse=True,
        )

        formatted_chunks = []
        for i, chunk in enumerate(sorted_context[:8]):  # Limit to top 8 chunks
            # Extract settlement metadata
            settlement_info = []
            if "location_entities" in chunk:
                locations = chunk["location_entities"]
                if locations:
                    settlement_info.append(f"Locations: {', '.join(locations)}")

            if "cost_entities" in chunk:
                costs = chunk["cost_entities"]
                if costs:
                    settlement_info.append(f"Costs: {', '.join(costs)}")

            if "topic_tags" in chunk:
                topics = chunk["topic_tags"]
                if topics:
                    settlement_info.append(f"Topics: {', '.join(topics)}")

            # Format chunk with metadata
            metadata_str = (
                f" [{', '.join(settlement_info)}]" if settlement_info else ""
            )

            formatted_chunk = f"[CHUNK {i+1}] (Relevance: {chunk.get('score', 0):.2f}, Settlement: {chunk.get('settlement_score', 0):.2f}){metadata_str}\n{chunk['text']}\n"
            formatted_chunks.append(formatted_chunk)

        return "\n".join(formatted_chunks)

    def _get_settlement_response_guidelines(
        self, intent_type: IntentType
    ) -> str:
        """Get response guidelines optimized for settlement domain."""
        common_guidelines = """
SETTLEMENT-SPECIFIC RESPONSE GUIDELINES:
1. Provide PRACTICAL, ACTIONABLE information relevant to international students in Nairobi
2. Include specific costs in KSh when available in context
3. Mention specific neighborhoods, institutions, or landmarks when relevant
4. Consider the international student perspective and adaptation challenges
5. Be culturally sensitive while providing honest, realistic guidance
6. Include safety considerations when relevant to the topic
7. Suggest next steps or resources when appropriate
8. Use clear, supportive language appropriate for someone navigating a new country
9. If costs are mentioned, indicate they may vary and suggest verification
10. Acknowledge when information might be outdated and suggest checking current details
"""

        intent_specific_guidelines = {
            IntentType.HOUSING_INQUIRY: """
11. Organize housing information by: location → cost → amenities → safety
12. Always mention deposit requirements and typical lease terms
13. Include utility cost expectations and what's typically included
14. Warn about common rental scams or red flags
15. Suggest viewing properties in person and in daylight
""",
            IntentType.TRANSPORTATION: """
11. Explain how matatu system works for newcomers
12. Include approximate travel times between key locations
13. Mention payment methods (cash, cards, mobile money)
14. Provide safety tips specific to each transport mode
15. Include backup transportation options
""",
            IntentType.SAFETY_CONCERN: """
11. Balance honesty about risks with practical solutions
12. Provide specific prevention strategies, not just warnings
13. Include emergency contact information when relevant
14. Mention community resources and support networks
15. Address both day and night safety considerations
""",
            IntentType.COST_INQUIRY: """
11. Present cost ranges rather than exact figures when appropriate
12. Explain factors that influence pricing
13. Include money-saving tips and budget-friendly alternatives
14. Mention seasonal price variations if relevant
15. Suggest ways to verify current pricing
""",
            IntentType.EMERGENCY_HELP: """
11. Prioritize immediate safety and action steps
12. Provide specific contact numbers and addresses
13. Include embassy information for serious legal/passport issues
14. Mention hospital locations for medical emergencies
15. Suggest follow-up resources and support
""",
        }

        return common_guidelines + intent_specific_guidelines.get(
            intent_type, ""
        )

    def validate_response_quality(
        self, response: str, intent_type: IntentType
    ) -> Dict[str, Any]:
        """Validate response quality for settlement content."""
        try:
            quality_metrics = {
                "has_practical_info": False,
                "mentions_costs": False,
                "includes_locations": False,
                "culturally_appropriate": True,
                "safety_aware": False,
                "actionable": False,
            }

            response_lower = response.lower()

            # Check for practical information
            practical_indicators = [
                "steps",
                "process",
                "procedure",
                "contact",
                "address",
                "phone",
                "requirements",
                "documents",
                "timeline",
                "hours",
                "location",
            ]
            quality_metrics["has_practical_info"] = any(
                indicator in response_lower
                for indicator in practical_indicators
            )

            # Check for cost information
            cost_indicators = [
                "ksh",
                "shilling",
                "cost",
                "price",
                "fee",
                "budget",
                "expensive",
                "cheap",
            ]
            quality_metrics["mentions_costs"] = any(
                indicator in response_lower for indicator in cost_indicators
            )

            # Check for location specificity
            location_indicators = [
                "westlands",
                "kilimani",
                "karen",
                "lavington",
                "nairobi",
                "cbd",
                "university",
                "campus",
                "neighborhood",
                "area",
                "street",
            ]
            quality_metrics["includes_locations"] = any(
                indicator in response_lower for indicator in location_indicators
            )

            # Check for safety awareness (especially for relevant intents)
            safety_indicators = [
                "safe",
                "safety",
                "secure",
                "avoid",
                "careful",
                "caution",
                "emergency",
            ]
            quality_metrics["safety_aware"] = any(
                indicator in response_lower for indicator in safety_indicators
            )

            # Check for actionable advice
            action_indicators = [
                "should",
                "can",
                "need to",
                "recommend",
                "suggest",
                "try",
                "contact",
                "visit",
                "check",
                "apply",
                "register",
                "download",
            ]
            quality_metrics["actionable"] = any(
                indicator in response_lower for indicator in action_indicators
            )

            # Calculate overall quality score
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)

            return {
                "quality_score": quality_score,
                "metrics": quality_metrics,
                "intent_appropriate": quality_score > 0.6,
                "settlement_optimized": quality_metrics["has_practical_info"]
                and quality_metrics["actionable"],
            }

        except Exception as e:
            logger.error(f"Response quality validation failed: {str(e)}")
            return {
                "quality_score": 0.5,
                "error": str(e),
                "intent_appropriate": False,
            }

    def generate_follow_up_questions(
        self, intent_type: IntentType, response: str
    ) -> List[str]:
        """Generate relevant follow-up questions for settlement topics."""
        try:
            follow_ups = {
                IntentType.HOUSING_INQUIRY: [
                    "Would you like information about specific neighborhoods?",
                    "Do you need guidance on rental agreements and deposits?",
                    "Are you interested in furnished or unfurnished options?",
                    "Would you like safety information for different areas?",
                ],
                IntentType.TRANSPORTATION: [
                    "Do you need specific matatu route information?",
                    "Would you like safety tips for using public transport?",
                    "Are you interested in ride-hailing app recommendations?",
                    "Do you need information about transportation costs?",
                ],
                IntentType.UNIVERSITY_INFO: [
                    "Do you need help with registration procedures?",
                    "Would you like information about campus facilities?",
                    "Are you interested in international student services?",
                    "Do you need guidance on academic requirements?",
                ],
                IntentType.COST_INQUIRY: [
                    "Would you like a breakdown of monthly living expenses?",
                    "Do you need budgeting tips for international students?",
                    "Are you interested in cost comparisons between areas?",
                    "Would you like money-saving recommendations?",
                ],
                IntentType.SAFETY_CONCERN: [
                    "Do you need emergency contact information?",
                    "Would you like area-specific safety guidance?",
                    "Are you interested in community safety resources?",
                    "Do you need personal safety tips for students?",
                ],
            }

            return follow_ups.get(
                intent_type,
                [
                    "Is there anything specific you'd like to know more about?",
                    "Do you need information about other settlement topics?",
                    "Would you like guidance on related procedures?",
                ],
            )[
                :3
            ]  # Return top 3 follow-ups

        except Exception as e:
            logger.error(f"Error generating follow-ups: {str(e)}")
            return []

    def get_response_stats(self) -> Dict[str, Any]:
        """Get response generation statistics."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "settlement_optimized": True,
            "multilingual_support": True,
            "supported_intents": len(IntentType),
            "features": [
                "Settlement-domain expertise",
                "Multilingual query processing",
                "Cultural sensitivity",
                "Practical guidance focus",
                "Cost and location specificity",
                "Safety awareness",
                "Quality validation",
            ],
        }
