import os
import time
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pytz
from openai import OpenAI

from config.settings import settings
from services.intent_recognizer import IntentType
from services.language_processor import LanguageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_generator")


class ResponseGenerator:
    """
    Optimized response generator for 100/100 performance with speed improvements.
    Uses gpt-4.1-mini throughout and Tavily search for current information.
    """

    def __init__(self):
        self.model = "gpt-4.1-mini"  # Fast, cost-efficient model
        self.temperature = settings.llm.temperature
        self.max_tokens = settings.llm.max_tokens

        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize language processor
        self.language_processor = LanguageProcessor()

        # Context quality thresholds
        self.min_context_relevance = 0.3
        self.min_chunks_for_response = 1

        # Essential settlement information
        self.essential_info = {
            "emergency_numbers": {
                "universal_emergency": "999",
                "police": "999",
                "ambulance": "999",
                "fire_service": "999",
                "red_cross": "0700 395 395",
                "aa_kenya": "0700 200 007",
            },
            "immigration_office": {
                "main_office": "Nyayo House, Uhuru Highway",
                "contact": "Department of Immigration Services",
                "phone": "+254 20 222 2022",
                "hours": "8:00 AM - 5:00 PM (Monday - Friday)",
                "services": "Visa applications, renewals, permits",
            },
            "key_hospitals": {
                "nairobi_hospital": {
                    "name": "The Nairobi Hospital",
                    "address": "Argwings Kodhek Road, Upper Hill, Nairobi",
                    "phone": "+254 20 2845000 / +254 703 082000",
                    "email": "inquiry@nbihosp.org",
                    "website": "https://www.nairobihosp.co.ke",
                    "notes": "Large private hospital with advanced specialist and diagnostic services.",
                },
                "aga_khan": {
                    "name": "Aga Khan University Hospital",
                    "address": "3rd Parklands Avenue / Limuru Road, Nairobi",
                    "phone": "+254 20 3740000 / +254 73 0011888",
                    "email": "akhn@akhskenya.org",
                    "website": "https://hospitals.aku.edu/nairobi/Pages/default.aspx",
                    "notes": "Tertiary teaching hospital with wide range of specialty services.",
                },
                "mp_shah": {
                    "name": "MP Shah Hospital",
                    "address": "Shivachi Road, Parklands, Nairobi",
                    "phone": "+254 20 3742763‑7",
                    "email": "info@mpshahhosp.org",
                    "website": "https://www.mpshahhospitals.com",
                    "notes": "Well-known multi-specialty private hospital.",
                },
                "knh": {
                    "name": "Kenyatta National Hospital",
                    "address": "Hospital Road, Upper Hill, Nairobi",
                    "phone": "+254 20 2726300",
                    "email": "knhadmin@knh.or.ke",
                    "website": "https://www.knh.or.ke",
                    "notes": "Kenya's largest public referral and teaching hospital.",
                },
                "mater": {
                    "name": "Mater Misericordiae Hospital",
                    "address": "Dunga Road, South B, Nairobi",
                    "phone": "+254 20 531197 / +254 72 4531199",
                    "email": "ceo@materkenya.com",
                    "website": "https://www.materkenya.com",
                    "notes": "Faith-based hospital, widely respected.",
                },
                "gertrude": {
                    "name": "Gertrude's Children's Hospital",
                    "address": "Muthaiga Road, Muthaiga, Nairobi",
                    "phone": "+254 20 7206000",
                    "email": "info@gerties.org",
                    "website": "https://www.gertschospital.com",
                    "notes": "Specialized pediatric hospital.",
                },
                "karenn": {
                    "name": "Karen Hospital",
                    "address": "Lang'ata Road, Karen, Nairobi",
                    "phone": "+254 20 6613000 / +254 71 9240000",
                    "email": "info@karenhospital.org",
                    "website": "https://www.karenhospital.org",
                    "notes": "Multi-specialty private hospital in Karen.",
                },
                "nairobi_womens": {
                    "name": "The Nairobi Women's Hospital",
                    "address": "Hurlingham Court / Argwings Kodhek Road, Nairobi",
                    "phone": "+254 709 667000",
                    "email": "info@nwch.co.ke",
                    "website": "https://nwh.co.ke",
                    "notes": "Focus on obstetrics, gynecology, and women's health.",
                },
                "coptic": {
                    "name": "Coptic Hospital",
                    "address": "Ngong Road, Nairobi",
                    "phone": "+254 20 2720402 / +254 709 572000",
                    "email": "info@coptichospital.org",
                    "website": "https://www.coptichospital.org",
                    "notes": "Private hospital offering various specialist services.",
                },
                "kenyatta_uni_hosp": {
                    "name": "Kenyatta University Teaching, Referral & Research Hospital",
                    "address": "Kahawa West, Northern Bypass, Nairobi",
                    "website": "https://kutrrh.go.ke/",
                    "notes": "University hospital, specialized teaching and referral facility.",
                },
                "mbagathi": {
                    "name": "Mbagathi District Hospital",
                    "address": "Ngumo Estate, Off Mbagathi Road, Nairobi",
                    "phone": "+254 20 2724712 / +254 721 311808",
                    "notes": "Public hospital serving Nairobi County.",
                },
                "mama_lucy": {
                    "name": "Mama Lucy Kibaki Hospital",
                    "address": "Umoja II, Kangundo Road, Nairobi",
                    "phone": "+254 20 8022676",
                    "email": "info@mamalucykibakihospital.or.ke",
                    "website": "https://mamalucykibakihospital.or.ke",
                    "notes": "Public district hospital under Nairobi County.",
                },
                "pumwani": {
                    "name": "Pumwani Maternity Hospital",
                    "address": "City Hall Way, Nairobi City County, Nairobi",
                    "phone": "+254 725 624489 / +254 738 041292",
                    "email": "info@nairobi.go.ke",
                    "website": "https://nairobi.go.ke/pumwani-maternity-hospital",
                    "notes": "Public maternity hospital in Nairobi.",
                },
                "st_francis": {
                    "name": "St. Francis Community Hospital",
                    "address": "Kasarani, Nairobi",
                    "phone": "+254 20 2012230",
                    "notes": "Faith-based community hospital.",
                },
                "nazareth": {
                    "name": "Nazareth Hospital",
                    "address": "Limuru Road, Nairobi",
                    "phone": "+254 20 2720450",
                    "notes": "Faith-based general medical services hospital.",
                },
            },
            "universities": {
                "uon": {
                    "name": "University of Nairobi",
                    "address": "University Way, Nairobi 00100, Kenya",
                    "admissions_phone": "+254‑20‑4910000",
                    "admissions_email": "admissions@uonbi.ac.ke",
                    "website": "https://www.uonbi.ac.ke",
                },
                "jkuat": {
                    "name": "Jomo Kenyatta University of Agriculture & Technology (Nairobi Campus)",
                    "address": "JUJA / Nairobi",
                    "website": "https://www.jkuat.ac.ke",
                },
                "mmu": {
                    "name": "Multimedia University of Kenya",
                    "address": "Magadi Road, Nairobi",
                    "website": "https://www.mmu.ac.ke",
                },
                "tuk": {
                    "name": "Technical University of Kenya",
                    "address": "Haile Selassie Avenue, Nairobi",
                    "website": "https://www.tukenya.ac.ke",
                },
                "kca": {
                    "name": "KCA University",
                    "address": "Thika Road, Ruaraka, Nairobi",
                    "phone": "0709‑813800",
                    "email": "ctle@kcau.ac.ke",
                    "website": "https://www.kcau.ac.ke",
                },
                "strathmore": {
                    "name": "Strathmore University",
                    "address": "Ole Sangale Road, Madaraka, Nairobi",
                    "phone": "0703‑034000",
                    "website": "https://www.strathmore.edu",
                },
                "daystar": {
                    "name": "Daystar University",
                    "address": "Athi River (main campus), Nairobi campus",
                    "website": "https://www.daystar.ac.ke",
                },
                "aiu": {
                    "name": "Africa International University (AIU)",
                    "address": "Nairobi, Kenya",
                    "website": "https://www.aiu.ac.ke",
                },
                "anaz": {
                    "name": "Africa Nazarene University",
                    "address": "Nairobi, Kenya",
                    "website": "https://www.anu.ac.ke",
                },
                "cuea": {
                    "name": "Catholic University of Eastern Africa",
                    "address": "Lang'ata Road, Nairobi",
                    "website": "https://www.cuea.edu",
                },
                "amref": {
                    "name": "Amref International University",
                    "address": "Nairobi, Kenya",
                    "website": "https://www.amref.ac.ke",
                },
                "pac": {
                    "name": "Pan Africa Christian University (PAC)",
                    "address": "Roysambu, Nairobi",
                    "website": "https://www.pacuniversity.ac.ke",
                },
                "mua": {
                    "name": "Management University of Africa",
                    "address": "Waiyaki Way, Nairobi",
                    "website": "https://www.mua.ac.ke",
                },
                "tangaza": {
                    "name": "Tangaza University",
                    "address": "Lang'ata, Nairobi",
                    "website": "https://tangaza.ac.ke",
                },
                "kwust": {
                    "name": "Kiriri Women's University of Science and Technology (KWUST)",
                    "address": "Githurai / Nairobi",
                    "website": "http://www.kwust.ac.ke",
                },
                "usiu": {
                    "name": "United States International University – Africa (USIU‑A)",
                    "address": "USIU Road, Kasarani, Nairobi",
                    "phone": "0730‑116000",
                    "website": "https://www.usiu.ac.ke",
                },
            },
        }

        # Empathy and validation phrases by emotional state
        self.empathy_responses = {
            "stress": [
                "That's completely understandable - settling in a new country can feel overwhelming",
                "Many international students feel this way when they first arrive",
                "It's normal to feel stressed about this - you're navigating a lot of new things",
            ],
            "confusion": [
                "This can definitely be confusing when you're new to the system",
                "These processes can seem complicated at first, but they become clearer",
                "Don't worry - this is a common question that many students ask",
            ],
            "anxiety": [
                "It's natural to feel anxious about this - it's an important decision",
                "Your concerns about this are completely valid",
                "Many students worry about this same thing - you're not alone",
            ],
            "urgency": [
                "I understand this is urgent for you - let me help you prioritize",
                "Time-sensitive situations like this can be stressful - here's what you need to know",
                "I can see this needs quick action - let's focus on immediate steps",
            ],
        }

        # Safety protocols by topic
        self.safety_protocols = {
            "housing": [
                "Always visit properties in person before paying any deposits",
                "Verify landlord credentials and ask for official rental agreements",
                "Check that the neighborhood is well-lit and has security presence",
                "Ensure the property has working locks and security features",
            ],
            "transportation": [
                "Avoid traveling alone late at night, especially in unfamiliar areas",
                "Use registered taxi services or ride-sharing apps rather than unofficial taxis",
                "Keep emergency numbers saved in your phone for quick access",
                "Trust your instincts - if something feels wrong, find an alternative route",
            ],
            "finance": [
                "Never share your M-Pesa PIN or banking details with anyone",
                "Be cautious of advance fee scams promising easy money or jobs",
                "Use ATMs inside banks or shopping malls rather than standalone machines",
                "Keep copies of important financial documents in a separate location",
            ],
            "general": [
                "Register with your embassy or consulate after arrival",
                "Keep photocopies of important documents separate from originals",
                "Stay aware of your surroundings, especially in crowded areas",
                "Have multiple ways to contact help in case of emergency",
            ],
        }

        # Standard off-topic response
        self.off_topic_response = """## DIRECT ANSWER
I'm specifically designed to help international students with settlement in Nairobi, Kenya. I don't have information about this topic as it falls outside my area of expertise.

## ADDITIONAL INFORMATION
I'm here to assist you with questions about:

**Housing and Accommodation**: Finding safe, affordable places to live near universities, understanding rental processes, neighborhood recommendations

**University Information**: Admission processes, academic requirements, campus facilities, student services for institutions like University of Nairobi, Strathmore, JKUAT, and USIU

**Immigration and Legal**: Student visa applications, permit renewals, immigration office locations and procedures

**Transportation**: Matatu routes, public transport, taxi services, getting around Nairobi safely and affordably

**Safety and Security**: Neighborhood safety, personal security measures, emergency procedures, areas to avoid

**Banking and Finance**: Opening bank accounts, using M-Pesa, money transfers, budgeting for student life

**Healthcare**: Finding medical services, health insurance options, hospitals and clinics that serve international patients

**Cultural Adaptation**: Understanding Kenyan customs, making local connections, adapting to life in Nairobi

## NEXT STEPS
1. Ask me anything about settling in Nairobi as an international student - I have comprehensive information to help make your transition smooth
2. If you have urgent settlement needs, I can prioritize the most critical information first
3. Consider joining international student groups at your university for peer support and additional advice
4. Feel free to ask follow-up questions to get more specific guidance on any settlement topic"""

    def get_current_nairobi_time(self):
        """Get current date and time in Nairobi (EAT - UTC+3)."""
        nairobi_tz = pytz.timezone("Africa/Nairobi")
        current_time = nairobi_tz.localize(datetime.now())
        formatted_time = current_time.strftime("%A, %B %d, %Y %H:%M")
        time_only = current_time.strftime("%H:%M")
        return formatted_time, time_only

    def search_web_for_current_info(
        self, query: str, intent_type: IntentType
    ) -> Optional[Dict]:
        """Search web for current information using Tavily AI search API."""
        try:
            # Import Tavily client
            try:
                from tavily import TavilyClient
            except ImportError:
                logger.warning(
                    "tavily-python library not installed. Install with: pip install tavily-python"
                )
                return None

            # Get API key from environment
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                logger.warning(
                    "TAVILY_API_KEY not found in environment variables. Skipping web search."
                )
                return None

            # Initialize Tavily client
            tavily_client = TavilyClient(api_key=tavily_api_key)

            # Construct targeted search queries based on intent
            search_queries = self._build_search_queries(query, intent_type)

            web_results = []

            # Limit to 2 searches to balance quality and cost
            for search_query in search_queries[:2]:
                try:
                    # Perform Tavily search - optimized for AI/RAG applications
                    start_time = time.perf_counter()
                    search_result = tavily_client.search(
                        query=search_query,
                        search_depth="basic",  # Fast and cost-efficient
                        max_results=3,  # Top 3 most relevant results
                        include_answer=True,  # Get AI-generated summary
                        include_raw_content=False,  # Don't need full HTML
                        include_images=False,  # No images needed
                    )
                    end_time = time.perf_counter()
                    print(f"Elapsed time is { end_time - start_time } seconds")

                    # Tavily returns structured data perfect for AI consumption
                    if search_result and search_result.get("results"):
                        # Extract the AI-generated answer (summary)
                        ai_summary = search_result.get("answer", "")

                        # Extract individual results with clean content
                        results_list = search_result.get("results", [])

                        # Build comprehensive content from results
                        content_parts = []

                        if ai_summary:
                            content_parts.append(f"Summary: {ai_summary}\n")

                        content_parts.append("Key Information:")
                        for idx, result in enumerate(results_list[:3], 1):
                            title = result.get("title", "")
                            content = result.get(
                                "content", ""
                            )  # Already cleaned by Tavily
                            url = result.get("url", "")
                            score = result.get("score", 0)

                            content_parts.append(
                                f"\n{idx}. {title}\n"
                                f"   {content}\n"
                                f"   Source: {url} (Relevance: {score:.2f})"
                            )

                        combined_content = "\n".join(content_parts)

                        web_results.append(
                            {
                                "query": search_query,
                                "content": combined_content,
                                "source": "tavily_search",
                                "results_count": len(results_list),
                                "has_answer": bool(ai_summary),
                            }
                        )

                        logger.info(
                            f"Successfully retrieved Tavily results for: {search_query} "
                            f"({len(results_list)} results)"
                        )
                    else:
                        logger.info(
                            f"No Tavily results found for query: {search_query}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Tavily search failed for query '{search_query}': {str(e)}"
                    )
                    continue

            if web_results:
                logger.info(
                    f"Tavily search completed successfully with {len(web_results)} result sets"
                )
            else:
                logger.info(
                    "Tavily search completed but no results were retrieved"
                )

            return {
                "results": web_results,
                "search_successful": len(web_results) > 0,
            }

        except Exception as e:
            logger.error(f"Tavily search orchestration failed: {str(e)}")
            return None

    def _build_search_queries(
        self, query: str, intent_type: IntentType
    ) -> List[str]:
        """Build targeted search queries based on intent type."""
        base_query = f"{query} Nairobi Kenya international students 2024"

        intent_specific_queries = {
            IntentType.HOUSING_INQUIRY: [
                f"student accommodation Nairobi 2024 prices areas {query}",
                f"international student housing Nairobi safety {query}",
            ],
            IntentType.UNIVERSITY_INFO: [
                f"Kenya universities admission international students 2024 {query}",
                f"University of Nairobi Strathmore JKUAT admission {query}",
            ],
            IntentType.IMMIGRATION_VISA: [
                f"Kenya student visa 2024 requirements process {query}",
                f"immigration office Nairobi visa renewal {query}",
            ],
            IntentType.TRANSPORTATION: [
                f"Nairobi public transport matatu routes 2024 {query}",
                f"student transport Nairobi safety tips {query}",
            ],
            IntentType.SAFETY_CONCERN: [
                f"Nairobi safety international students 2024 areas {query}",
                f"student safety Kenya crime prevention {query}",
            ],
            IntentType.COST_INQUIRY: [
                f"cost of living Nairobi students 2024 budget {query}",
                f"student expenses Kenya accommodation food transport {query}",
            ],
            IntentType.BANKING_FINANCE: [
                f"Kenya banking international students M-Pesa 2024 {query}",
                f"student bank accounts Nairobi requirements {query}",
            ],
            IntentType.HEALTHCARE: [
                f"Nairobi hospitals international students 2024 {query}",
                f"student health insurance Kenya medical services {query}",
            ],
        }

        return intent_specific_queries.get(intent_type, [base_query])

    def detect_emotional_state(self, query: str) -> Dict[str, Any]:
        """Detect emotional state from query - optimized version."""
        try:
            # Quick check for obvious emotional indicators
            query_lower = query.lower()

            # Emergency/high urgency
            if any(
                word in query_lower
                for word in ["emergency", "urgent", "help", "scared", "danger"]
            ):
                return {
                    "primary_emotion": "urgency",
                    "intensity": "high",
                    "indicators": ["urgent language detected"],
                    "needs_validation": True,
                }

            # Stress indicators
            if any(
                word in query_lower
                for word in ["overwhelmed", "stressed", "too much", "difficult"]
            ):
                return {
                    "primary_emotion": "stress",
                    "intensity": "medium",
                    "indicators": ["stress language detected"],
                    "needs_validation": True,
                }

            # Confusion indicators
            if any(
                word in query_lower
                for word in [
                    "confused",
                    "don't understand",
                    "unclear",
                    "how do i",
                ]
            ):
                return {
                    "primary_emotion": "confusion",
                    "intensity": "low",
                    "indicators": ["confusion language detected"],
                    "needs_validation": True,
                }

            # Anxiety indicators
            if any(
                word in query_lower
                for word in ["worried", "anxious", "nervous", "afraid"]
            ):
                return {
                    "primary_emotion": "anxiety",
                    "intensity": "medium",
                    "indicators": ["anxiety language detected"],
                    "needs_validation": True,
                }

            # Default neutral state
            return {
                "primary_emotion": "neutral",
                "intensity": "low",
                "indicators": [],
                "needs_validation": False,
            }

        except Exception as e:
            logger.warning(f"Emotion detection failed: {str(e)}")
            return {
                "primary_emotion": "neutral",
                "intensity": "medium",
                "indicators": [],
                "needs_validation": False,
            }

    def assess_crisis_indicators(
        self, query: str, emotional_state: Dict
    ) -> Dict[str, Any]:
        """Assess if query contains crisis indicators requiring immediate support."""
        crisis_keywords = [
            "emergency",
            "urgent",
            "help",
            "crisis",
            "danger",
            "threat",
            "scared",
            "afraid",
            "attacked",
            "robbed",
            "stolen",
            "lost",
            "stranded",
            "sick",
            "injured",
            "hospital",
            "police",
            "can't",
            "desperate",
            "nowhere",
        ]

        query_lower = query.lower()
        crisis_indicators = [
            word for word in crisis_keywords if word in query_lower
        ]

        # Determine crisis level
        crisis_level = "none"
        if len(crisis_indicators) >= 3 or any(
            word in query_lower
            for word in ["emergency", "help", "urgent", "danger"]
        ):
            crisis_level = "high"
        elif (
            len(crisis_indicators) >= 2
            or emotional_state.get("intensity") == "high"
        ):
            crisis_level = "medium"
        elif len(crisis_indicators) >= 1:
            crisis_level = "low"

        return {
            "crisis_level": crisis_level,
            "indicators": crisis_indicators,
            "needs_emergency_info": crisis_level in ["medium", "high"],
            "needs_immediate_support": crisis_level == "high",
        }

    def generate_response(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        intent_info: Dict[str, Any],
        conversation_context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive, optimized responses with speed improvements.
        Uses parallel processing where possible and gpt-4.1-mini throughout.
        """
        try:
            # Process language detection and translation
            language_result = self.language_processor.detect_and_process_query(
                query
            )
            english_query = language_result["english_query"]
            original_language = language_result["detected_language"]
            needs_translation = language_result["needs_translation"]

            # Handle off-topic queries immediately
            if intent_info["intent_type"] == IntentType.OFF_TOPIC:
                logger.info(
                    "Off-topic query detected, returning standard response"
                )

                response_content = self.off_topic_response

                if needs_translation and original_language != "english":
                    try:
                        translated_response = (
                            self.language_processor.translate_response(
                                response_content, original_language
                            )
                        )
                    except:
                        translated_response = response_content
                else:
                    translated_response = response_content

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
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "current_time": None,
                    "settlement_optimized": True,
                    "response_style": "comprehensive_off_topic",
                    "empathy_applied": False,
                    "safety_protocols_added": False,
                    "crisis_level": "none",
                }

            # Get current time
            current_time_full, current_time = self.get_current_nairobi_time()

            # OPTIMIZATION: Run emotion detection and web search in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                emotion_future = executor.submit(
                    self.detect_emotional_state, english_query
                )
                web_future = executor.submit(
                    self.search_web_for_current_info,
                    english_query,
                    intent_info["intent_type"],
                )

                # Get results (blocks until both complete)
                emotional_state = emotion_future.result()
                web_info = web_future.result()

            # Assess crisis indicators
            crisis_assessment = self.assess_crisis_indicators(
                english_query, emotional_state
            )

            # Evaluate and enhance context
            context_evaluation = self._evaluate_and_enhance_context(
                retrieved_context, english_query, intent_info, crisis_assessment
            )

            # Format comprehensive context
            enhanced_context = self._format_comprehensive_context(
                retrieved_context, web_info, intent_info, context_evaluation
            )

            # Generate comprehensive response
            response_content = self._generate_comprehensive_response(
                english_query,
                enhanced_context,
                intent_info,
                emotional_state,
                crisis_assessment,
                current_time_full,
                context_evaluation,
            )

            # Apply final validation and safety checks
            validated_response = self._apply_final_validation_and_safety(
                response_content,
                intent_info,
                crisis_assessment,
                emotional_state,
            )

            # Translate if needed
            if needs_translation and original_language != "english":
                try:
                    translated_response = (
                        self.language_processor.translate_response(
                            validated_response, original_language
                        )
                    )
                    translation_quality = (
                        self.language_processor.validate_translation_quality(
                            validated_response,
                            translated_response,
                            original_language,
                        )
                    )
                except Exception as e:
                    # Continuation from Part 1 - Translation section
                    logger.warning(f"Translation failed: {str(e)}")
                    translated_response = validated_response
                    translation_quality = None
            else:
                translated_response = validated_response
                translation_quality = None

            # Generate token usage estimate
            token_usage = {
                "prompt_tokens": len(english_query.split()) * 4,
                "completion_tokens": len(validated_response.split()) * 4,
                "total_tokens": (len(english_query) + len(validated_response))
                * 4,
            }

            logger.info("Generated comprehensive response with optimizations")

            return {
                "response": translated_response,
                "original_response": validated_response
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
                "response_style": "comprehensive_empathetic",
                "empathy_applied": emotional_state.get(
                    "needs_validation", False
                ),
                "safety_protocols_added": self._safety_protocols_applied(
                    intent_info
                ),
                "crisis_level": crisis_assessment["crisis_level"],
                "emotional_state": emotional_state["primary_emotion"],
                "web_search_used": web_info is not None
                and web_info.get("search_successful", False),
                "context_enhanced": context_evaluation["was_enhanced"],
                "comprehensive_score": self._calculate_comprehensive_score(
                    context_evaluation, emotional_state, crisis_assessment
                ),
            }

        except Exception as e:
            logger.error(f"Error generating comprehensive response: {str(e)}")
            return self._generate_error_response(
                intent_info,
                original_language
                if "original_language" in locals()
                else "english",
            )

    def _evaluate_and_enhance_context(
        self,
        retrieved_context: List[Dict[str, Any]],
        query: str,
        intent_info: Dict[str, Any],
        crisis_assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate context and identify enhancement opportunities."""

        if not retrieved_context:
            context_quality = 0.0
            relevant_contexts = []
        else:
            relevant_contexts = [
                ctx
                for ctx in retrieved_context
                if ctx.get("score", 0) >= self.min_context_relevance
            ]
            context_quality = sum(
                ctx.get("score", 0) for ctx in relevant_contexts
            ) / max(len(relevant_contexts), 1)

        # Identify essential information needed
        essential_needed = self._identify_comprehensive_info_needed(
            query, intent_info, crisis_assessment
        )

        # Determine enhancement strategy
        enhancement_strategy = "comprehensive"
        if crisis_assessment["crisis_level"] in ["medium", "high"]:
            enhancement_strategy = "crisis_focused"
        elif context_quality < 0.4:
            enhancement_strategy = "knowledge_supplemented"

        return {
            "sufficient_context": True,
            "context_quality": context_quality,
            "relevant_count": len(relevant_contexts),
            "essential_needed": essential_needed,
            "was_enhanced": len(essential_needed) > 0,
            "enhancement_strategy": enhancement_strategy,
            "comprehensive_score": min(context_quality + 0.3, 1.0),
        }

    def _identify_comprehensive_info_needed(
        self, query: str, intent_info: Dict, crisis_assessment: Dict
    ) -> List[str]:
        """Identify comprehensive information needed for complete response."""
        needed_info = []
        query_lower = query.lower()
        intent_type = intent_info["intent_type"]

        # Crisis information
        if crisis_assessment["needs_emergency_info"]:
            needed_info.extend(["emergency_numbers", "crisis_contacts"])

        # Intent-specific information
        if intent_type == IntentType.HOUSING_INQUIRY:
            needed_info.extend(
                ["housing_areas", "rental_process", "safety_tips"]
            )
            if any(word in query_lower for word in ["cost", "price", "budget"]):
                needed_info.append("housing_costs")

        elif intent_type == IntentType.UNIVERSITY_INFO:
            needed_info.extend(["university_contacts", "academic_processes"])
            if any(word in query_lower for word in ["admission", "apply"]):
                needed_info.append("admission_requirements")

        elif intent_type == IntentType.IMMIGRATION_VISA:
            needed_info.extend(["immigration_office", "visa_requirements"])

        elif intent_type == IntentType.TRANSPORTATION:
            needed_info.extend(["transport_options", "safety_transport"])

        elif intent_type == IntentType.SAFETY_CONCERN:
            needed_info.extend(
                ["safety_protocols", "emergency_numbers", "safe_areas"]
            )

        elif intent_type == IntentType.BANKING_FINANCE:
            needed_info.extend(["banking_info", "mpesa_guide"])

        elif intent_type == IntentType.HEALTHCARE:
            needed_info.extend(["hospitals", "health_insurance"])

        # Always include general safety
        if intent_type not in [
            IntentType.SAFETY_CONCERN,
            IntentType.EMERGENCY_HELP,
        ]:
            needed_info.append("general_safety")

        return list(set(needed_info))

    def _format_comprehensive_context(
        self,
        retrieved_context: List[Dict[str, Any]],
        web_info: Optional[Dict],
        intent_info: Dict[str, Any],
        context_evaluation: Dict[str, Any],
    ) -> str:
        """Format comprehensive context including all available information sources."""

        context_parts = []

        # Add retrieved RAG context
        if retrieved_context:
            relevant_contexts = [
                ctx
                for ctx in retrieved_context
                if ctx.get("score", 0) >= self.min_context_relevance
            ]

            if relevant_contexts:
                context_parts.append("RETRIEVED SETTLEMENT INFORMATION:")
                sorted_contexts = sorted(
                    relevant_contexts,
                    key=lambda x: x.get("score", 0),
                    reverse=True,
                )[:7]

                for i, chunk in enumerate(sorted_contexts, 1):
                    context_parts.append(
                        f"Source {i} (Relevance: {chunk.get('score', 0):.2f}):"
                    )
                    context_parts.append(chunk["text"].strip())
                    context_parts.append("")

        # Add web search results
        if web_info and web_info.get("search_successful"):
            context_parts.append("CURRENT WEB INFORMATION:")
            for result in web_info["results"]:
                context_parts.append(f"Recent Information - {result.get('query')}:")
                context_parts.append(result["content"])
                context_parts.append("")

        # Add essential information
        essential_needed = context_evaluation.get("essential_needed", [])
        if essential_needed:
            context_parts.append("ESSENTIAL SETTLEMENT INFORMATION:")

            for info_type in essential_needed:
                if info_type == "emergency_numbers":
                    context_parts.append("EMERGENCY CONTACTS:")
                    for service, number in self.essential_info[
                        "emergency_numbers"
                    ].items():
                        context_parts.append(
                            f"- {service.replace('_', ' ').title()}: {number}"
                        )
                    context_parts.append("")

                elif info_type == "immigration_office":
                    context_parts.append("IMMIGRATION OFFICE:")
                    for key, value in self.essential_info[
                        "immigration_office"
                    ].items():
                        context_parts.append(
                            f"- {key.replace('_', ' ').title()}: {value}"
                        )
                    context_parts.append("")

                elif info_type == "hospitals":
                    context_parts.append("KEY HOSPITALS:")
                    for hospital_key, hospital_info in self.essential_info[
                        "key_hospitals"
                    ].items():
                        context_parts.append(
                            f"- {hospital_info.get('name', '')}: {hospital_info.get('address', '')}, {hospital_info.get('phone', '')}"
                        )
                    context_parts.append("")

                elif info_type == "university_contacts":
                    context_parts.append("UNIVERSITY CONTACTS:")
                    for uni_key, uni_info in self.essential_info[
                        "universities"
                    ].items():
                        context_parts.append(
                            f"- {uni_info.get('name', '')}: {uni_info.get('address', '')}"
                        )
                    context_parts.append("")

        if not context_parts:
            context_parts = [
                "Limited specific information available, will provide comprehensive general guidance."
            ]

        return "\n".join(context_parts)

    def _generate_comprehensive_response(
        self,
        query: str,
        context_text: str,
        intent_info: Dict[str, Any],
        emotional_state: Dict[str, Any],
        crisis_assessment: Dict[str, Any],
        current_time: str,
        context_evaluation: Dict[str, Any],
    ) -> str:
        """Generate comprehensive response using gpt-4.1-mini."""

        # Build empathy component
        empathy_component = ""
        if emotional_state.get("needs_validation", False):
            emotion = emotional_state.get("primary_emotion", "neutral")
            if emotion in self.empathy_responses:
                empathy_component = f"EMPATHY_VALIDATION: {self.empathy_responses[emotion][0]}\n\n"

        # Build crisis component
        crisis_component = ""
        if crisis_assessment["needs_immediate_support"]:
            crisis_component = "CRISIS_PRIORITY: This appears to be an urgent situation requiring immediate attention.\n\n"

        # Get comprehensive system prompt
        system_prompt = self._get_comprehensive_system_prompt(
            intent_info["intent_type"], emotional_state, crisis_assessment
        )

        # Build comprehensive user prompt
        user_prompt = f"""[CURRENT CONTEXT - NAIROBI, KENYA]
Time: {current_time}
Query Analysis: Intent={intent_info["intent_type"].value}, Emotion={emotional_state.get("primary_emotion", "neutral")}, Crisis Level={crisis_assessment["crisis_level"]}

[STUDENT QUERY]
"{query}"

{empathy_component}{crisis_component}[COMPREHENSIVE INFORMATION SOURCES]
{context_text}

[RESPONSE REQUIREMENTS]
Create a comprehensive, empathetic response that achieves 100/100 performance by:

1. DIRECT ANSWER (Must be substantial and complete):
- Provide a thorough, specific answer that directly addresses their question
- Include concrete details, numbers, locations, and actionable information
- Don't cut short - give them everything they need to know immediately
- Use settlement-specific knowledge and current information

2. ADDITIONAL INFORMATION (Must be comprehensive and valuable):
- Provide extensive context, background, and related information
- Include safety considerations, cost information, and practical tips
- Add information about alternatives, options, and considerations
- Cover related topics they might need to know
- Include specific contacts, websites, and resources

3. NEXT STEPS (Must be actionable and complete):
- Give detailed, sequential steps they can take immediately
- Include specific contact information, websites, and locations
- Provide timeline expectations and preparation requirements
- Offer multiple pathways and backup options

CRITICAL REQUIREMENTS:
- Be genuinely helpful and comprehensive, not bureaucratic
- Include safety protocols relevant to their situation
- Provide specific Nairobi locations, contacts, and current information
- Use encouraging, supportive language appropriate to their emotional state
- Make sure each section is substantial and complete
- Focus on practical, actionable guidance they can use right away"""

        try:
            start_time = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,  # gpt-4.1-mini for speed and cost
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
            end_time = time.perf_counter()
            print(f"Elapsed time is { end_time - start_time } seconds")

            response_content = response.choices[0].message.content
            return self._ensure_comprehensive_structure(response_content)

        except Exception as e:
            logger.error(f"Error generating comprehensive response: {str(e)}")
            return self._generate_comprehensive_fallback(
                query, intent_info, emotional_state
            )

    def _get_comprehensive_system_prompt(
        self,
        intent_type: IntentType,
        emotional_state: Dict,
        crisis_assessment: Dict,
    ) -> str:
        """Get comprehensive system prompt optimized for 100/100 performance."""
        base_prompt = """You are SettleBot, an expert assistant for international students settling in Nairobi, Kenya. Your goal is to achieve 100/100 performance by being genuinely comprehensive, empathetic, and helpful.

RESPONSE STRUCTURE (Always use exactly these sections):
## DIRECT ANSWER
[Provide a complete, thorough answer that fully addresses their question. This must be substantial (5+ sentences) with specific details, locations, costs, and actionable information. Don't cut short.]

## ADDITIONAL INFORMATION
[Provide comprehensive background, context, safety considerations, alternatives, and everything else they should know. This should be extensive and valuable.]

## NEXT STEPS
[Give detailed, actionable steps they can take immediately. Include specific contacts, locations, timelines, and multiple options.]

CORE PRINCIPLES:
1. Be genuinely comprehensive - give them everything they need to know
2. Include specific Nairobi details - locations, costs in KSh, contact numbers, current information
3. Add safety considerations systematically - this is critical for student wellbeing
4. Use empathetic, supportive language that validates their concerns
5. Provide multiple options and alternatives when possible
6. Include practical details like operating hours, required documents, costs
7. Make each section substantial and complete - never give superficial answers

EMPATHY REQUIREMENTS:
- Acknowledge the emotional aspect of their situation
- Use validating language that shows understanding
- Frame challenges as navigable rather than overwhelming
- Provide confidence-building statements"""

        # Add intent-specific guidance
        intent_specific = {
            IntentType.EMERGENCY_HELP: """
EMERGENCY PRIORITY:
- Lead with immediate emergency numbers and contacts
- Provide specific step-by-step emergency procedures
- Include multiple emergency resources and backup options
- Be direct and action-oriented while remaining supportive""",
            IntentType.HOUSING_INQUIRY: """
HOUSING FOCUS:
- Include specific neighborhoods with detailed descriptions
- Provide current rental price ranges in KSh
- Add comprehensive safety considerations for each area
- Include rental process details and scam prevention
- Cover transportation access from housing areas to universities""",
            IntentType.UNIVERSITY_INFO: """
UNIVERSITY FOCUS:
- Provide detailed information about specific universities mentioned
- Include admission processes, requirements, and timelines
- Cover student support services and campus facilities
- Add information about academic culture and expectations
- Include contact details for international student offices""",
            IntentType.SAFETY_CONCERN: """
SAFETY FOCUS:
- Provide comprehensive safety protocols and precautions
- Include specific areas to avoid with reasons why
- Give detailed personal security measures
- Add emergency procedures and contact information
- Cover both day and night safety considerations""",
            IntentType.IMMIGRATION_VISA: """
IMMIGRATION FOCUS:
- Include detailed visa procedures and requirements
- Provide specific office locations, hours, and contact information
- Cover required documents with detailed descriptions
- Include timelines, costs, and renewal procedures
- Add backup options and common problem solutions""",
            IntentType.COST_INQUIRY: """
COST FOCUS:
- Provide detailed cost breakdowns in KSh
- Include multiple price ranges (budget/medium/premium options)
- Cover hidden costs and additional expenses to consider
- Add money-saving tips and student discounts
- Include comparison across different areas/options""",
        }

        # Add emotional state guidance
        emotional_guidance = {
            "stress": "Use extra validation and break information into manageable steps. Emphasize that challenges are normal and navigable.",
            "anxiety": "Provide reassurance while giving concrete, actionable information. Focus on what they can control.",
            "urgency": "Prioritize immediate actionable steps while ensuring comprehensive information.",
            "confusion": "Explain processes clearly with step-by-step guidance. Use simple, clear language.",
        }

        # Add crisis response guidance
        crisis_guidance = ""
        if crisis_assessment["crisis_level"] == "high":
            crisis_guidance = """
CRISIS RESPONSE MODE:
- Prioritize immediate safety and emergency information
- Provide direct emergency contacts and procedures
- Use calm, clear, directive language
- Include multiple support options and resources"""

        return (
            base_prompt
            + intent_specific.get(intent_type, "")
            + "\n"
            + emotional_guidance.get(
                emotional_state.get("primary_emotion", "neutral"), ""
            )
            + crisis_guidance
        )

    def _apply_final_validation_and_safety(
        self,
        response: str,
        intent_info: Dict[str, Any],
        crisis_assessment: Dict[str, Any],
        emotional_state: Dict[str, Any],
    ) -> str:
        """Apply final validation and safety protocol integration."""

        # Ensure proper structure
        response = self._ensure_comprehensive_structure(response)

        # Add safety protocols if missing
        response = self._integrate_safety_protocols(
            response, intent_info["intent_type"], crisis_assessment
        )

        # Add empathy if emotional state detected but not addressed
        if emotional_state.get(
            "needs_validation"
        ) and not self._has_empathy_language(response):
            response = self._add_empathy_validation(response, emotional_state)

        # Ensure crisis information if needed
        if (
            crisis_assessment["needs_immediate_support"]
            and "999" not in response
        ):
            response = self._add_crisis_information(response)

        return response

    def _integrate_safety_protocols(
        self, response: str, intent_type: IntentType, crisis_assessment: Dict
    ) -> str:
        """Systematically integrate relevant safety protocols."""

        # Map intent types to safety categories
        intent_safety_map = {
            IntentType.HOUSING_INQUIRY: ["housing", "general"],
            IntentType.TRANSPORTATION: ["transportation", "general"],
            IntentType.BANKING_FINANCE: ["finance", "general"],
            IntentType.SAFETY_CONCERN: ["general"],
            IntentType.EMERGENCY_HELP: ["general"],
        }

        relevant_safety = intent_safety_map.get(intent_type, ["general"])

        # Check if safety information already present
        has_safety = any(
            protocol in response.lower()
            for protocols in self.safety_protocols.values()
            for protocol in [p.lower()[:20] for p in protocols]
        )

        if not has_safety:
            # Add safety section before NEXT STEPS
            safety_content = "\n\n**IMPORTANT SAFETY CONSIDERATIONS:**\n"

            for safety_category in relevant_safety:
                if safety_category in self.safety_protocols:
                    safety_content += f"\n{safety_category.title()} Safety:\n"
                    for protocol in self.safety_protocols[safety_category][:2]:
                        safety_content += f"- {protocol}\n"

            # Insert before NEXT STEPS
            if "## NEXT STEPS" in response:
                response = response.replace(
                    "## NEXT STEPS", safety_content + "\n## NEXT STEPS"
                )
            else:
                response += safety_content

        return response

    def _has_empathy_language(self, response: str) -> bool:
        """Check if response contains empathetic language."""
        empathy_indicators = [
            "understand",
            "understandable",
            "normal",
            "common",
            "many students",
            "feel",
            "concern",
            "worry",
            "stress",
            "challenging",
            "difficult",
        ]
        response_lower = response.lower()
        return any(
            indicator in response_lower for indicator in empathy_indicators
        )

    def _add_empathy_validation(
        self, response: str, emotional_state: Dict
    ) -> str:
        """Add empathy validation to response."""
        emotion = emotional_state.get("primary_emotion", "neutral")

        if emotion in self.empathy_responses:
            validation = f"\n\n{self.empathy_responses[emotion][0]}\n"

            # Insert after DIRECT ANSWER
            if "## ADDITIONAL INFORMATION" in response:
                response = response.replace(
                    "## ADDITIONAL INFORMATION",
                    validation + "## ADDITIONAL INFORMATION",
                )
            else:
                response = response.replace(
                    "## DIRECT ANSWER", "## DIRECT ANSWER" + validation
                )

        return response

    def _add_crisis_information(self, response: str) -> str:
        """Add crisis information for urgent situations."""
        crisis_info = "\n\n**EMERGENCY CONTACTS:**\n"
        for service, number in self.essential_info["emergency_numbers"].items():
            crisis_info += f"- {service.replace('_', ' ').title()}: {number}\n"

        # Insert after DIRECT ANSWER
        if "## ADDITIONAL INFORMATION" in response:
            response = response.replace(
                "## ADDITIONAL INFORMATION",
                crisis_info + "\n## ADDITIONAL INFORMATION",
            )
        else:
            response += crisis_info

        return response

    def _ensure_comprehensive_structure(self, response: str) -> str:
        """Ensure response has comprehensive three-section structure."""
        required_sections = [
            "## DIRECT ANSWER",
            "## ADDITIONAL INFORMATION",
            "## NEXT STEPS",
        ]

        # Check if all sections are present
        missing_sections = [
            section for section in required_sections if section not in response
        ]

        if missing_sections:
            # Rebuild response with proper structure
            lines = response.split("\n")
            content_blocks = {"direct": [], "additional": [], "next_steps": []}
            current_section = "direct"

            for line in lines:
                if "## DIRECT ANSWER" in line:
                    current_section = "direct"
                    continue
                elif "## ADDITIONAL INFORMATION" in line:
                    current_section = "additional"
                    continue
                elif "## NEXT STEPS" in line:
                    current_section = "next_steps"
                    continue

                if line.strip():
                    content_blocks[current_section].append(line)

            # Rebuild with proper structure
            rebuilt_response = "## DIRECT ANSWER\n"
            if content_blocks["direct"]:
                rebuilt_response += "\n".join(content_blocks["direct"])
            else:
                rebuilt_response += "I understand you're looking for specific information about this topic."

            rebuilt_response += "\n\n## ADDITIONAL INFORMATION\n"
            if content_blocks["additional"]:
                rebuilt_response += "\n".join(content_blocks["additional"])
            else:
                rebuilt_response += "For comprehensive support with your settlement needs, consider reaching out to your university's international student services or relevant Kenyan authorities for the most current and detailed information."

            rebuilt_response += "\n\n## NEXT STEPS\n"
            if content_blocks["next_steps"]:
                rebuilt_response += "\n".join(content_blocks["next_steps"])
            else:
                rebuilt_response += "1. Contact the relevant offices or services mentioned above for specific guidance\n2. Gather any required documents or information\n3. Follow up as needed and don't hesitate to ask for clarification"

            return rebuilt_response

        return response

    def _generate_comprehensive_fallback(
        self, query: str, intent_info: Dict, emotional_state: Dict
    ) -> str:
        """Generate comprehensive fallback when main generation fails."""
        intent_name = intent_info["intent_type"].value.replace("_", " ").title()

        fallback = "## DIRECT ANSWER\n"
        fallback += f"I understand you're asking about {intent_name.lower()} for your settlement in Nairobi. While I'm experiencing some technical difficulty accessing all my resources right now, I can provide you with essential guidance.\n\n"

        # Add empathy if needed
        if emotional_state.get("needs_validation"):
            fallback += f"{self.empathy_responses.get(emotional_state.get('primary_emotion', 'stress'), ['This is a common concern for international students'])[0]}.\n\n"

        fallback += "## ADDITIONAL INFORMATION\n"
        fallback += f"For comprehensive assistance with {intent_name.lower()}, your best resources include:\n\n"
        fallback += (
            "- Your university's international student services office\n"
        )
        fallback += "- Relevant Kenyan government departments and offices\n"
        fallback += "- Fellow international students who have experience with similar situations\n"
        fallback += "- Official websites and documentation for current requirements and procedures\n\n"

        # Add safety if relevant
        if intent_info["intent_type"] in [
            IntentType.SAFETY_CONCERN,
            IntentType.EMERGENCY_HELP,
        ]:
            fallback += "**EMERGENCY CONTACTS:**\n"
            for service, number in self.essential_info[
                "emergency_numbers"
            ].items():
                fallback += f"- {service.replace('_', ' ').title()}: {number}\n"
            fallback += "\n"

        fallback += "## NEXT STEPS\n"
        fallback += f"1. Contact your university's international student services for immediate guidance about {intent_name.lower()}\n"
        fallback += "2. Reach out to the appropriate Kenyan government offices for official information and requirements\n"
        fallback += "3. Connect with other international students through university groups or online communities\n"
        fallback += "4. Feel free to ask me again for more specific guidance - I'm here to help make your settlement successful\n"

        return fallback

    def _safety_protocols_applied(self, intent_info: Dict) -> bool:
        """Check if safety protocols were applied based on intent type."""
        safety_relevant_intents = [
            IntentType.HOUSING_INQUIRY,
            IntentType.TRANSPORTATION,
            IntentType.BANKING_FINANCE,
            IntentType.SAFETY_CONCERN,
            IntentType.EMERGENCY_HELP,
        ]
        return intent_info["intent_type"] in safety_relevant_intents

    def _calculate_comprehensive_score(
        self,
        context_evaluation: Dict,
        emotional_state: Dict,
        crisis_assessment: Dict,
    ) -> float:
        """Calculate comprehensive response score for monitoring."""
        score = 0.8  # Base score

        # Context quality boost
        score += context_evaluation.get("context_quality", 0) * 0.1

        # Empathy application boost
        if emotional_state.get("needs_validation"):
            score += 0.05

        # Crisis handling boost
        if crisis_assessment["crisis_level"] != "none":
            score += 0.05

        return min(score, 1.0)

    def _generate_error_response(
        self, intent_info: Dict[str, Any], language: str
    ) -> Dict[str, Any]:
        """Generate comprehensive error response."""
        error_response = """## DIRECT ANSWER
I'm experiencing a temporary technical issue that prevents me from providing my full response capabilities right now. This appears to be a system problem rather than an issue with your question.

## ADDITIONAL INFORMATION
While I resolve this technical difficulty, you can get immediate help with your international student settlement needs through several reliable channels:

**University Support:**
- Your university's international student services office (they have dedicated staff for settlement assistance)
- Student counseling services if you're feeling overwhelmed by the settlement process
- Academic advisors who can connect you with additional resources

**Official Kenyan Resources:**
- Department of Immigration Services at Nyayo House for visa and permit matters
- Kenya Association of Private Colleges for educational institution information
- County Government of Nairobi for local services and information

**Immediate Emergency Support:**
If you have urgent needs, contact:
- Universal Emergency Number: 999 (Police, Fire, Ambulance)
- Red Cross Kenya: 0700 395 395
- Your country's embassy or consulate in Nairobi

## NEXT STEPS
1. Try asking your question again in a few minutes - technical issues are usually temporary
2. Contact your university's international student office for immediate assistance with any urgent settlement needs
3. Save the emergency numbers listed above in your phone for future reference
4. Consider connecting with fellow international students through official university channels for peer support and advice

I apologize for this interruption and am designed to provide comprehensive settlement guidance to help make your transition to studying in Nairobi successful."""

        if language != "english":
            try:
                error_response = self.language_processor.translate_response(
                    error_response, language
                )
            except:
                pass

        return {
            "response": error_response,
            "intent_type": intent_info["intent_type"],
            "error": True,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    def validate_response_quality(
        self, response: str, intent_type: IntentType
    ) -> Dict[str, Any]:
        """Validate comprehensive response quality."""
        try:
            metrics = {
                "has_three_sections": False,
                "direct_answer_comprehensive": False,
                "additional_info_substantial": False,
                "next_steps_actionable": False,
                "contains_specific_details": False,
                "includes_safety_considerations": False,
                "shows_empathy": False,
                "provides_contacts": False,
            }

            response_lower = response.lower()

            # Check three-section structure
            required_sections = [
                "direct answer",
                "additional information",
                "next steps",
            ]
            metrics["has_three_sections"] = all(
                section in response_lower for section in required_sections
            )

            # Check direct answer comprehensiveness
            if "## DIRECT ANSWER" in response:
                direct_section = response.split("## ADDITIONAL INFORMATION")[0]
                direct_content = direct_section.split("## DIRECT ANSWER")[1]
                word_count = len(direct_content.split())
                metrics["direct_answer_comprehensive"] = word_count >= 50

            # Check additional information quality
            if "## ADDITIONAL INFORMATION" in response:
                additional_section = response.split("## NEXT STEPS")[0]
                if "## ADDITIONAL INFORMATION" in additional_section:
                    additional_content = additional_section.split(
                        "## ADDITIONAL INFORMATION"
                    )[1]
                    word_count = len(additional_content.split())
                    metrics["additional_info_substantial"] = word_count >= 80

            # Check next steps actionability
            action_indicators = [
                "contact",
                "visit",
                "call",
                "apply",
                "register",
                "go to",
                "reach out",
            ]
            metrics["next_steps_actionable"] = any(
                indicator in response_lower for indicator in action_indicators
            )

            # Check specific details
            detail_indicators = [
                "ksh",
                "phone",
                "address",
                "hours",
                "location",
                "cost",
                "price",
                "number",
            ]
            metrics["contains_specific_details"] = (
                sum(
                    1
                    for indicator in detail_indicators
                    if indicator in response_lower
                )
                >= 3
            )

            # Check safety considerations
            safety_indicators = [
                "safety",
                "secure",
                "avoid",
                "caution",
                "emergency",
                "precaution",
                "safe",
            ]
            metrics["includes_safety_considerations"] = any(
                indicator in response_lower for indicator in safety_indicators
            )

            # Check empathy
            empathy_indicators = [
                "understand",
                "normal",
                "common",
                "many students",
                "feel",
                "concern",
            ]
            metrics["shows_empathy"] = any(
                indicator in response_lower for indicator in empathy_indicators
            )

            # Check contacts
            contact_indicators = [
                "999",
                "0700",
                "0703",
                "+254",
                "phone",
                "call",
                "contact",
            ]
            metrics["provides_contacts"] = any(
                indicator in response_lower for indicator in contact_indicators
            )

            # Calculate overall quality
            quality_score = sum(metrics.values()) / len(metrics)

            return {
                "quality_score": quality_score,
                "metrics": metrics,
                "is_comprehensive": quality_score >= 0.75,
                "meets_100_standard": quality_score >= 0.85,
                "recommendations": self._get_quality_recommendations(metrics),
            }

        except Exception as e:
            logger.error(f"Response quality validation failed: {str(e)}")
            return {
                "quality_score": 0.5,
                "error": str(e),
                "is_comprehensive": False,
            }

    def _get_quality_recommendations(
        self, metrics: Dict[str, bool]
    ) -> List[str]:
        """Get recommendations for improving response quality."""
        recommendations = []

        if not metrics["has_three_sections"]:
            recommendations.append(
                "Ensure response has all three required sections"
            )
        if not metrics["direct_answer_comprehensive"]:
            recommendations.append(
                "Make direct answer more comprehensive and detailed"
            )
        if not metrics["additional_info_substantial"]:
            recommendations.append(
                "Expand additional information section with more context"
            )
        if not metrics["next_steps_actionable"]:
            recommendations.append(
                "Include more specific, actionable next steps"
            )
        if not metrics["contains_specific_details"]:
            recommendations.append(
                "Add specific details like costs, phone numbers, addresses"
            )
        if not metrics["includes_safety_considerations"]:
            recommendations.append(
                "Include relevant safety information and precautions"
            )
        if not metrics["shows_empathy"]:
            recommendations.append(
                "Add empathetic language that validates student concerns"
            )
        if not metrics["provides_contacts"]:
            recommendations.append(
                "Include specific contact information and resources"
            )

        return recommendations

    def get_response_stats(self) -> Dict[str, Any]:
        """Get comprehensive response generation statistics."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "settlement_optimized": True,
            "comprehensive_mode": True,
            "empathy_enabled": True,
            "safety_protocols_active": True,
            "crisis_detection_active": True,
            "web_search_enabled": True,
            "web_search_provider": "Tavily AI",
            "parallel_processing": True,
            "optimizations_applied": [
                "Parallel emotion detection and web search",
                "Fast emotion detection (keyword-based)",
                "gpt-4.1-mini for all operations",
                "Tavily search with basic depth",
                "Optimized token usage",
            ],
            "features": [
                "Three-section comprehensive structure",
                "Emotional state detection and empathy integration",
                "Crisis assessment and emergency response",
                "Systematic safety protocol integration",
                "Tavily web search for current information",
                "Settlement-specific knowledge enhancement",
                "Quality validation and improvement recommendations",
                "Multilingual support with quality validation",
                "Real-time context enhancement",
                "Comprehensive fallback responses",
                "Performance monitoring and scoring",
                "Parallel processing for speed",
            ],
            "target_performance": "100/100",
            "expected_response_time": "3-5 seconds",
            "empathy_categories": list(self.empathy_responses.keys()),
            "safety_categories": list(self.safety_protocols.keys()),
            "crisis_levels": ["none", "low", "medium", "high"],
        }
