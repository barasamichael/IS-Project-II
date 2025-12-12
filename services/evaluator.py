import json
import logging

from pathlib import Path
from datetime import datetime

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from tqdm import tqdm

from config.settings import ROOT_DIR

from services.intent_recognizer import IntentRecognizer
from services.response_generator import ResponseGenerator
from services.vector_db import VectorDBService

# Add BLEU score capability
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk

    nltk.download("punkt", quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logging.warning("NLTK not available. Install with: pip install nltk")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


class InternationalStudentRAGEvaluator:
    """Evaluation system specifically designed for international student assistant."""

    def __init__(
        self,
        vector_db_service: Optional[VectorDBService] = None,
        intent_recognizer: Optional[IntentRecognizer] = None,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        self.vector_db_service = vector_db_service or VectorDBService()
        self.intent_recognizer = intent_recognizer or IntentRecognizer()
        self.response_generator = response_generator or ResponseGenerator()

        self.eval_dir = ROOT_DIR / "tests" / "eval_data"
        if not self.eval_dir.exists():
            self.eval_dir.mkdir(parents=True)

    def calculate_bleu_score(self, candidate: str, reference: str) -> float:
        """Calculate BLEU-4 score for response evaluation."""
        if not BLEU_AVAILABLE:
            return 0.0

        try:
            # Tokenize
            candidate_tokens = candidate.lower().split()
            reference_tokens = reference.lower().split()

            # Calculate BLEU-4
            smoother = SmoothingFunction()
            score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                smoothing_function=smoother.method1,
            )
            return score
        except:
            return 0.0

    def create_international_student_eval_set(
        self, output_path: Optional[Path] = None
    ) -> Path:
        """Create evaluation set tailored for international students in Nairobi."""
        output_path = (
            output_path or self.eval_dir / "international_student_eval.csv"
        )

        # Comprehensive evaluation questions covering international student needs
        eval_questions = [
            # HOUSING QUERIES
            {
                "id": "house_001",
                "query": "What are safe and affordable housing options for international students in Nairobi?",
                "expected_intent": "housing_inquiry",
                "expected_topic": "housing",
                "expected_answer_contains": [
                    "Kilimani",
                    "Kileleshwa",
                    "Westlands",
                    "safe",
                    "affordable",
                    "students",
                ],
                "notes": "Should provide specific neighborhoods with safety and cost considerations",
                "priority": "high",
                "reference_response": "You can find safe and affordable student housing in several Nairobi neighborhoods. Kilimani is popular with students due to its proximity to universities and good security. Westlands offers modern apartments with security features. Kileleshwa is quieter and family-friendly. Expect to pay KSh 15,000-30,000 per month for a bedsitter or one-bedroom apartment in these areas. Always visit properties in person, verify landlord credentials, and ensure the area has good lighting and security presence.",
            },
            {
                "id": "house_002",
                "query": "How much does it cost to rent a one-bedroom apartment in Kilimani?",
                "expected_intent": "cost_inquiry",
                "expected_topic": "housing",
                "expected_answer_contains": [
                    "KES",
                    "rent",
                    "Kilimani",
                    "bedroom",
                    "cost",
                ],
                "notes": "Should provide cost ranges in KES",
                "priority": "high",
                "reference_response": "In Kilimani, a one-bedroom apartment typically costs between KSh 20,000 to KSh 40,000 per month depending on the specific location and amenities. Apartments closer to Yaya Centre or along Argwings Kodhek Road are more expensive. You'll also need to budget for a deposit (usually 1-2 months rent), utilities (KSh 3,000-5,000), and connection fees. Furnished apartments cost about 20-30% more than unfurnished ones.",
            },
            {
                "id": "house_003",
                "query": "I'm worried about finding safe accommodation as an international student. Any advice?",
                "expected_intent": "reassurance_seeking",
                "expected_topic": "housing",
                "expected_answer_contains": [
                    "safe",
                    "security",
                    "advice",
                    "students",
                    "precautions",
                ],
                "notes": "Should provide reassurance and practical safety tips",
                "priority": "high",
                "reference_response": "It's completely normal to feel concerned about accommodation safety as an international student. Here's how to find secure housing: Focus on established neighborhoods like Kilimani, Westlands, and Kileleshwa which have good security and are popular with students. Always visit properties during the day and evening to assess the area. Look for apartments with security guards, CCTV, and controlled access. Ask other international students for recommendations through university groups. Trust your instincts - if something feels wrong, continue searching.",
            },
            # UNIVERSITY QUERIES
            {
                "id": "univ_001",
                "query": "What documents do I need for university admission in Kenya?",
                "expected_intent": "procedural_query",
                "expected_topic": "education",
                "expected_answer_contains": [
                    "documents",
                    "admission",
                    "transcript",
                    "certificate",
                    "visa",
                ],
                "notes": "Should list specific required documents",
                "priority": "high",
                "reference_response": "For university admission in Kenya, you'll need: Academic transcripts and certificates from your previous education (officially translated if not in English), completed application form, passport-size photographs, copy of passport, student visa or permit, medical certificate, yellow fever vaccination certificate, and proof of financial support. Some universities may require additional documents like recommendation letters or a personal statement. Contact your specific university's international office for their exact requirements.",
            },
            {
                "id": "univ_002",
                "query": "Compare University of Nairobi and Strathmore University for international students",
                "expected_intent": "comparison_query",
                "expected_topic": "education",
                "expected_answer_contains": [
                    "University of Nairobi",
                    "Strathmore",
                    "compare",
                    "international",
                ],
                "notes": "Should provide balanced comparison",
                "priority": "medium",
                "reference_response": "University of Nairobi is Kenya's largest public university with lower tuition fees and diverse programs, but larger class sizes. It has strong research facilities and is well-established. Strathmore University is a private institution with smaller classes, modern facilities, and strong industry connections, but higher tuition fees. Strathmore tends to have better support services for international students and more flexible programs. Both are well-recognized, so choose based on your budget, preferred class size, and specific program requirements.",
            },
            # SAFETY QUERIES
            {
                "id": "safe_001",
                "query": "Is it safe to walk alone at night in Westlands?",
                "expected_intent": "safety_concern",
                "expected_topic": "safety",
                "expected_answer_contains": [
                    "Westlands",
                    "night",
                    "safe",
                    "precautions",
                    "avoid",
                ],
                "notes": "Should provide honest safety assessment with precautions",
                "priority": "critical",
                "reference_response": "Westlands is generally safer than many Nairobi areas, but walking alone at night isn't recommended anywhere in the city. If you must walk at night in Westlands, stick to well-lit main roads like Waiyaki Way and areas near shopping centers. Avoid quiet side streets and parks after dark. Use ride-sharing apps like Uber or Bolt instead of walking. If walking is unavoidable, stay alert, don't use your phone openly, and walk confidently. The area near Sarit Centre and Westgate Mall has good security presence.",
            },
            {
                "id": "safe_002",
                "query": "Emergency contacts for international students in Nairobi",
                "expected_intent": "emergency_help",
                "expected_topic": "safety",
                "expected_answer_contains": [
                    "emergency",
                    "999",
                    "police",
                    "hospital",
                    "embassy",
                ],
                "notes": "Should provide comprehensive emergency contact list",
                "priority": "critical",
                "reference_response": "Essential emergency contacts: Police, Fire, Ambulance: 999 (universal emergency number), Tourist Police: +254 20 2729992, Red Cross: 0700 395 395, AA Kenya (roadside assistance): 0700 200 007. Major hospitals: Nairobi Hospital (+254 20 2845000), Aga Khan Hospital (+254 20 3740000), MP Shah Hospital (+254 20 3742763). Contact your country's embassy for assistance with passport/visa issues. Save these numbers in your phone and keep written copies in your accommodation.",
            },
            # TRANSPORTATION QUERIES
            {
                "id": "trans_001",
                "query": "How to get from JKIA airport to Westlands safely?",
                "expected_intent": "transportation",
                "expected_topic": "transport",
                "expected_answer_contains": [
                    "JKIA",
                    "airport",
                    "Westlands",
                    "taxi",
                    "uber",
                    "safe",
                ],
                "notes": "Should provide multiple transport options with safety tips",
                "priority": "high",
                "reference_response": "From JKIA to Westlands, use these safe options: Official airport taxis (KSh 2,500-3,500), Uber/Bolt (KSh 1,500-2,500), or airport shuttle services (KSh 1,000-2,000). Avoid unofficial taxis outside the terminal. Book Uber/Bolt from inside the airport using the designated pickup points. The journey takes 45-90 minutes depending on traffic. Keep your luggage secure and don't display expensive items. If taking a taxi, ensure the meter is working or agree on the fare beforehand.",
            },
            {
                "id": "trans_002",
                "query": "What is the cost of transport around Nairobi?",
                "expected_intent": "cost_inquiry",
                "expected_topic": "transport",
                "expected_answer_contains": [
                    "cost",
                    "transport",
                    "matatu",
                    "uber",
                    "KES",
                ],
                "notes": "Should provide cost ranges for different transport modes",
                "priority": "medium",
                "reference_response": "Nairobi transport costs vary by mode: Matatus (public buses): KSh 20-100 for most city routes, Uber/Bolt: KSh 150-800 depending on distance and time, Regular taxis: KSh 300-1,500, Boda boda (motorcycle): KSh 100-500, City buses: KSh 30-70. Budget KSh 200-500 daily for transport if using a mix of options. Matatus are cheapest but less comfortable. Uber/Bolt are safest for students, especially at night.",
            },
            # BANKING & FINANCE QUERIES
            {
                "id": "bank_001",
                "query": "How do I open a bank account as an international student in Kenya?",
                "expected_intent": "procedural_query",
                "expected_topic": "finance",
                "expected_answer_contains": [
                    "bank account",
                    "international student",
                    "documents",
                    "requirements",
                ],
                "notes": "Should provide step-by-step process",
                "priority": "high",
                "reference_response": "To open a bank account in Kenya: Visit a bank branch with your passport, student visa/permit, admission letter from your university, passport photos, and initial deposit (usually KSh 1,000-5,000). Popular student-friendly banks include KCB, Equity, and Cooperative Bank. You'll need proof of residence (utility bill or tenancy agreement). The account opening process takes 30-60 minutes. Consider banks with campus branches for convenience. Many banks offer student accounts with reduced fees.",
            },
            {
                "id": "bank_002",
                "query": "What is M-Pesa and how do I use it?",
                "expected_intent": "explanation_query",
                "expected_topic": "finance",
                "expected_answer_contains": [
                    "M-Pesa",
                    "mobile money",
                    "Kenya",
                    "how to use",
                ],
                "notes": "Should explain M-Pesa system clearly",
                "priority": "medium",
                "reference_response": "M-Pesa is Kenya's mobile money system that lets you send, receive, and store money using your phone. To start: Get a Safaricom SIM card, visit any M-Pesa agent with your ID/passport, register for the service (free), deposit money with an agent. You can then send money to others, pay bills, buy airtime, and withdraw cash at thousands of agents nationwide. Transaction fees range from KSh 1-100 depending on amount. It's widely accepted and very convenient for daily transactions in Kenya.",
            },
            # VISA & IMMIGRATION QUERIES
            {
                "id": "visa_001",
                "query": "How do I renew my student visa in Kenya?",
                "expected_intent": "procedural_query",
                "expected_topic": "legal",
                "expected_answer_contains": [
                    "student visa",
                    "renewal",
                    "immigration",
                    "documents",
                    "process",
                ],
                "notes": "Should provide renewal procedure",
                "priority": "high",
                "reference_response": "To renew your student visa: Apply 60 days before expiry at the Immigration Department (Nyayo House, Uhuru Highway). Required documents: completed application form, current passport, current visa, letter from your university confirming enrollment, fee payment receipt (KSh 10,000 for renewal), medical certificate if required. Processing takes 2-4 weeks. Apply early to avoid overstaying penalties. Keep copies of all documents and your receipt.",
            },
            # CULTURAL ADAPTATION QUERIES
            {
                "id": "cult_001",
                "query": "What should I know about Kenyan culture and etiquette?",
                "expected_intent": "cultural_adaptation",
                "expected_topic": "culture",
                "expected_answer_contains": [
                    "culture",
                    "etiquette",
                    "Kenya",
                    "respect",
                    "customs",
                ],
                "notes": "Should provide cultural guidance",
                "priority": "medium",
                "reference_response": "Kenyan cultural etiquette: Greet people warmly - handshakes are common, use both hands when giving or receiving items, dress modestly especially in rural areas and religious sites, respect elders and address them formally, remove shoes when entering homes, avoid pointing with fingers (use open hand), be patient as 'Kenyan time' can mean events start later, learn basic Swahili greetings (Jambo, Asante), bargaining is acceptable in markets but not in shops, and avoid discussing politics unless invited. Kenyans are generally friendly and helpful to visitors.",
            },
            # HEALTHCARE QUERIES
            {
                "id": "health_001",
                "query": "Where can I find good healthcare facilities in Nairobi?",
                "expected_intent": "healthcare",
                "expected_topic": "health",
                "expected_answer_contains": [
                    "healthcare",
                    "hospital",
                    "clinic",
                    "Nairobi",
                    "medical",
                ],
                "notes": "Should list major hospitals and clinics",
                "priority": "medium",
                "reference_response": "Top healthcare facilities in Nairobi: The Nairobi Hospital (Argwings Kodhek Road) - comprehensive services, Aga Khan Hospital (Parklands) - high-quality private care, MP Shah Hospital (Parklands) - good general services, Kenyatta National Hospital (Hospital Road) - largest public hospital, Karen Hospital (Karen) - modern private facility. For routine care, consider clinics like Meridian Medical Centre or AAR Healthcare. Many accept international insurance. Always carry your insurance documents and emergency contacts.",
            },
            # ENTERTAINMENT & SOCIAL QUERIES
            {
                "id": "social_001",
                "query": "What are good places for international students to socialize in Nairobi?",
                "expected_intent": "entertainment_social",
                "expected_topic": "lifestyle",
                "expected_answer_contains": [
                    "socialize",
                    "students",
                    "places",
                    "meet",
                    "friends",
                ],
                "notes": "Should suggest social venues and activities",
                "priority": "low",
                "reference_response": "Great places for students to socialize: Westgate Mall and Sarit Centre (shopping and dining), Karura Forest (walking and picnics), The Junction and Two Rivers Mall (entertainment and restaurants), university sports clubs and societies, international student groups on Facebook, coffee shops in Westlands and Kilimani, Uhuru Park (events and relaxation), National Museum and cultural centers, co-working spaces like iHub or Nailab, and weekend markets like Maasai Market. Join university orientation programs and student associations to meet people with similar interests.",
            },
            # SHOPPING QUERIES
            {
                "id": "shop_001",
                "query": "Where can I buy affordable groceries and household items?",
                "expected_intent": "shopping_markets",
                "expected_topic": "shopping",
                "expected_answer_contains": [
                    "groceries",
                    "affordable",
                    "shopping",
                    "market",
                    "supermarket",
                ],
                "notes": "Should suggest budget-friendly shopping options",
                "priority": "medium",
                "reference_response": "Affordable shopping options: Naivas, Quickmart, and Cleanshelf supermarkets for groceries, City Market (downtown) for fresh produce and bargains, local markets in your neighborhood for vegetables and fruits, Gikomba Market for household items and clothes (bargain carefully), Tuskys and Choppies supermarkets for budget options. Shop at local dukas (small shops) for daily needs. Compare prices between supermarkets and markets - markets are usually cheaper but require bargaining. Buy in bulk when possible to save money.",
            },
            # ACADEMIC CONVERSION QUERIES
            {
                "id": "acad_001",
                "query": "How do I convert my GPA to the Kenyan grading system?",
                "expected_intent": "academic_conversion",
                "expected_topic": "academics",
                "expected_answer_contains": [
                    "GPA",
                    "convert",
                    "grading system",
                    "Kenya",
                    "equivalent",
                ],
                "notes": "Should explain grade conversion process",
                "priority": "medium",
                "reference_response": "Kenyan universities typically use different grading systems. Most use: First Class (70-100%), Second Class Upper (60-69%), Second Class Lower (50-59%), Pass (40-49%). For US GPA conversion: 4.0 GPA ≈ First Class, 3.5-3.9 ≈ Second Upper, 3.0-3.4 ≈ Second Lower, 2.5-2.9 ≈ Pass. However, each university has specific conversion formulas. Contact your university's registrar or international office for official conversion as requirements vary between institutions. Some universities may require credential evaluation from recognized agencies.",
            },
            # REASSURANCE & SUPPORT QUERIES
            {
                "id": "support_001",
                "query": "I'm feeling overwhelmed as a new international student. Is this normal?",
                "expected_intent": "reassurance_seeking",
                "expected_topic": "general",
                "expected_answer_contains": [
                    "normal",
                    "overwhelmed",
                    "support",
                    "adaptation",
                    "help",
                ],
                "notes": "Should provide emotional support and reassurance",
                "priority": "high",
                "reference_response": "Feeling overwhelmed as a new international student is completely normal and very common. You're adjusting to a new country, culture, education system, and being away from home - that's a lot of change at once. Most students experience this during their first few months. It gets easier as you settle in and make friends. Reach out to your university's international student office for support, join student groups to meet others in similar situations, maintain contact with family and friends back home, establish routines, and don't hesitate to seek counseling services if available. Remember, this adjustment period is temporary, and thousands of students have successfully navigated this same experience.",
            },
            # LOCATION-SPECIFIC QUERIES
            {
                "id": "loc_001",
                "query": "Tell me about living in Karen area as a student",
                "expected_intent": "neighborhood_guide",
                "expected_topic": "location",
                "expected_answer_contains": [
                    "Karen",
                    "living",
                    "area",
                    "students",
                    "amenities",
                ],
                "notes": "Should provide comprehensive area information",
                "priority": "medium",
                "reference_response": "Karen is an upscale suburb popular with expatriates and middle-class families. Pros: quiet and leafy environment, good security, Karen Hospital nearby, shopping at Karen Waterfront and The Hub, close to Wilson Airport. Cons: higher rent (KSh 25,000-60,000 for student housing), farther from most universities (30-45 minutes to city center), limited nightlife, fewer public transport options. Best for students who prioritize quiet study environment and safety over proximity to campus. Consider if you have reliable transport or can afford regular taxi rides to university.",
            },
        ]

        # Save as CSV
        df = pd.DataFrame(eval_questions)
        df.to_csv(output_path, index=False)

        logger.info(
            f"Created international student evaluation set with {len(eval_questions)} questions at {output_path}"
        )
        return output_path

    def run_comprehensive_evaluation(
        self,
        eval_file: Optional[Path] = None,
        output_path: Optional[Path] = None,
        include_bleu: bool = True,
        reference_responses: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation on international student assistant."""
        eval_file = (
            eval_file or self.eval_dir / "international_student_eval.csv"
        )
        output_path = output_path or self.eval_dir / "evaluation_results.json"

        if not eval_file.exists():
            logger.warning(f"Evaluation file not found: {eval_file}")
            logger.info("Creating international student evaluation set...")
            eval_file = self.create_international_student_eval_set()

        try:
            eval_df = pd.read_csv(eval_file)
            logger.info(
                f"Loaded {len(eval_df)} evaluation questions from {eval_file}"
            )
        except Exception as e:
            logger.error(f"Error loading evaluation file: {str(e)}")
            return {
                "status": "error",
                "message": f"Error loading evaluation file: {str(e)}",
            }

        # Extract reference responses from eval data if not provided
        if include_bleu and reference_responses is None:
            reference_responses = {}
            for _, row in eval_df.iterrows():
                if "reference_response" in row and pd.notna(
                    row["reference_response"]
                ):
                    reference_responses[row["query"]] = row[
                        "reference_response"
                    ]

        # Run evaluation on each question
        results = []
        failed_queries = 0

        for _, row in tqdm(
            eval_df.iterrows(),
            total=len(eval_df),
            desc="Evaluating International Student Assistant",
        ):
            try:
                query = row["query"]

                # Recognize intent
                intent_info = self.intent_recognizer.recognize_intent(query)

                # Retrieve relevant context with student-focused search
                retrieved_chunks = []
                if intent_info["intent_type"] != "off_topic":
                    retrieved_chunks = (
                        self.vector_db_service.search_for_students(
                            query=query, top_k=5, prioritize_practical=True
                        )
                        if hasattr(
                            self.vector_db_service, "search_for_students"
                        )
                        else self.vector_db_service.search(query=query, top_k=5)
                    )

                # Generate response
                response_data = self.response_generator.generate_response(
                    query=query,
                    retrieved_context=retrieved_chunks,
                    intent_info=intent_info,
                )

                # Calculate BLEU score if enabled and reference available
                bleu_score = 0.0
                if (
                    include_bleu
                    and reference_responses
                    and query in reference_responses
                ):
                    bleu_score = self.calculate_bleu_score(
                        response_data["response"], reference_responses[query]
                    )

                # Evaluate response quality
                evaluation_metrics = self._evaluate_response_quality(
                    query, response_data, row, retrieved_chunks
                )

                result = {
                    "id": row.get("id", ""),
                    "query": query,
                    "response": response_data["response"],
                    "expected_intent": row.get("expected_intent", ""),
                    "actual_intent": intent_info["intent_type"].value,
                    "intent_match": evaluation_metrics["intent_match"],
                    "expected_topic": row.get("expected_topic", ""),
                    "actual_topic": intent_info["topic"].value,
                    "topic_match": evaluation_metrics["topic_match"],
                    "expected_answer_contains": evaluation_metrics[
                        "expected_contains"
                    ],
                    "contains_expected": evaluation_metrics[
                        "contains_expected"
                    ],
                    "student_relevance_score": evaluation_metrics[
                        "student_relevance_score"
                    ],
                    "practical_info_score": evaluation_metrics[
                        "practical_info_score"
                    ],
                    "empathy_score": evaluation_metrics["empathy_score"],
                    "overall_score": evaluation_metrics["overall_score"],
                    "bleu_score": bleu_score,
                    "priority": row.get("priority", "medium"),
                    "token_usage": response_data.get("token_usage", {}),
                    "urgency": intent_info.get("urgency", "low"),
                    "chunks_retrieved": len(retrieved_chunks),
                    "avg_chunk_relevance": sum(
                        c.get("score", 0) for c in retrieved_chunks
                    )
                    / len(retrieved_chunks)
                    if retrieved_chunks
                    else 0,
                }

                results.append(result)

            except Exception as e:
                logger.error(
                    f"Error evaluating query '{row.get('query', '')}': {str(e)}"
                )
                failed_queries += 1
                results.append(
                    {
                        "id": row.get("id", ""),
                        "query": row.get("query", ""),
                        "error": str(e),
                        "overall_score": 0,
                        "bleu_score": 0,
                        "priority": row.get("priority", "medium"),
                    }
                )

        # Calculate comprehensive metrics
        evaluation_report = self._generate_evaluation_report(
            results, failed_queries, include_bleu=include_bleu
        )

        # Save results
        with open(output_path, "w") as f:
            json.dump(evaluation_report, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {output_path}")
        logger.info(
            f"Overall Score: {evaluation_report['overall_metrics']['average_score']:.3f}"
        )
        if include_bleu and "bleu_metrics" in evaluation_report:
            logger.info(
                f"Average BLEU Score: {evaluation_report['bleu_metrics']['average_bleu']:.3f}"
            )
        logger.info(
            f"High Priority Score: {evaluation_report['priority_metrics'].get('critical', {}).get('avg_score', 0):.3f}"
        )

        return evaluation_report

    def _evaluate_response_quality(
        self,
        query: str,
        response_data: Dict[str, Any],
        expected: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate response quality with international student-specific criteria."""

        response_text = response_data.get("response", "")

        # Intent matching
        expected_intent = expected.get("expected_intent", "")
        actual_intent = (
            response_data.get("intent_type", "").value
            if hasattr(response_data.get("intent_type", ""), "value")
            else str(response_data.get("intent_type", ""))
        )
        intent_match = expected_intent == actual_intent

        # Topic matching
        expected_topic = expected.get("expected_topic", "")
        actual_topic = (
            response_data.get("topic", "").value
            if hasattr(response_data.get("topic", ""), "value")
            else str(response_data.get("topic", ""))
        )
        topic_match = expected_topic == actual_topic

        # Content matching
        expected_contains = expected.get("expected_answer_contains", [])
        if isinstance(expected_contains, str):
            try:
                expected_contains = eval(expected_contains)
            except:
                expected_contains = [expected_contains]

        contains_expected = (
            all(
                term.lower() in response_text.lower()
                for term in expected_contains
            )
            if expected_contains
            else True
        )

        # International student specific scoring
        student_relevance_score = self._calculate_student_relevance_score(
            response_text, query
        )
        practical_info_score = self._calculate_practical_info_score(
            response_text, retrieved_chunks
        )
        empathy_score = self._calculate_empathy_score(
            response_text, response_data.get("intent_type", "")
        )

        # Calculate overall score
        base_score = 0.0
        if intent_match:
            base_score += 0.2
        if topic_match:
            base_score += 0.2
        if contains_expected:
            base_score += 0.2

        # Add student-specific scoring components
        base_score += student_relevance_score * 0.15
        base_score += practical_info_score * 0.15
        base_score += empathy_score * 0.1

        # Priority boost for critical queries
        priority = expected.get("priority", "medium")
        if priority == "critical" and (contains_expected and intent_match):
            base_score += 0.05

        overall_score = min(base_score, 1.0)

        return {
            "intent_match": intent_match,
            "topic_match": topic_match,
            "expected_contains": expected_contains,
            "contains_expected": contains_expected,
            "student_relevance_score": student_relevance_score,
            "practical_info_score": practical_info_score,
            "empathy_score": empathy_score,
            "overall_score": overall_score,
        }

    def _calculate_student_relevance_score(
        self, response_text: str, query: str
    ) -> float:
        """Calculate how relevant the response is to international student needs."""
        response_lower = response_text.lower()

        # International student context indicators
        student_terms = [
            "international student",
            "foreign student",
            "student",
            "study abroad",
            "international",
            "foreign",
            "new to kenya",
            "new to nairobi",
        ]

        # Practical guidance indicators
        practical_terms = [
            "cost",
            "price",
            "budget",
            "affordable",
            "safe",
            "safety",
            "how to",
            "where to",
            "documents",
            "requirements",
            "process",
            "steps",
            "tip",
        ]

        # Location-specific indicators
        location_terms = [
            "nairobi",
            "kenya",
            "kilimani",
            "westlands",
            "karen",
            "eastleigh",
            "cbd",
            "upperhill",
            "parklands",
            "kileleshwa",
        ]

        score = 0.0

        # Score based on student-specific content
        student_mentions = sum(
            1 for term in student_terms if term in response_lower
        )
        score += min(student_mentions / len(student_terms), 0.4)

        # Score based on practical information
        practical_mentions = sum(
            1 for term in practical_terms if term in response_lower
        )
        score += min(practical_mentions / len(practical_terms), 0.4)

        # Score based on location specificity
        location_mentions = sum(
            1 for term in location_terms if term in response_lower
        )
        score += min(location_mentions / len(location_terms), 0.2)

        return min(score, 1.0)

    def _calculate_practical_info_score(
        self, response_text: str, retrieved_chunks: List[Dict]
    ) -> float:
        """Calculate how much practical, actionable information is provided."""
        response_lower = response_text.lower()

        score = 0.0

        # Check for specific practical elements
        practical_elements = {
            "costs": ["kes", "ksh", "shilling", "cost", "price", "fee"],
            "locations": [
                "address",
                "located",
                "area",
                "district",
                "road",
                "avenue",
            ],
            "contacts": ["phone", "email", "contact", "call", "reach"],
            "procedures": [
                "step",
                "process",
                "procedure",
                "how to",
                "apply",
                "register",
            ],
            "timing": [
                "hours",
                "time",
                "schedule",
                "open",
                "close",
                "deadline",
            ],
            "safety": ["safe", "avoid", "careful", "precaution", "security"],
        }

        for category, terms in practical_elements.items():
            if any(term in response_lower for term in terms):
                score += 0.15

        # Bonus for structured information (lists, numbered steps)
        if any(
            pattern in response_text
            for pattern in ["1.", "2.", "•", "-", "Step"]
        ):
            score += 0.1

        # Check if retrieved chunks had practical info
        practical_chunks = sum(
            1
            for chunk in retrieved_chunks
            if chunk.get("cost_mentioned") or chunk.get("safety_mentioned")
        )
        if practical_chunks > 0:
            score += 0.1

        return min(score, 1.0)

    def _calculate_empathy_score(
        self, response_text: str, intent_type: str
    ) -> float:
        """Calculate empathy and supportiveness of the response."""
        response_lower = response_text.lower()

        # Empathetic language indicators
        empathy_terms = [
            "understand",
            "know how you feel",
            "normal",
            "common",
            "don't worry",
            "help you",
            "support",
            "here for you",
            "not alone",
            "many students",
            "it's okay",
            "take time",
            "gradually",
            "step by step",
        ]

        # Reassuring language
        reassurance_terms = [
            "will be fine",
            "get better",
            "adapt",
            "settle in",
            "comfortable",
            "easier",
            "manageable",
            "doable",
            "possible",
            "achievable",
        ]

        # Harsh or dismissive language (negative scoring)
        harsh_terms = [
            "just",
            "simply",
            "obviously",
            "clearly",
            "of course",
            "everyone knows",
        ]

        score = 0.0

        # Positive scoring for empathy
        empathy_count = sum(
            1 for term in empathy_terms if term in response_lower
        )
        score += min(empathy_count / len(empathy_terms), 0.5)

        reassurance_count = sum(
            1 for term in reassurance_terms if term in response_lower
        )
        score += min(reassurance_count / len(reassurance_terms), 0.3)

        # Special boost for reassurance-seeking queries
        if "reassurance" in str(intent_type).lower():
            if any(
                term in response_lower
                for term in empathy_terms + reassurance_terms
            ):
                score += 0.2

        # Negative scoring for harsh language
        harsh_count = sum(1 for term in harsh_terms if term in response_lower)
        score -= harsh_count * 0.1

        return max(min(score, 1.0), 0.0)

    def _generate_evaluation_report(
        self,
        results: List[Dict],
        failed_queries: int,
        include_bleu: bool = False,
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""

        # Filter out failed results for metric calculations
        successful_results = [r for r in results if "error" not in r]

        if not successful_results:
            return {"status": "error", "message": "No successful evaluations"}

        # Overall metrics
        overall_metrics = {
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "failed_queries": failed_queries,
            "success_rate": len(successful_results) / len(results),
            "average_score": sum(
                r.get("overall_score", 0) for r in successful_results
            )
            / len(successful_results),
            "intent_accuracy": sum(
                1 for r in successful_results if r.get("intent_match", False)
            )
            / len(successful_results),
            "topic_accuracy": sum(
                1 for r in successful_results if r.get("topic_match", False)
            )
            / len(successful_results),
            "content_accuracy": sum(
                1
                for r in successful_results
                if r.get("contains_expected", False)
            )
            / len(successful_results),
        }

        # BLEU metrics if enabled
        if include_bleu:
            bleu_scores = [
                r.get("bleu_score", 0)
                for r in successful_results
                if r.get("bleu_score", 0) > 0
            ]
            overall_metrics["bleu_metrics"] = {
                "average_bleu": sum(bleu_scores) / len(bleu_scores)
                if bleu_scores
                else 0,
                "bleu_scores_available": len(bleu_scores),
                "high_bleu_responses": sum(
                    1 for score in bleu_scores if score > 0.3
                ),
                "max_bleu": max(bleu_scores) if bleu_scores else 0,
                "min_bleu": min(bleu_scores) if bleu_scores else 0,
            }

        # Student-specific metrics
        student_metrics = {
            "avg_student_relevance": sum(
                r.get("student_relevance_score", 0) for r in successful_results
            )
            / len(successful_results),
            "avg_practical_info_score": sum(
                r.get("practical_info_score", 0) for r in successful_results
            )
            / len(successful_results),
            "avg_empathy_score": sum(
                r.get("empathy_score", 0) for r in successful_results
            )
            / len(successful_results),
            "high_student_relevance_count": sum(
                1
                for r in successful_results
                if r.get("student_relevance_score", 0) > 0.7
            ),
            "high_empathy_count": sum(
                1 for r in successful_results if r.get("empathy_score", 0) > 0.7
            ),
        }

        # Priority-based metrics
        priority_groups = {}
        for priority in ["critical", "high", "medium", "low"]:
            priority_results = [
                r for r in successful_results if r.get("priority") == priority
            ]
            if priority_results:
                priority_groups[priority] = {
                    "count": len(priority_results),
                    "avg_score": sum(
                        r.get("overall_score", 0) for r in priority_results
                    )
                    / len(priority_results),
                    "intent_accuracy": sum(
                        1
                        for r in priority_results
                        if r.get("intent_match", False)
                    )
                    / len(priority_results),
                    "avg_student_relevance": sum(
                        r.get("student_relevance_score", 0)
                        for r in priority_results
                    )
                    / len(priority_results),
                }

                # Add BLEU metrics for priority groups if enabled
                if include_bleu:
                    priority_bleu = [
                        r.get("bleu_score", 0)
                        for r in priority_results
                        if r.get("bleu_score", 0) > 0
                    ]
                    priority_groups[priority]["avg_bleu"] = (
                        sum(priority_bleu) / len(priority_bleu)
                        if priority_bleu
                        else 0
                    )

        # Intent type performance
        intent_performance = {}
        for result in successful_results:
            intent = result.get("actual_intent", "unknown")
            if intent not in intent_performance:
                intent_performance[intent] = {
                    "count": 0,
                    "total_score": 0,
                    "correct_intent": 0,
                    "bleu_scores": [],
                }

            intent_performance[intent]["count"] += 1
            intent_performance[intent]["total_score"] += result.get(
                "overall_score", 0
            )
            if result.get("intent_match", False):
                intent_performance[intent]["correct_intent"] += 1

            # Add BLEU scores if available
            if include_bleu and result.get("bleu_score", 0) > 0:
                intent_performance[intent]["bleu_scores"].append(
                    result.get("bleu_score", 0)
                )

        # Calculate averages for intent performance
        for intent, stats in intent_performance.items():
            if stats["count"] > 0:
                stats["avg_score"] = stats["total_score"] / stats["count"]
                stats["intent_accuracy"] = (
                    stats["correct_intent"] / stats["count"]
                )
                if include_bleu and stats["bleu_scores"]:
                    stats["avg_bleu"] = sum(stats["bleu_scores"]) / len(
                        stats["bleu_scores"]
                    )

        # Token usage analysis
        token_stats = {
            "total_tokens": sum(
                r.get("token_usage", {}).get("total_tokens", 0)
                for r in successful_results
            ),
            "avg_tokens_per_query": sum(
                r.get("token_usage", {}).get("total_tokens", 0)
                for r in successful_results
            )
            / len(successful_results)
            if successful_results
            else 0,
            "avg_prompt_tokens": sum(
                r.get("token_usage", {}).get("prompt_tokens", 0)
                for r in successful_results
            )
            / len(successful_results)
            if successful_results
            else 0,
            "avg_completion_tokens": sum(
                r.get("token_usage", {}).get("completion_tokens", 0)
                for r in successful_results
            )
            / len(successful_results)
            if successful_results
            else 0,
        }

        # Retrieval analysis
        retrieval_stats = {
            "avg_chunks_retrieved": sum(
                r.get("chunks_retrieved", 0) for r in successful_results
            )
            / len(successful_results),
            "avg_chunk_relevance": sum(
                r.get("avg_chunk_relevance", 0) for r in successful_results
            )
            / len(successful_results),
            "queries_with_no_chunks": sum(
                1
                for r in successful_results
                if r.get("chunks_retrieved", 0) == 0
            ),
        }

        # Quality thresholds
        quality_analysis = {
            "excellent_responses": sum(
                1
                for r in successful_results
                if r.get("overall_score", 0) >= 0.8
            ),
            "good_responses": sum(
                1
                for r in successful_results
                if 0.6 <= r.get("overall_score", 0) < 0.8
            ),
            "poor_responses": sum(
                1 for r in successful_results if r.get("overall_score", 0) < 0.4
            ),
        }

        # Recommendations based on performance
        recommendations = self._generate_recommendations(
            overall_metrics,
            student_metrics,
            priority_groups,
            intent_performance,
        )

        return {
            "status": "success",
            "evaluation_summary": {
                "total_queries_evaluated": len(results),
                "overall_system_score": overall_metrics["average_score"],
                "key_strengths": self._identify_strengths(
                    overall_metrics, student_metrics, intent_performance
                ),
                "key_weaknesses": self._identify_weaknesses(
                    overall_metrics, student_metrics, intent_performance
                ),
            },
            "overall_metrics": overall_metrics,
            "student_specific_metrics": student_metrics,
            "priority_metrics": priority_groups,
            "intent_performance": intent_performance,
            "token_usage": token_stats,
            "retrieval_analysis": retrieval_stats,
            "quality_distribution": quality_analysis,
            "recommendations": recommendations,
            "detailed_results": results,
        }

    def _identify_strengths(
        self,
        overall_metrics: Dict,
        student_metrics: Dict,
        intent_performance: Dict,
    ) -> List[str]:
        """Identify system strengths based on metrics."""
        strengths = []

        if overall_metrics["average_score"] >= 0.75:
            strengths.append("High overall response quality")

        if overall_metrics["intent_accuracy"] >= 0.8:
            strengths.append("Excellent intent recognition accuracy")

        if student_metrics["avg_student_relevance"] >= 0.7:
            strengths.append(
                "Strong international student context understanding"
            )

        if student_metrics["avg_empathy_score"] >= 0.6:
            strengths.append("Good empathetic and supportive responses")

        if student_metrics["avg_practical_info_score"] >= 0.6:
            strengths.append("Effective delivery of practical information")

        # Check for strong intent categories
        strong_intents = [
            intent
            for intent, stats in intent_performance.items()
            if stats.get("avg_score", 0) >= 0.8 and stats.get("count", 0) >= 2
        ]
        if strong_intents:
            strengths.append(
                f"Excellent performance on: {', '.join(strong_intents)}"
            )

        # Check BLEU performance if available
        if "bleu_metrics" in overall_metrics:
            if overall_metrics["bleu_metrics"]["average_bleu"] >= 0.3:
                strengths.append(
                    "Good response similarity to reference answers"
                )

        return strengths or ["System shows basic functionality"]

    def _identify_weaknesses(
        self,
        overall_metrics: Dict,
        student_metrics: Dict,
        intent_performance: Dict,
    ) -> List[str]:
        """Identify system weaknesses based on metrics."""
        weaknesses = []

        if overall_metrics["average_score"] < 0.5:
            weaknesses.append("Low overall response quality needs improvement")

        if overall_metrics["intent_accuracy"] < 0.6:
            weaknesses.append("Intent recognition accuracy needs improvement")

        if overall_metrics["content_accuracy"] < 0.6:
            weaknesses.append(
                "Response content doesn't adequately match expected answers"
            )

        if student_metrics["avg_student_relevance"] < 0.5:
            weaknesses.append(
                "Insufficient focus on international student needs"
            )

        if student_metrics["avg_empathy_score"] < 0.4:
            weaknesses.append("Responses lack empathy and emotional support")

        if student_metrics["avg_practical_info_score"] < 0.5:
            weaknesses.append("Insufficient practical, actionable information")

        # Check for weak intent categories
        weak_intents = [
            intent
            for intent, stats in intent_performance.items()
            if stats.get("avg_score", 0) < 0.5 and stats.get("count", 0) >= 2
        ]
        if weak_intents:
            weaknesses.append(f"Poor performance on: {', '.join(weak_intents)}")

        # Check BLEU performance if available
        if "bleu_metrics" in overall_metrics:
            if overall_metrics["bleu_metrics"]["average_bleu"] < 0.2:
                weaknesses.append(
                    "Low response similarity to reference answers"
                )

        return weaknesses or ["No significant weaknesses identified"]

    def _generate_recommendations(
        self,
        overall_metrics: Dict,
        student_metrics: Dict,
        priority_groups: Dict,
        intent_performance: Dict,
    ) -> List[str]:
        """Generate actionable recommendations based on evaluation results."""
        recommendations = []

        # Overall performance recommendations
        if overall_metrics["average_score"] < 0.6:
            recommendations.append(
                "Consider retraining or fine-tuning the response generation model"
            )

        if overall_metrics["intent_accuracy"] < 0.7:
            recommendations.append(
                "Improve intent recognition patterns, especially for international student contexts"
            )

        # Student-specific recommendations
        if student_metrics["avg_student_relevance"] < 0.6:
            recommendations.append(
                "Enhance training data with more international student-specific content"
            )

        if student_metrics["avg_empathy_score"] < 0.5:
            recommendations.append(
                "Improve response templates to include more empathetic and supportive language"
            )

        if student_metrics["avg_practical_info_score"] < 0.6:
            recommendations.append(
                "Ensure knowledge base contains more practical, actionable information"
            )

        # Priority-specific recommendations
        critical_performance = priority_groups.get("critical", {}).get(
            "avg_score", 1.0
        )
        if critical_performance < 0.8:
            recommendations.append(
                "URGENT: Improve handling of critical queries (safety, emergencies)"
            )

        # Intent-specific recommendations
        poor_intents = [
            intent
            for intent, stats in intent_performance.items()
            if stats.get("avg_score", 0) < 0.5
        ]
        if poor_intents:
            recommendations.append(
                f"Focus improvement efforts on these intent types: {', '.join(poor_intents)}"
            )

        # BLEU-specific recommendations
        if "bleu_metrics" in overall_metrics:
            if overall_metrics["bleu_metrics"]["average_bleu"] < 0.2:
                recommendations.append(
                    "Improve response consistency and alignment with expected answers"
                )

        # Knowledge base recommendations
        recommendations.extend(
            [
                "Regularly update cost information and exchange rates",
                "Ensure safety information reflects current conditions",
                "Add more location-specific details for popular student areas",
                "Include more peer experiences and testimonials",
            ]
        )

        return recommendations

    def run_focused_evaluation(
        self,
        focus_area: str,
        num_queries: int = 10,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run focused evaluation on specific areas (housing, safety, etc.)."""

        focus_queries = {
            "housing": [
                "Where should international students live in Nairobi?",
                "How much does student accommodation cost?",
                "Is it safe to live in Kilimani as a student?",
                "What should I look for when renting an apartment?",
                "Are there student hostels near universities?",
            ],
            "safety": [
                "Is Nairobi safe for international students?",
                "What areas should I avoid at night?",
                "Emergency contacts for students in Kenya",
                "How to stay safe using public transport?",
                "What to do if I lose my passport?",
            ],
            "university": [
                "How do I apply to Kenyan universities?",
                "What documents do I need for admission?",
                "Compare University of Nairobi vs Strathmore",
                "When does the academic year start?",
                "How do I convert my grades to Kenyan system?",
            ],
            "finance": [
                "How to open a bank account as international student?",
                "What is M-Pesa and how do I use it?",
                "Cost of living for students in Nairobi",
                "How to send money from abroad?",
                "Student visa fees and requirements",
            ],
        }

        if focus_area not in focus_queries:
            raise ValueError(
                f"Focus area '{focus_area}' not supported. Choose from: {list(focus_queries.keys())}"
            )

        queries = focus_queries[focus_area][:num_queries]

        results = []
        for i, query in enumerate(queries):
            try:
                # Process query
                intent_info = self.intent_recognizer.recognize_intent(query)
                retrieved_chunks = (
                    self.vector_db_service.search_for_students(query, top_k=5)
                    if hasattr(self.vector_db_service, "search_for_students")
                    else self.vector_db_service.search(query, top_k=5)
                )
                response_data = self.response_generator.generate_response(
                    query, retrieved_chunks, intent_info
                )

                # Simple scoring for focused evaluation
                score = self._calculate_focus_score(
                    response_data, focus_area, query
                )

                results.append(
                    {
                        "query_id": f"{focus_area}_{i+1:02d}",
                        "query": query,
                        "response": response_data["response"],
                        "intent_type": intent_info["intent_type"].value,
                        "score": score,
                        "chunks_retrieved": len(retrieved_chunks),
                        "focus_area": focus_area,
                    }
                )

            except Exception as e:
                logger.error(
                    f"Error in focused evaluation query '{query}': {str(e)}"
                )
                results.append(
                    {
                        "query_id": f"{focus_area}_{i+1:02d}",
                        "query": query,
                        "error": str(e),
                        "score": 0,
                        "focus_area": focus_area,
                    }
                )

        # Calculate focused metrics
        successful_results = [r for r in results if "error" not in r]
        avg_score = (
            sum(r["score"] for r in successful_results)
            / len(successful_results)
            if successful_results
            else 0
        )

        focused_report = {
            "focus_area": focus_area,
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "average_score": avg_score,
            "results": results,
            "timestamp": str(datetime.now()),
        }

        # Save if output path specified
        if output_path:
            with open(output_path, "w") as f:
                json.dump(focused_report, f, indent=2)
            logger.info(f"Focused evaluation results saved to {output_path}")

        return focused_report

    def _calculate_focus_score(
        self, response_data: Dict, focus_area: str, query: str
    ) -> float:
        """Calculate score for focused evaluation."""
        response_text = response_data.get("response", "").lower()

        # Area-specific scoring criteria
        scoring_criteria = {
            "housing": [
                "accommodation",
                "rent",
                "cost",
                "safe",
                "area",
                "neighborhood",
            ],
            "safety": [
                "safe",
                "security",
                "police",
                "emergency",
                "avoid",
                "precaution",
            ],
            "university": [
                "university",
                "admission",
                "document",
                "academic",
                "requirement",
            ],
            "finance": [
                "bank",
                "money",
                "cost",
                "mpesa",
                "transfer",
                "account",
            ],
        }

        criteria = scoring_criteria.get(focus_area, [])

        # Basic content matching
        matches = sum(1 for term in criteria if term in response_text)
        content_score = min(matches / len(criteria), 1.0) if criteria else 0.5

        # Response quality indicators
        quality_score = 0.0
        if len(response_text) > 100:  # Adequate length
            quality_score += 0.2
        if any(
            indicator in response_text
            for indicator in ["kes", "ksh", "cost", "price"]
        ):  # Cost info
            quality_score += 0.1
        if any(
            indicator in response_text
            for indicator in ["contact", "address", "location"]
        ):  # Practical info
            quality_score += 0.1

        return min((content_score * 0.7) + (quality_score * 0.3), 1.0)

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a quick summary of the evaluation system capabilities."""

        return {
            "evaluator_info": {
                "name": "International Student RAG Evaluator",
                "version": "1.0",
                "focus": "International students in Nairobi, Kenya",
                "supported_languages": ["English"],
                "evaluation_areas": [
                    "housing",
                    "safety",
                    "university",
                    "finance",
                    "transportation",
                    "healthcare",
                    "cultural_adaptation",
                ],
            },
            "evaluation_types": {
                "comprehensive": "Full evaluation across all student needs",
                "focused": "Targeted evaluation for specific areas",
                "bleu_scoring": "Response similarity to reference answers",
            },
            "metrics_tracked": {
                "intent_recognition": "Accuracy of understanding user intentions",
                "student_relevance": "Relevance to international student needs",
                "empathy_scoring": "Emotional support and understanding",
                "practical_information": "Actionable and useful content delivery",
                "safety_prioritization": "Appropriate handling of safety concerns",
                "bleu_score": "Response quality compared to reference answers",
            },
            "output_formats": ["JSON", "HTML", "CSV"],
            "integration_ready": True,
        }
