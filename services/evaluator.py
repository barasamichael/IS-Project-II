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
from services.vector_db import InternationalStudentVectorDB

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


class InternationalStudentRAGEvaluator:
    """Evaluation system specifically designed for international student assistant."""

    def __init__(
        self,
        vector_db_service: Optional[InternationalStudentVectorDB] = None,
        intent_recognizer: Optional[IntentRecognizer] = None,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        self.vector_db_service = (
            vector_db_service or InternationalStudentVectorDB()
        )
        self.intent_recognizer = intent_recognizer or IntentRecognizer()
        self.response_generator = response_generator or ResponseGenerator()

        self.eval_dir = ROOT_DIR / "tests" / "eval_data"
        if not self.eval_dir.exists():
            self.eval_dir.mkdir(parents=True)

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
                    )

                # Generate response
                response_data = self.response_generator.generate_response(
                    query=query,
                    retrieved_context=retrieved_chunks,
                    intent_info=intent_info,
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
                        "priority": row.get("priority", "medium"),
                    }
                )

        # Calculate comprehensive metrics
        evaluation_report = self._generate_evaluation_report(
            results, failed_queries
        )

        # Save results
        with open(output_path, "w") as f:
            json.dump(evaluation_report, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {output_path}")
        logger.info(
            f"Overall Score: {evaluation_report['overall_metrics']['average_score']:.3f}"
        )
        logger.info(
            f"High Priority Score: {evaluation_report['priority_metrics']['critical_avg_score']:.3f}"
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
        self, results: List[Dict], failed_queries: int
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

        # Intent type performance
        intent_performance = {}
        for result in successful_results:
            intent = result.get("actual_intent", "unknown")
            if intent not in intent_performance:
                intent_performance[intent] = {
                    "count": 0,
                    "total_score": 0,
                    "correct_intent": 0,
                }

            intent_performance[intent]["count"] += 1
            intent_performance[intent]["total_score"] += result.get(
                "overall_score", 0
            )
            if result.get("intent_match", False):
                intent_performance[intent]["correct_intent"] += 1

        # Calculate averages for intent performance
        for intent, stats in intent_performance.items():
            if stats["count"] > 0:
                stats["avg_score"] = stats["total_score"] / stats["count"]
                stats["intent_accuracy"] = (
                    stats["correct_intent"] / stats["count"]
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

    def generate_html_report(
        self, eval_results: Dict[str, Any], output_path: Optional[Path] = None
    ) -> Path:
        """Generate comprehensive HTML report for international student evaluation."""
        output_path = (
            output_path
            or self.eval_dir / "international_student_evaluation_report.html"
        )

        # Generate HTML content
        html_content = self._create_html_report_content(eval_results)

        # Save HTML report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML evaluation report generated at {output_path}")
        return output_path

    def _create_html_report_content(self, eval_results: Dict[str, Any]) -> str:
        """Create HTML content for the evaluation report."""
        overall_score = eval_results.get("overall_metrics", {}).get(
            "average_score", 0
        )
        student_relevance = eval_results.get(
            "student_specific_metrics", {}
        ).get("avg_student_relevance", 0)
        empathy_score = eval_results.get("student_specific_metrics", {}).get(
            "avg_empathy_score", 0
        )

        def get_score_class(score):
            if score >= 0.8:
                return "score-excellent"
            elif score >= 0.6:
                return "score-good"
            elif score >= 0.4:
                return "score-fair"
            else:
                return "score-poor"

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>International Student Assistant Evaluation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h3 {{ color: #7f8c8d; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
        .score-excellent {{ color: #27ae60; }}
        .score-good {{ color: #f39c12; }}
        .score-fair {{ color: #e67e22; }}
        .score-poor {{ color: #e74c3c; }}
        .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; }}
        .strengths {{ background: #d1edff; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .weaknesses {{ background: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .priority-critical {{ background-color: #ffebee; }}
        .priority-high {{ background-color: #fff3e0; }}
        .priority-medium {{ background-color: #f3e5f5; }}
        .priority-low {{ background-color: #e8f5e8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 International Student Assistant Evaluation Report</h1>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Overall Score</h3>
                <div class="metric-value {get_score_class(overall_score)}">{overall_score:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>Student Relevance</h3>
                <div class="metric-value {get_score_class(student_relevance)}">{student_relevance:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>Empathy Score</h3>
                <div class="metric-value {get_score_class(empathy_score)}">{empathy_score:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>Success Rate</h3>
                <div class="metric-value {get_score_class(eval_results.get('overall_metrics', {}).get('success_rate', 0))}">{eval_results.get('overall_metrics', {}).get('success_rate', 0):.2%}</div>
            </div>
        </div>

        <h2>📊 Key Insights</h2>
        <div class="strengths">
            <h3>💪 Strengths</h3>
            <ul>
        """

        for strength in eval_results.get("evaluation_summary", {}).get(
            "key_strengths", []
        ):
            html_content += f"<li>{strength}</li>"

        html_content += """
            </ul>
        </div>
        
        <div class="weaknesses">
            <h3>⚠️ Areas for Improvement</h3>
            <ul>
        """

        for weakness in eval_results.get("evaluation_summary", {}).get(
            "key_weaknesses", []
        ):
            html_content += f"<li>{weakness}</li>"

        html_content += """
            </ul>
        </div>
        
        <div class="recommendations">
            <h3>🎯 Recommendations</h3>
            <ul>
        """

        for recommendation in eval_results.get("recommendations", []):
            html_content += f"<li>{recommendation}</li>"

        html_content += """
            </ul>
        </div>

        <h2>📈 Performance by Intent Type</h2>
        <table>
            <tr>
                <th>Intent Type</th>
                <th>Count</th>
                <th>Average Score</th>
                <th>Intent Accuracy</th>
            </tr>
        """

        intent_performance = eval_results.get("intent_performance", {})
        for intent, stats in sorted(
            intent_performance.items(),
            key=lambda x: x[1].get("avg_score", 0),
            reverse=True,
        ):
            score_class = get_score_class(stats.get("avg_score", 0))
            html_content += f"""
            <tr>
                <td>{intent.replace('_', ' ').title()}</td>
                <td>{stats.get('count', 0)}</td>
                <td class="{score_class}">{stats.get('avg_score', 0):.2%}</td>
                <td>{stats.get('intent_accuracy', 0):.2%}</td>
            </tr>
            """

        html_content += """
        </table>

        <h2>🎯 Performance by Priority</h2>
        <table>
            <tr>
                <th>Priority Level</th>
                <th>Count</th>
                <th>Average Score</th>
                <th>Intent Accuracy</th>
                <th>Student Relevance</th>
            </tr>
        """

        priority_metrics = eval_results.get("priority_metrics", {})
        for priority in ["critical", "high", "medium", "low"]:
            if priority in priority_metrics:
                stats = priority_metrics[priority]
                score_class = get_score_class(stats.get("avg_score", 0))
                html_content += f"""
                <tr class="priority-{priority}">
                    <td>{priority.title()}</td>
                    <td>{stats.get('count', 0)}</td>
                    <td class="{score_class}">{stats.get('avg_score', 0):.2%}</td>
                    <td>{stats.get('intent_accuracy', 0):.2%}</td>
                    <td>{stats.get('avg_student_relevance', 0):.2%}</td>
                </tr>
                """

        html_content += f"""
        </table>

        <h2>💰 Resource Usage</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Tokens</h3>
                <div class="metric-value">{eval_results.get('token_usage', {}).get('total_tokens', 0):,}</div>
            </div>
            <div class="metric-card">
                <h3>Avg Tokens/Query</h3>
                <div class="metric-value">{eval_results.get('token_usage', {}).get('avg_tokens_per_query', 0):.0f}</div>
            </div>
            <div class="metric-card">
                <h3>Avg Chunks Retrieved</h3>
                <div class="metric-value">{eval_results.get('retrieval_analysis', {}).get('avg_chunks_retrieved', 0):.1f}</div>
            </div>
            <div class="metric-card">
                <h3>Chunk Relevance</h3>
                <div class="metric-value">{eval_results.get('retrieval_analysis', {}).get('avg_chunk_relevance', 0):.2%}</div>
            </div>
        </div>

        <h2>📝 Detailed Results Summary</h2>
        <p><strong>Total Queries Evaluated:</strong> {eval_results.get('overall_metrics', {}).get('total_queries', 0)}</p>
        <p><strong>Successful Evaluations:</strong> {eval_results.get('overall_metrics', {}).get('successful_queries', 0)}</p>
        <p><strong>Failed Evaluations:</strong> {eval_results.get('overall_metrics', {}).get('failed_queries', 0)}</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Excellent Responses</h3>
                <div class="metric-value score-excellent">{eval_results.get('quality_distribution', {}).get('excellent_responses', 0)}</div>
            </div>
            <div class="metric-card">
                <h3>Good Responses</h3>
                <div class="metric-value score-good">{eval_results.get('quality_distribution', {}).get('good_responses', 0)}</div>
            </div>
            <div class="metric-card">
                <h3>Poor Responses</h3>
                <div class="metric-value score-poor">{eval_results.get('quality_distribution', {}).get('poor_responses', 0)}</div>
            </div>
        </div>

        <footer style="margin-top: 40px; text-align: center; color: #7f8c8d; border-top: 1px solid #ecf0f1; padding-top: 20px;">
            <p>Report generated on {eval_results.get('timestamp', 'Unknown')}</p>
            <p>International Student Assistant Evaluation Framework v1.0</p>
        </footer>
    </div>
</body>
</html>
        """

        return html_content

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
                retrieved_chunks = self.vector_db_service.search_for_students(
                    query, top_k=5
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

    def benchmark_against_baseline(
        self, baseline_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Benchmark current system against baseline performance."""

        if not baseline_file or not baseline_file.exists():
            logger.warning(
                "No baseline file found. Running current evaluation as new baseline."
            )
            return self.run_comprehensive_evaluation()

        try:
            # Load baseline results
            with open(baseline_file, "r") as f:
                baseline_results = json.load(f)

            # Run current evaluation
            current_results = self.run_comprehensive_evaluation()

            # Compare metrics
            comparison = {
                "baseline_date": baseline_results.get("timestamp", "Unknown"),
                "current_date": current_results.get(
                    "timestamp", str(datetime.now())
                ),
                "improvements": [],
                "regressions": [],
                "metric_comparison": {},
            }

            # Compare key metrics
            baseline_metrics = baseline_results.get("overall_metrics", {})
            current_metrics = current_results.get("overall_metrics", {})

            key_metrics = [
                "average_score",
                "intent_accuracy",
                "content_accuracy",
            ]

            for metric in key_metrics:
                baseline_val = baseline_metrics.get(metric, 0)
                current_val = current_metrics.get(metric, 0)
                change = current_val - baseline_val

                comparison["metric_comparison"][metric] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "change": change,
                    "change_percentage": (change / baseline_val * 100)
                    if baseline_val > 0
                    else 0,
                }

                if change > 0.05:  # Significant improvement
                    comparison["improvements"].append(
                        f"{metric}: +{change:.2%}"
                    )
                elif change < -0.05:  # Significant regression
                    comparison["regressions"].append(f"{metric}: {change:.2%}")

            # Add student-specific metric comparisons
            baseline_student = baseline_results.get(
                "student_specific_metrics", {}
            )
            current_student = current_results.get(
                "student_specific_metrics", {}
            )

            student_metrics = [
                "avg_student_relevance",
                "avg_empathy_score",
                "avg_practical_info_score",
            ]

            for metric in student_metrics:
                baseline_val = baseline_student.get(metric, 0)
                current_val = current_student.get(metric, 0)
                change = current_val - baseline_val

                comparison["metric_comparison"][f"student_{metric}"] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "change": change,
                    "change_percentage": (change / baseline_val * 100)
                    if baseline_val > 0
                    else 0,
                }

            # Overall assessment
            overall_baseline = baseline_metrics.get("average_score", 0)
            overall_current = current_metrics.get("average_score", 0)

            if overall_current > overall_baseline + 0.05:
                comparison["overall_assessment"] = "IMPROVED"
            elif overall_current < overall_baseline - 0.05:
                comparison["overall_assessment"] = "REGRESSED"
            else:
                comparison["overall_assessment"] = "STABLE"

            return {
                "comparison": comparison,
                "current_results": current_results,
                "baseline_results": baseline_results,
            }

        except Exception as e:
            logger.error(f"Error in baseline comparison: {str(e)}")
            return {"error": str(e)}

    def export_performance_trends(
        self, results_history: List[Dict], output_path: Optional[Path] = None
    ) -> Path:
        """Export performance trends over multiple evaluation runs."""

        output_path = output_path or self.eval_dir / "performance_trends.json"

        trends = {
            "evaluation_dates": [],
            "overall_scores": [],
            "intent_accuracy": [],
            "student_relevance": [],
            "empathy_scores": [],
            "recommendations_timeline": [],
        }

        for result in results_history:
            if result.get("status") == "success":
                # Extract date
                trends["evaluation_dates"].append(
                    result.get("timestamp", "Unknown")
                )

                # Extract metrics
                overall_metrics = result.get("overall_metrics", {})
                student_metrics = result.get("student_specific_metrics", {})

                trends["overall_scores"].append(
                    overall_metrics.get("average_score", 0)
                )
                trends["intent_accuracy"].append(
                    overall_metrics.get("intent_accuracy", 0)
                )
                trends["student_relevance"].append(
                    student_metrics.get("avg_student_relevance", 0)
                )
                trends["empathy_scores"].append(
                    student_metrics.get("avg_empathy_score", 0)
                )

                # Track recommendations
                recommendations = result.get("recommendations", [])
                trends["recommendations_timeline"].append(
                    {
                        "date": result.get("timestamp", "Unknown"),
                        "recommendations": recommendations,
                    }
                )

        # Calculate trends
        if len(trends["overall_scores"]) >= 2:
            recent_trend = (
                trends["overall_scores"][-1] - trends["overall_scores"][-2]
            )
            trends["recent_trend"] = (
                "improving"
                if recent_trend > 0
                else "declining"
                if recent_trend < 0
                else "stable"
            )

        # Save trends
        with open(output_path, "w") as f:
            json.dump(trends, f, indent=2)

        logger.info(f"Performance trends exported to {output_path}")
        return output_path

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
                "baseline_comparison": "Performance comparison against previous results",
                "trend_analysis": "Performance tracking over time",
            },
            "metrics_tracked": {
                "intent_recognition": "Accuracy of understanding user intentions",
                "student_relevance": "Relevance to international student needs",
                "empathy_scoring": "Emotional support and understanding",
                "practical_information": "Actionable and useful content delivery",
                "safety_prioritization": "Appropriate handling of safety concerns",
            },
            "output_formats": ["JSON", "HTML", "CSV"],
            "integration_ready": True,
        }
