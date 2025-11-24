import json
import logging
from typing import Dict
from typing import List
from typing import Optional

import openai
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("language_processor")


class LanguageProcessor:
    """
    LLM-powered language detection and translation optimized for settlement queries.
    Uses OpenAI for both detection and translation - more reliable and context-aware.
    """

    def __init__(self):
        self.openai_client = openai.OpenAI()

        # Supported languages for international students
        self.supported_languages = {
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "pt": "portuguese",
            "it": "italian",
            "de": "german",
            "ru": "russian",
            "zh": "chinese",
            "ja": "japanese",
            "ko": "korean",
            "hi": "hindi",
            "ar": "arabic",
            "sw": "swahili",
            "am": "amharic",
            "so": "somali",
            "ha": "hausa",
            "yo": "yoruba",
            "ig": "igbo",
            "zu": "zulu",
            "xh": "xhosa",
            "af": "afrikaans",
            "ny": "chewa",
            "st": "sesotho",
            "tn": "setswana",
        }

        # Country-language mapping for context
        self.country_languages = {
            "kenya": ["sw", "en"],
            "uganda": ["en", "sw"],
            "tanzania": ["sw", "en"],
            "rwanda": ["fr", "en"],
            "burundi": ["fr", "en"],
            "somalia": ["so", "ar", "en"],
            "ethiopia": ["am", "en"],
            "sudan": ["ar", "en"],
            "nigeria": ["en", "ha", "yo", "ig"],
            "ghana": ["en"],
            "south_africa": ["en", "af", "zu", "xh"],
            "malawi": ["en", "ny"],
            "congo": ["fr"],
        }

        self.primary_language = settings.language.primary_language
        self.detection_enabled = settings.language.detection_enabled

        logger.info("LLM-only Language Processor initialized successfully")

    def detect_and_process_query(self, query: str) -> Dict[str, str]:
        """
        Detect language and translate using LLM with settlement context awareness.

        Args:
            query: User query in any supported language

        Returns:
            Dictionary with detection and translation results
        """
        if not self.detection_enabled:
            return {
                "detected_language": "english",
                "original_query": query,
                "english_query": query,
                "needs_translation": False,
                "detection_method": "disabled",
            }

        try:
            # Single LLM call for detection + translation
            result = self._detect_and_translate_llm(query)

            logger.info(
                f"LLM detected: {result['detected_language']} (confidence: {result['confidence']:.2f})"
            )

            return {
                "detected_language": result["detected_language"],
                "original_query": query,
                "english_query": result["english_query"],
                "needs_translation": result["needs_translation"],
                "confidence": result["confidence"],
                "detection_method": "llm_optimized",
            }

        except Exception as e:
            logger.error(f"LLM language processing failed: {str(e)}")
            # Safe fallback
            return {
                "detected_language": "english",
                "original_query": query,
                "english_query": query,
                "needs_translation": False,
                "detection_method": "fallback",
                "error": str(e),
            }

    def _detect_and_translate_llm(self, query: str) -> Dict[str, any]:
        """
        Combined detection and translation using LLM with settlement context.

        Args:
            query: Input query

        Returns:
            Dictionary with detection and translation results
        """
        prompt = f"""Analyze this query from an international student about settlement in Nairobi, Kenya.

Query: "{query}"

Tasks:
1. Detect the language
2. If not English, translate to English while preserving settlement context
3. Keep ALL location names unchanged (Westlands, Kilimani, Karen, Lavington, etc.)
4. Keep ALL institution names unchanged (University of Nairobi, Strathmore, JKUAT, USIU, etc.)
5. Keep currency references as "KSh"
6. Keep transport terms unchanged (matatu, boda boda)
7. Preserve the student's intent and urgency level

Respond with JSON only:
{{
    "detected_language": "language_name",
    "language_code": "2-letter code",
    "english_query": "translated text or original if English", 
    "needs_translation": true/false,
    "confidence": 0.0-1.0,
    "preserved_terms": ["term1", "term2"]
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert language processor specializing in international student settlement queries for Nairobi, Kenya. Provide accurate language detection and context-preserving translation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            result = json.loads(response.choices[0].message.content.strip())

            # Validate and normalize result
            detected_lang = result.get("detected_language", "english").lower()
            lang_code = result.get("language_code", "en").lower()

            # Ensure language code is supported
            if lang_code not in self.supported_languages:
                lang_code = "en"
                detected_lang = "english"

            return {
                "detected_language": detected_lang,
                "language_code": lang_code,
                "english_query": result.get("english_query", query),
                "needs_translation": result.get("needs_translation", False),
                "confidence": result.get("confidence", 0.9),
                "preserved_terms": result.get("preserved_terms", []),
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"LLM processing failed: {str(e)}")
            raise

    def translate_response(self, response: str, target_language: str) -> str:
        """
        Translate English response to target language using LLM.

        Args:
            response: English response text
            target_language: Target language name or code

        Returns:
            Translated response
        """
        if not self.detection_enabled or target_language.lower() in [
            "english",
            "en",
        ]:
            return response

        try:
            target_code = self._get_language_code(target_language)
            if target_code == "en":
                return response

            return self._translate_response_llm(
                response, target_code, target_language
            )

        except Exception as e:
            logger.error(f"Response translation failed: {str(e)}")
            return response

    def _translate_response_llm(
        self, text: str, target_code: str, target_language: str
    ) -> str:
        """
        Translate response using LLM with settlement context preservation.
        """
        try:
            language_name = self.supported_languages.get(
                target_code, target_language
            )

            prompt = f"""Translate this settlement response for international students in Nairobi to {language_name}.

English response: "{text}"

CRITICAL TRANSLATION RULES:
1. Preserve ALL location names exactly: Westlands, Kilimani, Karen, Lavington, Kileleshwa, Parklands, Hurlingham, etc.
2. Keep ALL institution names unchanged: University of Nairobi, Strathmore University, JKUAT, USIU, Kenyatta University, etc.
3. Keep currency as "KSh" (do not translate)
4. Keep transport terms: matatu, boda boda, M-Pesa (do not translate)
5. Preserve all contact numbers and addresses exactly
6. Keep emergency numbers: 999, +254 numbers
7. Maintain helpful, supportive tone for students
8. Keep technical visa/permit terminology
9. Preserve the three-section structure (## DIRECT ANSWER, ## ADDITIONAL INFORMATION, ## NEXT STEPS)

Translate to {language_name}:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert translator for international student settlement information in Nairobi, Kenya. Translate to {language_name} while preserving ALL crucial settlement details, location names, and institutional information exactly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )

            translation = response.choices[0].message.content.strip()

            # Clean formatting artifacts
            if translation.startswith('"') and translation.endswith('"'):
                translation = translation[1:-1]

            logger.info(f"Successfully translated response to {language_name}")
            return translation

        except Exception as e:
            logger.error(f"LLM response translation failed: {str(e)}")
            return text

    def _get_language_code(self, language: str) -> str:
        """Convert language name to 2-letter code."""
        if len(language) == 2:
            return language.lower()

        # Reverse mapping
        name_to_code = {
            name: code for code, name in self.supported_languages.items()
        }
        return name_to_code.get(language.lower(), "en")

    def validate_translation_quality(
        self, original: str, translated: str, target_lang: str
    ) -> Dict[str, any]:
        """
        Validate translation quality with settlement-specific checks.
        """
        try:
            # Length ratio check
            length_ratio = len(translated) / len(original) if original else 0

            # Critical settlement terms that must be preserved
            critical_terms = [
                "Nairobi",
                "Westlands",
                "Kilimani",
                "Karen",
                "Lavington",
                "Kileleshwa",
                "Parklands",
                "Hurlingham",
                "Riverside",
                "University of Nairobi",
                "Strathmore",
                "JKUAT",
                "USIU",
                "KSh",
                "matatu",
                "boda boda",
                "M-Pesa",
                "999",
                "Immigration",
                "Nyayo House",
            ]

            # Count preserved terms
            preserved_count = 0
            total_critical_terms = 0

            for term in critical_terms:
                if term in original:
                    total_critical_terms += 1
                    if term in translated:
                        preserved_count += 1

            preservation_score = (
                preserved_count / total_critical_terms
                if total_critical_terms > 0
                else 1.0
            )

            # Calculate quality score
            quality_score = 1.0
            issues = []

            # Length validation
            if length_ratio < 0.3 or length_ratio > 3.5:
                quality_score -= 0.2
                issues.append("Unusual length difference")

            # Term preservation validation
            if preservation_score < 0.9:
                quality_score -= 0.3
                issues.append("Critical settlement terms not preserved")

            # Structure preservation (for responses)
            required_sections = [
                "## DIRECT ANSWER",
                "## ADDITIONAL INFORMATION",
                "## NEXT STEPS",
            ]
            if any(section in original for section in required_sections):
                preserved_sections = sum(
                    1 for section in required_sections if section in translated
                )
                if preserved_sections < len(
                    [s for s in required_sections if s in original]
                ):
                    quality_score -= 0.2
                    issues.append("Response structure not preserved")

            return {
                "quality_score": max(quality_score, 0.0),
                "length_ratio": length_ratio,
                "preservation_score": preservation_score,
                "preserved_terms": preserved_count,
                "total_critical_terms": total_critical_terms,
                "issues": issues,
                "settlement_optimized": True,
            }

        except Exception as e:
            logger.error(f"Translation quality validation failed: {str(e)}")
            return {
                "quality_score": 0.5,
                "issues": ["Validation failed"],
                "error": str(e),
            }

    def get_language_stats(self) -> Dict[str, any]:
        """Get comprehensive language processing statistics."""
        return {
            "processor_type": "llm_only",
            "supported_languages": list(self.supported_languages.values()),
            "total_languages": len(self.supported_languages),
            "detection_enabled": self.detection_enabled,
            "primary_language": self.primary_language,
            "detection_method": "llm_context_aware",
            "translation_provider": "llm_settlement_optimized",
            "supported_language_codes": list(self.supported_languages.keys()),
            "supported_countries": list(self.country_languages.keys()),
            "advantages": [
                "No external library dependencies",
                "Context-aware settlement translation",
                "Combined detection and translation",
                "Perfect term preservation",
                "No async compatibility issues",
                "Settlement domain optimization",
                "Consistent quality across all languages",
            ],
            "features": [
                "Single LLM call for detection + translation",
                "Settlement context preservation",
                "Critical term protection",
                "Structure preservation for responses",
                "Quality validation with settlement metrics",
                "Robust error handling",
                "Optimized for student queries",
            ],
        }

    def get_supported_languages_by_country(self, country: str) -> List[str]:
        """Get supported languages for a specific country."""
        country_key = country.lower().replace(" ", "_")
        lang_codes = self.country_languages.get(country_key, ["en"])
        return [self.supported_languages.get(code, code) for code in lang_codes]

    def detect_country_context(self, query: str) -> Optional[str]:
        """Detect country context from query for language prioritization."""
        country_indicators = {
            "kenya": ["kenya", "nairobi", "mombasa", "kisumu", "nakuru"],
            "uganda": ["uganda", "kampala", "entebbe"],
            "tanzania": ["tanzania", "dar es salaam", "dodoma", "arusha"],
            "somalia": ["somalia", "mogadishu", "somali"],
            "ethiopia": ["ethiopia", "addis ababa", "ethiopian"],
            "nigeria": ["nigeria", "lagos", "abuja", "nigerian"],
            "ghana": ["ghana", "accra", "ghanaian"],
            "south_africa": [
                "south africa",
                "johannesburg",
                "cape town",
                "durban",
            ],
        }

        query_lower = query.lower()
        for country, indicators in country_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return country

        return None

    def test_translation_quality(
        self, test_cases: List[Dict]
    ) -> Dict[str, any]:
        """
        Test translation quality with predefined settlement queries.

        Args:
            test_cases: List of {"query": str, "language": str, "expected_terms": List[str]}

        Returns:
            Test results with quality metrics
        """
        results = []

        for test_case in test_cases:
            try:
                # Process query
                result = self.detect_and_process_query(test_case["query"])

                # Check if expected terms are preserved
                expected_terms = test_case.get("expected_terms", [])
                preserved_terms = [
                    term
                    for term in expected_terms
                    if term in result["english_query"]
                ]

                quality_metrics = {
                    "query": test_case["query"],
                    "expected_language": test_case["language"],
                    "detected_language": result["detected_language"],
                    "language_correct": test_case["language"].lower()
                    in result["detected_language"].lower(),
                    "english_query": result["english_query"],
                    "expected_terms": expected_terms,
                    "preserved_terms": preserved_terms,
                    "preservation_rate": len(preserved_terms)
                    / len(expected_terms)
                    if expected_terms
                    else 1.0,
                    "confidence": result["confidence"],
                }

                results.append(quality_metrics)

            except Exception as e:
                results.append(
                    {
                        "query": test_case["query"],
                        "error": str(e),
                        "preservation_rate": 0.0,
                    }
                )

        # Calculate overall statistics
        successful_tests = [r for r in results if "error" not in r]
        avg_preservation = (
            sum(r["preservation_rate"] for r in successful_tests)
            / len(successful_tests)
            if successful_tests
            else 0
        )
        avg_confidence = (
            sum(r["confidence"] for r in successful_tests)
            / len(successful_tests)
            if successful_tests
            else 0
        )
        language_accuracy = (
            sum(1 for r in successful_tests if r["language_correct"])
            / len(successful_tests)
            if successful_tests
            else 0
        )

        return {
            "total_tests": len(test_cases),
            "successful_tests": len(successful_tests),
            "failed_tests": len(test_cases) - len(successful_tests),
            "average_preservation_rate": avg_preservation,
            "average_confidence": avg_confidence,
            "language_detection_accuracy": language_accuracy,
            "detailed_results": results,
        }
