import re
import logging
from typing import Dict
from typing import List
from typing import Optional

import openai
from googletrans import Translator
from langdetect import detect
from langdetect import LangDetectError

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("language_processor")


class LanguageProcessor:
    """Handles language detection, translation, and multilingual query processing."""

    def __init__(self):
        self.translator = Translator()
        self.openai_client = openai.OpenAI()

        # Language mappings
        self.supported_languages = {
            "en": "english",
            "fr": "french",
            "sw": "swahili",
            "es": "spanish",
            "ar": "arabic",
            "pt": "portuguese",
        }

        self.primary_language = settings.language.primary_language
        self.detection_enabled = settings.language.detection_enabled

    def detect_and_process_query(self, query: str) -> Dict[str, str]:
        """
        Detect language and convert query to English for RAG processing.

        Args:
            query: User query in any supported language

        Returns:
            Dictionary with detected language, original query, and English query
        """
        if not self.detection_enabled:
            return {
                "detected_language": "english",
                "original_query": query,
                "english_query": query,
                "needs_translation": False,
            }

        try:
            # Handle mixed language queries
            languages_detected = self._detect_mixed_languages(query)

            if len(languages_detected) > 1:
                # Multiple languages detected - find dominant one
                dominant_lang = self._get_dominant_language(
                    query, languages_detected
                )
                logger.info(
                    f"Mixed languages detected, dominant: {dominant_lang}"
                )
            else:
                dominant_lang = (
                    languages_detected[0] if languages_detected else "en"
                )

            # Convert to English if needed
            if dominant_lang != "en":
                english_query = self._translate_to_english(
                    query, dominant_lang)
                needs_translation = True
            else:
                english_query = query
                needs_translation = False

            return {
                "detected_language": self.supported_languages.get(
                    dominant_lang, "english"
                ),
                "original_query": query,
                "english_query": english_query,
                "needs_translation": needs_translation,
                "confidence": self._get_detection_confidence(
                    query, dominant_lang
                ),
            }

        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return {
                "detected_language": "english",
                "original_query": query,
                "english_query": query,
                "needs_translation": False,
                "error": str(e),
            }

    def translate_response(self, response: str, target_language: str) -> str:
        """
        Translate response to target language.

        Args:
            response: Response text in English
            target_language: Target language code or name

        Returns:
            Translated response text
        """
        if not self.detection_enabled or target_language.lower() == "english":
            return response

        try:
            # Get language code
            target_code = self._get_language_code(target_language)

            if target_code == "en":
                return response

            # Use GPT for high-quality translation of settlement-specific content
            if target_code in ["sw", "fr"]:
                translated = self._gpt_translate(response, target_language)
                if translated:
                    return translated

            # Fallback to Google Translate
            result = self.translator.translate(response, dest=target_code)
            return result.text

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return response

    def _detect_mixed_languages(self, text: str) -> List[str]:
        """Detect multiple languages in text."""
        # Split text into segments and detect each
        segments = re.split(r"[.!?]+", text)
        detected_langs = []

        for segment in segments:
            segment = segment.strip()
            if len(segment) > 10:  # Only check substantial segments
                try:
                    lang = detect(segment)
                    if lang not in detected_langs:
                        detected_langs.append(lang)
                except LangDetectError:
                    continue

        # Fallback to full text detection
        if not detected_langs:
            try:
                main_lang = detect(text)
                detected_langs.append(main_lang)
            except LangDetectError:
                detected_langs.append("en")  # Default to English

        return detected_langs

    def _get_dominant_language(self, text: str, languages: List[str]) -> str:
        """Determine dominant language in mixed-language text."""
        if len(languages) == 1:
            return languages[0]

        # Count characters for each language using simple heuristics
        lang_scores = {lang: 0 for lang in languages}

        # Swahili indicators
        swahili_words = [
            "ni",
            "na",
            "ya",
            "wa",
            "kwa",
            "kutoka",
            "kwenda",
            "nyumba",
            "bei",
            "usalama",
        ]
        if any(word in text.lower() for word in swahili_words):
            lang_scores["sw"] = lang_scores.get("sw", 0) + 10

        # French indicators
        french_words = [
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            "le",
            "la",
            "les",
            "un",
            "une",
        ]
        if any(word in text.lower() for word in french_words):
            lang_scores["fr"] = lang_scores.get("fr", 0) + 10

        # English indicators (default if mixed)
        english_words = [
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        ]
        if any(word in text.lower() for word in english_words):
            lang_scores["en"] = lang_scores.get("en", 0) + 5

        # Return language with highest score, default to first detected
        if max(lang_scores.values()) > 0:
            return max(lang_scores, key=lang_scores.get)

        return languages[0]

    def _translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English with context awareness."""
        try:
            # Use GPT for settlement-specific translation
            if source_lang in ["sw", "fr"]:
                gpt_translation = self._gpt_translate_to_english(
                    text, source_lang
                )
                if gpt_translation:
                    return gpt_translation

            # Fallback to Google Translate
            result = self.translator.translate(
                text, src=source_lang, dest="en")
            return result.text

        except Exception as e:
            logger.error(f"Translation to English failed: {str(e)}")
            return text

    def _gpt_translate_to_english(
        self, text: str, source_lang: str
    ) -> Optional[str]:
        """Use GPT for context-aware translation to English."""
        try:
            lang_names = {
                "sw": "Swahili",
                "fr": "French",
                "es": "Spanish",
                "ar": "Arabic",
                "pt": "Portuguese",
            }

            prompt = f"""
            Translate this {lang_names.get(source_lang, source_lang)} text to English.
            This is a query about international student settlement in Nairobi, Kenya.
            Preserve the meaning and context related to:
            - Housing and accommodation
            - University life and education
            - Transportation and locations in Nairobi
            - Costs and finances
            - Safety and security
            - Cultural adaptation
            - Legal matters (visas, permits)

            Text to translate: "{text}"

            Provide only the English translation, no explanations.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"GPT translation failed: {str(e)}")
            return None

    def _gpt_translate(self, text: str, target_language: str) -> Optional[str]:
        """Use GPT for context-aware translation from English."""
        try:
            prompt = f"""
            Translate this English response to {target_language}.
            This is information about international student settlement in Nairobi, Kenya.
            Maintain accuracy for:
            - Location names (keep as-is: Westlands, Kilimani, etc.)
            - Institution names (universities, hospitals, etc.)
            - Costs and prices (maintain currency: KSh)
            - Technical terms (visa, permit, etc.)
            - Transportation terms (matatu, boda boda, etc.)

            Text to translate: "{text}"

            Provide only the {target_language} translation, no explanations.
            """

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"GPT response translation failed: {str(e)}")
            return None

    def _get_language_code(self, language: str) -> str:
        """Convert language name to language code."""
        if len(language) == 2:
            return language.lower()

        code_mapping = {
            "english": "en",
            "french": "fr",
            "swahili": "sw",
            "spanish": "es",
            "arabic": "ar",
            "portuguese": "pt",
        }

        return code_mapping.get(language.lower(), "en")

    def _get_detection_confidence(self, text: str, detected_lang: str) -> float:
        """Calculate confidence score for language detection."""
        try:
            # Use langdetect's confidence if available
            from langdetect import detect_langs

            lang_probs = detect_langs(text)
            for lang_prob in lang_probs:
                if lang_prob.lang == detected_lang:
                    return round(lang_prob.prob, 2)

            return 0.5  # Default confidence

        except Exception:
            return 0.5

    def validate_translation_quality(
        self, original: str, translated: str, target_lang: str
    ) -> Dict[str, any]:
        """
        Validate translation quality using simple heuristics.

        Returns:
            Dictionary with quality metrics and suggestions
        """
        try:
            # Basic length check
            length_ratio = len(translated) / len(original) if original else 0

            # Check for untranslated segments
            has_untranslated = any(
                word in translated.lower()
                for word in original.lower().split()
                if len(word) > 6  # Skip short common words
            )

            # Settlement-specific term preservation
            preserved_terms = [
                "Nairobi",
                "Westlands",
                "Kilimani",
                "Karen",
                "Lavington",
                "KSh",
                "matatu",
                "boda boda",
                "mpesa",
                "nyama choma",
            ]

            properly_preserved = all(
                term in translated
                for term in preserved_terms
                if term in original
            )

            quality_score = 1.0
            issues = []

            if length_ratio < 0.3 or length_ratio > 3.0:
                quality_score -= 0.3
                issues.append("Unusual length difference")

            if has_untranslated and target_lang != "en":
                quality_score -= 0.2
                issues.append("Contains untranslated segments")

            if not properly_preserved:
                quality_score -= 0.2
                issues.append("Important terms not preserved")

            return {
                "quality_score": max(quality_score, 0.0),
                "length_ratio": length_ratio,
                "issues": issues,
                "properly_preserved": properly_preserved,
            }

        except Exception as e:
            logger.error(f"Translation validation failed: {str(e)}")
            return {
                "quality_score": 0.5,
                "issues": ["Validation failed"],
                "error": str(e),
            }

    def preprocess_multilingual_query(self, query: str) -> str:
        """Preprocess query to handle common multilingual patterns."""
        # Handle code-switching patterns common in East Africa
        patterns = [
            # Swahili-English mixing
            (r"\bni\s+(\w+)", r"I am \1"),
            (r"\bnataka\s+", "I want "),
            (r"\bninahitaji\s+", "I need "),
            (r"\bwapi\s+", "where "),
            (r"\bvipi\s+", "how "),
            # French-English mixing
            (r"\bje\s+(\w+)", r"I \1"),
            (r"\bou\s+est\s+", "where is "),
            (r"\bcombien\s+", "how much "),
        ]

        processed_query = query
        for pattern, replacement in patterns:
            processed_query = re.sub(
                pattern, replacement, processed_query, flags=re.IGNORECASE
            )

        return processed_query

    def get_language_stats(self) -> Dict[str, any]:
        """Get statistics about language processing."""
        return {
            "supported_languages": list(self.supported_languages.values()),
            "detection_enabled": self.detection_enabled,
            "primary_language": self.primary_language,
            "translation_provider": "hybrid_gpt_google",
            "features": [
                "Mixed language detection",
                "Context-aware translation",
                "Settlement domain optimization",
                "Quality validation",
            ],
        }
