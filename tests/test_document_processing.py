"""Module 6: Document Processing — SB-TECH-2026-001 §5.2"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("SETTLEBOT_API_KEY", "test-api-key-for-unit-tests")
os.environ.setdefault("SETTLEBOT_LOCALE", "nairobi")

from config.settings import settings

_LOCALE = settings.locale
_NEIGHBORHOOD = _LOCALE.key_neighborhoods[0] if _LOCALE else "Westlands"
_CURRENCY_SYM = _LOCALE.currency_symbol if _LOCALE else "KSh"


def _make_processor(tmp_dir: Path):
    """
    Build a DocumentProcessor with mocked SemanticChunker and EmbeddingService.
    :param tmp_dir: Path - Temporary directory for data files.
    :return: DocumentProcessor instance.
    """
    mock_chunker = MagicMock()

    def _create_chunks_passthrough(text: str, doc_id: str, **kw):
        """Return a single chunk whose text mirrors the input so entity extraction works."""
        mc = MagicMock()
        mc.text = text
        mc.semantic_score = 0.75
        mc.topic_coherence = 0.8
        mc.chunk_type = "paragraph"
        return [mc]

    mock_chunker.create_chunks.side_effect = _create_chunks_passthrough
    mock_chunker.strategy = "settlement_optimized"

    mock_embed_svc = MagicMock()
    mock_embed_svc.embed_query.return_value = None

    with (
        patch("services.document_processor.SemanticChunker", return_value=mock_chunker),
        patch(
            "services.document_processor.EmbeddingService", return_value=mock_embed_svc
        ),
        patch(
            "services.document_processor.nlp",
            MagicMock(return_value=MagicMock(ents=[])),
        ),
    ):
        from services.document_processor import DocumentProcessor

        dp = DocumentProcessor(
            raw_dir=str(tmp_dir / "raw"),
            processed_dir=str(tmp_dir / "processed"),
            chunk_dir=str(tmp_dir / "chunks"),
            dedup_dir=str(tmp_dir / "dedup"),
            embedding_service=mock_embed_svc,
            locale=_LOCALE,
            enable_deduplication=False,
        )
        dp.semantic_chunker = mock_chunker
    return dp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_process_txt_file() -> None:
    """Valid .txt file produces at least one chunk and returns doc_id in metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dp = _make_processor(tmp_path)

        txt_file = tmp_path / "test_doc.txt"
        txt_file.write_text(
            f"International student guide. Housing near {_NEIGHBORHOOD}. Visa requirements.",
            encoding="utf-8",
        )
        result = dp.process_document(txt_file)

        assert result is not None
        assert "doc_id" in result
        assert result.get("num_chunks", 0) >= 1


def test_unsupported_extension_raises() -> None:
    """.exe file raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dp = _make_processor(tmp_path)

        exe_file = tmp_path / "malware.exe"
        exe_file.write_bytes(b"MZ")

        with pytest.raises(ValueError):
            dp.process_document(exe_file)


def test_settlement_score_is_float_in_range() -> None:
    """Every chunk produced by _create_settlement_chunks has settlement_score in [0.0, 1.0]."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dp = _make_processor(tmp_path)

        text = f"International student housing near {_NEIGHBORHOOD}. Cost {_CURRENCY_SYM} 20000."
        chunks, _ = dp._create_settlement_chunks(text, "doc_test", "/tmp/test.txt")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert 0.0 <= chunk.settlement_score <= 1.0, (
                f"settlement_score {chunk.settlement_score} out of [0.0, 1.0]"
            )


def test_location_entities_extracted() -> None:
    """
    A chunk whose text contains the first locale neighborhood has that
    neighborhood in location_entities.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dp = _make_processor(tmp_path)
        text = f"Find student accommodation near {_NEIGHBORHOOD} university."
        chunks, _ = dp._create_settlement_chunks(text, "doc_loc", "/tmp/loc.txt")

        assert len(chunks) >= 1
        found = any(_NEIGHBORHOOD in (c.location_entities or []) for c in chunks)
        assert found, (
            f"Expected '{_NEIGHBORHOOD}' in location_entities of at least one chunk"
        )


def test_cost_entities_extracted() -> None:
    """
    A chunk containing '{currency_symbol} 15,000' yields at least one
    match in cost_entities.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dp = _make_processor(tmp_path)
        text = f"Rent costs {_CURRENCY_SYM} 15,000 per month for a bedsitter."
        chunks, _ = dp._create_settlement_chunks(text, "doc_cost", "/tmp/cost.txt")

        assert len(chunks) >= 1
        found = any(len(c.cost_entities or []) > 0 for c in chunks)
        assert found, (
            f"Expected at least one cost_entity matching '{_CURRENCY_SYM} 15,000'"
        )


def test_document_index_persists() -> None:
    """After process_document(), calling list_documents() includes the doc_id."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dp = _make_processor(tmp_path)

        txt_file = tmp_path / "persist_test.txt"
        txt_file.write_text(
            f"Student accommodation guide for {_NEIGHBORHOOD} area.",
            encoding="utf-8",
        )
        metadata = dp.process_document(txt_file)
        assert metadata is not None

        doc_ids = [d["doc_id"] for d in dp.list_documents()]
        assert metadata["doc_id"] in doc_ids


def test_delete_document_removes_files() -> None:
    """
    After process_document() and then delete_document(doc_id), the chunks
    file and processed text file no longer exist on disk.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dp = _make_processor(tmp_path)

        txt_file = tmp_path / "delete_test.txt"
        txt_file.write_text(
            "International student settlement information guide.",
            encoding="utf-8",
        )
        metadata = dp.process_document(txt_file)
        assert metadata is not None

        doc_id = metadata["doc_id"]
        chunks_path = Path(metadata["chunks_path"])
        processed_path = Path(metadata["processed_path"])

        dp.delete_document(doc_id)

        assert not chunks_path.exists(), "Chunks file should be deleted"
        assert not processed_path.exists(), "Processed text file should be deleted"
