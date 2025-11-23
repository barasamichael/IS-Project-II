import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from datetime import datetime

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from config.settings import settings
from config.settings import ROOT_DIR
from services.vector_db import VectorDBService
from services.embeddings import EmbeddingService
from services.intent_recognizer import IntentRecognizer
from services.language_processor import LanguageProcessor
from services.response_generator import ResponseGenerator
from services.document_processor import DocumentProcessor

# Setup rich console
console = Console()

# Setup logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("settlebot-cli")

# Create Typer app
app = typer.Typer(
    help="SettleBot CLI - Settlement Assistant for International Students in Nairobi"
)

# Initialize services
embedding_service = EmbeddingService()
vector_db_service = VectorDBService(embedding_service=embedding_service)
intent_recognizer = IntentRecognizer()
language_processor = LanguageProcessor()
response_generator = ResponseGenerator()
document_processor = DocumentProcessor(
    embedding_service=embedding_service,
    enable_deduplication=settings.deduplication.enabled,
    similarity_threshold=settings.deduplication.similarity_threshold,
)

# Conversation memory for interactive sessions
conversation_memory: List[Dict[str, Any]] = []


@app.command()
def status():
    """Display comprehensive system status."""
    console.print("[bold blue]SettleBot System Status[/bold blue]")

    # System health table
    health_table = Table(title="Service Health")
    health_table.add_column("Service", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Details", style="dim")

    # Check services
    services = [
        (
            "OpenAI API",
            "Available" if os.getenv("OPENAI_API_KEY") else "Missing",
            "Required for embeddings and responses",
        ),
        (
            "Vector Database",
            "Ready",
            f"Collection: {settings.vector_db.collection_name}",
        ),
        (
            "Document Processor",
            "Ready",
            f"Strategy: {settings.chunking.strategy}",
        ),
        (
            "Language Processor",
            "Ready" if settings.language.detection_enabled else "Disabled",
            f"Languages: {len(settings.language.supported_languages)}",
        ),
        ("Embedding Service", "Ready", f"Model: {settings.embedding.model}"),
        ("Response Generator", "Ready", f"Model: {settings.llm.model}"),
    ]

    for service, status, details in services:
        health_table.add_row(service, status, details)

    console.print(health_table)

    # Configuration table
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    config_items = [
        ("Environment", settings.environment),
        ("Chunking Strategy", settings.chunking.strategy),
        ("Chunk Size", str(settings.chunking.chunk_size)),
        ("Chunk Overlap", str(settings.chunking.chunk_overlap)),
        ("Deduplication", str(settings.deduplication.enabled)),
        ("Language Detection", str(settings.language.detection_enabled)),
        ("Settlement Optimization", "Enabled"),
        ("Domain", "Nairobi Kenya Settlement"),
    ]

    for setting, value in config_items:
        config_table.add_row(setting, value)

    console.print(config_table)


@app.command()
def process_document(
    file_path: str = typer.Argument(..., help="Path to the document file")
):
    """Process a settlement document."""
    console.print(
        f"Processing settlement document: [bold blue]{file_path}[/bold blue]"
    )

    try:
        file_path = Path(file_path)

        if not file_path.exists():
            console.print(f"[bold red]File not found:[/bold red] {file_path}")
            return

        if not document_processor.is_file_supported(file_path):
            console.print(
                f"[bold red]Unsupported file type:[/bold red] {file_path.suffix}"
            )
            console.print(
                f"Supported formats: {', '.join(document_processor.get_supported_extensions())}"
            )
            return

        # Process document
        metadata = document_processor.process_document(file_path)

        if metadata:
            console.print(
                "[bold green]Document processed successfully![/bold green]"
            )

            # Display processing results
            results_table = Table(title="Processing Results")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="green")

            results_table.add_row("Document ID", metadata["doc_id"])
            results_table.add_row("Document Type", metadata["doc_type"])
            results_table.add_row(
                "Number of Chunks", str(metadata["num_chunks"])
            )
            results_table.add_row(
                "Chunking Strategy", metadata["chunking_strategy"]
            )

            if "avg_settlement_score" in metadata:
                results_table.add_row(
                    "Settlement Relevance",
                    f"{metadata['avg_settlement_score']:.3f}",
                )

            console.print(results_table)

            # Generate embeddings
            console.print("Generating embeddings...")
            embedding_service.embed_chunks(metadata["chunks_path"])

            # Index in vector database
            console.print("Indexing in vector database...")
            vector_db_service.index_chunks(metadata["chunks_path"])

            console.print(
                "[bold green]Document processing complete![/bold green]"
            )
        else:
            console.print("[bold red]Document processing failed[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def process_folder(
    folder_path: str = typer.Argument(
        ..., help="Path to folder containing documents"
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Process subfolders"
    ),
):
    """Process all documents in a folder."""
    console.print(f"Processing folder: [bold blue]{folder_path}[/bold blue]")

    try:
        folder_path = Path(folder_path)

        if not folder_path.exists() or not folder_path.is_dir():
            console.print(
                f"[bold red]Folder not found:[/bold red] {folder_path}"
            )
            return

        # Get files to process
        pattern = "**/*" if recursive else "*"
        files = [
            f
            for f in folder_path.glob(pattern)
            if f.is_file() and document_processor.is_file_supported(f)
        ]

        if not files:
            console.print("[bold yellow]No supported files found[/bold yellow]")
            return

        console.print(f"Found {len(files)} supported files")

        # Process files
        successful = 0
        failed = 0

        for file_path in files:
            try:
                console.print(f"Processing: {file_path.name}")
                metadata = document_processor.process_document(file_path)

                if metadata:
                    # Generate embeddings and index
                    embedding_service.embed_chunks(metadata["chunks_path"])
                    vector_db_service.index_chunks(metadata["chunks_path"])
                    successful += 1
                    console.print(
                        f"  ✓ Processed: {metadata['num_chunks']} chunks"
                    )
                else:
                    failed += 1
                    console.print("  ✗ Failed to process")

            except Exception as e:
                failed += 1
                console.print(f"  ✗ Error: {str(e)}")

        # Summary
        console.print("\n[bold green]Processing complete![/bold green]")
        console.print(f"Successful: {successful}, Failed: {failed}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text"),
    top_k: int = typer.Option(15, "--top-k", "-k", help="Number of results"),
    show_chunks: bool = typer.Option(
        False, "--show-chunks", "-c", help="Show retrieved chunks"
    ),
    language: str = typer.Option(
        "auto",
        "--language",
        "-l",
        help="Query language (auto-detect if 'auto')",
    ),
):
    """Query the settlement knowledge base."""
    console.print(f"Processing query: [bold blue]{query_text}[/bold blue]")

    try:
        # Language processing
        if language == "auto":
            language_result = language_processor.detect_and_process_query(
                query_text
            )
            english_query = language_result["english_query"]
            detected_lang = language_result["detected_language"]

            if language_result["needs_translation"]:
                console.print(f"Detected language: {detected_lang}")
                console.print(f"English query: {english_query}")
        else:
            english_query = query_text
            detected_lang = language
            language_result = {"needs_translation": False}

        # Intent recognition - Updated to use new method
        intent_info = intent_recognizer.get_intent_info(english_query)

        # Display intent analysis
        intent_table = Table(title="Query Analysis")
        intent_table.add_column("Aspect", style="cyan")
        intent_table.add_column("Value", style="green")

        intent_table.add_row("Intent Type", intent_info["intent_type"].value)
        intent_table.add_row("Topic", intent_info["topic"].value)
        intent_table.add_row("Confidence", f"{intent_info['confidence']:.2f}")

        console.print(intent_table)

        # Retrieve context
        if intent_info["intent_type"].value != "off_topic":
            retrieved_chunks = vector_db_service.search(
                english_query, top_k=top_k
            )
        else:
            retrieved_chunks = []

        # Generate response
        response_data = response_generator.generate_response(
            query=query_text,
            retrieved_context=retrieved_chunks,
            intent_info=intent_info,
        )

        # Display response
        console.print("\n[bold green]Response:[/bold green]")
        response_text = response_data.get("response", "No response generated")

        console.print(
            Panel.fit(
                response_text, title="SettleBot Response", border_style="green"
            )
        )

        # Show additional information
        if response_data.get("token_usage"):
            token_info = response_data["token_usage"]
            console.print(
                f"\nTokens used: {token_info['total_tokens']} (prompt: {token_info['prompt_tokens']}, completion: {token_info['completion_tokens']})"
            )

        if retrieved_chunks:
            console.print(f"Retrieved {len(retrieved_chunks)} relevant chunks")

        # Show chunks if requested
        if show_chunks and retrieved_chunks:
            console.print("\n[bold yellow]Retrieved Chunks:[/bold yellow]")
            for i, chunk in enumerate(retrieved_chunks[:5], 1):
                chunk_text = chunk.get("text", "")[:200] + "..."
                score = chunk.get("score", 0.0)
                settlement_score = chunk.get("settlement_score", 0.0)

                console.print(
                    f"\n{i}. [dim]Score: {score:.3f}, Settlement: {settlement_score:.3f}[/dim]"
                )
                console.print(f"   {chunk_text}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def interactive():
    """Start an interactive settlement assistance session."""
    console.print("[bold blue]SettleBot Interactive Session[/bold blue]")
    console.print(
        "Ask questions about settling in Nairobi as an international student."
    )
    console.print("Type 'quit' to exit, 'clear' to clear conversation history")
    console.print()

    session_memory = []

    while True:
        try:
            user_input = typer.prompt("\nYour question")

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print(
                    "[bold blue]Session ended. Best of luck with your settlement in Nairobi![/bold blue]"
                )
                break
            elif user_input.lower() == "clear":
                session_memory.clear()
                console.print("[yellow]Conversation history cleared[/yellow]")
                continue

            # Language processing
            language_result = language_processor.detect_and_process_query(
                user_input
            )
            english_query = language_result["english_query"]

            if language_result["needs_translation"]:
                console.print(
                    f"[dim]Detected language: {language_result['detected_language']}[/dim]"
                )

            # Intent recognition - Updated to use new method
            intent_info = intent_recognizer.get_intent_info(english_query)

            # Retrieve context
            retrieved_chunks = []
            if intent_info["intent_type"].value != "off_topic":
                retrieved_chunks = vector_db_service.search(
                    english_query, top_k=12
                )

            # Generate response
            response_data = response_generator.generate_response(
                query=user_input,
                retrieved_context=retrieved_chunks,
                intent_info=intent_info,
            )

            # Display response with intent info
            console.print(
                f"[dim]Intent: {intent_info['intent_type'].value} | Topic: {intent_info['topic'].value} | Confidence: {intent_info['confidence']:.2f}[/dim]"
            )
            console.print(
                f"[bold green]SettleBot:[/bold green] {response_data.get('response', 'No response generated')}"
            )

            # Update session memory
            session_memory.append(
                {
                    "user": user_input,
                    "response": response_data.get("response", ""),
                    "intent": intent_info["intent_type"].value,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Keep memory manageable
            if len(session_memory) > 20:
                session_memory = session_memory[-15:]

        except KeyboardInterrupt:
            console.print(
                "\n[bold blue]Session ended. Best of luck with your settlement![/bold blue]"
            )
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def list_documents(
    show_details: bool = typer.Option(
        False, "--details", "-d", help="Show detailed information"
    ),
):
    """List all processed documents."""
    try:
        documents = document_processor.list_documents()

        if not documents:
            console.print("[yellow]No documents found.[/yellow]")
            return

        # Create table
        table = Table(title=f"Settlement Documents ({len(documents)})")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Filename", style="green")
        table.add_column("Type", style="blue")
        table.add_column("Chunks", style="magenta", justify="right")
        table.add_column("Size (MB)", style="yellow", justify="right")
        table.add_column("Settlement Score", style="red", justify="right")

        if show_details:
            table.add_column("Processed Date", style="dim")

        for doc in documents:
            size_mb = (
                doc.get("file_size", 0) / (1024 * 1024)
                if doc.get("file_size")
                else 0
            )
            settlement_score = doc.get("avg_settlement_score", 0)

            row = [
                doc["doc_id"][:8] + "...",
                doc["file_name"],
                doc["doc_type"],
                str(doc["num_chunks"]),
                f"{size_mb:.2f}",
                f"{settlement_score:.3f}",
            ]

            if show_details:
                processed_date = datetime.fromtimestamp(
                    doc.get("processed_date", 0)
                ).strftime("%Y-%m-%d %H:%M")
                row.append(processed_date)

            table.add_row(*row)

        console.print(table)

        # Show statistics
        stats = document_processor.get_processing_stats()
        console.print(f"\nTotal documents: {stats['total_documents']}")
        console.print(f"Supported formats: {stats['supported_formats']}")
        console.print(
            f"Settlement optimization: {stats.get('settlement_optimized', False)}"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def rebuild_index():
    """Rebuild the vector database index."""
    console.print("[bold blue]Rebuilding vector database index[/bold blue]")

    try:
        if not typer.confirm(
            "This will recreate the entire index. Continue?", default=False
        ):
            console.print("Operation cancelled.")
            return

        documents = document_processor.list_documents()

        if not documents:
            console.print("[yellow]No documents found to index.[/yellow]")
            return

        console.print(f"Found {len(documents)} documents to index")

        # Recreate collection
        vector_db_service.initialize_collection(recreate=True)

        # Index documents
        for i, doc in enumerate(documents, 1):
            console.print(f"Indexing {doc['file_name']} ({i}/{len(documents)})")
            vector_db_service.index_chunks(doc["chunks_path"])

        # Final stats
        stats = vector_db_service.get_collection_stats()
        console.print("[bold green]Index rebuilt successfully![/bold green]")
        console.print(f"Total vectors: {stats['count']}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def search_topic(
    topic: str = typer.Argument(..., help="Settlement topic to search"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
):
    """Search by settlement topic (housing, transport, safety, etc.)."""
    console.print(f"Searching for topic: [bold blue]{topic}[/bold blue]")

    try:
        results = vector_db_service.search_by_topic(topic, top_k)

        if not results:
            console.print("[yellow]No results found for this topic.[/yellow]")
            return

        # Display results
        for i, result in enumerate(results, 1):
            score = result.get("score", 0.0)
            text = result.get("text", "")[:200] + "..."

            console.print(f"\n{i}. [dim]Score: {score:.3f}[/dim]")
            console.print(f"   {text}")

        console.print(f"\nFound {len(results)} results for topic '{topic}'")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def test_languages():
    """Test multilingual query processing."""
    console.print("[bold blue]Testing Multilingual Support[/bold blue]")

    test_queries = [
        ("Where can I find housing in Nairobi?", "english"),
        ("Où puis-je trouver un logement à Nairobi?", "french"),
        ("Niweze kupata nyumba wapi Nairobi?", "swahili"),
        ("¿Dónde puedo encontrar alojamiento en Nairobi?", "spanish"),
    ]

    for query, expected_lang in test_queries:
        console.print(f"\nTesting query: [cyan]{query}[/cyan]")

        try:
            # Test language detection
            lang_result = language_processor.detect_and_process_query(query)

            console.print(f"  Detected: {lang_result['detected_language']}")
            console.print(f"  English: {lang_result['english_query']}")
            console.print(
                f"  Translation needed: {lang_result['needs_translation']}"
            )

        except Exception as e:
            console.print(f"  [red]Error: {str(e)}[/red]")

    console.print("\n[bold green]Multilingual testing complete![/bold green]")


@app.command()
def process_url(
    url: str = typer.Argument(..., help="URL to process"),
    name: str = typer.Option(
        None, "--name", "-n", help="Custom name for the document"
    ),
):
    """Process content from a URL for settlement information."""
    console.print(f"Processing URL: [bold blue]{url}[/bold blue]")

    try:
        # Validate URL format
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        # Process URL
        metadata = document_processor.process_url(url, name)

        if metadata:
            console.print(
                "[bold green]URL processed successfully![/bold green]"
            )

            # Display processing results
            results_table = Table(title="URL Processing Results")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="green")

            results_table.add_row("Document ID", metadata["doc_id"])
            results_table.add_row("Document Type", metadata["doc_type"])
            results_table.add_row(
                "Number of Chunks", str(metadata["num_chunks"])
            )
            results_table.add_row("Source URL", metadata["file_path"])

            if "avg_settlement_score" in metadata:
                results_table.add_row(
                    "Settlement Relevance",
                    f"{metadata['avg_settlement_score']:.3f}",
                )

            console.print(results_table)

            # Generate embeddings
            console.print("Generating embeddings...")
            embedding_service.embed_chunks(metadata["chunks_path"])

            # Index in vector database
            console.print("Indexing in vector database...")
            vector_db_service.index_chunks(metadata["chunks_path"])

            console.print("[bold green]URL processing complete![/bold green]")
        else:
            console.print("[bold red]URL processing failed[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def process_sitemap(
    sitemap_url: str = typer.Argument(..., help="Sitemap URL to process"),
    max_pages: int = typer.Option(
        50, "--max-pages", "-m", help="Maximum pages to process"
    ),
    settlement_only: bool = typer.Option(
        True, "--settlement-only", help="Only process settlement-relevant pages"
    ),
):
    """Process multiple pages from a sitemap for settlement content."""
    console.print(f"Processing sitemap: [bold blue]{sitemap_url}[/bold blue]")
    console.print(
        f"Max pages: {max_pages}, Settlement filter: {settlement_only}"
    )

    try:
        # Process sitemap
        results = document_processor.process_sitemap(sitemap_url, max_pages)

        if results:
            console.print(
                "[bold green]Sitemap processed successfully![/bold green]"
            )

            # Display summary statistics
            summary_table = Table(title="Sitemap Processing Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")

            total_chunks = sum(result["num_chunks"] for result in results)
            avg_settlement_score = sum(
                result.get("avg_settlement_score", 0) for result in results
            ) / len(results)

            summary_table.add_row("Pages Processed", str(len(results)))
            summary_table.add_row("Total Chunks Created", str(total_chunks))
            summary_table.add_row(
                "Avg Settlement Score", f"{avg_settlement_score:.3f}"
            )
            summary_table.add_row("Source Sitemap", sitemap_url)

            console.print(summary_table)

            # Show top settlement-relevant pages
            if results:
                console.print(
                    "\n[bold yellow]Top Settlement-Relevant Pages:[/bold yellow]"
                )
                sorted_results = sorted(
                    results,
                    key=lambda x: x.get("avg_settlement_score", 0),
                    reverse=True,
                )

                for i, result in enumerate(sorted_results[:5], 1):
                    score = result.get("avg_settlement_score", 0)
                    chunks = result["num_chunks"]
                    name = (
                        result["file_name"][:50] + "..."
                        if len(result["file_name"]) > 50
                        else result["file_name"]
                    )

                    console.print(f"  {i}. {name}")
                    console.print(
                        f"     Settlement Score: {score:.3f} | Chunks: {chunks}"
                    )

            # Process embeddings and indexing for all results
            console.print("\nGenerating embeddings for all pages...")
            for result in results:
                embedding_service.embed_chunks(result["chunks_path"])

            console.print("Indexing all pages in vector database...")
            for result in results:
                vector_db_service.index_chunks(result["chunks_path"])

            console.print(
                "[bold green]Sitemap processing complete![/bold green]"
            )
        else:
            console.print(
                "[bold yellow]No settlement-relevant pages found in sitemap[/bold yellow]"
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def validate_intent():
    """Validate intent recognizer patterns and embeddings."""
    console.print("[bold blue]Validating Intent Recognition System[/bold blue]")

    try:
        # Get validation results
        validation_results = intent_recognizer.validate_patterns()

        console.print(f"Valid patterns: {validation_results['valid_patterns']}")
        console.print(
            f"Invalid patterns: {validation_results['invalid_patterns']}"
        )
        console.print(f"Overall health: {validation_results['overall_health']}")

        # Show pattern details table
        details_table = Table(title="Pattern Validation Details")
        details_table.add_column("Intent Type", style="cyan")
        details_table.add_column("Status", style="green")
        details_table.add_column("Examples", style="yellow")

        for intent_type, details in validation_results[
            "pattern_details"
        ].items():
            status_color = "green" if details["status"] == "valid" else "red"
            example_count = details.get("example_count", 0)

            details_table.add_row(
                intent_type,
                f"[{status_color}]{details['status']}[/{status_color}]",
                str(example_count),
            )

        console.print(details_table)

        # Get intent recognizer stats
        stats = intent_recognizer.get_stats()
        console.print(f"\nTotal intents: {stats['total_intents']}")
        console.print(
            f"Classification method: {stats['classification_method']}"
        )
        console.print(f"Cache enabled: {stats['cache_enabled']}")

    except Exception as e:
        console.print(
            f"[bold red]Error validating intent recognizer:[/bold red] {str(e)}"
        )


@app.command()
def test_intent(
    test_query: str = typer.Argument(
        ..., help="Query to test intent recognition"
    )
):
    """Test intent recognition for a specific query."""
    console.print(
        f"Testing intent recognition for: [bold blue]{test_query}[/bold blue]"
    )

    try:
        # Get intent classification
        intent_info = intent_recognizer.get_intent_info(test_query)

        # Display results table
        results_table = Table(title="Intent Classification Results")
        results_table.add_column("Property", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Intent Type", intent_info["intent_type"].value)
        results_table.add_row("Topic", intent_info["topic"].value)
        results_table.add_row("Confidence", f"{intent_info['confidence']:.3f}")
        results_table.add_row(
            "Settlement Relevance", f"{intent_info['settlement_relevance']:.3f}"
        )
        results_table.add_row(
            "Classification Method", intent_info["classification_method"]
        )
        results_table.add_row("Off Topic", str(intent_info["is_off_topic"]))

        console.print(results_table)

        # Show semantic scores if available
        if intent_info.get("semantic_scores"):
            console.print(
                "\n[bold yellow]Semantic Scores by Intent:[/bold yellow]"
            )
            semantic_scores = intent_info["semantic_scores"]

            # Sort by score for better display
            sorted_scores = sorted(
                semantic_scores.items(), key=lambda x: x[1], reverse=True
            )

            for intent_type, score in sorted_scores[:5]:  # Show top 5
                console.print(f"  {intent_type.value}: {score:.3f}")

        # Show off-topic indicators if any
        if intent_info.get("off_topic_indicators"):
            console.print("\n[bold red]Off-topic indicators:[/bold red]")
            for indicator in intent_info["off_topic_indicators"]:
                console.print(f"  - {indicator}")

    except Exception as e:
        console.print(f"[bold red]Error testing intent:[/bold red] {str(e)}")


@app.command()
def clear_intent_cache():
    """Clear the intent recognizer embedding cache."""
    try:
        intent_recognizer.clear_cache()
        console.print(
            "[bold green]Intent recognizer cache cleared successfully![/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Error clearing cache:[/bold red] {str(e)}")


@app.command()
def rebuild_intent_cache():
    """Rebuild the intent recognizer embedding cache."""
    try:
        intent_recognizer.rebuild_cache()
        console.print(
            "[bold green]Intent recognizer cache rebuilt successfully![/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Error rebuilding cache:[/bold red] {str(e)}")


@app.command()
def web_stats():
    """Display statistics for web-processed documents."""
    console.print("[bold blue]Web Processing Statistics[/bold blue]")

    try:
        documents = document_processor.list_documents()
        web_docs = [
            doc
            for doc in documents
            if doc.get("source_type") in ["web_url", "web_sitemap"]
        ]

        if not web_docs:
            console.print("[yellow]No web documents found.[/yellow]")
            return

        # Create statistics table
        stats_table = Table(title=f"Web Documents ({len(web_docs)})")
        stats_table.add_column("Source", style="cyan")
        stats_table.add_column("Count", style="green")
        stats_table.add_column("Avg Chunks", style="blue")
        stats_table.add_column("Avg Settlement Score", style="magenta")

        # Group by source type
        url_docs = [
            doc for doc in web_docs if doc.get("source_type") == "web_url"
        ]
        sitemap_docs = [
            doc for doc in web_docs if doc.get("source_type") == "web_sitemap"
        ]

        if url_docs:
            avg_chunks = sum(doc["num_chunks"] for doc in url_docs) / len(
                url_docs
            )
            avg_score = sum(
                doc.get("avg_settlement_score", 0) for doc in url_docs
            ) / len(url_docs)
            stats_table.add_row(
                "Direct URLs",
                str(len(url_docs)),
                f"{avg_chunks:.1f}",
                f"{avg_score:.3f}",
            )

        if sitemap_docs:
            avg_chunks = sum(doc["num_chunks"] for doc in sitemap_docs) / len(
                sitemap_docs
            )
            avg_score = sum(
                doc.get("avg_settlement_score", 0) for doc in sitemap_docs
            ) / len(sitemap_docs)
            stats_table.add_row(
                "Sitemap Pages",
                str(len(sitemap_docs)),
                f"{avg_chunks:.1f}",
                f"{avg_score:.3f}",
            )

        console.print(stats_table)

        # Show top performing web documents
        if web_docs:
            console.print(
                "\n[bold yellow]Top Settlement-Relevant Web Documents:[/bold yellow]"
            )
            sorted_docs = sorted(
                web_docs,
                key=lambda x: x.get("avg_settlement_score", 0),
                reverse=True,
            )

            for i, doc in enumerate(sorted_docs[:5], 1):
                score = doc.get("avg_settlement_score", 0)
                chunks = doc["num_chunks"]
                url = doc["file_path"]

                console.print(f"  {i}. {url}")
                console.print(
                    f"     Settlement Score: {score:.3f} | Chunks: {chunks}"
                )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def validate_url(
    url: str = typer.Argument(
        ..., help="URL to validate for settlement content"
    )
):
    """Validate if a URL contains settlement-relevant content without processing."""
    console.print(f"Validating URL: [bold blue]{url}[/bold blue]")

    try:
        from langchain_community.document_loaders import WebBaseLoader

        # Load content
        loader = WebBaseLoader(url)
        documents = loader.load()

        if not documents:
            console.print("[bold red]No content found at URL[/bold red]")
            return

        text = "\n\n".join(doc.page_content for doc in documents)

        if not text.strip():
            console.print("[bold red]Empty content at URL[/bold red]")
            return

        # Check settlement relevance using the processor's method
        is_relevant = document_processor._is_settlement_relevant(text)

        # Calculate basic statistics
        word_count = len(text.split())
        char_count = len(text)

        # Check for specific settlement keywords
        settlement_keywords = [
            "international student",
            "nairobi",
            "housing",
            "accommodation",
            "visa",
            "university",
            "transport",
            "safety",
            "cost of living",
        ]

        keyword_matches = [
            keyword
            for keyword in settlement_keywords
            if keyword.lower() in text.lower()
        ]

        # Display validation results
        validation_table = Table(title="URL Validation Results")
        validation_table.add_column("Metric", style="cyan")
        validation_table.add_column("Value", style="green")

        validation_table.add_row(
            "Settlement Relevant", "✓ Yes" if is_relevant else "✗ No"
        )
        validation_table.add_row("Word Count", str(word_count))
        validation_table.add_row("Character Count", str(char_count))
        validation_table.add_row(
            "Settlement Keywords Found", str(len(keyword_matches))
        )

        if keyword_matches:
            validation_table.add_row("Keywords", ", ".join(keyword_matches[:5]))

        console.print(validation_table)

        if is_relevant:
            console.print(
                "[bold green]✓ URL contains settlement-relevant content[/bold green]"
            )
            console.print(
                "Recommended: Process this URL with 'process-url' command"
            )
        else:
            console.print(
                "[bold yellow]⚠ URL may not contain settlement-relevant content[/bold yellow]"
            )
            console.print("Consider reviewing content before processing")

    except Exception as e:
        console.print(f"[bold red]Error validating URL:[/bold red] {str(e)}")


@app.command()
def delete_document(
    doc_id: str = typer.Argument(..., help="Document ID to delete")
):
    """Delete a document and all associated files (chunks, embeddings, etc.)."""
    console.print(f"Deleting document: [bold red]{doc_id}[/bold red]")

    try:
        # Get document info first
        doc_info = document_processor.get_document_info(doc_id)

        if not doc_info:
            console.print(f"[bold red]Document not found:[/bold red] {doc_id}")
            return

        console.print(f"Document: {doc_info['file_name']}")

        # Confirm deletion
        if not typer.confirm(
            f"Are you sure you want to delete '{doc_info['file_name']}'? This action cannot be undone.",
            default=False,
        ):
            console.print("Deletion cancelled.")
            return

        # Delete from vector database first
        try:
            # Remove chunks from vector database
            chunks_path = Path(doc_info.get("chunks_path", ""))
            if chunks_path.exists():
                console.print("Removing chunks from vector database...")
                # Note: ChromaDB doesn't have a direct way to delete by doc_id
                # We'll need to rebuild the index or implement chunk-level deletion
                console.print(
                    "[yellow]Warning: Vector database cleanup requires manual index rebuild[/yellow]"
                )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Vector database cleanup failed: {str(e)}[/yellow]"
            )

        # Delete document using document processor
        success = document_processor.delete_document(doc_id)

        if success:
            console.print(
                "[bold green]Document deleted successfully![/bold green]"
            )

            # Show what was deleted
            deleted_table = Table(title="Deleted Files")
            deleted_table.add_column("Type", style="cyan")
            deleted_table.add_column("Status", style="green")

            deleted_table.add_row("Document metadata", "✓ Removed")
            deleted_table.add_row("Processed text", "✓ Removed")
            deleted_table.add_row("Chunks", "✓ Removed")
            deleted_table.add_row("Embeddings", "✓ Removed")
            deleted_table.add_row("Vector database", "⚠ Requires rebuild")

            console.print(deleted_table)
            console.print(
                "\n[yellow]Note: Run 'rebuild-index' to clean up vector database[/yellow]"
            )

        else:
            console.print("[bold red]Document deletion failed[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def update_document(
    doc_id: str = typer.Argument(..., help="Document ID to update"),
    file_path: str = typer.Argument(..., help="Path to new document file"),
):
    """Update an existing document by replacing it with a new file."""
    console.print(f"Updating document: [bold blue]{doc_id}[/bold blue]")

    try:
        # Check if document exists
        doc_info = document_processor.get_document_info(doc_id)

        if not doc_info:
            console.print(f"[bold red]Document not found:[/bold red] {doc_id}")
            console.print(
                "Use 'process-document' to add a new document instead."
            )
            return

        # Check if new file exists
        new_file_path = Path(file_path)
        if not new_file_path.exists():
            console.print(
                f"[bold red]New file not found:[/bold red] {file_path}"
            )
            return

        if not document_processor.is_file_supported(new_file_path):
            console.print(
                f"[bold red]Unsupported file type:[/bold red] {new_file_path.suffix}"
            )
            return

        # Show current document info
        console.print(f"Current document: [cyan]{doc_info['file_name']}[/cyan]")
        console.print(f"New document: [cyan]{new_file_path.name}[/cyan]")

        # Confirm update
        if not typer.confirm(
            f"Replace '{doc_info['file_name']}' with '{new_file_path.name}'?",
            default=False,
        ):
            console.print("Update cancelled.")
            return

        console.print("Updating document...")

        # Step 1: Delete old document
        console.print("1. Removing old document...")
        delete_success = document_processor.delete_document(doc_id)

        if not delete_success:
            console.print("[bold red]Failed to remove old document[/bold red]")
            return

        # Step 2: Process new document
        console.print("2. Processing new document...")

        # Process the new document
        metadata = document_processor.process_document(new_file_path)

        if not metadata:
            console.print("[bold red]Failed to process new document[/bold red]")
            return

        new_doc_id = metadata["doc_id"]

        # Step 3: Generate embeddings
        console.print("3. Generating embeddings...")
        embedding_service.embed_chunks(metadata["chunks_path"])

        # Step 4: Update vector database
        console.print("4. Updating vector database...")
        vector_db_service.index_chunks(metadata["chunks_path"])

        # Success summary
        console.print("[bold green]Document updated successfully![/bold green]")

        update_table = Table(title="Update Results")
        update_table.add_column("Metric", style="cyan")
        update_table.add_column("Old Value", style="yellow")
        update_table.add_column("New Value", style="green")

        update_table.add_row("Document ID", doc_id, new_doc_id)
        update_table.add_row(
            "Filename", doc_info["file_name"], new_file_path.name
        )
        update_table.add_row(
            "Chunks", str(doc_info["num_chunks"]), str(metadata["num_chunks"])
        )
        update_table.add_row(
            "Document Type", doc_info["doc_type"], metadata["doc_type"]
        )

        if "avg_settlement_score" in metadata:
            old_score = doc_info.get("avg_settlement_score", 0)
            update_table.add_row(
                "Settlement Score",
                f"{old_score:.3f}",
                f"{metadata['avg_settlement_score']:.3f}",
            )

        console.print(update_table)

        if new_doc_id != doc_id:
            console.print(
                f"\n[yellow]Note: Document ID changed from {doc_id} to {new_doc_id}[/yellow]"
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def get_document_info(
    doc_id: str = typer.Argument(..., help="Document ID to get information for")
):
    """Get detailed information about a document."""
    console.print(f"Document Information: [bold blue]{doc_id}[/bold blue]")

    try:
        # Get document info
        doc_info = document_processor.get_document_info(doc_id)

        if not doc_info:
            console.print(f"[bold red]Document not found:[/bold red] {doc_id}")
            return

        # Basic information table
        info_table = Table(title="Document Details")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("Document ID", doc_info["doc_id"])
        info_table.add_row("Filename", doc_info["file_name"])
        info_table.add_row("Document Type", doc_info["doc_type"])
        info_table.add_row("File Path", str(doc_info["file_path"]))

        # File size
        file_size = doc_info.get("file_size", 0)
        if file_size > 0:
            size_mb = file_size / (1024 * 1024)
            info_table.add_row("File Size", f"{size_mb:.2f} MB")

        # Processing info
        info_table.add_row("Number of Chunks", str(doc_info["num_chunks"]))

        if "avg_settlement_score" in doc_info:
            info_table.add_row(
                "Settlement Score", f"{doc_info['avg_settlement_score']:.3f}"
            )

        # Dates
        if "last_modified" in doc_info and doc_info["last_modified"]:
            last_mod = datetime.fromtimestamp(
                doc_info["last_modified"]
            ).strftime("%Y-%m-%d %H:%M")
            info_table.add_row("Last Modified", last_mod)

        if "processed_date" in doc_info:
            processed = datetime.fromtimestamp(
                doc_info["processed_date"]
            ).strftime("%Y-%m-%d %H:%M")
            info_table.add_row("Processed Date", processed)

        # Processing configuration
        if "chunking_strategy" in doc_info:
            info_table.add_row(
                "Chunking Strategy", doc_info["chunking_strategy"]
            )

        if "settlement_optimized" in doc_info:
            info_table.add_row(
                "Settlement Optimized", str(doc_info["settlement_optimized"])
            )

        console.print(info_table)

        # File paths table
        paths_table = Table(title="Associated Files")
        paths_table.add_column("Type", style="cyan")
        paths_table.add_column("Path", style="green")
        paths_table.add_column("Exists", style="yellow")

        # Check file existence
        paths_to_check = [
            ("Original File", doc_info["file_path"]),
            ("Processed Text", doc_info.get("processed_path")),
            ("Chunks File", doc_info.get("chunks_path")),
        ]

        # Check for embeddings file
        embeddings_dir = ROOT_DIR / "data" / "embeddings"
        embeddings_file = (
            embeddings_dir / f"{doc_info['doc_id']}_embeddings.npz"
        )
        paths_to_check.append(("Embeddings", str(embeddings_file)))

        for file_type, file_path in paths_to_check:
            if file_path:
                exists = "✓" if Path(file_path).exists() else "✗"
                exists_style = "green" if exists == "✓" else "red"
                paths_table.add_row(
                    file_type,
                    str(file_path),
                    f"[{exists_style}]{exists}[/{exists_style}]",
                )

        console.print(paths_table)

        # Show chunk statistics if chunks file exists
        chunks_path = doc_info.get("chunks_path")
        if chunks_path and Path(chunks_path).exists():
            try:
                chunk_count = 0
                total_chars = 0
                total_words = 0

                with open(chunks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            chunk = json.loads(line)
                            chunk_count += 1
                            text = chunk.get("text", "")
                            total_chars += len(text)
                            total_words += len(text.split())
                        except json.JSONDecodeError:
                            continue

                if chunk_count > 0:
                    stats_table = Table(title="Chunk Statistics")
                    stats_table.add_column("Metric", style="cyan")
                    stats_table.add_column("Value", style="green")

                    stats_table.add_row("Total Chunks", str(chunk_count))
                    stats_table.add_row("Total Characters", f"{total_chars:,}")
                    stats_table.add_row("Total Words", f"{total_words:,}")
                    stats_table.add_row(
                        "Avg Characters per Chunk",
                        f"{total_chars // chunk_count:,}",
                    )
                    stats_table.add_row(
                        "Avg Words per Chunk", f"{total_words // chunk_count:,}"
                    )

                    console.print(stats_table)

            except Exception as e:
                console.print(
                    f"[yellow]Could not read chunk statistics: {str(e)}[/yellow]"
                )

        # Vector database status
        try:
            vector_stats = vector_db_service.get_collection_stats()
            console.print(
                f"\nVector Database: {vector_stats['count']} total vectors"
            )
        except Exception as e:
            console.print(
                f"\n[yellow]Vector database info unavailable: {str(e)}[/yellow]"
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def find_document(
    search_term: str = typer.Argument(
        ..., help="Search term (filename, doc_id, or keyword)"
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of results"
    ),
):
    """Find documents by filename, document ID, or keyword."""
    console.print(
        f"Searching for documents: [bold blue]{search_term}[/bold blue]"
    )

    try:
        documents = document_processor.list_documents()

        if not documents:
            console.print("[yellow]No documents found in the system.[/yellow]")
            return

        # Search in multiple fields
        matches = []
        search_lower = search_term.lower()

        for doc in documents:
            score = 0

            # Exact doc_id match (highest priority)
            if doc["doc_id"].lower() == search_lower:
                score += 100
            elif search_lower in doc["doc_id"].lower():
                score += 50

            # Filename match
            if search_lower in doc["file_name"].lower():
                score += 30

            # File path match
            if search_lower in str(doc["file_path"]).lower():
                score += 20

            # Doc type match
            if search_lower in doc["doc_type"].lower():
                score += 10

            if score > 0:
                matches.append((doc, score))

        if not matches:
            console.print(
                f"[yellow]No documents found matching '{search_term}'[/yellow]"
            )
            return

        # Sort by score and limit results
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:limit]

        # Display results
        results_table = Table(title=f"Search Results ({len(matches)} found)")
        results_table.add_column("Doc ID", style="cyan", no_wrap=True)
        results_table.add_column("Filename", style="green")
        results_table.add_column("Type", style="blue")
        results_table.add_column("Chunks", style="magenta", justify="right")
        results_table.add_column(
            "Settlement Score", style="red", justify="right"
        )
        results_table.add_column("Match Score", style="yellow", justify="right")

        for doc, match_score in matches:
            settlement_score = doc.get("avg_settlement_score", 0)
            results_table.add_row(
                doc["doc_id"][:12] + "...",
                doc["file_name"][:40]
                + ("..." if len(doc["file_name"]) > 40 else ""),
                doc["doc_type"],
                str(doc["num_chunks"]),
                f"{settlement_score:.3f}",
                str(match_score),
            )

        console.print(results_table)

        console.print(f"\nFound {len(matches)} matching documents")
        console.print(
            "Use 'get-document-info <doc_id>' for detailed information"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


@app.command()
def check_health():
    """Check the health of all system components."""
    console.print("[bold blue]SettleBot System Health Check[/bold blue]")

    try:
        # Check vector database health
        health_results = vector_db_service.health_check()

        health_table = Table(title="System Health Status")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="green")
        health_table.add_column("Details", style="dim")

        # Vector DB health
        db_status = (
            "✓ Healthy" if health_results["overall_health"] else "✗ Unhealthy"
        )
        status_style = "green" if health_results["overall_health"] else "red"

        health_table.add_row(
            "Vector Database",
            f"[{status_style}]{db_status}[/{status_style}]",
            f"Vectors: {health_results.get('vector_count', 0)}",
        )

        # Embedding service health
        embedding_status = (
            "✓ Available"
            if health_results["embedding_service_available"]
            else "✗ Unavailable"
        )
        embedding_style = (
            "green" if health_results["embedding_service_available"] else "red"
        )

        health_table.add_row(
            "Embedding Service",
            f"[{embedding_style}]{embedding_status}[/{embedding_style}]",
            f"Model: {embedding_service.model_name}",
        )

        # Search functionality
        search_status = (
            "✓ Working" if health_results["search_functional"] else "✗ Failed"
        )
        search_style = "green" if health_results["search_functional"] else "red"

        health_table.add_row(
            "Search Function",
            f"[{search_style}]{search_status}[/{search_style}]",
            "Semantic search capability",
        )

        # Intent recognition health
        intent_validation = intent_recognizer.validate_patterns()
        intent_status = (
            "✓ Working" if intent_validation["overall_health"] else "✗ Issues"
        )
        intent_style = "green" if intent_validation["overall_health"] else "red"

        health_table.add_row(
            "Intent Recognition",
            f"[{intent_style}]{intent_status}[/{intent_style}]",
            f"Valid patterns: {intent_validation['valid_patterns']}",
        )

        # Language processing health
        lang_stats = language_processor.get_language_stats()
        lang_status = (
            "✓ Ready" if lang_stats["detection_enabled"] else "⚠ Disabled"
        )
        lang_style = "green" if lang_stats["detection_enabled"] else "yellow"

        health_table.add_row(
            "Language Processing",
            f"[{lang_style}]{lang_status}[/{lang_style}]",
            f"Languages: {lang_stats['total_languages']}",
        )

        # Response generation health
        response_stats = response_generator.get_response_stats()
        resp_status = "✓ Ready"
        resp_style = "green"

        health_table.add_row(
            "Response Generator",
            f"[{resp_style}]{resp_status}[/{resp_style}]",
            f"Model: {response_stats['model']}",
        )

        console.print(health_table)

        # Overall system status
        overall_healthy = (
            health_results["overall_health"]
            and intent_validation["overall_health"]
        )
        overall_status = (
            "System Healthy" if overall_healthy else "System Issues Detected"
        )
        overall_style = "bold green" if overall_healthy else "bold red"

        console.print(f"\n[{overall_style}]{overall_status}[/{overall_style}]")

        if not overall_healthy:
            console.print("\n[bold yellow]Recommended Actions:[/bold yellow]")
            if not health_results["overall_health"]:
                console.print(
                    "  - Check vector database connection and rebuild index if needed"
                )
            if not intent_validation["overall_health"]:
                console.print("  - Rebuild intent recognition cache")

    except Exception as e:
        console.print(f"[bold red]Health check failed:[/bold red] {str(e)}")


@app.command()
def process_urls_file(
    file_path: str = typer.Argument(
        ..., help="Path to text file containing URLs (one per line)"
    ),
    max_urls: int = typer.Option(
        100, "--max-urls", "-m", help="Maximum number of URLs to process"
    ),
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip URL validation before processing"
    ),
    settlement_only: bool = typer.Option(
        True,
        "--settlement-only",
        help="Only process URLs with settlement-relevant content",
    ),
    batch_size: int = typer.Option(
        10, "--batch-size", "-b", help="Number of URLs to process in each batch"
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error",
        help="Continue processing if individual URLs fail",
    ),
):
    """Process multiple URLs from a text file with deduplication and robust error handling."""
    console.print(
        f"Processing URLs from file: [bold blue]{file_path}[/bold blue]"
    )

    try:
        file_path = Path(file_path)
        if not file_path.exists():
            console.print(f"[bold red]File not found:[/bold red] {file_path}")
            return

        # Read URLs from file
        console.print("Reading URLs from file...")
        urls = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                url = line.strip()
                # Skip empty lines and comments
                if url and not url.startswith("#"):
                    # Basic URL validation and normalization
                    if not url.startswith(("http://", "https://")):
                        url = f"https://{url}"
                    urls.append((line_num, url))

        if not urls:
            console.print("[yellow]No valid URLs found in file.[/yellow]")
            return

        console.print(f"Found {len(urls)} URLs in file")

        # Limit URLs if needed
        if len(urls) > max_urls:
            console.print(
                f"[yellow]Limiting to {max_urls} URLs (use --max-urls to change)[/yellow]"
            )
            urls = urls[:max_urls]

        # Check for already processed URLs (deduplication)
        console.print("Checking for already processed URLs...")
        processed_urls = set()
        new_urls = []
        skipped_existing = 0

        for line_num, url in urls:
            # Generate URL hash for checking
            url_hash = hashlib.md5(url.encode()).hexdigest()

            # Check if already processed by looking in document index
            already_processed = False
            for doc_id, doc_info in document_processor.document_index.items():
                if (
                    doc_info.get("file_path") == url
                    or doc_info.get("doc_id") == url_hash
                ):
                    already_processed = True
                    processed_urls.add(url)
                    break

            if already_processed:
                skipped_existing += 1
                console.print(f"  [dim]Skipping already processed: {url}[/dim]")
            elif url in [
                u[1] for u in new_urls
            ]:  # Check for duplicates in current file
                skipped_existing += 1
                console.print(f"  [dim]Skipping duplicate in file: {url}[/dim]")
            else:
                new_urls.append((line_num, url))

        if skipped_existing > 0:
            console.print(
                f"[yellow]Skipped {skipped_existing} already processed/duplicate URLs[/yellow]"
            )

        if not new_urls:
            console.print(
                "[green]All URLs have already been processed![/green]"
            )
            return

        console.print(f"Processing {len(new_urls)} new URLs...")

        # Validate URLs if not skipped
        valid_urls = []
        if not skip_validation:
            console.print("Validating URLs for settlement content...")

            for line_num, url in new_urls:
                try:
                    # Quick validation without full processing
                    if settlement_only:
                        # Use document processor's validation method
                        from langchain_community.document_loaders import (
                            WebBaseLoader,
                        )

                        loader = WebBaseLoader(url)
                        documents = loader.load()

                        if documents and documents[0].page_content:
                            text = documents[0].page_content
                            if document_processor._is_settlement_relevant(text):
                                valid_urls.append((line_num, url))
                                console.print(f"  Valid: {url}")
                            else:
                                console.print(
                                    f"  Not settlement relevant: {url}"
                                )
                        else:
                            console.print(f"  No content: {url}")
                    else:
                        # Just check if URL is accessible
                        import requests

                        response = requests.head(
                            url, timeout=10, allow_redirects=True
                        )
                        if response.status_code < 400:
                            valid_urls.append((line_num, url))
                            console.print(f"  Accessible: {url}")
                        else:
                            console.print(
                                f"  Inaccessible ({response.status_code}): {url}"
                            )

                except Exception as e:
                    console.print(f"  Validation failed: {url} - {str(e)}")
                    if continue_on_error:
                        continue
                    else:
                        raise
        else:
            valid_urls = new_urls

        if not valid_urls:
            console.print("[yellow]No valid URLs to process.[/yellow]")
            return

        console.print(
            f"Processing {len(valid_urls)} validated URLs in batches of {batch_size}"
        )

        # Process URLs in batches
        successful = 0
        failed = 0
        results = []

        for i in range(0, len(valid_urls), batch_size):
            batch = valid_urls[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(valid_urls) + batch_size - 1) // batch_size

            console.print(
                f"\n[bold blue]Batch {batch_num}/{total_batches}[/bold blue] ({len(batch)} URLs)"
            )

            for line_num, url in batch:
                try:
                    console.print(f"Processing: {url}")

                    # Generate custom name from URL
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    custom_name = f"line{line_num}_{parsed.netloc}_{parsed.path.replace('/', '_')}"

                    # Process the URL
                    metadata = document_processor.process_url(url, custom_name)

                    if metadata:
                        # Generate embeddings
                        console.print("  Generating embeddings...")
                        embedding_service.embed_chunks(metadata["chunks_path"])

                        # Index in vector database
                        console.print("  Indexing in vector database...")
                        vector_db_service.index_chunks(metadata["chunks_path"])

                        successful += 1
                        results.append(metadata)

                        settlement_score = metadata.get(
                            "avg_settlement_score", 0
                        )
                        console.print(
                            f"  Success: {metadata['num_chunks']} chunks, settlement score: {settlement_score:.3f}"
                        )
                    else:
                        failed += 1
                        console.print("  Failed: No content processed")

                except Exception as e:
                    failed += 1
                    error_msg = str(e)
                    console.print(f"  Error: {error_msg}")

                    if not continue_on_error:
                        console.print(
                            "[bold red]Stopping due to error (use --continue-on-error to continue)[/bold red]"
                        )
                        break
                    else:
                        console.print("  Continuing with next URL...")

            # Small delay between batches to be respectful
            if i + batch_size < len(valid_urls):
                console.print("  [dim]Brief pause between batches...[/dim]")
                import time

                time.sleep(2)

        # Generate comprehensive summary
        console.print("\n[bold green]URL Processing Complete![/bold green]")

        summary_table = Table(title="Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")

        summary_table.add_row("Total URLs in file", str(len(urls)))
        summary_table.add_row(
            "Already processed/duplicates", str(skipped_existing)
        )
        summary_table.add_row("New URLs validated", str(len(valid_urls)))
        summary_table.add_row("Successfully processed", str(successful))
        summary_table.add_row("Failed", str(failed))

        if results:
            total_chunks = sum(result["num_chunks"] for result in results)
            avg_settlement_score = sum(
                result.get("avg_settlement_score", 0) for result in results
            ) / len(results)
            summary_table.add_row("Total chunks created", str(total_chunks))
            summary_table.add_row(
                "Average settlement score", f"{avg_settlement_score:.3f}"
            )

        console.print(summary_table)

        # Show top settlement-relevant URLs
        if results:
            console.print(
                "\n[bold yellow]Top Settlement-Relevant URLs:[/bold yellow]"
            )
            sorted_results = sorted(
                results,
                key=lambda x: x.get("avg_settlement_score", 0),
                reverse=True,
            )

            for i, result in enumerate(sorted_results[:5], 1):
                score = result.get("avg_settlement_score", 0)
                chunks = result["num_chunks"]
                url = result["file_path"]

                console.print(f"  {i}. {url}")
                console.print(
                    f"     Settlement Score: {score:.3f} | Chunks: {chunks}"
                )

        # Generate processing log
        log_file = (
            Path(file_path).parent
            / f"{Path(file_path).stem}_processing_log.txt"
        )
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                f"URL Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Source file: {file_path}\n")
            f.write(f"Total URLs: {len(urls)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n\n")

            f.write("Successfully processed URLs:\n")
            for result in results:
                f.write(
                    f"- {result['file_path']} (score: {result.get('avg_settlement_score', 0):.3f})\n"
                )

            f.write("\nAlready processed URLs:\n")
            for url in processed_urls:
                f.write(f"- {url}\n")

        console.print(f"\n[dim]Processing log saved to: {log_file}[/dim]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise


@app.command()
def validate_urls_file(
    file_path: str = typer.Argument(
        ..., help="Path to text file containing URLs"
    ),
    max_check: int = typer.Option(
        50, "--max-check", "-m", help="Maximum number of URLs to check"
    ),
    settlement_only: bool = typer.Option(
        True, "--settlement-only", help="Only validate settlement relevance"
    ),
):
    """Validate URLs in a file without processing them."""
    console.print(
        f"Validating URLs from file: [bold blue]{file_path}[/bold blue]"
    )

    try:
        file_path = Path(file_path)
        if not file_path.exists():
            console.print(f"[bold red]File not found:[/bold red] {file_path}")
            return

        # Read URLs
        urls = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                url = line.strip()
                if url and not url.startswith("#"):
                    if not url.startswith(("http://", "https://")):
                        url = f"https://{url}"
                    urls.append((line_num, url))

        if not urls:
            console.print("[yellow]No valid URLs found in file.[/yellow]")
            return

        console.print(f"Validating {min(len(urls), max_check)} URLs...")

        valid_count = 0
        accessible_count = 0
        settlement_relevant = 0
        already_processed = 0

        validation_table = Table(title="URL Validation Results")
        validation_table.add_column("Line", style="cyan", width=5)
        validation_table.add_column("Status", style="green", width=15)
        validation_table.add_column("URL", style="blue")
        validation_table.add_column("Notes", style="yellow")

        for line_num, url in urls[:max_check]:
            status = ""
            notes = ""

            try:
                # Check if already processed
                url_hash = hashlib.md5(url.encode()).hexdigest()
                is_processed = any(
                    doc_info.get("file_path") == url
                    or doc_info.get("doc_id") == url_hash
                    for doc_info in document_processor.document_index.values()
                )

                if is_processed:
                    status = "PROCESSED"
                    notes = "Already in database"
                    already_processed += 1
                else:
                    # Check accessibility
                    import requests

                    response = requests.head(
                        url, timeout=10, allow_redirects=True
                    )

                    if response.status_code < 400:
                        accessible_count += 1

                        if settlement_only:
                            # Check settlement relevance
                            from langchain_community.document_loaders import (
                                WebBaseLoader,
                            )

                            loader = WebBaseLoader(url)
                            documents = loader.load()

                            if documents and documents[0].page_content:
                                if document_processor._is_settlement_relevant(
                                    documents[0].page_content
                                ):
                                    status = "VALID"
                                    notes = "Settlement relevant"
                                    settlement_relevant += 1
                                    valid_count += 1
                                else:
                                    status = "LOW RELEVANCE"
                                    notes = "Not settlement relevant"
                            else:
                                status = "NO CONTENT"
                                notes = "Empty page"
                        else:
                            status = "ACCESSIBLE"
                            notes = "URL accessible"
                            valid_count += 1
                    else:
                        status = "INACCESSIBLE"
                        notes = f"HTTP {response.status_code}"

            except Exception as e:
                status = "ERROR"
                notes = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)

            validation_table.add_row(
                str(line_num),
                status,
                url[:60] + "..." if len(url) > 60 else url,
                notes[:30] + "..." if len(notes) > 30 else notes,
            )

        console.print(validation_table)

        # Summary
        summary_table = Table(title="Validation Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")
        summary_table.add_column("Percentage", style="yellow")

        total_checked = min(len(urls), max_check)

        summary_table.add_row("Total URLs in file", str(len(urls)), "100%")
        summary_table.add_row(
            "URLs checked",
            str(total_checked),
            f"{(total_checked/len(urls)*100):.1f}%",
        )
        summary_table.add_row(
            "Already processed",
            str(already_processed),
            f"{(already_processed/total_checked*100):.1f}%",
        )
        summary_table.add_row(
            "Accessible",
            str(accessible_count),
            f"{(accessible_count/total_checked*100):.1f}%",
        )

        if settlement_only:
            summary_table.add_row(
                "Settlement relevant",
                str(settlement_relevant),
                f"{(settlement_relevant/total_checked*100):.1f}%",
            )
            summary_table.add_row(
                "Recommended for processing",
                str(settlement_relevant),
                f"{(settlement_relevant/total_checked*100):.1f}%",
            )
        else:
            summary_table.add_row(
                "Valid for processing",
                str(valid_count),
                f"{(valid_count/total_checked*100):.1f}%",
            )

        console.print(summary_table)

        if settlement_relevant > 0 or valid_count > 0:
            console.print(
                f"\n[green]Found {settlement_relevant if settlement_only else valid_count} URLs suitable for processing[/green]"
            )
            console.print("Use 'process-urls-file' command to process them.")
        else:
            console.print(
                "\n[yellow]No suitable URLs found for processing[/yellow]"
            )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold yellow]Warning:[/bold yellow] OPENAI_API_KEY environment variable not set"
        )
        console.print("Set it by running: export OPENAI_API_KEY=your_key_here")
        console.print()

    console.print(
        "[bold green]SettleBot CLI - Settlement Assistant for Nairobi[/bold green]"
    )
    console.print(
        "Helping international students navigate settlement in Nairobi, Kenya"
    )
    console.print()

    app()
