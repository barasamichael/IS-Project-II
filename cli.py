import os
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
from services.document_processor import DocumentProcessor
from services.embeddings import EmbeddingService
from services.intent_recognizer import IntentRecognizer
from services.language_processor import LanguageProcessor
from services.response_generator import ResponseGenerator
from services.vector_db import VectorDBService

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

        # Intent recognition
        intent_info = intent_recognizer.recognize_intent(english_query)

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

            # Intent recognition
            intent_info = intent_recognizer.recognize_intent(english_query)

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
