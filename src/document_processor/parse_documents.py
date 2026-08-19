import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import logging
import re
from datetime import datetime
from typing import List, Optional, Union, Dict, Any, Iterator

from bs4 import BeautifulSoup
import markdownify
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)



class SecEdgarAdvancedLoader(BaseLoader):
    """
    A LangChain loader for SEC EDGAR 'full-submission.txt' files.

    Features:
    - Extracts specific document types (10-K, 10-Q, EX-21, etc.).
    - Converts Financial HTML Tables to Markdown for better RAG retrieval.
    - Flattens "Layout Tables" to avoid polluting embeddings with noise.
    - Captures global metadata (CIK, Company Name, Filing Date).
    """

    def __init__(
        self,
        file_path: str,
        target_types: Union[str, List[str]] = ["10-K"],
        ticker: Optional[str] = None,
    ):
        """
        Args:
            file_path: Path to the .txt submission file.
            target_types: The document types to extract (e.g., "10-K", "EX-21").
                          Case-insensitive.
            ticker: The stock ticker this specific filing belongs to (recovered from the
                    downloaded filename, not the raw multi-ticker LLM extraction), attached to
                    every chunk's metadata when present.
        """
        self.file_path = file_path
        if isinstance(target_types, str):
            self.target_types = [target_types.upper()]
        else:
            self.target_types = [t.upper() for t in target_types]
        self.ticker = ticker

    def _parse_header_metadata(self, content: str) -> Dict[str, Any]:
        """Extracts global metadata from the SEC-HEADER block."""
        metadata = {}
        # Regex for common header fields
        patterns = {
            "cik": r"CENTRAL INDEX KEY:\s+(\d+)",
            "company_name": r"COMPANY CONFORMED NAME:\s+(.+)",
            "filing_date": r"FILED AS OF DATE:\s+(\d+)",
            "fiscal_year_end": r"FISCAL YEAR END:\s+(\d+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metadata[key] = match.group(1).strip()

        return metadata

    def _is_data_table(self, table_soup) -> bool:
        """
        Heuristic to determine if a table contains data (keep structure)
        or just layout (flatten).

        Logic: Data tables usually have a high density of numbers/digits.
        """
        text = table_soup.get_text()
        if not text.strip():
            return False

        # Count digits
        num_digits = sum(c.isdigit() for c in text)
        total_chars = len(text)

        # If > 10% of characters are digits, or it contains specific financial keywords
        # we treat it as a data table.
        # (Thresholds can be tuned based on specific needs)
        if num_digits > 5 and (num_digits / total_chars > 0.05):
            return True

        # Check for headers often found in financial tables
        headers = [th.get_text().lower() for th in table_soup.find_all("th")]
        financial_keywords = [
            "revenue",
            "income",
            "assets",
            "liabilities",
            "total",
            "ended",
        ]
        if any(k in " ".join(headers) for k in financial_keywords):
            return True

        return False

    def _process_html_content(self, html_content: str) -> str:
        """Cleans HTML: Converts data tables to Markdown, flattens layout tables."""
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Remove Script/Style/Hidden tags to reduce noise
        for tag in soup(["script", "style", "ix:header"]):
            tag.decompose()

        # 2. Process Tables
        for table in soup.find_all("table"):
            if self._is_data_table(table):
                # Convert Financial Tables to Markdown
                md_table = markdownify.markdownify(str(table), heading_style="ATX")
                # Add markers so the chunker can respect table boundaries later
                table.replace_with(f"\n\n[START_TABLE]\n{md_table}\n[END_TABLE]\n\n")
            else:
                # Layout Table: Unwrap (remove <table> tags but keep text)
                # This prevents "Address blocks" from looking like grid data
                table.unwrap()

        # 3. Extract text
        return soup.get_text(separator="\n", strip=True)

    def lazy_load(self) -> Iterator[Document]:
        """
        Yields documents one by one.
        """
        try:
            # We still read the file content to find regex matches,
            # but we delay the heavy HTML parsing.
            # NOTE: For multi-GB files, you would need to memory-map (mmap) this,
            # but for <500MB SEC filings, reading into string is usually fine.
            # The memory bottleneck is usually the HTML DOM, not the raw text.
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()
        except FileNotFoundError:
            logger.debug(f"Error: File not found at {self.file_path}")
            return

        # 1. Parse Metadata once
        header_end = raw_content.find("</SEC-HEADER>")
        global_metadata = {}
        if header_end != -1:
            global_metadata = self._parse_header_metadata(raw_content[:header_end])

        # 2. Iterate through regex matches (Lazy)
        doc_pattern = re.compile(
            r"<DOCUMENT>\s*<TYPE>([^\n]+).*?<TEXT>(.*?)</TEXT>",
            re.DOTALL | re.IGNORECASE,
        )

        # finditer returns an iterator, it doesn't create a list of all matches
        for match in doc_pattern.finditer(raw_content):
            doc_type = match.group(1).strip().upper()

            if doc_type in self.target_types:
                # Heavy lifting happens HERE, only for the current document
                content_html = match.group(2)
                clean_text = self._process_html_content(content_html)

                doc_metadata = global_metadata.copy()
                doc_metadata.update({"source": self.file_path, "doc_type": doc_type})
                if self.ticker:
                    doc_metadata["ticker"] = self.ticker

                yield Document(page_content=clean_text, metadata=doc_metadata)

    def load(self) -> List[Document]:
        documents = []

        try:
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()
        except FileNotFoundError:
            logger.debug("Error: File not found at %s", self.file_path)
            return []

        # 1. Get Global Metadata
        header_end = raw_content.find("</SEC-HEADER>")
        if header_end != -1:
            header_content = raw_content[:header_end]
            global_metadata = self._parse_header_metadata(header_content)
        else:
            global_metadata = {}

        # 2. Find All Documents matching target types
        # This regex looks for <DOCUMENT> ... <TYPE> ... <TEXT> ... </TEXT>
        # capturing the Type and the Content.
        doc_pattern = re.compile(
            r"<DOCUMENT>\s*<TYPE>([^\n]+).*?<TEXT>(.*?)</TEXT>",
            re.DOTALL | re.IGNORECASE,
        )

        for match in doc_pattern.finditer(raw_content):
            doc_type = match.group(1).strip().upper()

            # Check if this doc_type is requested
            if doc_type in self.target_types:
                doc_content_html = match.group(2)

                # Process the content
                clean_text = self._process_html_content(doc_content_html)

                # Combine metadata
                doc_metadata = global_metadata.copy()
                doc_metadata.update({"source": self.file_path, "doc_type": doc_type})
                if self.ticker:
                    doc_metadata["ticker"] = self.ticker

                documents.append(
                    Document(page_content=clean_text, metadata=doc_metadata)
                )

        return documents


class DocumentProcessor:
    def __init__(self, collection_name: str):
        self.config = settings
        self.vector_generation_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Chroma(
            persist_directory=self.config.CHROMA_PERSIST_PATH,
            embedding_function=self.vector_generation_model,
            collection_name=collection_name,
        )

    def extract_data(
        self, file_path: str, filings: List[str] = ["10-K"], ticker: Optional[str] = None
    ) -> None:
        logger.debug(
            "Starting data extraction from file: %s and filings: %s",
            file_path,
            str(filings),
        )
        try:
            start = datetime.now()
            document_loader = SecEdgarAdvancedLoader(
                file_path=file_path, target_types=filings, ticker=ticker
            )
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["[START_TABLE]", "[END_TABLE]", "\n\n", ". "],
                chunk_size=4000,
                chunk_overlap=200,
            )
            for doc in document_loader.lazy_load():
                chunks = text_splitter.split_documents([doc])
                self.vector_store.add_documents(documents=chunks)
            end = datetime.now()
            logger.info("Successfully loaded data into vector store")
            logger.debug("Time elapsed: %s", str(end - start))
        except Exception as e:
            logger.exception(
                "Error while extracting data from file: %s with error: %s",
                file_path,
                str(e),
                exc_info=True,
            )
            raise e
