import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import logging
import re
from typing import List, Optional, Union, Dict, Any, Iterator

from bs4 import BeautifulSoup
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings
from src.utils.timing import log_duration

logger = logging.getLogger(__name__)

# Reserve headroom under the text splitter's chunk_size for the
# "[START_TABLE]\n...\n[END_TABLE]\n\n" wrapper markers so a table chunk we hand
# to the splitter is never itself larger than chunk_size (which would force the
# splitter to cut it further, severing data rows from their header).
_TABLE_MARKER_OVERHEAD = 64

_NUMERIC_CELL_PATTERN = re.compile(
    r"^\(?-?\$?\s*[\d,]+(\.\d+)?%?\)?$|^-$|^—$|^n/a$", re.IGNORECASE
)

_FINANCIAL_HEADER_KEYWORDS = [
    "revenue",
    "income",
    "assets",
    "liabilities",
    "total",
    "ended",
    "cash flow",
    "expenses",
    "earnings",
    "equity",
]


def table_to_grid(table_soup) -> List[List[str]]:
    """
    Flatten an HTML `<table>` into a rectangular text grid, resolving `colspan`/`rowspan`.

    Unlike naively reading `<td>` text in document order (what `markdownify` effectively
    does), this walks each row tracking which columns are already occupied by a cell
    carried down from a previous row's `rowspan`, so merged header/label cells land under
    the correct column in every row they span rather than only their first row.

    Args:
        table_soup: A BeautifulSoup `<table>` tag.

    Returns:
        A list of rows, each a list of cell text strings, all rows padded to the same
        (maximum) column count with `""` for any cell a malformed table left empty.
    """
    rows = table_soup.find_all("tr")
    grid: List[List[Optional[str]]] = []
    for r, tr in enumerate(rows):
        while len(grid) <= r:
            grid.append([])
        row = grid[r]
        col = 0
        for cell in tr.find_all(["td", "th"]):
            while col < len(row) and row[col] is not None:
                col += 1
            try:
                colspan = int(cell.get("colspan", 1) or 1)
            except ValueError:
                colspan = 1
            try:
                rowspan = int(cell.get("rowspan", 1) or 1)
            except ValueError:
                rowspan = 1
            text = cell.get_text(" ", strip=True)
            for rr in range(r, r + max(rowspan, 1)):
                while len(grid) <= rr:
                    grid.append([])
                target_row = grid[rr]
                while len(target_row) < col + colspan:
                    target_row.append(None)
                for cc in range(col, col + colspan):
                    target_row[cc] = text
            col += colspan

    max_cols = max((len(row) for row in grid), default=0)
    padded = [
        ["" if cell is None else cell for cell in row + [None] * (max_cols - len(row))]
        for row in grid
    ]
    return _collapse_redundant_columns(padded)


def _collapse_redundant_columns(grid: List[List[str]]) -> List[List[str]]:
    """
    Drop spacer columns and merge duplicate-content columns produced by fine-grained
    EDGAR table layouts.

    SEC HTML tables routinely split what's visually one column into several `<td>`s
    purely for pixel alignment (a blank spacer column between the `$` sign and the
    number, a `colspan` label cell that `table_to_grid` fills identically into every
    column it spans, ...). Left alone these show up as long runs of blank cells or
    the same value repeated 2-3x across adjacent columns in the rendered Markdown.
    Both transforms here are lossless: a column empty in *every* row carries no data
    to lose, and two adjacent columns identical in *every* row carry the same data
    twice, so merging them keeps exactly one copy.
    """
    if not grid or not grid[0]:
        return grid

    num_cols = len(grid[0])
    non_empty_cols = [c for c in range(num_cols) if any(row[c] for row in grid)]
    if not non_empty_cols:
        return grid
    grid = [[row[c] for c in non_empty_cols] for row in grid]

    num_cols = len(grid[0])
    kept_cols = [0]
    for c in range(1, num_cols):
        if all(row[c] == row[kept_cols[-1]] for row in grid):
            continue
        kept_cols.append(c)
    return [[row[c] for c in kept_cols] for row in grid]


def _header_row_count(table_soup, grid: List[List[str]]) -> int:
    """
    Number of leading rows in `grid` that make up the table header.

    Prefers HTML-semantic signals over guessing from cell content: an explicit `<thead>`
    wins outright, then leading `<tr>`s made mostly of `<th>` cells. Both are reliable
    even when a header row is purely numeric (e.g. "2023 2022"), which the last-resort
    non-numeric-content heuristic would otherwise mistake for a data row.
    """
    if table_soup.find("thead") is not None:
        count = len(table_soup.find("thead").find_all("tr"))
        if 0 < count < len(grid):
            return count

    count = 0
    for tr in table_soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        th_fraction = sum(1 for c in cells if c.name == "th") / len(cells)
        if th_fraction >= 0.5:
            count += 1
        else:
            break
    if count:
        return min(count, len(grid))

    # No <th> tags at all in this table: fall back to a content heuristic.
    count = 0
    for row in grid:
        non_empty = [c for c in row if c]
        if non_empty and all(not _NUMERIC_CELL_PATTERN.match(c) for c in non_empty):
            count += 1
        else:
            break
    return max(count, 1) if grid else 0


def is_data_table(table_soup, grid: Optional[List[List[str]]] = None) -> bool:
    """
    Decide whether a table carries tabular financial data (keep structure) or is
    being used purely for page layout (flatten to plain text).

    Uses the resolved grid rather than raw digit-density of the table's full text, since
    a layout table (e.g. an address block laid out in a grid) can contain incidental
    digits, and a genuine data table's numeric density can be diluted by long text labels
    in its first column. Two independent signals are checked: (1) what fraction of
    non-header, non-empty cells look like numbers/currency/percentages, and (2) whether
    the header row(s) contain common financial-statement vocabulary.

    Args:
        table_soup: A BeautifulSoup `<table>` tag.
        grid: Optional pre-computed `table_to_grid(table_soup)` result, to avoid
            recomputing it when the caller already has one.

    Returns:
        `True` if the table should be preserved as structured Markdown, `False` if it
        should be flattened/unwrapped as layout.
    """
    grid = grid if grid is not None else table_to_grid(table_soup)
    if not grid or not any(any(cell for cell in row) for row in grid):
        return False

    header_rows = _header_row_count(table_soup, grid)
    data_cells = [cell for row in grid[header_rows:] for cell in row if cell]
    if data_cells:
        numeric_cells = sum(1 for c in data_cells if _NUMERIC_CELL_PATTERN.match(c))
        if numeric_cells / len(data_cells) > 0.3:
            return True

    header_text = " ".join(
        cell.lower() for row in grid[:header_rows] for cell in row if cell
    )
    if any(keyword in header_text for keyword in _FINANCIAL_HEADER_KEYWORDS):
        return True

    return False


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def grid_to_markdown_chunks(
    grid: List[List[str]], header_row_count: int, max_chars: int
) -> List[str]:
    """
    Render a table grid as one or more Markdown tables, each within `max_chars`.

    Splits only between data rows, never inside one, and repeats the (merged) header
    row + separator at the top of every chunk, so a table too large for a single chunk
    still leaves every chunk self-contained: a chunk of rows retrieved on its own still
    carries its column labels instead of being bare numbers with no header for context.

    Args:
        grid: A rectangular cell-text grid, as returned by `table_to_grid`.
        header_row_count: How many leading rows of `grid` are header rows (merged
            column-wise into a single Markdown header row).
        max_chars: Soft cap on each returned chunk's length; a single data row is never
            split, so a pathological row wider than `max_chars` still returns whole.

    Returns:
        One or more Markdown table strings (header + separator + some data rows each),
        in original row order. Empty list if `grid` is empty.
    """
    if not grid:
        return []

    header_row_count = min(max(header_row_count, 1), len(grid))
    header_rows = grid[:header_row_count]
    data_rows = grid[header_row_count:]
    num_cols = len(header_rows[0]) if header_rows else 0

    merged_header = []
    for c in range(num_cols):
        parts = []
        for hr in header_rows:
            val = hr[c] if c < len(hr) else ""
            if val and val not in parts:
                parts.append(val)
        merged_header.append(" ".join(parts))

    header_line = "| " + " | ".join(_escape_md_cell(h) for h in merged_header) + " |"
    sep_line = "| " + " | ".join("---" for _ in merged_header) + " |"
    header_block = f"{header_line}\n{sep_line}"

    chunks: List[str] = []
    current_rows: List[str] = []
    current_len = len(header_block)
    for row in data_rows:
        row_line = "| " + " | ".join(_escape_md_cell(c) for c in row) + " |"
        row_len = len(row_line) + 1
        if current_rows and current_len + row_len > max_chars:
            chunks.append(header_block + "\n" + "\n".join(current_rows))
            current_rows = []
            current_len = len(header_block)
        current_rows.append(row_line)
        current_len += row_len

    chunks.append(header_block + "\n" + "\n".join(current_rows) if current_rows else header_block)
    return chunks


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
        table_max_chars: int = 4000 - _TABLE_MARKER_OVERHEAD,
    ):
        """
        Args:
            file_path: Path to the .txt submission file.
            target_types: The document types to extract (e.g., "10-K", "EX-21").
                          Case-insensitive.
            ticker: The stock ticker this specific filing belongs to (recovered from the
                    downloaded filename, not the raw multi-ticker LLM extraction), attached to
                    every chunk's metadata when present.
            table_max_chars: Soft cap on the Markdown length of a single table chunk,
                    should be set to (downstream text splitter's chunk_size -
                    `_TABLE_MARKER_OVERHEAD`) so a table this loader emits never needs to
                    be split further by the chunker, which would otherwise sever data
                    rows from their header (see `grid_to_markdown_chunks`).
        """
        self.file_path = file_path
        if isinstance(target_types, str):
            self.target_types = [target_types.upper()]
        else:
            self.target_types = [t.upper() for t in target_types]
        self.ticker = ticker
        self.table_max_chars = table_max_chars

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

    def _process_html_content(self, html_content: str) -> str:
        """Cleans HTML: Converts data tables to Markdown, flattens layout tables."""
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Remove Script/Style/Hidden tags to reduce noise
        for tag in soup(["script", "style", "ix:header"]):
            tag.decompose()

        # 2. Process Tables
        for table in soup.find_all("table"):
            grid = table_to_grid(table)
            if is_data_table(table, grid=grid):
                # Convert to one or more Markdown tables (colspan/rowspan-aware, never
                # larger than table_max_chars so the chunker won't split a table mid-row).
                header_rows = _header_row_count(table, grid)
                md_chunks = grid_to_markdown_chunks(grid, header_rows, self.table_max_chars)
                wrapped = "".join(
                    f"\n\n[START_TABLE]\n{chunk}\n[END_TABLE]\n\n" for chunk in md_chunks
                )
                table.replace_with(wrapped)
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
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
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
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
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


_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 200

_TABLE_SPAN_PATTERN = re.compile(r"\[START_TABLE\]\n.*?\n\[END_TABLE\]", re.DOTALL)


def split_document_preserving_tables(
    doc: Document, chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Chunk a document's text, never splitting a `[START_TABLE]...[END_TABLE]` span.

    Feeding `RecursiveCharacterTextSplitter` the whole document with the table markers
    as split separators (the previous approach) isn't reliable: when a table plus its
    surrounding narrative text together exceed `chunk_size`, the splitter's recursive
    behavior can use "[END_TABLE]" as a separator relative to the *following* prose,
    which detaches the closing marker from its table into the next chunk even though
    the table's own Markdown is well under `chunk_size` on its own. Instead, table spans
    (already pre-sized under `chunk_size` by `grid_to_markdown_chunks`) are pulled out
    and emitted as their own single chunk verbatim; only the plain-text stretches
    between/around them go through the standard recursive splitter.

    Args:
        doc: A document whose `page_content` may contain zero or more
            `[START_TABLE]...[END_TABLE]` spans (as produced by
            `SecEdgarAdvancedLoader._process_html_content`).
        chunk_size: Max characters per plain-text chunk (table chunks are passed through
            as-is regardless of size, on the assumption they were already pre-sized).
        chunk_overlap: Character overlap between adjacent plain-text chunks.

    Returns:
        Chunks in original document order, each copying `doc.metadata`; table chunks
        additionally get `metadata["is_table"] = True`.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    def _text_chunks(text: str) -> List[Document]:
        return [
            Document(page_content=piece, metadata=dict(doc.metadata))
            for piece in text_splitter.split_text(text)
            if piece.strip()
        ]

    chunks: List[Document] = []
    last_end = 0
    for match in _TABLE_SPAN_PATTERN.finditer(doc.page_content):
        chunks.extend(_text_chunks(doc.page_content[last_end : match.start()]))
        table_metadata = dict(doc.metadata)
        table_metadata["is_table"] = True
        chunks.append(Document(page_content=match.group(0), metadata=table_metadata))
        last_end = match.end()
    chunks.extend(_text_chunks(doc.page_content[last_end:]))
    return chunks


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
            with log_duration(logger, f"Document extraction+chunking+embedding ({file_path})"):
                document_loader = SecEdgarAdvancedLoader(
                    file_path=file_path,
                    target_types=filings,
                    ticker=ticker,
                    table_max_chars=_CHUNK_SIZE - _TABLE_MARKER_OVERHEAD,
                )
                for doc in document_loader.lazy_load():
                    chunks = split_document_preserving_tables(
                        doc, chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
                    )
                    with log_duration(logger, f"Embed+store {len(chunks)} chunk(s) ({file_path})"):
                        self.vector_store.add_documents(documents=chunks)
            logger.info("Successfully loaded data into vector store")
        except Exception as e:
            logger.exception(
                "Error while extracting data from file: %s with error: %s",
                file_path,
                str(e),
                exc_info=True,
            )
            raise e
