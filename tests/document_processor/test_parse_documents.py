from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.document_processor.parse_documents import (
    _header_row_count,
    grid_to_markdown_chunks,
    is_data_table,
    split_document_preserving_tables,
    table_to_grid,
)


def _table(html: str):
    return BeautifulSoup(f"<table>{html}</table>", "html.parser").find("table")


def test_table_to_grid_resolves_colspan_and_rowspan():
    table = _table(
        """
        <tr><th rowspan="2">Item</th><th colspan="2">Year Ended Dec 31,</th></tr>
        <tr><th>2023</th><th>2022</th></tr>
        <tr><td>Revenue</td><td>1,234</td><td>1,000</td></tr>
        """
    )

    grid = table_to_grid(table)

    assert grid == [
        ["Item", "Year Ended Dec 31,", "Year Ended Dec 31,"],
        ["Item", "2023", "2022"],
        ["Revenue", "1,234", "1,000"],
    ]


def test_header_row_count_uses_th_tags_not_numeric_content():
    # The second header row is purely numeric ("2023"/"2022"), which a naive
    # numeric-content heuristic would mistake for a data row.
    table = _table(
        """
        <tr><th rowspan="2">Item</th><th colspan="2">Year Ended Dec 31,</th></tr>
        <tr><th>2023</th><th>2022</th></tr>
        <tr><td>Revenue</td><td>1,234</td><td>1,000</td></tr>
        """
    )
    grid = table_to_grid(table)

    assert _header_row_count(table, grid) == 2


def test_is_data_table_true_for_financial_table():
    table = _table(
        """
        <tr><th>Item</th><th>2023</th><th>2022</th></tr>
        <tr><td>Revenue</td><td>1,234</td><td>1,000</td></tr>
        <tr><td>Net income</td><td>(200)</td><td>150</td></tr>
        """
    )

    assert is_data_table(table) is True


def test_is_data_table_false_for_layout_table():
    table = _table(
        """
        <tr><td>123 Main St</td><td>Suite 400</td></tr>
        <tr><td>City, ST 12345</td><td></td></tr>
        """
    )

    assert is_data_table(table) is False


def test_is_data_table_false_for_empty_table():
    table = _table("<tr><td></td><td></td></tr>")

    assert is_data_table(table) is False


def test_grid_to_markdown_chunks_repeats_header_when_splitting():
    grid = [["Item", "2023", "2022"]] + [
        [f"Line {i}", f"{i * 111},000", f"{i * 99},500"] for i in range(50)
    ]

    chunks = grid_to_markdown_chunks(grid, header_row_count=1, max_chars=500)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    for chunk in chunks:
        lines = chunk.splitlines()
        assert lines[0] == "| Item | 2023 | 2022 |"
        assert lines[1] == "| --- | --- | --- |"


def test_grid_to_markdown_chunks_never_splits_a_single_row():
    # A single data row wider than max_chars is still returned whole, not truncated.
    grid = [["Item", "Value"], ["Line item", "x" * 1000]]

    chunks = grid_to_markdown_chunks(grid, header_row_count=1, max_chars=50)

    assert len(chunks) == 1
    assert "x" * 1000 in chunks[0]


def test_grid_to_markdown_chunks_escapes_pipe_characters():
    grid = [["Item", "Value"], ["A | B", "1|2"]]

    chunks = grid_to_markdown_chunks(grid, header_row_count=1, max_chars=4000)

    assert "A \\| B" in chunks[0]
    assert "1\\|2" in chunks[0]


def test_split_document_preserving_tables_never_splits_a_table_span():
    # A table (well under chunk_size on its own) followed by enough prose that the
    # *combined* table+prose text exceeds chunk_size. Regression test for a real bug:
    # handing this to RecursiveCharacterTextSplitter with table markers as separators
    # let it split the table's own [END_TABLE] marker away from its [START_TABLE],
    # even though the table itself was small enough to never need splitting.
    table_md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    prose = "This is filler narrative text. " * 200  # >> chunk_size
    content = f"Intro paragraph.\n\n[START_TABLE]\n{table_md}\n[END_TABLE]\n\n{prose}"
    doc = Document(page_content=content, metadata={"ticker": "AAPL"})

    chunks = split_document_preserving_tables(doc, chunk_size=500, chunk_overlap=50)

    table_chunks = [c for c in chunks if "[START_TABLE]" in c.page_content]
    assert len(table_chunks) == 1
    assert table_chunks[0].page_content.count("[START_TABLE]") == 1
    assert table_chunks[0].page_content.count("[END_TABLE]") == 1
    assert table_chunks[0].metadata["is_table"] is True
    assert table_chunks[0].metadata["ticker"] == "AAPL"

    assert all(len(c.page_content) <= 500 for c in chunks if not c.metadata.get("is_table"))


def test_split_document_preserving_tables_keeps_oversized_row_whole():
    # Mirrors grid_to_markdown_chunks: a table chunk larger than chunk_size (e.g. a
    # pathological single wide row) is passed through untouched rather than truncated.
    big_table = "[START_TABLE]\n" + ("x" * 2000) + "\n[END_TABLE]"
    doc = Document(page_content=big_table, metadata={})

    chunks = split_document_preserving_tables(doc, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].page_content == big_table
