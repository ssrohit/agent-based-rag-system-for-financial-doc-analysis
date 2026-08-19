from typing import Optional

import pytest
from langchain_core.documents import Document


@pytest.fixture
def make_document():
    def _make(content: str, doc_id: Optional[str] = None, **metadata) -> Document:
        return Document(page_content=content, metadata=metadata, id=doc_id)

    return _make
