from src.services.hybrid_retrieval import (
    bm25_top_n,
    build_bm25_index,
    reciprocal_rank_fusion,
    select_top_k_by_score,
    tokenize,
)


def test_tokenize_lowercases_and_splits_on_non_word_chars():
    assert tokenize("Apple's Revenue Grew 12%!") == ["apple", "s", "revenue", "grew", "12"]


def test_bm25_top_n_ranks_relevant_document_first(make_document):
    docs = [
        make_document("The weather today is sunny and warm.", doc_id="1"),
        make_document("Total revenue increased due to strong iPhone sales.", doc_id="2"),
        make_document("A cat sat on a mat.", doc_id="3"),
    ]
    index = build_bm25_index(docs)

    results = bm25_top_n(index, docs, "revenue and iPhone sales", n=2)

    assert results[0].id == "2"


def test_reciprocal_rank_fusion_dedupes_by_id(make_document):
    shared = make_document("Shared chunk content", doc_id="shared")
    vector_ranked = [shared, make_document("Only in vector", doc_id="v2")]
    bm25_ranked = [make_document("Only in bm25", doc_id="b2"), shared]

    fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked])

    fused_ids = [doc.id for doc in fused]
    assert fused_ids.count("shared") == 1
    assert set(fused_ids) == {"shared", "v2", "b2"}


def test_reciprocal_rank_fusion_orders_by_combined_score(make_document):
    doc_a = make_document("a", doc_id="a")
    doc_b = make_document("b", doc_id="b")
    doc_c = make_document("c", doc_id="c")

    # doc_a: rank 1 in list 1, rank 2 in list 2 -> 1/61 + 1/62 (highest combined score).
    # doc_c: rank 1 in list 2 only -> 1/61.
    # doc_b: rank 2 in list 1 only -> 1/62 (lowest).
    vector_ranked = [doc_a, doc_b]
    bm25_ranked = [doc_c, doc_a]

    fused = reciprocal_rank_fusion([vector_ranked, bm25_ranked], k=60)

    assert [doc.id for doc in fused] == ["a", "c", "b"]


def test_reciprocal_rank_fusion_falls_back_when_id_missing(make_document):
    same_content = make_document("identical content", doc_id=None)
    also_same_content = make_document("identical content", doc_id=None)

    fused = reciprocal_rank_fusion([[same_content], [also_same_content]])

    assert len(fused) == 1


def test_select_top_k_by_score_sorts_and_truncates(make_document):
    docs = [make_document("a", doc_id="a"), make_document("b", doc_id="b"), make_document("c", doc_id="c")]
    scores = [0.1, 0.9, 0.5]

    top = select_top_k_by_score(docs, scores, k=2)

    assert [doc.id for doc in top] == ["b", "c"]


def test_select_top_k_by_score_does_not_mutate_input(make_document):
    original_metadata = {"source": "filing.txt"}
    doc = make_document("content", doc_id="a", **original_metadata)
    original_metadata_snapshot = dict(doc.metadata)

    top = select_top_k_by_score([doc], [0.42], k=1)

    assert doc.metadata == original_metadata_snapshot
    assert "rerank_score" not in doc.metadata
    assert top[0].metadata["rerank_score"] == 0.42
