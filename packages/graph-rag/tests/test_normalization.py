from graph_rag.models import Triple
from graph_rag.normalization import normalize_text, normalize_triple


def test_normalize_text_produces_stable_entity_key() -> None:
    assert normalize_text("  ACME   Corp. ") == "acme corp"


def test_normalize_triple_discards_incomplete_fact() -> None:
    assert normalize_triple(Triple(subject="Alice", relation="works at", object="  ")) is None


def test_normalize_triple_discards_degenerate_self_facts() -> None:
    """Repeated navigation labels do not become ranking edges."""

    assert normalize_triple(Triple(subject="Price", relation="price", object="Price")) is None
    assert normalize_triple(Triple(subject="Price", relation="price", object="Variants")) is None
