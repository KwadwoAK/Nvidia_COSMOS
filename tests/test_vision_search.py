from vision_search import build_search_text, suggest_search_terms


def test_build_search_text_includes_summary_and_frame_captions():
    text = build_search_text(
        "## Overview\nVehicle observed.",
        [
            {"description": "A red car near a traffic light."},
            {"description": "A pedestrian crosses the street."},
        ],
    )

    assert "## Overview" in text
    assert "=== Per-frame captions ===" in text
    assert "A red car near a traffic light." in text
    assert "A pedestrian crosses the street." in text


def test_suggest_search_terms_returns_reasonable_terms():
    terms = suggest_search_terms(
        [
            {"description": "A red vehicle is parked near a sidewalk and storefront."},
            {"description": "A pedestrian approaches the storefront and vehicle."},
        ],
        max_terms=10,
    )

    assert isinstance(terms, list)
    assert len(terms) <= 10
    assert all(isinstance(t, str) and t for t in terms)
