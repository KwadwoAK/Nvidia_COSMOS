from summarys.summary_templates import metadata_line, parse_template_id_from_summary
from summarys.summary_templates import style_key_from_label


def test_metadata_line_template_id_round_trip():
    header = metadata_line(style="formal", engine="gemma4", vision_model="Cosmos-Reason2-8B")
    summary_text = f"{header}\n\nGenerated summary body."

    assert parse_template_id_from_summary(summary_text) == "cosmos_summary_v1"


def test_style_key_from_label_defaults_to_formal_for_unknown_label():
    assert style_key_from_label("Unknown Style") == "formal"
