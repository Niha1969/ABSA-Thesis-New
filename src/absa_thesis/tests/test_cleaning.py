from src.absa_thesis.etl.clean_normalize import clean_text

def test_clean_text_basic():
    assert clean_text("  hi\nthere ") == "hi there"
    assert clean_text(123) == ""
