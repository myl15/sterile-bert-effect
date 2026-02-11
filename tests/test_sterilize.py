import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.sterilize import sterilize_line


class TestSterilize:
    def test_removes_urls(self):
        text = "Visit https://example.com for details"
        result = sterilize_line(text, remove_urls=True, remove_digits=False,
                                remove_shared_punct=False)
        assert "https" not in result
        assert "example" not in result

    def test_removes_www_urls(self):
        text = "Go to www.example.org/page for info"
        result = sterilize_line(text, remove_urls=True, remove_digits=False,
                                remove_shared_punct=False)
        assert "www" not in result

    def test_removes_digits(self):
        text = "There are 42 cats and 7 dogs"
        result = sterilize_line(text, remove_urls=False, remove_digits=True,
                                remove_shared_punct=False)
        assert "42" not in result
        assert "7" not in result
        assert "cats" in result

    def test_removes_shared_punct(self):
        text = "Hello, world! How are you?"
        result = sterilize_line(text, remove_urls=False, remove_digits=False,
                                remove_shared_punct=True)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
        assert "Hello" in result

    def test_preserves_language_specific_chars(self):
        text = "Les \u00e9l\u00e8ves \u00e9tudient \u00e0 l'\u00e9cole fran\u00e7aise"
        result = sterilize_line(text, remove_urls=True, remove_digits=True,
                                remove_shared_punct=True)
        assert "\u00e9l\u00e8ves" in result
        assert "fran\u00e7aise" in result

    def test_collapses_whitespace(self):
        text = "Too   many    spaces   here"
        result = sterilize_line(text, remove_urls=False, remove_digits=False,
                                remove_shared_punct=False)
        assert "  " not in result

    def test_full_sterilization(self):
        text = "On Jan. 1, 2024, visit http://example.com for 50% off!"
        result = sterilize_line(text)
        assert not any(c.isdigit() for c in result)
        assert "http" not in result
        assert "." not in result
        assert "!" not in result

    def test_empty_result(self):
        text = "123 456 789"
        result = sterilize_line(text)
        assert result == ""

    def test_url_before_punct(self):
        """URLs must be removed before punctuation to match correctly."""
        text = "See http://a.b.c/path for details."
        result = sterilize_line(text)
        assert "http" not in result
