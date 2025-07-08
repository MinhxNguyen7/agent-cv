"""Unit tests for formatting utilities."""

import unittest
from agent_cv.utils.formatting import normalize_markup


class TestNormalizeMarkup(unittest.TestCase):
    """Test cases for normalize_markup function."""
    
    def test_empty_string(self):
        """Test with empty string."""
        result = normalize_markup("")
        self.assertEqual(result, "")
    
    def test_whitespace_only(self):
        """Test with whitespace-only string."""
        result = normalize_markup("   \n\t  ")
        self.assertEqual(result, "")
    
    def test_single_tag(self):
        """Test with single opening and closing tag."""
        markup = "<div>content</div>"
        expected = "<div>\n\tcontent\n</div>"
        result = normalize_markup(markup)
        self.assertEqual(result, expected)
    
    def test_nested_tags(self):
        """Test with nested tags."""
        markup = "<div><p>text</p></div>"
        expected = "<div>\n\t<p>\n\t\ttext\n\t</p>\n</div>"
        result = normalize_markup(markup)
        self.assertEqual(result, expected)
    
    def test_multiple_lines_with_whitespace(self):
        """Test with multiple lines containing extra whitespace."""
        markup = """  <div>
            <p>   content   </p>
        </div>  """
        expected = "<div>\n\t<p>\n\t\tcontent\n\t</p>\n</div>"
        result = normalize_markup(markup)
        self.assertEqual(result, expected)
    
    def test_custom_tab_size(self):
        """Test with custom tab size."""
        markup = "<div><p>text</p></div>"
        expected = "<div>\n\t\t<p>\n\t\t\t\ttext\n\t\t</p>\n</div>"
        result = normalize_markup(markup, tab_size=2)
        self.assertEqual(result, expected)
    
    def test_deeply_nested(self):
        """Test with deeply nested structure."""
        markup = "<div><section><article><p>deep</p></article></section></div>"
        expected = "<div>\n\t<section>\n\t\t<article>\n\t\t\t<p>\n\t\t\t\tdeep\n\t\t\t</p>\n\t\t</article>\n\t</section>\n</div>"
        result = normalize_markup(markup)
        self.assertEqual(result, expected)
    
    def test_empty_lines_removed(self):
        """Test that empty lines are removed."""
        markup = """<div>

        <p>text</p>

        </div>"""
        expected = "<div>\n\t<p>\n\t\ttext\n\t</p>\n</div>"
        result = normalize_markup(markup)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()