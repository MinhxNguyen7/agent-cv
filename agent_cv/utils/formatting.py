"""Utilities for formatting and normalizing markup."""

import re
from typing import List


def normalize_markup(markup: str, tab_size: int = 1) -> str:
    """
    Normalize HTML-like markup with proper tab indentation and whitespace cleanup.
    
    Args:
        markup: The markup string to normalize
        tab_size: Number of tabs to use for each indentation level
        
    Returns:
        Normalized markup string with proper indentation and cleaned whitespace
    """
    if not markup.strip():
        return ""
    
    # Remove leading/trailing whitespace from the entire string
    markup = markup.strip()
    
    # Split content by tags to handle inline vs block content
    # Find all tags and text content
    parts = re.split(r'(<[^>]+>)', markup)
    parts = [part.strip() for part in parts if part.strip()]  # Remove empty parts
    
    normalized_lines: List[str] = []
    indent_level = 0
    
    for part in parts:
        # Check if this is a closing tag
        if part.startswith('</'):
            indent_level = max(0, indent_level - 1)
            indented_line = '\t' * (indent_level * tab_size) + part
            normalized_lines.append(indented_line)
        # Check if this is an opening tag
        elif part.startswith('<'):
            indented_line = '\t' * (indent_level * tab_size) + part
            normalized_lines.append(indented_line)
            indent_level += 1
        # This is text content
        else:
            indented_line = '\t' * (indent_level * tab_size) + part
            normalized_lines.append(indented_line)
    
    return '\n'.join(normalized_lines)