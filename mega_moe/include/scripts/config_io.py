"""Read JSON5 with an additional shell-style # line-comment syntax."""
from pathlib import Path

import json5


def strip_hash_comments(text):
    """Preserve strings, JSON5 comments and source positions while masking # comments."""
    result = list(text)
    state = None
    i = 0
    while i < len(text):
        char = text[i]
        pair = text[i:i + 2]
        if state in ('"', "'"):
            if char == '\\':
                i += 2
                continue
            if char == state:
                state = None
        elif state == 'block':
            if pair == '*/':
                state = None
                i += 2
                continue
        elif state in ('line', 'hash'):
            if char in '\r\n\u2028\u2029':
                state = None
            elif state == 'hash':
                result[i] = ' '
        elif char in ('"', "'"):
            state = char
        elif pair in ('//', '/*'):
            state = 'line' if pair == '//' else 'block'
            i += 2
            continue
        elif char == '#':
            state = 'hash'
            result[i] = ' '
        i += 1
    return ''.join(result)


def load_config(path):
    return json5.loads(strip_hash_comments(Path(path).read_text(encoding='utf-8')))
