"""CPU tests for the project's annotated JSON5 configuration format."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from config_io import load_config, strip_hash_comments
import json5


class ConfigTest(unittest.TestCase):
    def test_json5_and_hash_comments_preserve_strings(self):
        text = '''{ # outer
            url: 'https://host/#anchor', # comment with " and '
            escaped: "a\\\"#b", // # and ' inside standard comment
            /* ' " # block comment */
            values: [1, 2,], # trailing comma
        }'''
        parsed = json5.loads(strip_hash_comments(text))
        self.assertEqual(parsed['url'], 'https://host/#anchor')
        self.assertEqual(parsed['escaped'], 'a"#b')
        self.assertEqual(parsed['values'], [1, 2])
        self.assertEqual(len(strip_hash_comments(text)), len(text))

    def test_config_comments_align_and_archive_verbatim(self):
        source = ROOT / 'configs/two_card.json5'
        text = source.read_text()
        self.assertEqual(len({line.index('#') for line in text.splitlines()}), 1)
        with tempfile.TemporaryDirectory() as directory:
            archived = Path(directory) / 'config.json5'
            archived.write_bytes(source.read_bytes())
            config = load_config(archived)
            self.assertEqual(config, load_config(source))
            self.assertEqual(config['profiling']['timestamp_frequency_mhz'], 1000)
            self.assertNotIn('frequency_mhz', config['device'])
            self.assertGreaterEqual(config['case']['warmup'], 0)


if __name__ == '__main__':
    unittest.main()
