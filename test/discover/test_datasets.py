"""Tests of datasets discovery

Copyright 2018 Spectre Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest
from unittest.mock import MagicMock, mock_open, patch
import json
from functools import partial
import os

import discover.datasets


class TestReadable(unittest.TestCase):
    def test_replaces_underscores_into_spaces(self):
        self.assertEqual('some name', discover.datasets.as_readable('some_name'))


DUMMY_DATASETS = [
    "dataset",
    "another",
    "and_even_this_one",
]
NOT_DATASETS = [
    "not_a_dataset",
    "neither_this.exe",
    "trustworthy.sh",
]
DUMMY_DATASETS_PATHS = [
    os.path.join(discover.datasets.DATASETS_ROOT, "dataset"),
    os.path.join(discover.datasets.DATASETS_ROOT, "another"),
    os.path.join(discover.datasets.DATASETS_ROOT, "and_even_this_one"),
]

dummy_store_listing = MagicMock(return_value=DUMMY_DATASETS + NOT_DATASETS)
dummy_dataset_recognition = MagicMock(side_effect=lambda name: name in DUMMY_DATASETS_PATHS)


@patch('os.path.isdir', new=dummy_dataset_recognition)
@patch('os.listdir', new=dummy_store_listing)
class TestGetDatasets(unittest.TestCase):
    def test_finds_only_directories(self):
        datasets = discover.datasets.get_datasets()
        selected = [dataset["value"] for dataset in datasets]
        self.assertSetEqual(set(DUMMY_DATASETS), set(selected))

    def test_follows_name_value_format(self):
        for dataset in discover.datasets.get_datasets():
            self.assertIn("name", dataset)
            self.assertIn("value", dataset)


class TestSubstituteTags(unittest.TestCase):
    def setUp(self):
        self.tags = {
            "$BLAH": "\"never again\"",
            "$NOPE": str({"1": 2, "3": "4"}).replace("'", '"')
        }
        self.text = """
        {
            "annoying_string": "$BLAH",
            "some_dictionary": "$NOPE",
            "unknown": "$TAG"
        }
        """

    def test_substitutes_known_tags(self):
        text = discover.datasets.substitute_tags(self.tags, self.text)
        self.assertNotIn("$BLAH", text)
        self.assertNotIn("$NOPE", text)

    def test_preserves_unknown_tags(self):
        text = discover.datasets.substitute_tags(self.tags, self.text)
        self.assertIn("$TAG", text)

    def test_produces_readable_json(self):
        text = discover.datasets.substitute_tags(self.tags, self.text)
        structured = json.loads(text)
        self.assertIn("annoying_string", structured)
        self.assertEqual(structured["annoying_string"], "never again")
        self.assertIn("some_dictionary", structured)
        self.assertIn("1", structured["some_dictionary"])
        self.assertEqual(structured["some_dictionary"]["1"], 2)
        self.assertIn("3", structured["some_dictionary"])
        self.assertEqual(structured["some_dictionary"]["3"], "4")
        self.assertIn("unknown", structured)
        self.assertEqual(structured["unknown"], "$TAG")


@patch('os.path.exists', new=MagicMock(return_value=True))
@patch('builtins.open', new=mock_open(read_data="\"blah\""))
class TestFileFromDisk(unittest.TestCase):
    def test_returns_file_content_as_a_response(self):
        content, code = discover.datasets.file_from_disk(None, "some_path")
        self.assertEqual("\"blah\"", content)
        self.assertEqual(code, 200)

    def test_returns_404_for_nonexistent(self):
        with patch('os.path.exists', return_value=False):
            _, code = discover.datasets.file_from_disk(None, "some_path")
        self.assertEqual(code, 404)

    def test_performs_substitution_if_specified(self):
        factory = lambda: partial(discover.datasets.substitute_tags, {"blah": "wololo"})
        content, _ = discover.datasets.file_from_disk(factory, "some_path")
        self.assertEqual(content, "wololo")
