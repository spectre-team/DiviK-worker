"""Tests of api module

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

import api


class TestReadable(unittest.TestCase):
    def test_replaces_underscores_into_spaces(self):
        self.assertEqual('some name', api.as_readable('some_name'))


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
    os.path.join(api.DATASETS_ROOT, "dataset"),
    os.path.join(api.DATASETS_ROOT, "another"),
    os.path.join(api.DATASETS_ROOT, "and_even_this_one"),
]

dummy_store_listing = MagicMock(return_value=DUMMY_DATASETS + NOT_DATASETS)
dummy_dataset_recognition = MagicMock(side_effect=lambda name: name in DUMMY_DATASETS_PATHS)


@patch('os.path.isdir', new=dummy_dataset_recognition)
@patch('os.listdir', new=dummy_store_listing)
class TestGetDatasets(unittest.TestCase):
    def test_finds_only_directories(self):
        datasets = api.get_datasets()
        selected = [dataset["value"] for dataset in datasets]
        self.assertSetEqual(set(DUMMY_DATASETS), set(selected))

    def test_follows_name_value_format(self):
        for dataset in api.get_datasets():
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
        text = api.substitute_tags(self.tags, self.text)
        self.assertNotIn("$BLAH", text)
        self.assertNotIn("$NOPE", text)

    def test_preserves_unknown_tags(self):
        text = api.substitute_tags(self.tags, self.text)
        self.assertIn("$TAG", text)

    def test_produces_readable_json(self):
        text = api.substitute_tags(self.tags, self.text)
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
        content, code = api.file_from_disk(None, "some_path")
        self.assertEqual("\"blah\"", content)
        self.assertEqual(code, 200)

    def test_returns_404_for_nonexistent(self):
        with patch('os.path.exists', return_value=False):
            _, code = api.file_from_disk(None, "some_path")
        self.assertEqual(code, 404)

    def test_performs_substitution_if_specified(self):
        factory = lambda: partial(api.substitute_tags, {"blah": "wololo"})
        content, _ = api.file_from_disk(factory, "some_path")
        self.assertEqual(content, "wololo")


class TestSchema(unittest.TestCase):
    @patch.object(api, 'unchanged_file')
    def test_passes_json_without_replacements(self, reader):
        schema = api.schema("inputs", "divik")
        self.assertIs(schema, reader.return_value)


class TestSchemaIntegration(unittest.TestCase):
    def test_return_readable_json(self):
        schema, _ = api.schema("inputs", "divik")
        try:
            json.loads(schema)
        except ValueError as ex:
            raise AssertionError(schema) from ex


class TestLayout(unittest.TestCase):
    @patch.object(api, 'file_with_datasets_substitution')
    def test_enumerates_available_datasets(self, reader):
        layout = api.layout("inputs", "divik")
        self.assertIs(layout, reader.return_value)


@patch('os.path.isdir', new=dummy_dataset_recognition)
@patch('os.listdir', new=dummy_store_listing)
class TestLayoutIntegration(unittest.TestCase):
    def test_returns_readable_json_with_datasets(self):
        layout, _ = api.layout("inputs", "divik")
        try:
            structured = json.loads(layout)
        except ValueError as ex:
            raise AssertionError(layout) from ex
        for element in structured:
            if element['key'] == "DatasetName":
                datasets = element['titleMap']
                for dataset in datasets:
                    self.assertIn(dataset['value'], DUMMY_DATASETS)
                break
        else:
            self.fail("DatasetName unspecified in layout")
