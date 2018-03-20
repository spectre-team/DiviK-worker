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
from unittest.mock import MagicMock, patch
import json
import os

import discover.datasets

import api


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
