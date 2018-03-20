"""Tests for analyses discovery methods

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
from unittest.mock import create_autospec, MagicMock, patch
import os

import discover.analyses as an


ROOT = os.path.abspath(os.sep)
SAMPLE_LISTINGS = {
    os.path.join(ROOT, 'data'): [
        'peptides-1',
        'peptides-2',
        'is this a directory'
    ],
    os.path.join(ROOT, 'data', 'peptides-1', 'divik'): [
        'blah',
        'wololo',
        'blaah'
    ],
    os.path.join(ROOT, 'data', 'peptides-2', 'divik'): [
        'sample analysis',
        'this is boring'
    ]
}
IS_DIR = [
    os.path.join(ROOT, 'data', 'peptides-1'),
    os.path.join(ROOT, 'data', 'peptides-1', 'divik', 'blah'),
    os.path.join(ROOT, 'data', 'peptides-1', 'divik', 'wololo'),
    os.path.join(ROOT, 'data', 'peptides-1', 'divik', 'blaah'),
    os.path.join(ROOT, 'data', 'peptides-2'),
    os.path.join(ROOT, 'data', 'peptides-2', 'divik', 'sample analysis'),
    os.path.join(ROOT, 'data', 'peptides-2', 'divik', 'this is boring')
]


mock_only_existing = create_autospec(an.only_existing, side_effect=lambda f: f)
mock_only_folders = create_autospec(
    an.only_folders,
    side_effect=lambda paths: [
        path for path in paths if path in IS_DIR or path + os.path.sep in IS_DIR
    ])
mock_listdir = create_autospec(
    os.listdir,
    side_effect=lambda path: SAMPLE_LISTINGS[path])


@patch.object(an, 'only_existing', new=mock_only_existing)
@patch.object(an, 'only_folders', new=mock_only_folders)
@patch.object(os, 'listdir', new=mock_listdir)
class TestFoldersIn(unittest.TestCase):
    def test_returns_direct_subdirectories(self):
        path = os.path.join(ROOT, 'data', 'peptides-1', 'divik')
        listing = an.folders_in(path)
        self.assertEqual(3, len(listing))
        for member in listing:
            for supposed_member in ['blah', 'wololo', 'blaah']:
                if member.endswith(supposed_member) or member.endswith(
                        supposed_member + os.path.sep):
                    break
            else:
                self.fail(supposed_member + ' not found')


class TestUserFriendlyName(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(ROOT, 'data', 'peptides-1', 'divik', 'sample divik')

    def test_contains_dataset_name(self):
        name = an.user_friendly_name(self.path)
        self.assertIn('peptides-1', name)

    def test_contains_analysis_name(self):
        name = an.user_friendly_name(self.path)
        self.assertIn('sample divik', name)


class TestAnalysisId(unittest.TestCase):
    def setUp(self):
        self.some_path = os.path.join(ROOT, 'data', 'peptides-1', 'divik', 'sample divik')
        self.other_path = os.path.join(ROOT, 'data', 'peptides-1', 'divik', 'unknown')

    def test_returns_consistent_id_for_the_same_analysis(self):
        first = an.analysis_id(self.some_path)
        second = an.analysis_id(self.some_path)
        self.assertEqual(first, second)

    def test_returns_different_id_for_different_analysis(self):
        first = an.analysis_id(self.some_path)
        second = an.analysis_id(self.other_path)
        self.assertNotEqual(first, second)


@patch.object(an, 'only_existing', new=mock_only_existing)
@patch.object(an, 'only_folders', new=mock_only_folders)
@patch.object(os, 'listdir', new=mock_listdir)
@patch.object(an, 'get_datasets', new=MagicMock(return_value=[
        {'value': 'peptides-1'}, {'value': 'peptides-2'}]))
class TestFindAnalysisResults(unittest.TestCase):
    def test_finds_results(self):
        results = an.find_analysis_results('divik')
        self.assertEqual(5, len(results))
