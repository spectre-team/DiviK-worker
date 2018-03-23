"""Tests summary aspect

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

import matlab_hooks as mh
import aspect._summary as summary


DUMMY_DETAILS = {
    'centroids': None,
    'quality': 0.0,
    'partition': None,
    'filters': None,
    'thresholds': None,
    'merged': None
}


class TestDepth(unittest.TestCase):
    def test_is_zero_for_None(self):
        self.assertEqual(0, summary.depth(None))

    def test_is_one_for_leaf(self):
        tree = mh.DivikResult(subregions=[None, None, None], **DUMMY_DETAILS)
        self.assertEqual(1, summary.depth(tree))

    def test_is_exact_for_mixed_tree(self):
        tree = mh.DivikResult(
            subregions=[
                None,
                mh.DivikResult(subregions=[None, None, None], **DUMMY_DETAILS),
                mh.DivikResult(subregions=[
                    None,
                    None,
                    mh.DivikResult(subregions=[None, None, None], **DUMMY_DETAILS)
                ], **DUMMY_DETAILS)
            ], **DUMMY_DETAILS
        )
        self.assertEqual(3, summary.depth(tree))
