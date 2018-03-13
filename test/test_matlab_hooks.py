import unittest
from unittest.mock import patch
from enum import Enum
from typing import NamedTuple

import numpy as np

import matlab_hooks as mh

try:
    with mh._matlab_paths():
        import MatlabAlgorithms.MsiAlgorithms
        import matlab
    HAS_MCR = True
except ImportError as ex:
    HAS_MCR = False
    print(ex)


mcr_only = unittest.skipUnless(HAS_MCR, "requires MCR")


@mcr_only
class TestEngine(unittest.TestCase):
    def test_initializes_engine(self):
        mh.engine()

    def test_engine_supports_divik(self):
        engine = mh.engine()
        self.assertTrue(hasattr(engine, 'divik'))


@mcr_only
class TestIsMatlab(unittest.TestCase):
    def test_passes_matlab_data(self):
        self.assertTrue(mh._is_matlab(matlab.double([1])))

    def test_discards_numpy_data(self):
        self.assertFalse(mh._is_matlab(np.array([1])))


class TestDivikDefaults(unittest.TestCase):
    def test_returns_proper_type(self):
        defaults = mh.divik_defaults()
        self.assertIsInstance(defaults, mh.DivikOptions)


class TestUnroll(unittest.TestCase):
    def test_flattens_named_tuple_into_name_value_format(self):
        SomeTuple = NamedTuple('SomeTuple', [
            ('blah', int),
            ('wololo', str),
        ])
        a_tuple = SomeTuple(blah=1, wololo='2')
        flat = mh._unroll(a_tuple)
        self.assertEqual(flat.index('blah') + 1, flat.index(1))
        self.assertEqual(flat.index('wololo') + 1, flat.index('2'))

    def test_simplifies_enums_into_values(self):
        class Dummy(Enum):
            Blah = 'blah_value'
        SomeTuple = NamedTuple('SomeTuple', [('blah', Dummy)])
        simplified = mh._unroll(SomeTuple(Dummy.Blah))
        self.assertEqual('blah', simplified[0])
        self.assertEqual('blah_value', simplified[1])


@mcr_only
class TestDivik(unittest.TestCase):
    def setUp(self):
        self.engine = mh.engine()
        self.options = mh.divik_defaults()
        self.observations = 1000
        self.dimensions = 100
        np.random.seed(0)
        self.data = np.vstack([
            np.random.randn(400, self.dimensions),
            100 + 3 * np.random.randn(400, self.dimensions),
            -300 + 2 * np.random.randn(200, self.dimensions),
        ])

    def test_segments_data(self):
        result = mh.divik(self.options, self.engine, self.data)
        self.assertEqual(len(result.partition), self.observations)
        self.assertEqual(len(result.merged), self.observations)
        self.assertEqual(result.centroids.shape[1], self.dimensions)
        self.assertEqual(result.centroids.shape[0], len(result.subregions))
