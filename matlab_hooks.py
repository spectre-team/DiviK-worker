"""Hooks & wrappers to MATLAB dedicated algorithms implementation for MSI

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

Note:

    There is still problem in running this code under anything different than
    Linux, as it says that flush is failing. Until we disable printouts, we are
    stuck with testing through containers.

Todo:

    * Recompile MATLAB libraries to disable all the printouts, including
      warnings

Example:

    import spdata.reader
    from matlab_hooks import DivikOptions, engine, divik
    with open('peptides-1.txt') as infile:
        peptides = spdata.reader.load_txt(infile)
    opts = DivikOptions(
        MaxK=10,
        Level=2,
        UseLevels=True,
        AmplitudeFiltration=True,
        VarianceFiltration=True,
        PercentSizeLimit=0.001,
        FeaturePreservationLimit=0.05,
        Metric='pearson',
        MaxComponentsForDecomposition=10,
        KmeansMaxIters=100,
    )
    eng = engine()
    return divik(opts, eng, peptides.spectra)

Example:

    import spdata.reader
    from matlab_hooks import divik_defaults, engine, divik
    with open('peptides-1.txt') as infile:
        peptides = spdata.reader.load_txt(infile)
    opts = divik_defaults()
    eng = engine()
    return divik(opts, eng, peptides.spectra)

"""

from contextlib import contextmanager
from enum import Enum
from itertools import chain
import os
import platform
from typing import Dict, NamedTuple, NewType, List, Union

import numpy as np


_MATLAB_SEARCH_PATHS = \
    ":/usr/local/MATLAB/MATLAB_Runtime/v91/runtime/glnxa64" + \
    ":/usr/local/MATLAB/MATLAB_Runtime/v91/bin/glnxa64" + \
    ":/usr/local/MATLAB/MATLAB_Runtime/v91/sys/os/glnxa64" + \
    ":/usr/local/MATLAB/MATLAB_Runtime/v91/sys/opengl/lib/glnxa64"

Engine = NewType('MatlabEngine', 'matlab_pysdk.runtime.deployablepackage.DeployablePackage')

@contextmanager
def _matlab_paths():
    local_system = platform.system()
    if local_system == 'Linux':
        old_env = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = old_env + _MATLAB_SEARCH_PATHS
    elif local_system == 'Darwin':
        raise NotImplementedError('OSX hosts are not supported.')
    yield
    if local_system == 'Linux':
        os.environ['LD_LIBRARY_PATH'] = old_env


def engine() -> Engine:
    """Create backend for calculations"""
    with _matlab_paths():
        from MatlabAlgorithms import MsiAlgorithms
        import matlab
    return MsiAlgorithms.initialize()


def _is_matlab(some_object):
    with _matlab_paths():
        from MatlabAlgorithms import MsiAlgorithms
        import matlab
    return type(some_object).__module__ == matlab.double.__module__


def _parse_list(list_of_matlab_objects):
    return [_conditional_parse(element) for element in list_of_matlab_objects]


def _parse_dict(dict_of_matlab_objects):
    return {
        key: _conditional_parse(dict_of_matlab_objects[key])
        for key in dict_of_matlab_objects
    }


def _conditional_parse(some_object):
    return np.array(some_object) if _is_matlab(some_object) else some_object


def _to_numpy(matlab_object):
    if isinstance(matlab_object, dict):
        return _parse_dict(matlab_object)
    if isinstance(matlab_object, (list, tuple)):
        return _parse_list(matlab_object)
    return _conditional_parse(matlab_object)


def _numpy_to_matlab(matrix: np.ndarray) -> 'matlab.double':
    if 0 < len(matrix.shape) < 3:
        matrix = matrix.reshape((matrix.shape[0], -1))
        listed = list(map(list, matrix))
        with _matlab_paths():
            from MatlabAlgorithms import MsiAlgorithms
            import matlab
        return matlab.double(listed)
    raise TypeError("Expected 2D matrix or vector. Got: " + repr(matrix))


class DistanceMetric(Enum):
    pearson = 'pearson'
    jaccard = 'jaccard'
    euclidean = 'euclidean'
    cosine = 'cosine'


DivikOptions = NamedTuple('DivikOptions', [
    ('MaxK', int),
    ('Level', int),
    ('UseLevels', bool),
    ('AmplitudeFiltration', bool),
    ('VarianceFiltration', bool),
    ('PercentSizeLimit', float),
    ('FeaturePreservationLimit', float),
    ('Metric', DistanceMetric),
    ('MaxComponentsForDecomposition', int),
    ('KmeansMaxIters', int),
])


def divik_defaults() -> DivikOptions:
    """Build default DiviK options"""
    return DivikOptions(
        MaxK=10,
        Level=3,
        UseLevels=True,
        AmplitudeFiltration=True,
        VarianceFiltration=True,
        PercentSizeLimit=0.001,
        FeaturePreservationLimit=0.05,
        Metric=DistanceMetric.pearson,
        MaxComponentsForDecomposition=10,
        KmeansMaxIters=100,
    )


def _unroll(options: NamedTuple) -> List:
    structured = options._asdict()
    chained = chain(*structured.items())
    return [
        element if not isinstance(element, Enum) else element.value
        for element in chained
    ]


FilterName = str
Table = np.ndarray  # 2D matrix
IntLabels = np.ndarray
BoolFilter = np.ndarray
DivikResult = NamedTuple('DivikResult', [
    ('centroids', Table),
    ('quality', float),
    ('partition', IntLabels),
    ('filters', Dict[FilterName, BoolFilter]),
    ('thresholds', Dict[FilterName, float]),
    ('merged', IntLabels),
    ('subregions', List[Union['DivikResult', None]]),
])


def _parse_divik(matlab_result) -> DivikResult:
    numpied = _to_numpy(matlab_result)
    centroids = numpied['centroids'].T
    quality = numpied['index']
    partition = numpied['partition'].astype(int).ravel()

    filter_names = [name for name in numpied if '_filter' in name]
    filters = {
        filter_name.replace('_filter', ''): numpied[filter_name].astype(bool).ravel()
        for filter_name in filter_names
    }
    thresholds = {
        filter_name.replace('_filter', ''): numpied[filter_name.replace('_filter', '_thr')]
        for filter_name in filter_names
    }

    merged = numpied['merged'].astype(int).ravel() if 'merged' in numpied else partition
    direct_subregions = np.max(partition)
    subregions = direct_subregions * [None]
    if 'subregions' in numpied:
        for idx, region in enumerate(numpied['subregions']):
            if not isinstance(region, np.ndarray):
                subregions[idx] = _parse_divik(region)
    return DivikResult(
        centroids=centroids,
        quality=quality,
        partition=partition,
        filters=filters,
        thresholds=thresholds,
        merged=merged,
        subregions=subregions
    )


def divik(divik_options: DivikOptions, backend: Engine, data: Table) -> DivikResult:
    """Segment data into clusters with DiviK algorithm

    For more details check module docstring.

    Args:
        divik_options (DivikOptions): configuration of DiviK algorithm
        backend (Engine): initialized MCR instance
        data (Table): 2D matrix with observations in rows, features in columns

    Returns:
        Normalized DivikResult, with predictable fields. In contrary to MATLAB
        version, no field disappear in this implementation. Filters and
        thresholds have been wrapped into dictionaries, since they may differ
        between levels. In future, they will be even more flexible, so such
        behavior is reasonable.

    """
    xy = _numpy_to_matlab(np.nan * np.zeros((data.shape[0], 2)))
    data = _numpy_to_matlab(data)
    result = backend.divik(
        data, xy,
        'PlotPartitions', False,
        'PlotRecursively', False,
        'DecompositionPlots', False,
        'DecompositionPlotsRecursively', False,
        'Cache', False,
        *_unroll(divik_options),
        nargout=2)
    return _parse_divik(result[1])
