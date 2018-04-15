"""Backend processing divik results to extract summary aspect

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
import json
import os
import pickle
from typing import Dict, List, NamedTuple, Union

import numpy as np

from common import NOT_FOUND, Response
import discover
import matlab_hooks as mh


DivikSummary = NamedTuple('DivikSummary', [
    ('depth', int),
    ('mean_cluster_size', float),
    ('std_cluster_size', float),
    ('number_of_clusters', int),
    ('size_reduction', float),
])


def depth(tree: mh.DivikResult) -> int:
    "Depth of decomposition tree"
    if tree is None:
        return 0
    return max(map(depth, tree.subregions)) + 1


def cluster_sizes(tree: mh.DivikResult) -> List[int]:
    "Sizes of final clusters"
    _, counts = np.unique(tree.merged, return_counts=True)
    return counts


def make_summary(tree: mh.DivikResult) -> DivikSummary:
    "Create divik result summary"
    sizes = cluster_sizes(tree)
    return DivikSummary(
        depth=depth(tree),
        mean_cluster_size=np.mean(sizes),
        std_cluster_size=np.std(sizes),
        number_of_clusters=len(sizes),
        size_reduction=1.0 - float(len(sizes)) / tree.merged.size
    )


ColumnDefinition = NamedTuple('ColumnDefinition', [('key', str), ('name', str)])
SimpleType = Union[str, int, float, bool]
Key = str
Description = str
Row = Dict[Key, SimpleType]
JsonTable = NamedTuple('JsonTable', [
    ('columns', List[ColumnDefinition]),
    ('data', List[Row])
])


def as_table(some_dict: Dict[str, SimpleType], explanation: Dict[Key, Description]) -> JsonTable:
    columns_definition = [
        ColumnDefinition(key="name", name="Description")._asdict(),
        ColumnDefinition(key="value", name="Value")._asdict()
    ]
    data = [{"name": explanation[key], "value": value} for key, value in some_dict.items()]
    return JsonTable(columns_definition, data)


DIVIK_SUMMARY_EXPLANATION = {
    'depth': 'Depth of the decomposition tree',
    'mean_cluster_size': 'Mean size of the final clusters',
    'std_cluster_size': 'Std. dev. of the final clusters sizes',
    'number_of_clusters': 'Total number of clusters at maximum depth',
    'size_reduction': 'Data volume reduced by'
}


def aspect(analysis_id: str) -> Response:
    try:
        analysis_path = discover.find_analysis_by_id('divik', analysis_id)
    except ValueError:
        return NOT_FOUND
    result_path = os.path.join(analysis_path, 'result.pkl')
    with open(result_path, 'rb') as result_file:
        result = pickle.load(result_file)
    summary = make_summary(result)
    tabular = as_table(summary._asdict(), DIVIK_SUMMARY_EXPLANATION)
    return json.dumps(tabular._asdict()), 200
