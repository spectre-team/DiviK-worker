"""Analyses discovery methods

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
from itertools import chain
from functools import partial
import hashlib
import os
from typing import List, NamedTuple

from common import DATASETS_ROOT
from .datasets import as_readable, get_datasets


only_existing = partial(filter, os.path.exists)
only_folders = partial(filter, os.path.isdir)
Path = str


def folders_in(path: str) -> List[Path]:
    "Find all direct subdirectories"
    return list(only_folders(os.path.join(path, name) for name in os.listdir(path)))


def analysis_directory(analysis_type: str, dataset_name: str) -> Path:
    "Directory for results of analyses of given type"
    return os.path.join(DATASETS_ROOT, dataset_name, analysis_type)


def user_friendly_name(analysis_path: Path) -> str:
    "Convert analysis path to its user-friendly name"
    analyses_root, analysis_name = os.path.split(analysis_path)
    dataset_path, _ = os.path.split(analyses_root)
    _, dataset_name = os.path.split(dataset_path)
    dataset_name = as_readable(dataset_name)
    return "{0}: {1}".format(dataset_name, analysis_name)


def analysis_id(analysis_path: Path) -> str:
    "Get id of the analysis"
    return hashlib.sha256(analysis_path.encode()).hexdigest()


AnalysisResult = NamedTuple('AnalysisResult', [
    ('name', str),
    ('id', str)
])


def find_all_analyses_paths(analysis_type: str) -> List[Path]:
    "Find paths of all available analyses"
    datasets = [dataset['value'] for dataset in get_datasets()]
    analysis_root = partial(analysis_directory, analysis_type)
    search_paths = only_existing(analysis_root(dataset) for dataset in datasets)
    analyses = chain.from_iterable(folders_in(path) for path in search_paths)
    return list(analyses)


def find_analysis_results(analysis_type: str) -> List[AnalysisResult]:
    "Find available results of analyses of given type"
    analyses = find_all_analyses_paths(analysis_type)
    return [
        AnalysisResult(name=user_friendly_name(path), id=analysis_id(path))
        for path in analyses
    ]

def find_analysis_by_id(analysis_type: str, some_id: str) -> Path:
    "Find location of analysis by its id"
    for analysis in find_all_analyses_paths(analysis_type):
        if analysis_id(analysis) == some_id:
            return analysis
    raise ValueError("Unknown id of {0} analysis: {1}".format(analysis_type, some_id))
