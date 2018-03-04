"""API for passing content of schema & layout files

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
from functools import partial
import json
from typing import Callable, Dict, List, Optional, Tuple
import os

import flask

app = flask.Flask(__name__)

FILESYSTEM_ROOT = os.path.abspath(os.sep)
DATASETS_ROOT = os.path.join(FILESYSTEM_ROOT, 'data')


def as_readable(dataset_name: str) -> str:
    """Convert name on disk to readable for the user"""
    return dataset_name.replace('_', ' ')


def is_dir(root: str, name: str) -> bool:
    """Check if name is dataset name in root directory"""
    return os.path.isdir(os.path.join(root, name))


def get_datasets() -> List[Dict[str, str]]:
    """"Get datasets available in the store"""
    is_dataset_name = partial(is_dir, DATASETS_ROOT)
    datasets = filter(is_dataset_name, os.listdir(DATASETS_ROOT))
    return [{"name": as_readable(name), "value": name} for name in datasets]


def substitute_tags(tag_map: Dict[str, str], text: str) -> str:
    """Substitute tags from the text for corresponding values in the map"""
    for tag, value in tag_map.items():
        text = text.replace('"' + tag + '"', value)
    return text


Substitutor = Optional[Callable[[str], str]]


def datasets_substitutor() -> Substitutor:
    """Factory of datasets substitutor"""
    datasets = get_datasets()
    parsed = json.dumps(datasets)
    return partial(substitute_tags, {'$DATASETS': parsed})


SubstitutorFactory = Callable[[], Substitutor]
Response = Tuple[str, int]
NOT_FOUND = "", 404


def file_from_disk(substitutor_factory: SubstitutorFactory, path: str) -> Response:
    """Read file from disk with subsitutions and return it as HTTP response"""
    if not os.path.exists(path):
        return NOT_FOUND
    with open(path) as disk_file:
        content = disk_file.read()
    if substitutor_factory is None:
        return content, 200
    substitutor = substitutor_factory()
    substituted = substitutor(content)
    return substituted, 200


file_with_datasets_substitution = partial(file_from_disk, datasets_substitutor)
unchanged_file = partial(file_from_disk, None)


@app.route('/schema/<string:endpoint>/<string:task_name>/')
def schema(endpoint:str, task_name: str) -> Response:
    """Get static schema description"""
    path = os.path.join('.', 'schema', endpoint, task_name + '.json')
    return unchanged_file(path)


@app.route('/layout/<string:endpoint>/<string:task_name>/')
def layout(endpoint: str, task_name: str) -> Response:
    """Get dynamic layout description"""
    path = os.path.join('.', 'layout', endpoint, task_name + '.json')
    return file_with_datasets_substitution(path)
