"""divik task definition for the purpose of use through Celery

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
from contextlib import contextmanager
from functools import partial
import json
import os
import pickle
import re
import shutil
import signal
import sys

import celery
from celery.utils.log import get_task_logger

import spdata.reader
import spdata.types as ty

from spectre_analyses.celery import app
import matlab_hooks as mh


FILESYSTEM_ROOT = os.path.abspath(os.sep)
DATA_ROOT = os.path.join(FILESYSTEM_ROOT, 'data')
STATUS_PATHS = {
    'all': FILESYSTEM_ROOT,
    'done': DATA_ROOT,
    'processing': os.path.join(FILESYSTEM_ROOT, 'temp'),
    'failed': os.path.join(FILESYSTEM_ROOT, 'failed')
}
Name = str
Path = str
DISALLOWED_CHARACTERS = "[^a-zA-Z0-9_-]"


def _data_path(dataset_name: Name) -> Path:
    dataset_name = re.sub(DISALLOWED_CHARACTERS, '_', dataset_name)
    return os.path.join(DATA_ROOT, dataset_name, 'text_data', 'data.txt')


def _load_data(dataset_name) -> ty.Dataset:
    path = _data_path(dataset_name)
    with open(path) as infile:
        return spdata.reader.load_txt(infile)


_DEFAULT = signal.SIG_DFL
_CELERY_REVOKE = signal.SIGTERM


class signal_trap:
    """Context manager to hijack Celery revoke signal handling"""
    def __init__(self, handler, signal_=_CELERY_REVOKE):
        self._previous_handler = _DEFAULT
        self._handler = handler
        self._signal = signal_

    def _hijack(self, sig_num, stack_frame):
        self._handler(sig_num, stack_frame)
        os.kill(os.getpid(), sig_num)

    def __enter__(self):
        self._previous_handler = (signal.signal(self._signal, self._hijack) or _DEFAULT)
        return self

    def __exit__(self, *args):
        signal.signal(self._signal, self._previous_handler)


def cleanup(path: str, *_):
    """Clean up analysis directory"""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _open_analysis(dataset_name: str, algorithm_name: str, analysis_name: str):
    path_components = dataset_name, algorithm_name, analysis_name
    analysis_root = os.path.join(STATUS_PATHS['processing'], *path_components)
    os.makedirs(analysis_root)
    cleanup_root = partial(cleanup, analysis_root)
    try:
        with signal_trap(cleanup_root):
            yield analysis_root
        dest_root = os.path.join(STATUS_PATHS['done'], *path_components)
    except Exception as ex:
        dest_root = os.path.join(STATUS_PATHS['failed'], *path_components)
        raise RuntimeError('Analysis failed.') from ex
    finally:
        shutil.move(analysis_root, dest_root)


def _simply_typed(result: mh.DivikResult):
    result = result._asdict()
    result['centroids'] = result['centroids'].tolist()
    result['partition'] = result['partition'].tolist()
    result['quality'] = float(result['quality'])
    result['filters'] = {
        key: result['filters'][key].tolist() for key in result['filters']
    }
    result['thresholds'] = {
        key: float(result['thresholds'][key]) for key in result['thresholds']
    }
    result['merged'] = result['merged'].tolist()
    result['subregions'] = [
        _simply_typed(subregion) if subregion is not None else None
        for subregion in result['subregions']
    ]
    return result


def _notify(task, status):
    task.update_state(state=status)
    # Line below updates the status in Celery Flower.
    # It is disabled since Flower disables TERMINATE button for custom state.
    #task.send_event('task-' + status.lower().replace(' ', '_'))


@contextmanager
def _status_notifier(task: celery.Task):
    old_outs = sys.stdout, sys.stderr
    rlevel = task.app.conf.worker_redirect_stdouts_level
    notify = partial(_notify, task)
    logger = get_task_logger('divik')
    task.app.log.redirect_stdouts_to_logger(logger, rlevel)
    yield notify
    sys.stdout, sys.stderr = old_outs



# TODO: Rename the arguments when frontend gets ready for generic forms
@app.task(task_track_started=True, ignore_result=True, bind=True, name="analysis.divik")
def divik(self, AnalysisName: str, DatasetName: str, **kwargs):
    # preprocessing of our current strange format
    analysis_details = DatasetName, divik.__name__, AnalysisName
    options = mh.DivikOptions(
        AmplitudeFiltration=True, VarianceFiltration=True,
        **kwargs)
    options = mh.DivikOptions(*options)

    with _status_notifier(self) as notify, \
            _open_analysis(*analysis_details) as tmp_path:
        notify('PRESERVING CONFIGURATION')
        config_path = os.path.join(tmp_path, 'options')
        with open(config_path + '.pkl', 'wb') as config_pkl:
            pickle.dump(options, config_pkl)
        with open(config_path + '.json', 'w') as config_json:
            json.dump(options._asdict(), config_json)

        notify('LOADING DATA')
        data = _load_data(DatasetName)
        notify('LAUNCHING MCR ENGINE')
        engine = mh.engine()
        notify('RUNNING DIVIK')
        result = mh.divik(options, engine, data.spectra)

        notify('PRESERVING RESULTS')
        result_path = os.path.join(tmp_path, 'result')
        with open(result_path + '.pkl', 'wb') as result_pkl:
            pickle.dump(result, result_pkl)
        with open(result_path + '.json', 'w') as result_json:
            simple_result = _simply_typed(result)
            json.dump(simple_result, result_json)
