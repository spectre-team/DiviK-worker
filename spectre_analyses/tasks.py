import json
import os
import pickle as pkl
import shutil
import sys

from celery.utils.log import get_task_logger

import spdata.reader
import spdata.types as ty

from spectre_analyses.celery import app
import matlab_hooks as mh


FILESYSTEM_ROOT = os.path.abspath(os.sep)
DATA_ROOT = os.path.join(FILESYSTEM_ROOT, 'data')
STATUS_PATHS = {
    'all': FILESYSTEM_ROOT,
    'done': os.path.join(FILESYSTEM_ROOT, 'data'),
    'processing': os.path.join(FILESYSTEM_ROOT, 'temp'),
    'failed': os.path.join(FILESYSTEM_ROOT, 'failed')
}
Name = str
Path = str


def _data_path(dataset_name: Name) -> Path:
    return os.path.join(DATA_ROOT, dataset_name, 'text_data', 'data.txt')


def _load_data(dataset_name) -> ty.Dataset:
    path = _data_path(dataset_name)
    with open(path) as infile:
        return spdata.reader.load_txt(infile)


def _initialize_analysis(dataset_name: str, analysis_type: str, analysis_name: str) -> str:
    analysis_root = os.path.join(
        STATUS_PATHS['processing'],
        dataset_name,
        analysis_type,
        analysis_name,
    )
    os.makedirs(analysis_root)
    return analysis_root


def _finalize_analysis(dataset_name: str, analysis_type: str,
                       analysis_name: str, exception: Exception=None):
    temp_root = os.path.join(
        STATUS_PATHS['processing'],
        dataset_name,
        analysis_type,
        analysis_name
    )
    if exception is None:
        dest_root = os.path.join(
            STATUS_PATHS['done'],
            dataset_name,
            analysis_type,
            analysis_name
        )
    else:
        dest_root = os.path.join(
            STATUS_PATHS['failed'],
            dataset_name,
            analysis_type,
            analysis_name
        )
    shutil.move(temp_root, dest_root)


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


logger = get_task_logger('divik')


class StatusNotifier(object):
    def __init__(self, task):
        self.task = task

    def __call__(self, status):
        self.task.update_state(state=status)
        # Line below updates the status in Celery Flower.
        # It is disabled since Flower disables TERMINATE button for custom state.
        #self.task.send_event('task-' + status.lower().replace(' ', '_'))


@app.task(task_track_started=True, ignore_result=True, bind=True)
def divik(self, dataset_name: str, options: mh.DivikOptions, analysis_name: str):
    old_outs = sys.stdout, sys.stderr
    rlevel = self.app.conf.worker_redirect_stdouts_level
    notify = StatusNotifier(self)
    try:
        self.app.log.redirect_stdouts_to_logger(logger, rlevel)

        notify('INITAILIZING')
        tmp_path = _initialize_analysis(dataset_name, divik.__name__, analysis_name)

        notify('PRESERVING CONFIGURATION')
        options = mh.DivikOptions(*options)
        config_path = os.path.join(tmp_path, 'options')
        with open(config_path + '.pkl', 'wb') as config_pkl:
            pkl.dump(options, config_pkl)
        with open(config_path + '.json', 'w') as config_json:
            json.dump(options._asdict(), config_json)

        notify('LOADING DATA')
        data = _load_data(dataset_name)
        notify('LAUNCHING MCR ENGINE')
        engine = mh.engine()
        notify('RUNNING DIVIK')
        result = mh.divik(options, engine, data.spectra)

        notify('PRESERVING RESULTS')
        result_path = os.path.join(tmp_path, 'result')
        with open(result_path + '.pkl', 'wb') as result_pkl:
            pkl.dump(result, result_pkl)
        with open(result_path + '.json', 'w') as result_json:
            simple_result = _simply_typed(result)
            json.dump(simple_result, result_json)

        notify('FINALIZING')
        _finalize_analysis(dataset_name, divik.__name__, analysis_name)

    except Exception as ex:
        _finalize_analysis(dataset_name, divik.__name__, analysis_name, ex)
        raise RuntimeError() from ex

    finally:
        sys.stdout, sys.stderr = old_outs
