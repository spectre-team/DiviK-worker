"""Celery initialization

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
import os
from celery import Celery

env=os.environ
CELERY_BROKER_URL=env.get('CELERY_BROKER_URL','amqp://guest:guest@rabbitmq'),
CELERY_RESULT_BACKEND=env.get('CELERY_RESULT_BACKEND','db+postgresql://guest:guest@postgresql/celery')

app = Celery('spectre_analyzes',
             broker=CELERY_BROKER_URL,
             backend=CELERY_RESULT_BACKEND)
app.autodiscover_tasks(packages=[
    'spectre_analyses'
], force=True)

if __name__ == '__main__':
    print(app.conf.humanize(with_defaults=False, censored=True))
    app.start()
