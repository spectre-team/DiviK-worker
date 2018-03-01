from celery import Celery

app = Celery('spectre_analyzes')
app.config_from_object('spectre_analyses.celery_config')
app.autodiscover_tasks(packages=[
    'spectre_analyses'
], force=True)

if __name__ == '__main__':
    print(app.conf.humanize(with_defaults=False, censored=True))
    app.start()
