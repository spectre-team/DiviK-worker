FROM spectreteam/python_msi:v4.0.4

COPY . /app

WORKDIR /app

RUN apt update &&\
  apt install --yes git &&\
  pip install -r requirements.txt &&\
  apt remove --yes git &&\
  apt-get autoremove --yes

CMD ["celery", "-A", "spectre_analyses", "worker", "--loglevel=info", "-n", "divik-worker@%h"]
