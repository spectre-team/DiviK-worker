FROM spectreteam/python_msi:v4.0.4

COPY . /app

WORKDIR /app

RUN apt-get -qq update &&\
  apt-get -qq install git &&\
  pip install -r requirements.txt &&\
  apt-get -qq remove git &&\
  apt-get autoremove -y &&\
  rm -rf /var/lib/apt/lists/*

RUN python -m unittest discover

ENV FLASK_APP api.py

ENTRYPOINT ["/bin/bash"]

CMD ["./run.sh"]
