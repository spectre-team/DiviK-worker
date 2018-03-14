#!/bin/bash

WORKING_DIRECTORY=$(pwd)
echo "Starting up in $WORKING_DIRECTORY"

# try watching interval
if [ "$1" == "" ]; then
    INTERVAL="15s"
else
    INTERVAL=$1
fi
echo "Setting update interval to $INTERVAL"

# start Flask API
echo "Starting Flask..."
flask run -p 80 -h 0.0.0.0 & sleep 1s

# start Celery worker
echo "Starting Celery worker..."
celery -A spectre_analyses worker --loglevel=info -n divik-worker@%h &

# watch them running
while sleep $INTERVAL; do

    if ! pgrep flask > /dev/null; then
        echo "Flask died."
        exit 1
    fi

    if ! pgrep celery > /dev/null; then
        echo "Celery died."
        exit 2
    fi

done
