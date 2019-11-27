#!/bin/sh

# Prepare log files and start outputting logs to stdout
touch /src/logs/gunicorn.log
touch /src/logs/access.log
echo Starting nginx
# Start Gunicorn processes
echo Starting Gunicorn
exec gunicorn main:app \
    --bind localhost:7000 \
    --worker-class aiohttp.worker.GunicornWebWorker \
    --workers 1 \
    --log-level=info \
    --log-file=/src/logs/gunicorn.log \
    --access-logfile=/src/logs/access.log &

exec nginx -g "daemon off;"