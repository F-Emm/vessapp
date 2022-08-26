FROM python:3.7.13

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE $PORT

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app

# CMD ["flask", "exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app", "run", "--host=0.0.0.0:$PORT"]
