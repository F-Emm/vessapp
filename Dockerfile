FROM python:3.7.13

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE $PORT

CMD ["flask", "run", "--host=0.0.0.0:$PORT"]
