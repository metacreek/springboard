FROM python:3.7
WORKDIR news-please-repo
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY config config
COPY commands commands

CMD ["python", "commands/start.py"]
