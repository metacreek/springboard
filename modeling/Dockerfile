FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-2:latest

RUN pip install gsutilwrap
COPY code /home
COPY data /home/data

WORKDIR /home

ENTRYPOINT ["python", "bert_model.py"]
