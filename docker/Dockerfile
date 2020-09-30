FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-2:latest

RUN pip install gsutilwrap

WORKDIR /home
COPY app.py /home

ENTRYPOINT ["python", "app.py"]