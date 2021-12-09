FROM python:3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
    && apt-get clean

WORKDIR /tmp
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install gunicorn

WORKDIR /app
COPY main.py jumpcutter.py ./
COPY templates templates

CMD ["gunicorn", "--bind", "0.0.0.0:80", "main:app", "-w", "1", "--threads", "1"]
