FROM python:3.8-slim

WORKDIR /app

COPY [ "requirements.txt","constraints.txt", "/app/" ]

RUN apt-get update && apt-get -y install procps && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt -c constraints.txt

COPY . /app

ENTRYPOINT ["python","validation.py"]