FROM python:3.10.0-slim

RUN pip install -U pip & pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "batch.py", "batch.py" ]
COPY [ "model.bin", "model.bin" ]

ENTRYPOINT [ "python", "batch.py" ]

# docker build -t ride-duration-prediction-service:v1 .
# docker run -it --rm -p 9696:9696  ride-duration-prediction-service:v1