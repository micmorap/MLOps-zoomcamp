FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY Pipfile Pipfile.lock /app/

RUN mkdir -p /app/output_hw04

RUN pip install pipenv

#RUN pipenv install --system --deploy
COPY requirements.txt /app/

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --no-cache-dir -r requirements.txt

COPY starter.py /app/

CMD ["python", "starter.py", "2023", "5" ]

