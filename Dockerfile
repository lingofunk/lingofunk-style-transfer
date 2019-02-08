FROM python:3.6-jessie

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install .

RUN python -c "import nltk; nltk.download('punkt')"
