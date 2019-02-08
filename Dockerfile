FROM python:3.6-jessie

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install .

RUN python -c "import nltk; nltk.download('punkt')"

CMD ["python", "-m", "lingofunk_transfer_style.server", "--port=8005", \
     "--vocab=model/yelp.vocab", "--model=model/model", "--embedding=model/yelp.d100.emb.txt"]
