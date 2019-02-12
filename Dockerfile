FROM python:3.6-jessie

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install .

RUN python -c "import nltk; nltk.download('punkt')"

CMD ["python", "-m", "lingofunk_transfer_style", "--mode=serve", "--port=8000", "--load_models", "--no_cuda"]
