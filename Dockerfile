FROM python:3.6-jessie

COPY setup.py /app/
WORKDIR /app

RUN mkdir lingofunk_transfer_style
RUN pip install --upgrade pip
RUN pip install -e .
RUN python -c "import nltk; nltk.download('punkt')"

COPY . /app
CMD ["python", "-m", "lingofunk_transfer_style", "--mode=serve", "--port=8000", "--load_models", "--no_cuda"]
