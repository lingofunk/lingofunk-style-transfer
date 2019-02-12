# ARAE for language style transfer

Original paper: [https://arxiv.org/abs/1706.04223](https://arxiv.org/abs/1706.04223) ("Adversarially Regularized Autoencoders (ICML 2018)" by Zhao, Kim, Zhang, Rush and LeCun). Original implementation: [https://github.com/jakezhaojb/ARAE](https://github.com/jakezhaojb/ARAE).

## Requirements

- Python 3.6
- PyTorch 1.0.1, numpy, nltk, flask

You made need to download the Punkt tokenizer models by running `python -c 'import nltk; nltk.download("punkt")'`.

## Usage

- Training: `python -m lingofunk_transfer_style --mode=train [--load_models]`
- CLI for style transfer and interpolation: `python -m lingofunk_transfer_style (--mode=transfer|--mode=interpolate) --load_models`
- HTTP API for style transfer and interpolation: `python -m lingofunk_transfer_style --mode=serve --port=8000 --load_models`.

Additional configuration can be done via command-line arguments (`python -m lingofunk_transfer_style -h` will list them).

## Running the API using Docker

```console
# Make sure the training output directory is at ./output 
docker build --tag=style_transfer lingofunk-transfer-style
docker run -v "$(pwd)/output":/app/output -p 8000:8000 style_transfer
```
