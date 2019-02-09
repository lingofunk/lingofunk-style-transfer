import argparse
import logging
import sys

import tensorflow as tf
from flask import Flask, jsonify, request, Response

from lingofunk_transfer_style import beam_search, config, greedy_decoding
from lingofunk_transfer_style.style_transfer import StyleTransferModel
from lingofunk_transfer_style.utils import get_batch, untokenize
from lingofunk_transfer_style.vocab import Vocabulary
from nltk import sent_tokenize, word_tokenize

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Transferrer:
    def __init__(self, vocab: Vocabulary, decoder):
        self._vocab = vocab
        self._decoder = decoder

    def transfer(self, text: str, is_positive: bool) -> str:
        text = text.lower()
        sentences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
        batch = get_batch(
            sentences,
            [int(is_positive)] * len(sentences),
            self._vocab.word2id,
            noisy=False,
        )
        _, transferred = self._decoder.rewrite(batch)
        return " ".join(map(untokenize, transferred))


class Server:
    def __init__(self, app: Flask, transferrer: Transferrer, port: int):
        self._app = app
        self._transferrer = transferrer
        self._port = port
        app.route("/api/transfer", methods=["GET", "POST"])(self.transfer)

    def transfer(self):
        if request.method == "POST":
            data = request.json
            logger.debug(data)
            text = data.get("text")
            is_positive = bool(int(data.get("is_positive")))
            return jsonify(text=self._transferrer.transfer(text, is_positive))
        else:
            return Response(status=501)

    def serve(self):
        self._app.run(host="0.0.0.0", port=self._port, debug=True, threaded=True)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="The port to listen on (default is 8001).",
    )
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--embedding", type=str, default="")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--beam", type=int, default=1)
    return parser.parse_args()


def main():
    args = load_args()
    vocab = Vocabulary(args.vocab, args.embedding, config.style_transfer.dim_emb)
    print("Vocabulary size:", vocab.size)

    with tf.Session() as sess:
        model = StyleTransferModel(config.style_transfer, vocab)
        print("Loading model from", args.model + ".*")
        model.saver.restore(sess, args.model)

        if args.beam > 1:
            decoder = beam_search.Decoder(
                sess, vocab, model, config.style_transfer, args.beam
            )
        else:
            decoder = greedy_decoding.Decoder(sess, vocab, model)

        app = Flask(__name__)
        transferrer = Transferrer(vocab, decoder)
        server = Server(app, transferrer, args.port)
        server.serve()


if __name__ == "__main__":
    main()
