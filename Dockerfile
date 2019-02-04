FROM ubuntu:16.04

ENV AM_I_IN_A_DOCKER_CONTAINER Yes

# adapted from daten-und-bass.io/blog/fixing-missing-locale-setting-in-ubuntu-docker-image/
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN apt-get install --allow-unauthenticated -y wget git make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl
RUN curl https://pyenv.run | bash
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:$PATH
RUN pyenv install 3.6.4 && pyenv global 3.6.4 && pyenv rehash

WORKDIR /opt/lingofunk
COPY . /opt/lingofunk/
RUN bash download_model.sh
RUN pip install --upgrade pip
RUN pip install .
RUN python -c "import nltk; nltk.download('punkt')"
EXPOSE 8005
CMD ["python", "-m", "lingofunk_transfer_style.server", "--port=8005", \
     "--vocab=model/yelp.vocab", "--model=model/model", "--embedding=model/yelp.d100.emb.txt"]
