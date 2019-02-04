FROM ubuntu:16.04

# adapted from daten-und-bass.io/blog/fixing-missing-locale-setting-in-ubuntu-docker-image/
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN apt-get install --allow-unauthenticated -y wget python3.6
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

RUN pip3.6 install --upgrade pip && pip3.6 install virtualenv
WORKDIR /opt/lingofunk
RUN virtualenv venv && venv/bin/activate

COPY . /opt/lingofunk/
RUN pip install .
RUN bash download_model.sh
EXPOSE 8001
CMD [ "python", "-m", "lingofunk_transfer_style.server", "--port=8001",
      "--vocab=model\yelp.vocab", "--model=model\model", "--embedding=model\yelp.d100.emb.txt" ]
