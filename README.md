lingofunk-transfer-style
========================

Yelp Review Style Transfer

Usage
-----

```console000
$ cd lingofunk-transfer-style
$ docker build --tag=style_transfer .
$ bash download_model.sh
$ docker run -v "$(pwd)":/opt/lingofunk/model -p 8005:8005 style_transfer
```

Installation
------------

### Requirements

Compatibility
-------------

Licence
-------
