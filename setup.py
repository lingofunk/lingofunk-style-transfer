import io
import os
import re

from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="lingofunk_transfer_style",
    version="0.1.0",
    url="https://github.com/lingofunk/lingofunk-transfer-style",
    license='MIT',

    author="Arthur Liss",

    description="Yelp Review Style Transfer",
    long_description=read("README.rst"),

    packages=['lingofunk_transfer_style'],

    install_requires=['Flask>=1.0.2', 'nltk>=3.4', 'numpy>=1.16.1', 'tensorflow>=1.12.0'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
)
