from setuptools import setup


setup(
    name="lingofunk_transfer_style",
    version="0.1.0",
    url="https://github.com/lingofunk/lingofunk-transfer-style",
    license='MIT',
    author="Arthur Liss",
    description="Yelp Review Style Transfer",

    packages=['lingofunk_transfer_style'],
    install_requires=['torch>=1.0.1', 'numpy', 'nltk', 'Flask'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
)
