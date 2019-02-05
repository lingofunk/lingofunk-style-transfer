from setuptools import setup


setup(
    name="lingofunk_transfer_style",
    version="0.1.0",
    url="https://github.com/lingofunk/lingofunk-transfer-style",
    license='MIT',
    author="Arthur Liss",
    description="Yelp Review Style Transfer",

    packages=['lingofunk_transfer_style'],
    install_requires=['Flask>=1.0.2', 'nltk>=3.4', 'numpy>=1.16.1', 'tensorflow>=1.12.0'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
)
