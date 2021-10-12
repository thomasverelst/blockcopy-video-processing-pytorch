import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "blockcopy",
    version = "0.1",
    author = "Thomas Verelst",
    author_email = "thomas.verelst@esat.kuleuven.be",
    description = (""),
    license = "MIT",
    keywords = "",
    url = "",
    packages=["blockcopy"],
    long_description=read('README.md'),
    classifiers=[
    ],
)