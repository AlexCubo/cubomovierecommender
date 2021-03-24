from setuptools import setup, find_packages
import os

def open_file(fname):
    """helper function to open a local file"""
    return open(os.path.join(os.path.dirname(__file__), fname))


setup(
    name='cubomovierecommender',
    version='0.0.1',
    author='Alessandro Cubeddu',
    author_email='alessandro.cubeddu@yahoo.co.uk',
    packages=find_packages(),
    url='https://github.com/AlexCubo/cubomovierecommender',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    description='Cubomovierecommender is a Python library for movie recommendations',
    long_description=open_file('README.md').read(),
    # end-user dependencies for your library
    install_requires=[
        'pandas', 
        'scikit-learn', 
        'fuzzywuzzy', 
        'python-Levenshtein',
        'bs4',
        'requests'
    ],
    # include additional data
    package_data= {
        'cubomovierecommender': ['data/*.csv', 'models/nmf.pickle']
    }
)
