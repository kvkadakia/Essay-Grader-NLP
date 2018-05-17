#!/bin/bash

cd resources/stanford-corenlp-full-2018-01-31
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
pip install nltk
pip install numpy
pip install -U spacy
python -m spacy download en
pip install pyenchant
pip install pycorenlp
python essay_grader.py


