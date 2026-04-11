# CS-5760 NLP PROJECT
Lucas Wengler 700726823

## Dependencies
All code can be exectued by syncing dependencies with uv. Install uv and Run:

"uv sync"

in the root directory


## 1 Corpus Collection

The raw corpus consisting of mostly collegiate student handbooks can be found in at data/raw.
Corpus metadata can be found at data/corpus_metadata.csv

A short statistics report on the loaded corpus can be found in the 1_Corpus_Construction.ipynb notebook

## 2.1 Text Processing

The script 2.1_text_processing.py contains the regex processing for the documents and prints the frequency
of each pattern as well as some examples.

## 2.2 Normalization

The notebook 2.2_normalization performs tokenization, lowering, stopword removal, and creates both a stemmed
and normalized corpus which are written 