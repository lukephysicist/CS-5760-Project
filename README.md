# CS-5760 NLP PROJECT
Lucas Wengler 700726823

### Dependencies
All code can be exectued by syncing dependencies with uv. Install uv and Run:

*uv sync*

in the root directory

## PART I

### 1 Corpus Collection

The raw corpus consisting of mostly collegiate student handbooks can be found in at data/raw.
Corpus metadata can be found at data/corpus_metadata.csv

A short statistics report on the loaded corpus can be found in the *1_Corpus_Construction.ipynb* notebook

### 2.1 Text Processing

The script 2.1_text_processing.py contains the regex processing for the documents and prints the frequency
of each pattern as well as some examples.


### 2.2 Normalization

The notebook 2.2_normalization performs tokenization, lowering, stopword removal, and creates both a stemmed
and normalized corpus which are written to data/normalized/

Note that it may take a few minutes to download the nltk package information

### 3 N-Gram Language Models

The script *3_ngram_lm.py* trains unigram, bigram, and trigram models trained on the lemmatized corpus. 
A list of common english words in the file *data/common_words.txt* is used for laplace smoothing

The first outputs are some randomly sampled sentences using a trigram. Note that the generated text 
contains no stopwords since we filtered them out in *2.2 Normalization*.

Next we randomly sample a test set of 100 tokens from the lemmatized corpus and use the rest to train
our ngram models. We compute the perplexity for the unsmoothed unigram, bigram, and trigram and the 
smoothed unigram and bigram. No test was done on the smoothed trigram because the model crashes my  
computer when I try to load it into memory. 

As expected, models with smoothing and more context perform better. The following is a sample output
from the script:

Random sentences from trigram

1 ) warrensburg following recovery treatment resource available alcoholic anonymous 660-747-6313 compass health network 844-853-8937 recovery lighthouse

2 ) conduct using technology based actual perceived imbalance power 5 stalking including cyberstalking – repeated harassing

3 ) user must abide find link full listing including appropriate use university facility may restricted revoked

========= LEMMATIZED CORPUS TESTING =========\n

Unigram Smoothed Perplexity: 5405.789251116528

Bigram Smoothed Perplexity: 75.07346795319043

Unigram Unsmoothed Perplexity: 5278.984029427125

Bigram Unsmoothed Perplexity: 1209.774379878437

Trigram Unsmoothed Perplexity: 1174.77396092099

### 4 Text Classification 

Google Gemini generated the messages in *data/4_student_message.jsonl*. These messages each contain
a message id, a true class label, and a text field. The messages are used in the text classification
model in the notebook *4_text_classification.ipynb*

A Leave-One-Out strategy is used to make a prediction for each message. In other words, the training
corpus for each prediction contains all messages except for the one being predicted on. 

The end of the notebook contains model results including precision, recall, f1 and a confusion matrix.
There is also analysis of the model's incorrect predictions.


## PART II

### Q1

The material for Part ii, question 1 is in the notebook *Partii_q1.ipynb*. The notebook shows the training
of a RNN on text from gutenburg.org. The end of the notebook contains temperature specific generations,
the training and calidation loss curves, and a discussion of parameters.


### Q2 & Q3

The notebook *Partii_q2.ipynb* contains the material for questions 2 AND 3 for part two of the project
after encoding and embedding some toy sentences written at the top of the notebook, we create each of the 
required layers. Note that the self-attention layer in the notebook contains the requirement for 
Q3 of the part ii of the project. It implements the Attention(Q,K,V) function. The other layers
requried can also be found below. The end of the notebook contains a sample of a sentence and its encoding.