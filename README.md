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

### 1 Sequence Labeling

The python script *5_sequence_labeling* loads the unnormalized corpus and uses the spacy library to tag
each word with a part of speech. It also performs named entity recognition on each token. These entities
are passed to a few regular expression patterns to check if they matched any of the classes in the instructions.
My regular expression patterns from part 1 were augmented with reccomendations from an llm. 

The script prints a frequency table of the parts of speech in the corpus, as well as the first 15 examples
of named entities found by spacy. Below is a sample output

Number of NOUNs: 30509
Number of NUMs: 2606
Number of SPACEs: 25117
Number of PROPNs: 13890
Number of VERBs: 13071
Number of SYMs: 425
Number of ADVs: 2206
Number of ADPs: 13187
Number of DETs: 9957
Number of PUNCTs: 15956
Number of PRONs: 3939
Number of CCONJs: 6787
Number of ADJs: 7968
Number of SCONJs: 1350
Number of AUXs: 5634
Number of PARTs: 3001
Number of Xs: 454
Number of INTJs: 109

    Label       Class                                               Text
0     391        date                                          2025-2026
1     383      office         the Office of the Provost, Student Affairs
2     391        date                                  Between 2018-2019
3     383      office  Conduct, Office of Civil Rights & Title IX Pol...
4     383      office                  The Office of Community Standards
5     383      office                         the Events Planning Office
6     383      office                           Norris University Center
7     383      office  the Office of Global Marketing and Communications
8     383      office                                  the Jacobs Center
9     383      office                                      Norris Center
10    383      office                         the Office of Civil Rights
11    383      office                        the Dean of Students Office
12    383      office                  the Office of Community Standards
13    383      office                             Office of Civil Rights
14    383  department  the Pritzker School of Law and Feinberg School...