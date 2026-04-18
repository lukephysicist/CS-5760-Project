import pickle
import numpy as np
import random
from nltk.corpus import stopwords
from collections import Counter
import random
from nltk.util import ngrams
    
def get_common_words(path):
    common_words = []
    with open(path, "r") as file:
        for line in file:
            word = line.strip()
            common_words.append(word)
        
    return common_words

def laplace_smooth(corpus_count, common_words):
    # Add count = 1 for most common non-stopword words from https://github.com/mahsu/IndexingExercise/blob/master/5000-words.txt
    common_5k = set([w.lower() for w in common_words])
    words_to_remove = set(stopwords.words("english"))
    words_to_remove.update(['.',',', ')','(', ':', '•', "’"])
    common_words = common_5k - words_to_remove

    common_count = Counter(common_words)
    top_tenK = [word for word, _ in common_count.most_common(5000)]
    
    for word in top_tenK:
        if (word not in corpus_count) and (word not in words_to_remove):
            corpus_count[word] = 1
    
def get_unigram_weights(corpus, smoothing=True):
    corpus_count = Counter(corpus)
    if smoothing:
        common_5k = get_common_words("data/common_words.txt")
        laplace_smooth(corpus_count, common_5k)
    
    total_count = sum(corpus_count.values())
    return {word: count / total_count for word, count in corpus_count.items()}


def insert_word(counts_dict, prev, target):
        if prev not in counts_dict:
            counts_dict[prev] = {target : 1}
        else:
            word_counts = counts_dict[prev]
            if target not in word_counts:
                word_counts[target] = 1
            else:
                word_counts[target] += 1

def get_bigram_weights(corpus, smoothing=True):
    counts = {}
    for i in range(1, len(corpus)):
        insert_word(counts, corpus[i-1], corpus[i])
        
    # laplace smoothing
    if smoothing:
        common_5k = get_common_words("data/common_words.txt")
        for v in counts.values():
            laplace_smooth(v, common_5k)

    # get weights
    for word_counts in counts.values():
        total_count_this_key = sum([count for count in word_counts.values()])
        for word in word_counts.keys():
            word_counts[word] /= total_count_this_key
    return counts


def get_trigram_weights(corpus, smoothing=True):
    counts = {}
    for i in range(2, len(corpus)):
        insert_word(counts, (corpus[i-2], corpus[i-1]), corpus[i])
    
    # laplace smoothing
    if smoothing:
        common_5k = get_common_words("data/common_words.txt")
        for v in counts.values():
            laplace_smooth(v, common_5k)
    
    # get weights
    for word_counts in counts.values():
        total_count_this_key = sum([count for count in word_counts.values()])
        for word in word_counts.keys():
            word_counts[word] /= total_count_this_key

    return counts
    


def n_gram_prediction(unigram, bigram, trigram, predict_on_this=None): 
    def unigram_pred():
        words = list(unigram.keys())
        probs = list(unigram.values())
        return random.choices(words, probs)[0]
    
    def bigram_pred(predict_on_this):
        assert isinstance(predict_on_this, str), "input most be a string for bigram"
        
        probs = bigram.get(predict_on_this)
        # backoff to unigram
        if not probs:
            return unigram_pred()

        words = list(probs.keys())
        probabilites = list(probs.values())
        return random.choices(words, probabilites)[0]
    
    def trigram_pred(predict_on_this):

        probs = trigram.get(predict_on_this)
        # backoff to bigram
        if not probs:
            return bigram_pred(predict_on_this[1])

        words = list(probs.keys())
        probabilites = list(probs.values())
        return random.choices(words, probabilites)[0]

    # UNIGRAM, generates a random word from the probability distribution
    if not predict_on_this:
        return unigram_pred()
    # BIGRAM, randomly samples probability distribution of the previous word with UNIGRAM backoff
    if isinstance(predict_on_this, str):
        return bigram_pred(predict_on_this)
    # BIGRAM, randomly samples probability distribution of the previous two words with BIGRAM backoff
    else:
        return trigram_pred(predict_on_this)
    

def sample(corpus, model="unigram", length=20):
    unigram = get_unigram_weights(corpus, smoothing=False)
    bigram = get_bigram_weights(corpus, smoothing=False)
    trigram = get_trigram_weights(corpus, smoothing=False)
    
    # Get first first word from unigram prediction
    sentence = [n_gram_prediction(unigram, bigram, trigram)]
    
    for i in list(range(length))[1:]:
        if model == "unigram":
            sentence.append(n_gram_prediction(unigram, bigram, trigram))
        if model == "bigram":
            sentence.append(n_gram_prediction(unigram, bigram, trigram, sentence[i-1]))
        if model == "trigram":
            try:
                sentence.append(n_gram_prediction(unigram, bigram, trigram, (sentence[i-2], sentence[i-1])))
            # Use bigram for second word
            except IndexError:
                sentence.append(n_gram_prediction(unigram, bigram, trigram, sentence[i-1]))
    
    return sentence


# Grabs a random portion of text to make the testing set
def get_train_test_split(corpus, test_size = 100):
    corpus_len = len(corpus)
    random_idx = random.randint(0, corpus_len - test_size)
    test_set = corpus[random_idx : random_idx + test_size]
    training_set = corpus[:random_idx] + corpus[random_idx + test_size:]
    
    return training_set, test_set

def compute_perplexity(unigram, bigram, trigram, 
                       test_set, 
                       model="unigram"): 
    def trigram_log_prob(tri, bi, uni, context, word):
        prob = tri.get(context, {}).get(word)
        if not prob:
            return bigram_log_prob(bi, uni, context[1], word)
        return prob

    def bigram_log_prob(bi, uni, context, word):
        prob = bi.get(context, {}).get(word)
        if not prob:
            return unigram_log_prob(uni, word)
        return prob

    def unigram_log_prob(uni, word):
        prob = uni.get(word)
        if prob:
            return np.log(prob)
        return None

    
    min_prob = np.log(min(prob for prob in unigram.values()))
    log_prob_sum = 0
    N = 0

    for i in range(len(test_set)):
        word = test_set[i]
        prob = None

        if model == "trigram":
            prob = trigram_log_prob(trigram, bigram, unigram, (test_set[i-2], test_set[i-1]), word)
        if model == "bigram":
            prob = bigram_log_prob(bigram, unigram, test_set[i-1], word)
        if model == "unigram":
            prob = unigram_log_prob(unigram, word)

        if not prob:
            log_prob_sum += min_prob
        else:
            log_prob_sum += prob
        N += 1

    return np.exp(-log_prob_sum / N)


if __name__ == "__main__":
    
    ## LEMMATIZED ##
    with open("data/normalized/lemmatized.pkl", "rb") as file:
        lemmatized = pickle.load(file)

    print("Random sentences from trigram")
    for i in range(1,4):
        sentence = " ".join(sample(lemmatized, "trigram", length=15))
        print(i, ")", sentence)

    print("\n\n========= LEMMATIZED CORPUS TESTING =========\n")
    train, test = get_train_test_split(lemmatized)

    unigram_smoothed = get_unigram_weights(train, smoothing=True)
    bigram_smoothed = get_bigram_weights(train, smoothing=True)
    print("Unigram Smoothed Perplexity:", compute_perplexity(unigram_smoothed, None, None, test, model="unigram"))
    print("Bigram Smoothed Perplexity:", compute_perplexity(unigram_smoothed, bigram_smoothed, None, test, model="bigram"))

    del unigram_smoothed
    del bigram_smoothed

    unigram_unsmoothed = get_unigram_weights(train, smoothing=False)
    bigram_unsmoothed = get_bigram_weights(train, smoothing=False)
    trigram_unsmoothed = get_trigram_weights(train, smoothing=False)
    
    print("Unigram Unsmoothed Perplexity:", compute_perplexity(unigram_unsmoothed, None, None, test, model="unigram"))
    print("Bigram Unsmoothed Perplexity:", compute_perplexity(unigram_unsmoothed, bigram_unsmoothed, None, test, model="bigram"))
    print("Trigram Unsmoothed Perplexity:", compute_perplexity(unigram_unsmoothed, bigram_unsmoothed, trigram_unsmoothed, test, model="trigram"))
