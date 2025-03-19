from nltk import *
from conllu import *
from treebanks import languages, train_corpus, test_corpus, conllu_corpus
from collections import defaultdict, Counter
from math import log, exp
from sys import float_info

# Bigram Model Class to estimate bigram probabilities of tokens: words and punctuation.
class BigramModel:

    # Default initialiser for BigramModel class.
    def __init__(self, trainingCorpus, z=100000):
        self.z = z
        self.n = 0  # total number of tokens
        self.unigramCounts = defaultdict(int)  # count of each word
        self.bigramCounts = defaultdict(int)   # count of word pairs
        self.uniqueFollowers = defaultdict(set)  # set of words that follow each word
        self.__trainModel(trainingCorpus)

    # Train model variables based on provided training corpus.
    def __trainModel(self, trainingCorpus):
        for sent in trainingCorpus:
            words = [token['form'] for token in sent]
            
            # Add sentence start marker
            self.__addCount("<s>")
            
            # Process first word
            if len(words) > 0:
                self.__addCount(words[0])
                self.bigramCounts[("<s>", words[0])] += 1
                self.uniqueFollowers["<s>"].add(words[0])
            
            # Process word pairs
            for i in range(1, len(words)):
                self.__addCount(words[i])
                self.bigramCounts[(words[i-1], words[i])] += 1
                self.uniqueFollowers[words[i-1]].add(words[i])
            
            # Add sentence end marker
            if len(words) > 0:
                self.__addCount("</s>")
                self.bigramCounts[(words[-1], "</s>")] += 1
                self.uniqueFollowers[words[-1]].add("</s>")

    # Increment counts for a word total number of tokens.
    def __addCount(self, word):
        self.n += 1
        self.unigramCounts[word] += 1
        
    # P(word) using Witten-Bell smoothing.
    def __unigramProbability(self, word):
        m = len(self.unigramCounts) 
        if self.unigramCounts[word] > 0:
            return self.unigramCounts[word] / (self.n + m)
        else:
            return m / (self.z * (self.n + m))

    # P(word|prevWord) using Witten-Bell smoothing.
    def bigramProbability(self, word, prevWord):
        precCount = self.unigramCounts[prevWord]
        
        # Calculate lambda
        if precCount == 0:
            lambd = 0
        else:
            # Count number of unique words that follow prevWord
            possibleFollowers = len(self.uniqueFollowers[prevWord])
            lambd = precCount / (precCount + possibleFollowers)

        # Calculate P(word|prevWord)
        if precCount == 0:
            prob_word_given_prev = 0
        else:
            prob_word_given_prev = self.bigramCounts[(prevWord, word)] / precCount

        # Get smoothed unigram probability
        unig_prob = self.__unigramProbability(word)
        
        # Return interpolated probability
        return (lambd * prob_word_given_prev) + ((1 - lambd) * unig_prob)

    # Safely add log probabilities: function provided with practical.
    def __logsumexp(self, vals):
        min_log_prob = -float_info.max
        if len(vals) == 0:
            return min_log_prob
        m = max(vals)
        if m == min_log_prob:
            return min_log_prob
        return m + log(sum([exp(val - m) for val in vals]))

    # Compute perplexity of test corpus using trained model.
    def computePerplexity(self, testCorpus):
        total_log_prob = 0.0
        # Count total number of tokens + one EOS token per sentence.
        total_tokens = 0 
        
        for sent in testCorpus:
            words = [token['form'] for token in sent]
            # Add 1 to token count for the EOS marker for this sentence
            total_tokens += len(words) + 1
            
            # Calculate probability of first word
            if len(words) > 0:
                prob = log(self.bigramProbability(words[0], "<s>"))
                total_log_prob = self.__logsumexp([total_log_prob, prob])
            
            # Calculate probabilities for rest of sentence
            for i in range(1, len(words)):
                prob = log(self.bigramProbability(words[i], words[i-1]))
                total_log_prob = self.__logsumexp([total_log_prob, prob])
            
            # Add probability of end of sentence
            if len(words) > 0:
                prob = log(self.bigramProbability("</s>", words[-1]))
                total_log_prob = self.__logsumexp([total_log_prob, prob])
        
        return total_log_prob
        # # Convert from natural log to log base 2 and normalize
        # avg_log_prob = total_log_prob / log(2) 
        # avg_log_prob = avg_log_prob / total_tokens 
        
        # return pow(2.0, -avg_log_prob)
