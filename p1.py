from nltk import *
from conllu import *
from treebanks import languages, train_corpus, test_corpus, conllu_corpus
from collections import defaultdict, Counter
from math import log, exp
from sys import float_info


class HidddenMarkovModel:

    def __init__(self, trainingCorpus, z = 100000): 
        self.z = z
        self.n = 0  # total number of tokens
        self.unigramCounts = defaultdict(int)
        self.bigramCount = defaultdict(int)
        self.wordTagCounts = defaultdict(lambda: defaultdict(int))  # counts of word-tag pairs
        self.tagOccurr = defaultdict(int)      # counts for tags
        self.uniqueWordsPerTag = defaultdict(set)  # set of unique words per tag
        self.uniqueTags = []                   # list of all unique tags
        self.__initialiseVars(trainingCorpus)

    def __initialiseVars(self, trainingCorpus): 
        # Get unique tags and initialize counts
        self.uniqueTags = list(set([token['upos'] for sent in trainingCorpus for token in sent]))
        
        # Process sentences
        for sent in trainingCorpus:
            # Add sentence markers and count bigrams
            for i in range(len(sent)):
                # Handle start of sentence
                if i == 0:
                    self.__addCount("<s>", "<s>")
                    self.bigramCount[("<s>", sent[i]['upos'])] += 1
                
                # Handle end of sentence
                if i == len(sent) - 1:
                    self.__addCount("</s>", "</s>")
                    self.bigramCount[(sent[i]['upos'], "</s>")] += 1
                else:
                    self.bigramCount[(sent[i]['upos'], sent[i+1]['upos'])] += 1
                
                # Count word-tag pairs
                word, tag = sent[i]['form'], sent[i]['upos']
                self.__addCount(word, tag)
                self.wordTagCounts[tag][word] += 1
                self.uniqueWordsPerTag[tag].add(word)

    def __addCount(self, word, tag):
        self.n += 1
        self.tagOccurr[tag] += 1
        self.unigramCounts[word] += 1

    def __unigramProbability(self, tag):
        """Calculate P(tag) using Witten-Bell smoothing"""
        m = len(self.tagOccurr)  # number of unique tags
        if self.tagOccurr[tag] > 0:
            return self.tagOccurr[tag] / (self.n + m)
        else:
            return m / (self.z * (self.n + m))

    def unigramEmissionProbability(self, word, tag):
        """Calculate P(word|tag) using Witten-Bell smoothing"""
        tag_word_counts = self.wordTagCounts[tag]
        t = len(self.uniqueWordsPerTag[tag])  # number of unique words seen with this tag
        n = self.tagOccurr[tag]               # total occurrences of this tag

        if word in tag_word_counts:
            return tag_word_counts[word] / (n + t)
        else:
            return t / (self.z * (n + t))

    def bigramTransitionProbability(self, precedingTag, tag):
        """Calculate P(tag|precedingTag) using Witten-Bell smoothing"""
        precCount = self.tagOccurr[precedingTag]
        
        # Calculate lambda
        if precCount == 0:
            lambd = 0
        else:
            # Count number of unique tags that follow precedingTag
            possibleFollowers = sum(1 for (tag1, _) in self.bigramCount if tag1 == precedingTag)
            lambd = precCount / (precCount + possibleFollowers)

        # Calculate P(tag|precedingTag)
        if precCount == 0:
            prob_tag_given_preceding = 0
        else:
            prob_tag_given_preceding = self.bigramCount[(precedingTag, tag)] / precCount

        # Get smoothed unigram probability
        unig_prob = self.__unigramProbability(tag)
        
        # Return interpolated probability
        return (lambd * prob_tag_given_preceding) + ((1 - lambd) * unig_prob)

    def __logsumexp(self, vals):
        """Add log probabilities safely"""
        min_log_prob = -float_info.max
        if len(vals) == 0:
            return min_log_prob
        m = max(vals)
        if m == min_log_prob:
            return min_log_prob
        return m + log(sum([exp(val - m) for val in vals]))

    def viterbi(self, sent):
        """Implement Viterbi algorithm to find most likely tag sequence"""
        # Initialize tables
        V = defaultdict(lambda: defaultdict(lambda: -float_info.max))  # Viterbi matrix
        backptr = defaultdict(dict)  # Backpointers
        
        # Initialize first position
        for tag in self.uniqueTags:
            trans_p = log(self.bigramTransitionProbability("<s>", tag))
            emit_p = log(self.unigramEmissionProbability(sent[0], tag))
            V[0][tag] = trans_p + emit_p
        
        # Recursion
        for t in range(1, len(sent)):
            for tag in self.uniqueTags:
                max_prob = -float_info.max
                max_prev = None
                
                for prev_tag in self.uniqueTags:
                    trans_p = log(self.bigramTransitionProbability(prev_tag, tag))
                    emit_p = log(self.unigramEmissionProbability(sent[t], tag))
                    prob = V[t-1][prev_tag] + trans_p + emit_p
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_prev = prev_tag
                
                V[t][tag] = max_prob
                backptr[t][tag] = max_prev
        
        # Termination
        max_prob = -float_info.max
        max_tag = None
        for tag in self.uniqueTags:
            prob = V[len(sent)-1][tag] + log(self.bigramTransitionProbability(tag, "</s>"))
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        
        # Backtrace
        tags = [max_tag]
        for t in range(len(sent)-1, 0, -1):
            tags.insert(0, backptr[t][tags[0]])
            
        return tags

    def posTaggingAccuracy(self, testCorpus):
        """Calculate accuracy of POS tagging on test corpus"""
        total_tags = 0
        correct_tags = 0
        
        for sent in testCorpus:
            words = [token["form"] for token in sent]
            true_tags = [token["upos"] for token in sent]
            pred_tags = self.viterbi(words)
            
            correct_tags += sum(1 for pred, true in zip(pred_tags, true_tags) if pred == true)
            total_tags += len(true_tags)
        
        return correct_tags / total_tags if total_tags > 0 else 0


## Experiments

# Import language 
lang = "en" 
#lang = "orv" 
# lang = "tr" 
train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))

print("------------")
print("Testing HMM probabilities now")

# for sent in train_sents:
#     for token in sent:
#         print(token['form'], '->', token['upos'], sep='', end=' ')
#         #print(token['form'], sep = '', end= "")
#     print()
    
# uniqueTags = set([token['upos'] for sent in train_sents for token in sent])
# tagCountOccurrences = Counter([token['upos'] for sent in train_sents for token in sent])
# print(uniqueTags)
# hmm = HidddenMarkovModel(trainingCorpus=train_sents)
# print(tagCountOccurrences)
# print(tagCountOccurrences['PROPN'])


hmm = HidddenMarkovModel(trainingCorpus=train_sents)
#hmm = HidddenMarkovModel("tr")

print("unig: ", hmm.unigramEmissionProbability("memphis", "PROPN")) 
print("unig: ", hmm.unigramEmissionProbability("show", "VERB")) 
print("unig: ", hmm.unigramEmissionProbability("me", "PRON")) 
print("unig: ", hmm.unigramEmissionProbability("petersburg", "PRON")) 
print("unig: ", hmm.unigramEmissionProbability("me", "VERB")) 
print("big: ", hmm.bigramTransitionProbability("ADP", "NOUN")) 
print("big: ", hmm.bigramTransitionProbability("NOUN", "NOUN")) 
print("big: ", hmm.bigramTransitionProbability("fgfsgs", "NOUN")) 

# what->PRON is->AUX the->DET cost->NOUN of->ADP
print(hmm.viterbi(["what", "is", "the", "cost", "of"]))
print(hmm.viterbi(["hello", "world"]))

print(hmm.posTaggingAccuracy(train_sents))
print(hmm.posTaggingAccuracy(test_sents))
#what->DET flights->NOUN are->VERB there->PRON from->ADP phoenix->PROPN to->ADP milwaukee->PROPN 