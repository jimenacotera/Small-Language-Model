from nltk import *
from conllu import *
from treebanks import languages, train_corpus, test_corpus, conllu_corpus
from collections import defaultdict, Counter
from math import log, exp
from sys import float_info

# Hidden Markov Model Class to estimate probabilities of tokens: words and punctuation.
# Uses Viterbi algorithm to find most likely tag sequence.
# Uses Witten-Bell smoothing to estimate probabilities.
# Probabilites are given in log form to avoid underflow.
class HidddenMarkovModel:

    # Default initialiser for HidddenMarkovModel class.
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

    # Initialise the variables and "train" the model.
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

    # P(tag) using Witten-Bell smoothing.
    def __unigramProbability(self, tag):
        m = len(self.tagOccurr)  
        if self.tagOccurr[tag] > 0:
            return log(self.tagOccurr[tag] / (self.n + m))
        else:
            return log(m / (self.z * (self.n + m)))

    # P(word|tag) using Witten-Bell smoothing in log space.
    def unigramEmissionProbability(self, word, tag):
        tag_word_counts = self.wordTagCounts[tag]
        t = len(self.uniqueWordsPerTag[tag])  
        n = self.tagOccurr[tag]               

        if word in tag_word_counts:
            return log(tag_word_counts[word] / (n + t))
        else:
            return log(t / (self.z * (n + t)))

    # P(tag|precedingTag) using Witten-Bell smoothing in log space.
    def bigramTransitionProbability(self, precedingTag, tag):
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

        # Get smoothed unigram probability not in log space
        unig_prob = exp(self.__unigramProbability(tag))  
        
        # Return probability in log space
        prob = (lambd * prob_tag_given_preceding) + ((1 - lambd) * unig_prob)
        return log(prob) if prob > 0 else -float_info.max # TODO

    # Safely add log probabilities: function provided with practical.
    def __logsumexp(self, vals):
        min_log_prob = -float_info.max
        if len(vals) == 0:
            return min_log_prob
        m = max(vals)
        if m == min_log_prob:
            return min_log_prob
        return m + log(sum([exp(val - m) for val in vals]))

    # Viterbi algorithm to find most likely tag sequence for given sentence.
    # Sentences do not include sentence markers so it is handled in the function.
    def viterbi(self, sent):
        # Initialize tables
        V = defaultdict(lambda: defaultdict(lambda: -float_info.max))  # Viterbi matrix
        backptr = defaultdict(dict)  # Backpointers
        
        # Initialize first position
        for tag in self.uniqueTags:
            trans_p = self.bigramTransitionProbability("<s>", tag)
            emit_p = self.unigramEmissionProbability(sent[0], tag)
            V[0][tag] = self.__logsumexp([trans_p, emit_p])
        
        # Recursion
        for t in range(1, len(sent)):
            for tag in self.uniqueTags:
                max_prob = -float_info.max
                max_prev = None
                
                for prev_tag in self.uniqueTags:
                    trans_p = self.bigramTransitionProbability(prev_tag, tag)
                    emit_p = self.unigramEmissionProbability(sent[t], tag)
                    prob = self.__logsumexp([V[t-1][prev_tag], trans_p, emit_p])
                    
                    if prob > max_prob:
                        max_prob = prob
                        max_prev = prev_tag
                
                V[t][tag] = max_prob
                backptr[t][tag] = max_prev
        
        # Termination
        max_prob = -float_info.max
        max_tag = None
        for tag in self.uniqueTags:
            prob = self.__logsumexp([V[len(sent)-1][tag], self.bigramTransitionProbability(tag, "</s>")])
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        
        # Backtrack and reconstruct the tag sequence
        tags = [max_tag]
        for t in range(len(sent)-1, 0, -1):
            tags.insert(0, backptr[t][tags[0]])
            
        return tags

    # Compute accuracy of POS tagging on provided test corpus.
    def posTaggingAccuracy(self, testCorpus):
        # Accuracy number of correct tags divided by total number of tags.
        total_tags = 0
        correct_tags = 0
        
        for sent in testCorpus:
            words = [token["form"] for token in sent]
            true_tags = [token["upos"] for token in sent]
            pred_tags = self.viterbi(words)
            
            correct_tags += sum(1 for pred, true in zip(pred_tags, true_tags) if pred == true)
            total_tags += len(true_tags)
        
        return correct_tags / total_tags if total_tags > 0 else 0

    # Compute perplexity of test corpus using the HMM model.
    def computePerplexity(self, testCorpus):
        total_log_prob = -float_info.max
        total_tokens = 0  # Will count actual tokens + one EOS token per sentence
        
        for sent in testCorpus:
            # Add 1 to token count for the EOS marker for this sentence
            total_tokens += len(sent) + 1
            
            # Get words and tags from sentence
            words = [token["form"] for token in sent]
            tags = [token["upos"] for token in sent]
            
            # Calculate probability of first word and tag
            if len(words) > 0:
                # P(t1|<s>) * P(w1|t1)
                first_trans = self.bigramTransitionProbability("<s>", tags[0])
                first_emit = self.unigramEmissionProbability(words[0], tags[0])
                total_log_prob = self.__logsumexp([total_log_prob, first_trans + first_emit])
            
            # Calculate probabilities for rest of sentence
            for i in range(1, len(words)):
                # P(ti|ti-1) * P(wi|ti)
                trans_prob = self.bigramTransitionProbability(tags[i-1], tags[i])
                emit_prob = self.unigramEmissionProbability(words[i], tags[i])
                total_log_prob = self.__logsumexp([total_log_prob, trans_prob + emit_prob])
            
            # Add probability of end of sentence
            if len(tags) > 0:
                # P(</s>|tn)
                eos_prob = self.bigramTransitionProbability(tags[-1], "</s>")
                total_log_prob = self.__logsumexp([total_log_prob, eos_prob])
        
        # Normalize by total tokens
        avg_log_prob = total_log_prob / total_tokens
        
        # Return perplexity
        return exp(-avg_log_prob)
