from nltk import *
from conllu import *
from treebanks import languages, train_corpus, test_corpus, conllu_corpus
from collections import defaultdict, Counter
from math import log, exp
from sys import float_info


from HiddenMarkovModel import HidddenMarkovModel
from BigramModel import BigramModel

## Experiments

def experimentHMM(lang):
        # Load data
        train_sents = conllu_corpus(train_corpus(lang))
        test_sents = conllu_corpus(test_corpus(lang))
        
        # Train model
        hmm = HidddenMarkovModel(trainingCorpus=train_sents)
        
        # Calculate accuracies
        train_acc = hmm.posTaggingAccuracy(train_sents)
        test_acc = hmm.posTaggingAccuracy(test_sents)
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Testing accuracy: {test_acc:.4f}")
        
        # Calculate perplexities
        train_perp = hmm.computePerplexity(train_sents)
        test_perp = hmm.computePerplexity(test_sents)
        print(f"Training perplexity: {train_perp:.2f}")
        print(f"Testing perplexity: {test_perp:.2f}")

def bigramExperiment(lang):
        # Load data
        train_sents = conllu_corpus(train_corpus(lang))
        test_sents = conllu_corpus(test_corpus(lang))
        
        # Train bigram model
        bigram_model = BigramModel(train_sents)
        
        # Calculate perplexities
        train_perp = bigram_model.computePerplexity(train_sents)
        test_perp = bigram_model.computePerplexity(test_sents)
        print(f"Training perplexity: {train_perp:.2f}")
        print(f"Testing perplexity: {test_perp:.2f}")
        print("-" * 40)

def main():
    languages = ["en", "tr", "orv"]

    for lang in languages:
        print(f"\nTesting language: {lang}")
        print("-" * 40)
        experimentHMM(lang)

        # # Add bigram model experiments
        print("-" * 40)
        print("Bigram Model Experiments")
        # print("=" * 40)
        bigramExperiment(lang)


main()