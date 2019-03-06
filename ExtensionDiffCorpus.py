import progressbar
from POSTagger import POSTagger
from nltk.corpus import brown
from ViterbiPOSTagger import ViterbiPOSTagger
from BeamSearchPOSTagger import BeamSearchPOSTagger
from ForwardBackwardPOSTagger import ForwardBackwardPOSTagger
from POSTagger import POSTagger
import nltk
import time

import sys

def getData(corpus="brown"):
    print corpus

if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        exit("Usage: python P1_POS <algorithm> <do_accuracy_test>")

    # print "Load..."
    # startTime = time.time()
    # # Add start and end marker
    # startWord = POSTagger.startWord
    # endWord = POSTagger.endWord
    # # data = brown.tagged_sents(tagset='universal')[0:100]
    # data = brown.tagged_sents(tagset='universal')
    # sents = [[(startWord, startWord)] + s + [(endWord, endWord)]  for s in data]

    # # Determine a tagset
    # tagset = set()
    # for s in sents:
    #     t = [t for (w,t) in s]
    #     tagset.update(t)

    # # Split data for training and testing
    # traingRation = 0.95
    # trainSents = sents[0: int(traingRation*len(sents))]
    # testSents = sents[int(traingRation*len(sents)):]

    # print ""
    # print "Train size: ", len(trainSents)
    # print "Test size: ", len(testSents)
    # print ""
    # tagger = ViterbiPOSTagger(tagset)
    # if sys.argv[1] == "beam":
    #     print "Select Beam search algorithm"
    #     tagger = BeamSearchPOSTagger(tagset, 1)
    # elif sys.argv[1] == "fwbw":
    #     print "Select forward and backward probabilities"
    #     tagger = ForwardBackwardPOSTagger(tagset)
    # else:
    #     print "Select Viterbi algorithm"

    # # First,
    # # Estimate the transition probabilities
    # # Estimate the emission probabilities 
    # print "Step 1: Estimating probabilities"
    # emissionProb = tagger.getEmissionProb(trainSents, tagset)
    # transitionProb = tagger.getTransitionProb(trainSents, tagset)
    # leaningTime = time.time()
    # print "Training time", (leaningTime - startTime)
    # print ""
    # if sys.argv[2].lower() == "yes":
    #     # Second, 
    #     # Apply a trained HMM on sentences from the testing data    
    #     print "Step 2: Applying HMM"
    #     bar = progressbar.ProgressBar(maxval=len(testSents), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #     bar.start()

    #     comparision = {
    #         "total": {"correct":0, "incorrect":0},
    #         "first": {"correct":1, "incorrect":0},
    #         "last": {"correct":1, "incorrect":0},
    #     }
    #     for tag in tagset:
    #         comparision[tag] = {"correct":0, "incorrect":0}

    #     index = 0

    #     cc = 0
    #     for sent in testSents:
    #         bar.update(index)
    #         if len(sent) > 100:
    #             continue
    #         words = [w.lower() for (w, t) in sent]
    #         Xs = [w for (w, t) in sent if t=="X"]
    #         if len(Xs) > 0:
    #             cc += 1
    #             continue

    #         predicted = tagger.apply(words, emissionProb, transitionProb)
    #         comparision = tagger.evaluate(sent, predicted, comparision)
    #         index += 1
    #     bar.finish()
    #     print cc
    #     predictingTime = time.time()
    #     print "Predicting time", (predictingTime - leaningTime)
    #     print ""
    #     # Third, 
    #     # Compare them with the gold-standard sequence of tags for that sentence
    #     print "Step 3: Evaluation"
    #     for tag in tagset:
    #         if tag == startWord or tag == endWord:
    #             continue
            
    #         if (comparision[tag]["correct"] + comparision[tag]["incorrect"]) == 0:
    #             print "Tag {0} {1}".format(tag, "Inf")
    #             continue

    #         accuracy = comparision[tag]["correct"]*100.0/(comparision[tag]["correct"] + comparision[tag]["incorrect"])
    #         print "Tag {0} {1:.2f}".format(tag, accuracy)

    #     accuracy = comparision["total"]["correct"]*100.0/(comparision["total"]["correct"] + comparision["total"]["incorrect"])
    #     print "Total {0:.2f}".format(accuracy)

    #     # accuracy = comparision["first"]["correct"]*100.0/(comparision["first"]["correct"] + comparision["first"]["incorrect"])
    #     # print "first {0:.2f}".format(accuracy)
    #     # accuracy = comparision["last"]["correct"]*100.0/(comparision["last"]["correct"] + comparision["last"]["incorrect"])
    #     # print "last {0:.2f}".format(accuracy)

    # print ""
    # if len(sys.argv) > 3:
    #     targetString = sys.argv[3]
        
    #     if targetString is not None:
    #         words = [startWord] + nltk.word_tokenize(targetString) + [endWord]
    #         predicted = tagger.apply(words, emissionProb, transitionProb)
    #         print "Test", words
    #         print predicted





