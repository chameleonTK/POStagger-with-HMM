import progressbar
from POSTagger import POSTagger
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import nps_chat
from nltk.corpus import conll2000

from ViterbiPOSTagger import ViterbiPOSTagger
from BeamSearchPOSTagger import BeamSearchPOSTagger
from ForwardBackwardPOSTagger import ForwardBackwardPOSTagger
from nltk import FreqDist, WittenBellProbDist,  LidstoneProbDist, SimpleGoodTuringProbDist
from nltk.util import ngrams
import nltk
import time

import sys

def getEmissionProb(sm, sents, tagset):
    # P(word|tag) = transitionProb[tag].prob(word)
    emission = []
    for s in sents:
        emission += [(w.lower(), t) for (w, t) in s]

    emissionProb = {}
    for tag in tagset:
        words = [w for (w, t) in emission if t == tag]
        if sm == "no":
            emissionProb[tag] = LidstoneProbDist(FreqDist(words), 0, bins=1e5)
        elif sm == "laplace":
            emissionProb[tag] = LidstoneProbDist(FreqDist(words), 1, bins=1e5)
        elif sm == "goodturing":
            emissionProb[tag] = SimpleGoodTuringProbDist(FreqDist(words), bins=1e5)
        else:
            emissionProb[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

    return emissionProb

def getTransitionProb(sm, sents, tagset):
    # P(nextTag|prevTag) = transitionProb[prevTag].prob(nextTag)
    transition = []
    for s in sents:
        tags = [t for (w, t) in s]
        transition += ngrams(tags,2)

    transitionProb = {}
    for tag in tagset:
        nextTags = [nextTag for (prevTag, nextTag) in transition if prevTag == tag]
        
        if sm == "no":
            transitionProb[tag] = LidstoneProbDist(FreqDist(nextTags), 0, bins=1e5)
        elif sm == "laplace":
            transitionProb[tag] = LidstoneProbDist(FreqDist(nextTags), 1, bins=1e5)
        elif sm == "goodturing":
            transitionProb[tag] = SimpleGoodTuringProbDist(FreqDist(nextTags), bins=1e5)
        else:
            transitionProb[tag] = WittenBellProbDist(FreqDist(nextTags), bins=1e5)

    return transitionProb

if __name__ == '__main__':

    startWord = POSTagger.startWord
    endWord = POSTagger.endWord
    data = brown.tagged_sents(tagset='universal')

    sents = [[(startWord, startWord)] + s + [(endWord, endWord)]  for s in data]

    # Determine a tagset
    tagset = set()
    for s in sents:
        t = [t for (w,t) in s]
        tagset.update(t)

    # Split data for training and testing
    traingRation = 0.95
    trainSents = sents[0: int(traingRation*len(sents))]
    testSents = sents[int(traingRation*len(sents)):]

    print ""
    print "Train size: ", len(trainSents)
    print "Test size: ", len(testSents)
    print ""

    smoothing = [
        ("No", 'no'), 
        ("Witten-Bell", 'wittenbell'), 
        ("Laplace", 'laplace'), 
        ("Good-Turing", 'goodturing')
    ]

    for (name, sm) in smoothing:

        tagger = ViterbiPOSTagger(tagset)
        startTime = time.time()
        emissionProb = getEmissionProb(sm, trainSents, tagset)
        transitionProb = getTransitionProb(sm, trainSents, tagset)
        leaningTime = time.time()

        print name, "Smoothing"
        print ""

        bar = progressbar.ProgressBar(maxval=len(testSents), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        comparision = {"total": {"correct":0, "incorrect":0}}
        for tag in tagset:
            comparision[tag] = {"correct":0, "incorrect":0}

        index = 0
        for sent in testSents:
            bar.update(index)
            if len(sent) > 100:
                continue
            words = [w.lower() for (w, t) in sent]

            predicted = tagger.apply(words, emissionProb, transitionProb)
            comparision = tagger.evaluate(sent, predicted, comparision)
            index += 1
        bar.finish()
        
        predictingTime = time.time()
        print "Predicting time", (predictingTime - leaningTime)
        print ""

        # for tag in tagset:
        #     if tag == startWord or tag == endWord:
        #         continue
            
        #     if (comparision[tag]["correct"] + comparision[tag]["incorrect"]) == 0:
        #         print "Tag {0} {1}".format(tag, "Inf")
        #         continue

        #     accuracy = comparision[tag]["correct"]*100.0/(comparision[tag]["correct"] + comparision[tag]["incorrect"])
        #     print "Tag {0} {1:.2f}".format(tag, accuracy)
            
        accuracy = comparision["total"]["correct"]*100.0/(comparision["total"]["correct"] + comparision["total"]["incorrect"])
        print name, "%Acc {0:.2f}".format(accuracy)





