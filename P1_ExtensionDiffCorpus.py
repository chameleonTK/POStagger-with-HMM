import progressbar
from POSTagger import POSTagger
from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import nps_chat
from nltk.corpus import conll2000

from ViterbiPOSTagger import ViterbiPOSTagger
from BeamSearchPOSTagger import BeamSearchPOSTagger
from ForwardBackwardPOSTagger import ForwardBackwardPOSTagger
from POSTagger import POSTagger
import nltk
import time

import sys

def getData(corpus="brown", categories=""):
    if corpus == "brown":
        if categories != "":
            return brown.tagged_sents(tagset='universal', categories=categories)

        return brown.tagged_sents(tagset='universal')
    elif corpus == "treebank":
        return treebank.tagged_sents(tagset='universal')
    elif corpus == "nps_chat":
        #Identifying Dialogue
        data = []
        posts = nps_chat.posts()
        words = nps_chat.tagged_words(tagset='universal')

        index = 0
        for sent in posts:
            data.append(words[index: index+len(sent)])
            index += len(sent)
        return data

    elif corpus == "conll2000":
        return conll2000.tagged_sents(tagset='universal')

    return brown.tagged_sents(tagset='universal')

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        exit("Usage: python P1_ExtensionDiffCorpus.py <corpus>")

    startWord = POSTagger.startWord
    endWord = POSTagger.endWord
    data = getData(sys.argv[1])
    if len(sys.argv) >= 3:
        data = getData(sys.argv[1], sys.argv[2])

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

    taggers = [
        ("Viterbi", ViterbiPOSTagger(tagset)), 
        ("Beam search", BeamSearchPOSTagger(tagset, 1)), 
        ("Forward-backward", ForwardBackwardPOSTagger(tagset))
    ]

    for (name, tagger) in taggers:
        startTime = time.time()
        emissionProb = tagger.getEmissionProb(trainSents, tagset)
        transitionProb = tagger.getTransitionProb(trainSents, tagset)
        leaningTime = time.time()

        print name, "Algorithm"
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

        for tag in tagset:
            if tag == startWord or tag == endWord:
                continue
            
            if (comparision[tag]["correct"] + comparision[tag]["incorrect"]) == 0:
                print "Tag {0} {1}".format(tag, "Inf")
                continue

            accuracy = comparision[tag]["correct"]*100.0/(comparision[tag]["correct"] + comparision[tag]["incorrect"])
            print "Tag {0} {1:.2f}".format(tag, accuracy)
            
        accuracy = comparision["total"]["correct"]*100.0/(comparision["total"]["correct"] + comparision["total"]["incorrect"])
        print name, "%Acc {0:.2f}".format(accuracy)





