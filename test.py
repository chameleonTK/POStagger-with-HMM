import progressbar
from POSTagger import POSTagger
from nltk.corpus import brown
from ViterbiPOSTagger import ViterbiPOSTagger
from BeamSearchPOSTagger import BeamSearchPOSTagger
from ForwardBackwardPOSTagger import ForwardBackwardPOSTagger
from POSTagger import POSTagger

if __name__ == '__main__':

    print "Load..."
    # Add start and end marker
    startWord = POSTagger.startWord
    endWord = POSTagger.endWord
    # data = brown.tagged_sents(tagset='universal')[0:100]
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

    print "Train size: ", len(trainSents)
    print "Test size: ", len(testSents)

    tagger = ViterbiPOSTagger(tagset)
    tagger = BeamSearchPOSTagger(tagset, 1)
    tagger = ForwardBackwardPOSTagger(tagset)
    # First,
    # Estimate the transition probabilities
    # Estimate the emission probabilities 
    print "Step 1: Estimating probabilities"
    emissionProb = tagger.getEmissionProb(trainSents, tagset)
    transitionProb = tagger.getTransitionProb(trainSents, tagset)

    print tagset
    print emissionProb["DET"].prob("the")