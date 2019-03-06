from nltk import FreqDist, WittenBellProbDist
from nltk.util import ngrams
from ViterbiPOSTagger import ViterbiPOSTagger

class BeamSearchPOSTagger(ViterbiPOSTagger):

    def __init__(self, tagset, K):
        self.tagset = tagset
        self.K = K
        if K > len(tagset) :
            self.K = len(tagset)
        
    
    def runthrough(self, vtb, sentence, emissionProb, transitionProb):
        # run for each word
        for i in range(2, len(sentence)-1):
            word = sentence[i]
            
            items = []
            for tag in self.tagset:
                items.append((tag, vtb[tag][i-1][1]))
            topTags = sorted(items, key=lambda item: (-1)*item[1])  

            for tag in self.tagset:
                maxProb = (None, 0)
                cc = 0
                for ttag in topTags:
                    prevTag = ttag[0]
                    p = vtb[prevTag][i-1][1] * transitionProb[prevTag].prob(tag) * emissionProb[tag].prob(word)
                    if p > maxProb[1] or maxProb[0] is None:
                        maxProb = (prevTag, p)
                    cc += 1
                    if cc >= self.K:
                        break
                vtb[tag][i] = maxProb

        return vtb