from nltk import FreqDist, WittenBellProbDist
from nltk.util import ngrams
from POSTagger import POSTagger

class ViterbiPOSTagger(POSTagger):
    tagset = set()
    
    def __init__(self, tagset):
        self.tagset = tagset

    def initialse(self, sentence, emissionProb, transitionProb):
        vtb = {}
        for tag in self.tagset:
            vtb[tag] = [None for w in sentence] + [None]

        # initialise
        for tag in self.tagset:
            word = sentence[1]
            p = transitionProb[self.startWord].prob(tag) * emissionProb[tag].prob(word)
            vtb[tag][1] = (self.startWord, p)
        
        return vtb

    def runthrough(self, vtb, sentence, emissionProb, transitionProb):
        # run for each word
        for i in range(2, len(sentence)-1):
            word = sentence[i]
            for tag in self.tagset:
                maxProb = (None, 0)
                for prevTag in self.tagset:
                    p = vtb[prevTag][i-1][1] * transitionProb[prevTag].prob(tag) * emissionProb[tag].prob(word)
                    if p > maxProb[1]:
                        maxProb = (prevTag, p)
                        
                vtb[tag][i] = maxProb

        return vtb

    def finalise(self, vtb, sentence, emissionProb, transitionProb):
        # finalise
        maxProb = (None, 0)
        n = len(sentence)-1
        for tag in self.tagset:
            p = vtb[tag][n-1][1] * transitionProb[tag].prob(self.endWord)
            if p > maxProb[1]:
                maxProb = (tag, p)

        return maxProb

    def backtrack(self, vtb, sentence, maxProb):
        # backtrack
        predictedPOS = [(self.endWord, self.endWord)]
        for i in range(len(sentence)-2, 0, -1):
            word = sentence[i]
            tag = maxProb[0]
            
            predictedPOS.insert(0, (word, tag))
            maxProb = vtb[tag][i]

        predictedPOS.insert(0, (self.startWord, self.startWord))
        return predictedPOS

    def apply(self, sentence, emissionProb, transitionProb):    
        vtb = self.initialse(sentence, emissionProb, transitionProb)
        vtb = self.runthrough(vtb, sentence, emissionProb, transitionProb)
        maxProb = self.finalise(vtb, sentence, emissionProb, transitionProb)
        
        if maxProb[0] is None:
            return [(w, None) for w in sentence]

        return self.backtrack(vtb, sentence, maxProb)
