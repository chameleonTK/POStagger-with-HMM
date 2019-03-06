from nltk import FreqDist, WittenBellProbDist
from nltk.util import ngrams
from POSTagger import POSTagger

class ForwardBackwardPOSTagger(POSTagger):
    tagset = set()
    
    def __init__(self, tagset):
        self.tagset = tagset


    def forward(self, sentence, emissionProb, transitionProb):
        fw = {}
        for tag in self.tagset:
            fw[tag] = [0 for w in sentence] + [0]

        # initialise
        for tag in self.tagset:
            word = sentence[1]
            fw[tag][1] = transitionProb[self.startWord].prob(tag) * emissionProb[tag].prob(word)
        
        # run for each word
        for i in range(2, len(sentence)-1):
            word = sentence[i]
            for tag in self.tagset:

                #print "FW", tag, i
                for prevTag in self.tagset:
                    fw[tag][i] += fw[prevTag][i-1] * transitionProb[prevTag].prob(tag) * emissionProb[tag].prob(word)
                    #print "\t{0:.2e}*{1:.2e}*{2:.2e} = {3:.2e}; {4:.2e}".format(fw[prevTag][i-1], transitionProb[prevTag].prob(tag), emissionProb[tag].prob(word), fw[prevTag][i-1] * transitionProb[prevTag].prob(tag) * emissionProb[tag].prob(word), fw[tag][i])
                
        # finalise
        #print "FW", self.endWord
        for tag in self.tagset:
            fw[self.endWord][len(sentence)-1] += fw[tag][len(sentence)-2] * transitionProb[tag].prob(self.endWord)
            #print "\t{0:.2e}*{1:.2e} = {2:.2e}; {3:.2e}".format(fw[tag][len(sentence)-2], transitionProb[tag].prob(self.endWord), fw[tag][len(sentence)-2] * transitionProb[tag].prob(self.endWord), fw[self.endWord][len(sentence)-1])

        return fw

    def backward(self, sentence, emissionProb, transitionProb):
        bw = {}
        for tag in self.tagset:
            bw[tag] = [0 for w in sentence] + [0]

        # initialise
        for tag in self.tagset:
            bw[tag][len(sentence)-2] = transitionProb[tag].prob(self.endWord)
    
        # run for each word
        for i in range(len(sentence)-3, 0, -1):
            nextWord = sentence[i+1]
            for tag in self.tagset:
                #print "BW", tag, i
                for nextTag in self.tagset:
                    bw[tag][i] += bw[nextTag][i+1] * transitionProb[tag].prob(nextTag) * emissionProb[nextTag].prob(nextWord)
                    # print "\t{0:.2e} * {1:.2e} * {2:.2e} = {3:.2e}; {4:.2e}".format(bw[nextTag][i+1], transitionProb[tag].prob(nextTag), emissionProb[nextTag].prob(nextWord), bw[nextTag][i+1] * transitionProb[tag].prob(nextTag) * emissionProb[nextTag].prob(nextWord), bw[tag][i])
                
        # finalise
        # print "BW", self.startWord
        for tag in self.tagset:
            bw[self.startWord][0] += bw[tag][1] * transitionProb[self.startWord].prob(tag) * emissionProb[tag].prob(sentence[1])

            # print "\t{0:.2e} * {1:.2e} * {2:.2e} = {3:.2e}; {4:.2e}".format(bw[tag][1], transitionProb[self.startWord].prob(tag), emissionProb[tag].prob(sentence[1]), bw[tag][1] * transitionProb[self.startWord].prob(tag) * emissionProb[tag].prob(sentence[1]), bw[self.startWord][0])

        return bw
    
    def printEmissionProb(self, sentence, tagset, emissionProb):
        print ""
        print "Emission Prob"
        print "\t",
        for word2 in sentence:
            print word2, "\t\t",
        print ""
        for tag in self.tagset:
            print tag, "\t",
            for word2 in sentence:
                print "{0:.4e}".format(emissionProb[tag].prob(word2)), "\t",
            print ""
    
    def printTransitionProb(self, sentence, tagset, transitionProb):
        print ""
        print "Transition Prob"
        print "\t",
        for tag in self.tagset:
            print tag, "\t\t",
        print ""
        for tag in self.tagset:
            print tag, "\t",
            for nextTag in self.tagset:
                print "{0:.4e}".format(transitionProb[tag].prob(nextTag)), "\t",
            print ""
        
        print ""

    def printProbTable(self, sentence, tagset, prob, label=""):
        print ""
        print label
        print "\t",
        for word2 in sentence:
            print word2, "\t\t",
        print ""
        for tag in self.tagset:
            print tag, "\t",
            i = 0
            for word2 in sentence:
                print "{0:.4e}".format(prob[tag][i]), "\t",
                i += 1
            print ""     

        print ""
            

    def apply(self, sentence, emissionProb, transitionProb):    
        forwardProb = self.forward(sentence, emissionProb, transitionProb)
        backwardProb = self.backward(sentence, emissionProb, transitionProb)

        # self.printEmissionProb(sentence, self.tagset, emissionProb)
        # self.printTransitionProb(sentence, self.tagset, emissionProb)
        # self.printProbTable(sentence, self.tagset, forwardProb, label="Forward Prob")
        # self.printProbTable(sentence, self.tagset, backwardProb, label="Backward Prob")

        assert "{0:.4e}".format(forwardProb[self.endWord][len(sentence)-1]) == "{0:.4e}".format(backwardProb[self.startWord][0])


        predictedPOS = [(self.startWord, self.startWord)]
        for i in range(1, len(sentence)-1):
            word = sentence[i]
            maxProb = (None, 0)
    
            acc = 0
            for tag in self.tagset:
                if tag == self.startWord or tag == self.endWord:
                    continue
                    
                p = forwardProb[tag][i] * backwardProb[tag][i]
                
                if maxProb[1] < p:
                    maxProb = (tag, p)
                acc += p

            # assert "{0:.4e}".format(forwardProb[self.endWord][len(sentence)-1]) == "{0:.4e}".format(acc)

            predictedPOS.append((word, maxProb[0]))



        predictedPOS.append((self.endWord, self.endWord))
        return predictedPOS
