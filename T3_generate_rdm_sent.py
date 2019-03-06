from nltk import corpus
from nltk.probability import FreqDist
from nltk.util import ngrams
from random import random

startWord = "<s>"
endWord = "</s>"

sents = corpus.gutenberg.sents('austen-emma.txt')
words = set(corpus.gutenberg.words('austen-emma.txt')).union([endWord])


bigrams = []
trigrams = []
for s in sents:
    s = [startWord, startWord] + s + [endWord] 
    bigrams += ngrams(s,2)
    trigrams += ngrams(s,3)

freqBigrams = FreqDist(bigrams)
freqTrigrams = FreqDist(trigrams)

sentence = [startWord, startWord]
while sentence[-1] != endWord:
    rand = random()
    acc = 0

    # print sentence[-2:], freqBigrams[tuple(sentence[-2:])]
    for word in words:
        if freqBigrams[tuple(sentence[-2:])] <= 0:
            continue

        prob = freqTrigrams[tuple(sentence[-2:]+[word])]*1.0 / freqBigrams[tuple(sentence[-2:])] 
        acc += prob
        if acc > rand:
            sentence += [word]
            break

print " ".join(sentence[2:-1])




