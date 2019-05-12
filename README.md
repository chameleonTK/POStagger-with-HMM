
# POS Tagger with HMM models

## Usage 

`````
python P1_POS <algorithm> <do_accuracy_test> <test_sentence>
`````

<algorithm>        : select one of these options; viterbi, beam, fwbw
<do_accuracy_test>    : select one of these options; yes, no
<test_sentence>    : (optional) testing sentence

## Example
python P1_POS.py viterbi yes "I like cats"

# Extension

`````
python P1_ExtensionDiffCorpus.py <corpus>
`````

<corpus> 	: select one of these options; brown, treebank, nps_chat, conll2000

===========

`````
python P1_ExtensionDiffSmoothing.py
`````