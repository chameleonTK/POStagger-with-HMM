
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
