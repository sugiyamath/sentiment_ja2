import pickle
import pprint
from sudachipy import tokenizer
from sudachipy import dictionary

with open("model.pkl", "rb") as f:
    vect, models = pickle.load(f)

tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

def tok(x):
    return ' '.join([m.dictionary_form() for m in tokenizer_obj.tokenize(x, mode)]).strip()

import sys
texts = [sys.argv[1]]


v = vect.transform(tok(x) for x in texts)
out = [{"text": x} for x in texts]
for n, m in models:
    for i, p in enumerate(m.predict_proba(v)[:,1]):
        out[i][n] = p

pprint.pprint(out)
