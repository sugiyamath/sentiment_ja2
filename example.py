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

texts = [
    "ティファの技って面白いですね",
    "エアリスが死んで悲しいよ",
    "クラウドって自称ソルジャーとかキモいね",
    "バレットって叫んでばっかでうるさくてムカつく",
    "セフィロスチョー強くてびっくり",
    "タークスとかいう闇組織こわ",]

v = vect.transform(tok(x) for x in texts)
out = [{"text": x} for x in texts]
for n, m in models:
    for i, p in enumerate(m.predict_proba(v)[:,1]):
        out[i][n] = p

pprint.pprint(out)
