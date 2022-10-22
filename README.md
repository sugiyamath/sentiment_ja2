# sentiment_ja2
日本語テキストを入力すると6種類の感情を予測します。https://qiita.com/sugiyamath2/items/192f8986ae956a53e231

### 事前準備

```bash
pip3 install scikit-learn sudachipy sudachidict_core
```

### 実行例

```python
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
```

[出力結果]

```
[{'angry': 0.40615498588424054,
  'disgust': 0.43617797134000547,
  'fear': 0.34428334475001915,
  'happy': 0.7088348451518252,
  'sad': 0.18659369619373317,
  'surprise': 0.5945579020526537,
  'text': 'ティファの技って面白いですね'},
 {'angry': 0.3255409884859278,
  'disgust': 0.6403053398396842,
  'fear': 0.5624700021052991,
  'happy': 0.011181721210346484,
  'sad': 0.9826757596031229,
  'surprise': 0.2381717810583594,
  'text': 'エアリスが死んで悲しいよ'},
 {'angry': 0.7500363481182517,
  'disgust': 0.9544050190803858,
  'fear': 0.8494007973218299,
  'happy': 0.2639460326558745,
  'sad': 0.38108966714181386,
  'surprise': 0.7168851050601532,
  'text': 'クラウドって自称ソルジャーとかキモいね'},
 {'angry': 0.9982501925510556,
  'disgust': 0.8589145740901739,
  'fear': 0.3952860506820674,
  'happy': 0.08769217193114408,
  'sad': 0.25645772617129226,
  'surprise': 0.6812433700747725,
  'text': 'バレットって叫んでばっかでうるさくてムカつく'},
 {'angry': 0.5782436616801234,
  'disgust': 0.5969887481607302,
  'fear': 0.8639368010938334,
  'happy': 0.275274952811296,
  'sad': 0.5639769085144961,
  'surprise': 0.9051187081720317,
  'text': 'セフィロスチョー強くてびっくり'},
 {'angry': 0.5908963133343687,
  'disgust': 0.6955870617493253,
  'fear': 0.8791544261974054,
  'happy': 0.1797083289465487,
  'sad': 0.5045448107756639,
  'surprise': 0.6820180108705666,
  'text': 'タークスとかいう闇組織こわ'}]
```

### 詳細
訓練とテストの詳細を知りたい人はtrain_test.ipynbをご覧ください。
