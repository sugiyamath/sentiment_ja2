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
[{'angry': 0.4224291515429129,
  'disgust': 0.4173733262323974,
  'fear': 0.359762507050206,
  'happy': 0.7178410456561672,
  'sad': 0.1611998429397086,
  'surprise': 0.5737928576153907,
  'text': 'ティファの技って面白いですね'},
 {'angry': 0.28790300106603023,
  'disgust': 0.6518970091207457,
  'fear': 0.5216814638042353,
  'happy': 0.012175882719503922,
  'sad': 0.9862391779299033,
  'surprise': 0.2140938162742958,
  'text': 'エアリスが死んで悲しいよ'},
 {'angry': 0.8017693086335025,
  'disgust': 0.914084995109833,
  'fear': 0.8251661193502758,
  'happy': 0.2468809876476842,
  'sad': 0.3667284373890585,
  'surprise': 0.6902082109170887,
  'text': 'クラウドって自称ソルジャーとかキモいね'},
 {'angry': 0.9991787157146794,
  'disgust': 0.8073388378968666,
  'fear': 0.40859210600758866,
  'happy': 0.07045632207386164,
  'sad': 0.2319066526171211,
  'surprise': 0.6695058352724972,
  'text': 'バレットって叫んでばっかでうるさくてムカつく'},
 {'angry': 0.5421001856739588,
  'disgust': 0.6279451519549495,
  'fear': 0.8930258596749754,
  'happy': 0.2795665261469537,
  'sad': 0.39058399704319413,
  'surprise': 0.8688049878299227,
  'text': 'セフィロスチョー強くてびっくり'},
 {'angry': 0.6322618613146697,
  'disgust': 0.6578651098916986,
  'fear': 0.8964255553614843,
  'happy': 0.18828917617500254,
  'sad': 0.5363194193208867,
  'surprise': 0.6429357498601701,
  'text': 'タークスとかいう闇組織こわ'}]
```

### 詳細
訓練とテストの詳細を知りたい人はtrain_test.ipynbをご覧ください。
