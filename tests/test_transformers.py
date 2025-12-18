import numpy as np
import pandas as pd

import starter.pipelines as pl


class _FakeToken:
    def __init__(self, text: str):
        self.text = text
        self.lemma_ = text.lower()

        self.is_stop = self.lemma_ in {"the", "a", "an", "and", "is", "to"}
        self.is_punct = False  # required by your SpacyTextLemmatizer
        self.is_alpha = text.isalpha()

        # deterministic POS for tests
        if self.lemma_ in {"dress", "shirt", "quality"}:
            self.pos_ = "NOUN"
        elif self.lemma_ in {"love", "like", "buy"}:
            self.pos_ = "VERB"
        elif self.lemma_ in {"great", "good", "bad"}:
            self.pos_ = "ADJ"
        else:
            self.pos_ = "X"


class _FakeDoc:
    def __init__(self, text: str):
        self._tokens = [_FakeToken(t) for t in text.split()]
        # deterministic NER: capitalized tokens are entities
        self.ents = [t for t in text.split() if t[:1].isupper()]

    def __iter__(self):
        return iter(self._tokens)


class _FakeNLP:
    def pipe(self, texts, batch_size=64):
        for t in texts:
            yield _FakeDoc(str(t))


def test_spacy_text_lemmatizer_outputs_strings_and_removes_stopwords_and_punct():
    nlp = _FakeNLP()
    tr = pl.SpacyTextLemmatizer(nlp=nlp)

    X = ["I love the dress and the quality", "This is bad"]
    out = tr.fit_transform(X)

    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(x, str) for x in out)

    # stopwords removed (the/and/is)
    assert "the" not in out[0].split()
    assert "and" not in out[0].split()
    assert "is" not in out[1].split()


def test_spacy_pos_ner_transformer_shape_and_basic_counts():
    nlp = _FakeNLP()
    tr = pl.SpacyPOSNERTransformer(nlp=nlp, batch_size=2)

    X = ["I love Dress great", "bad quality"]
    feats = tr.fit_transform(X)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (2, 4)  # noun, verb, adj, ents

    # "Dress" is NOUN + entity (capitalized)
    noun_count, verb_count, adj_count, ent_count = feats[0]
    assert noun_count >= 1
    assert verb_count >= 1
    assert adj_count >= 1
    assert ent_count >= 1


def test_transformer_sentiment_scorer_signed_scores(monkeypatch):
    # Patch HF pipeline factory BEFORE instantiating the transformer,
    # because your code initializes the pipeline in __init__.
    def _fake_pipeline(task, model=None, device=None):
        def _run(texts, batch_size=64):
            out = []
            for t in texts:
                if "love" in str(t).lower() or "great" in str(t).lower() or "good" in str(t).lower():
                    out.append({"label": "POSITIVE", "score": 0.9})
                else:
                    out.append({"label": "NEGATIVE", "score": 0.8})
            return out
        return _run

    monkeypatch.setattr(pl, "pipeline", _fake_pipeline, raising=True)

    # Reset class cache to ensure our patched pipeline gets used
    pl.TransformerSentimentScorer._sent_pipeline = None

    tr = pl.TransformerSentimentScorer(model_name="dummy", batch_size=2, text_column="Review")

    X = pd.DataFrame({"Review": ["I love it", "terrible"]})
    scores = tr.fit_transform(X)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2, 1)
    assert scores[0, 0] > 0
    assert scores[1, 0] < 0