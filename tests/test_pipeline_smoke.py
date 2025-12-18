import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

import starter.pipelines as pl


class _FakeToken:
    def __init__(self, text: str):
        self.text = text
        self.lemma_ = text.lower()

        self.is_stop = self.lemma_ in {"the", "a", "an", "and", "is", "to"}
        self.is_punct = False
        self.is_alpha = text.isalpha()

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
        self.ents = [t for t in text.split() if t[:1].isupper()]

    def __iter__(self):
        return iter(self._tokens)


class _FakeNLP:
    def pipe(self, texts, batch_size=64):
        for t in texts:
            yield _FakeDoc(str(t))


def _tiny_df():
    return pd.DataFrame(
        {
            "Age": [25, 40, 31, 22, 55, 29],
            "Positive Feedback Count": [0, 3, 1, 10, 2, 0],
            "Clothing ID": [1001, 1002, 1001, 1003, 1002, 1004],
            "Division Name": ["General", "General", "Petite", "General", "Petite", "General"],
            "Department Name": ["Dresses", "Tops", "Dresses", "Dresses", "Tops", "Tops"],
            "Class Name": ["Casual", "Work", "Casual", "Party", "Work", "Casual"],
            "Review": [
                "I love this Dress great quality",
                "bad quality",
                "love it",
                "great Dress",
                "bad",
                "good shirt love",
            ],
        }
    )


def test_full_pipeline_fit_predict(monkeypatch):
    # Patch plt.show (your evaluate function uses it; safe to patch anyway)
    monkeypatch.setattr(pl.plt, "show", lambda *args, **kwargs: None, raising=True)

    # Patch HF pipeline factory BEFORE transformer creation
    def _fake_pipeline(task, model=None, device=None):
        def _run(texts, batch_size=64):
            out = []
            for t in texts:
                t = str(t).lower()
                if "love" in t or "great" in t or "good" in t:
                    out.append({"label": "POSITIVE", "score": 0.9})
                else:
                    out.append({"label": "NEGATIVE", "score": 0.8})
            return out
        return _run

    monkeypatch.setattr(pl, "pipeline", _fake_pipeline, raising=True)
    pl.TransformerSentimentScorer._sent_pipeline = None  # ensure patched pipeline used

    df = _tiny_df()
    y = np.array([1, 0, 1, 1, 0, 1])

    num_features = ["Age", "Positive Feedback Count"]
    cat_features = ["Clothing ID", "Division Name", "Department Name", "Class Name"]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    nlp = _FakeNLP()

    # Your lemmatizer expects strings; ColumnTransformer will pass a Series for a single column.
    # So we ravel -> then lemmatize -> then TFIDF.
    tfidf_pipeline = Pipeline(
        steps=[
            ("reshape", FunctionTransformer(lambda x: pd.Series(x).astype(str).tolist(), validate=False)),
            ("lemmatize", FunctionTransformer(lambda x: pl.SpacyTextLemmatizer(nlp=nlp).fit_transform(x), validate=False)),
            ("tfidf", TfidfVectorizer(min_df=1)),
        ]
    )

    text_features = FeatureUnion(
        transformer_list=[
            ("tfidf", tfidf_pipeline),
            ("sent", pl.TransformerSentimentScorer(model_name="dummy", batch_size=8, text_column="Review")),
            ("nlp", pl.SpacyPOSNERTransformer(nlp=nlp, batch_size=8)),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features),
            ("text", text_features, "Review"),
        ]
    )

    model = make_pipeline(
        preprocessing,
        RandomForestClassifier(n_estimators=10, max_features=10, random_state=42),
    )

    model.fit(df, y)
    preds = model.predict(df)

    assert preds.shape == (len(df),)
    assert set(np.unique(preds)).issubset({0, 1})