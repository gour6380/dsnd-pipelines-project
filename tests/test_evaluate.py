import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier

import starter.pipelines as pl


def test_evaluate_classifier_returns_metrics_and_cm(monkeypatch):
    # Patch notebook-only globals:
    monkeypatch.setattr(pl, "display", lambda x: x, raising=False)
    monkeypatch.setattr(pl.plt, "show", lambda *args, **kwargs: None, raising=True)

    X_tr = pd.DataFrame({"x": [0, 1, 2, 3]})
    y_tr = np.array([0, 0, 1, 1])
    X_te = pd.DataFrame({"x": [4, 5]})
    y_te = np.array([1, 1])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_tr, y_tr)

    metrics_df, cm = pl.evaluate_classifier(model, X_tr, y_tr, X_te, y_te, label="dummy")

    # Your implementation uses index=["train","test"] (not a "set" column)
    assert list(metrics_df.index) == ["train", "test"]
    assert set(metrics_df.columns) == {"accuracy", "precision", "recall", "f1"}

    # Confusion matrix is always 2x2 because you call labels=[0,1]
    assert cm.shape == (2, 2)