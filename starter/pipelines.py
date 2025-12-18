import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import torch
from transformers import pipeline

import spacy
from spacy.util import is_package

import matplotlib.pyplot as plt

class SpacyTextLemmatizer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for text lemmatization using spaCy.

    This transformer converts words with similar meanings (e.g., "good",
    "better") into their base lemma form and removes stop words. It is
    designed to be used inside an sklearn Pipeline.
    """

    def __init__(self, nlp):
        """
        Initialize the SpacyTextLemmatizer.

        Parameters
        ----------
        nlp : spacy.language.Language
            A loaded spaCy NLP pipeline (e.g., en_core_web_sm) used
            for tokenization and lemmatization.
        """
        self.nlp = nlp

    def fit(self, X, y=None):
        """
        Fit method (no-op).

        This method exists to comply with the scikit-learn transformer API.
        No fitting is required since spaCy models are pre-trained.

        Parameters
        ----------
        X : iterable of str
            Input text data.
        y : None, optional
            Ignored.

        Returns
        -------
        self : SpacyTextLemmatizer
            The fitted transformer instance.
        """
        return self

    def transform(self, X):
        """
        Transform text data by lemmatizing and removing stop words.

        Parameters
        ----------
        X : iterable of str
            Raw text input.

        Returns
        -------
        list of str
            Lemmatized text with stop words and punctuation removed.
        """
        # Ensure all inputs are strings
        texts = [str(text) for text in X]

        # Lemmatize text efficiently using spaCy's pipe
        lemmatized_texts = [
            " ".join(
                token.lemma_
                for token in doc
                if not token.is_stop and not token.is_punct
            )
            for doc in self.nlp.pipe(texts)
        ]

        return lemmatized_texts

class SpacyPOSNERTransformer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to extract simple linguistic features
    using spaCy Part-of-Speech (POS) tagging and Named Entity Recognition (NER).

    For each input text, the transformer computes:
      - Number of nouns
      - Number of verbs
      - Number of adjectives
      - Number of named entities
    """

    def __init__(self, nlp, batch_size=64):
        """
        Initialize the transformer.

        Parameters
        ----------
        nlp : spacy.language.Language
            A loaded spaCy NLP pipeline (e.g., en_core_web_sm) used to
            process text and extract POS and NER features.
        batch_size : int, default=64
            Batch size used when processing text with spaCy for
            improved performance.
        """
        self.nlp = nlp
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """
        Fit method (no-op).

        This method exists to comply with the scikit-learn transformer API.
        No fitting is required because spaCy models are pre-trained.

        Parameters
        ----------
        X : iterable of str
            Raw text input.
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : SpacyPOSNERTransformer
            The fitted transformer instance.
        """
        return self

    def transform(self, X):
        """
        Transform raw text into POS and NER count features.

        Parameters
        ----------
        X : iterable of str
            Raw text input.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (n_samples, 4) where each row contains:
            [noun_count, verb_count, adjective_count, named_entity_count]
        """
        features = []

        # Process text efficiently in batches
        for doc in self.nlp.pipe(X, batch_size=self.batch_size):
            noun_count = sum(token.pos_ == "NOUN" for token in doc)
            verb_count = sum(token.pos_ == "VERB" for token in doc)
            adj_count = sum(token.pos_ == "ADJ" for token in doc)
            ner_count = len(doc.ents)

            features.append([
                noun_count,
                verb_count,
                adj_count,
                ner_count
            ])

        return np.array(features)

class TransformerSentimentScorer(BaseEstimator, TransformerMixin):
    """
    scikit-learn transformer that converts raw text into a numeric sentiment score
    using a Hugging Face Transformers sentiment-analysis pipeline.

    Output is a single feature per sample:
      +score for POSITIVE predictions
      -score for NEGATIVE predictions

    Notes
    -----
    - The underlying HF pipeline is cached at the class level so it is initialized
      only once (faster across multiple fit/transform calls).
    - Device selection prefers CUDA, then Apple MPS, else CPU.
    """

    _sent_pipeline = None  # class-level cache (shared across instances)

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        batch_size: int = 64,
        text_column: str = "Review",
    ):
        """
        Parameters
        ----------
        model_name : str, default="distilbert-base-uncased-finetuned-sst-2-english"
            Hugging Face model name for sentiment analysis.
        batch_size : int, default=64
            Batch size passed to the HF pipeline.
        text_column : str, default="Review"
            Column name to read text from when X is a pandas DataFrame.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.text_column = text_column

        # Initialize the shared pipeline once for speed
        if TransformerSentimentScorer._sent_pipeline is None:
            if torch.cuda.is_available():
                device = 0
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1

            TransformerSentimentScorer._sent_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=device,
            )

    def fit(self, X, y=None):
        """
        Fit method (no-op) to comply with the scikit-learn transformer API.

        Parameters
        ----------
        X : array-like
            Input samples (ignored).
        y : array-like, optional
            Target values (ignored).

        Returns
        -------
        self : TransformerSentimentScorer
            The transformer instance.
        """
        return self

    def transform(self, X):
        """
        Transform input text into a numeric sentiment score feature.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray or iterable of str
            Input text. Supported formats:
              - DataFrame: text is taken from `self.text_column`
              - ndarray: text is assumed to be in the first column (X[:, 0])
              - iterable: treated as a sequence of text strings

        Returns
        -------
        numpy.ndarray
            Array of shape (n_samples, 1) containing signed sentiment scores.
        """
        # --- Extract texts from common input formats ---
        if isinstance(X, pd.DataFrame):
            texts = X[self.text_column].astype(str).tolist()
        elif isinstance(X, np.ndarray):
            texts = X[:, 0].astype(str).tolist()
        else:
            texts = pd.Series(X).astype(str).tolist()

        # --- Run sentiment model in batches ---
        results = TransformerSentimentScorer._sent_pipeline(texts, batch_size=self.batch_size)

        # --- Convert labels to signed scores (+ for POSITIVE, - for NEGATIVE) ---
        scores = np.array(
            [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results],
            dtype=float,
        ).reshape(-1, 1)

        return scores


def evaluate_classifier(model, X_tr, y_tr, X_te, y_te, *, pos_label=1, label="Model"):
    """
    Evaluate a classification model on training and test data.

    This function compares model performance on the training and test
    sets to assess generalization. It computes common classification
    metrics (accuracy, precision, recall, and F1-score), prints a
    detailed classification report for the test set, and visualizes
    the test confusion matrix.

    Parameters
    ----------
    model : object
        A fitted classification model that implements a `predict` method.
    X_tr : array-like or pandas.DataFrame
        Feature matrix used for training.
    y_tr : array-like or pandas.Series
        True labels for the training data.
    X_te : array-like or pandas.DataFrame
        Feature matrix used for testing.
    y_te : array-like or pandas.Series
        True labels for the test data.
    pos_label : int or str, default=1
        Label of the positive class used when computing precision,
        recall, and F1-score for binary classification.
    label : str, default="Model"
        Descriptive name of the model used in printed output and plot titles.

    Returns
    -------
    metrics : pandas.DataFrame
        DataFrame containing accuracy, precision, recall, and F1-score
        for both the training and test sets.
    cm : numpy.ndarray
        Confusion matrix computed on the test set.

    Notes
    -----
    - Large differences between training and test metrics may indicate
      overfitting or underfitting.
    - Precision, recall, and F1-score are computed with `zero_division=0`
      to safely handle edge cases with no predicted positive samples.
    """
    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)

    def _row(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
            "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
            "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        }

    metrics = pd.DataFrame(
        [_row(y_tr, yhat_tr), _row(y_te, yhat_te)],
        index=["train", "test"],
    )
    print(f"\n=== {label}: train vs test ===")
    display(metrics)

    print("\nTest classification report:")
    print(classification_report(y_te, yhat_te, digits=4, zero_division=0))

    cm = confusion_matrix(y_te, yhat_te, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm)
    ax.set_title(f"{label} — Confusion Matrix (test)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_yticks([0, 1], ["0", "1"])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    plt.show()
    return metrics, cm
