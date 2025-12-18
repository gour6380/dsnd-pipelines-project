# DSND Pipelines Project — Text Classification with NLP Pipelines

This project demonstrates how to build **production-ready, testable machine-learning pipelines** using **scikit-learn**, **spaCy**, and **Hugging Face Transformers**.  
It is designed for the **Udacity Data Scientist Nanodegree (DSND)** and focuses on:

- Custom NLP transformers (lemmatization, POS/NER features, sentiment scoring)
- End-to-end sklearn pipelines combining numeric, categorical, and text features
- Proper **train vs test evaluation** with accuracy, precision, recall, F1, and confusion matrix
- Fully automated **pytest-based unit tests** that validate pipeline correctness

---

## Getting Started

These instructions will help you set up the project locally and run the model pipelines and tests.

### Dependencies

The project requires Python **3.9+** and the following libraries:

```
pandas
numpy
scikit-learn
spacy
torch
transformers
matplotlib
pytest
```

Additionally, the spaCy English model is required:

```
en_core_web_sm
```

---

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:gour6380/dsnd-pipelines-project.git
   cd dsnd-pipelines-project
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```

4. **Download spaCy language model**
   ```bash
   python3 -m spacy download en_core_web_sm
   ```

---

## Testing

Automated unit tests are written using **pytest** to ensure that all transformers, pipelines, and evaluation logic work correctly.

### Running Tests

From the project root directory, run:

```bash
python3 -m pytest -q
```

---

### Break Down Tests

The test suite is located in the `tests/` directory and covers the following components:

#### 1. Transformer Tests (`test_transformers.py`)
- Verifies that:
  - `SpacyTextLemmatizer` returns cleaned, lemmatized text
  - Stop words and punctuation are removed correctly
  - `SpacyPOSNERTransformer` outputs the correct feature shape `(n_samples, 4)`
  - `TransformerSentimentScorer` produces signed sentiment scores
- Uses **fake spaCy and Hugging Face pipelines** to avoid heavy downloads

#### 2. Pipeline Smoke Test (`test_pipeline_smoke.py`)
- Confirms that:
  - A full sklearn pipeline (numeric + categorical + text features) can `fit()` and `predict()`
  - Custom transformers integrate correctly inside `ColumnTransformer` and `FeatureUnion`
  - Predictions are valid binary outputs

#### 3. Evaluation Tests (`test_evaluate.py`)
- Ensures that:
  - `evaluate_classifier` returns a metrics DataFrame indexed by `train` and `test`
  - Accuracy, precision, recall, and F1 are computed correctly
  - A valid 2×2 confusion matrix is produced
- Notebook-specific functions like `display()` and `plt.show()` are safely monkey-patched

---

## Project Instructions

Students are expected to:

1. Build reusable **custom sklearn transformers** for NLP tasks
2. Combine numeric, categorical, and text features into a single pipeline
3. Train and evaluate a classification model using proper metrics
4. Demonstrate understanding of **train vs test performance**
5. Write automated tests that validate:
   - Transformer behavior
   - Pipeline integration
   - Evaluation correctness
6. Ensure the project runs both in **Jupyter notebooks** and **pure Python environments**

---

## Built With

- [scikit-learn](https://scikit-learn.org/) — Machine learning pipelines and models  
- [spaCy](https://spacy.io/) — NLP processing, lemmatization, POS, and NER  
- [Hugging Face Transformers](https://huggingface.co/transformers/) — Pretrained sentiment analysis models  
- [PyTorch](https://pytorch.org/) — Backend for transformer models  
- [pytest](https://docs.pytest.org/) — Automated testing framework  
- [matplotlib](https://matplotlib.org/) — Visualization (confusion matrix)

---

## License

This project is licensed under the terms specified in the  
[LICENSE](LICENSE.txt) file.
