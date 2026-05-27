# California Yoga Studio Reviews
### A Google Reviews NLP Analysis for Yoga Studio Owners

*2026-05*

---

## Project overview

My friend recently opened her own yoga business. She is in the first steps of starting her own yoga brand. She currently rents a studio with having plans to buy one of her own in the near future. I wanted to give her something useful for her business progression — actual data. Consequently, this project analyses **13,938 Google Reviews** across **423 California yoga studios** to understand what drives customer satisfaction, what distinguishes thriving studios from those that permanently closed, and what operational failures generate the most damaging negative reviews.

California was selected as the focus state because the Western US leads in yoga studio concentration nationally and California is widely considered a trendsetter for US wellness culture. The dataset is also the largest single-state Google Reviews archive available, ensuring sufficient volume after filtering.

---

## Dataset

- **Source:** [UCSD Google Local Dataset](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/) — 70.5M California reviews
- **Filtered to:** 423 yoga studios (Yoga studio, Yoga instructor, Bikram yoga studio categories)
- **Reviews:** 13,938 total — 10,547 with text for NLP, 3,391 rating-only
- **Date range:** 2007–2021 (snapshot taken August 2021)
- **Studio status:** 360 active, 63 permanently closed

---

## Repository structure

```
├── yoga_metadata_prep.ipynb       # Download and filter studio metadata
├── yoga_review_data_prep.ipynb    # Download and filter review data
├── yoga_EDA.ipynb                 # Exploratory data analysis
├── yoga_VADER.ipynb               # Sentiment analysis
├── yoga_LDA.ipynb                 # Topic modelling
├── yoga_classifier.ipynb          # Classifier design and future work
└── requirements.txt
```

Run notebooks in order — each saves intermediate files to Google Drive
that are loaded by the next notebook.

---

## Methodology

### Exploratory data analysis
Rating distributions, review volume patterns, geographic concentration,
temporal trends, opening hours analysis, and text length analysis.
Statistical comparisons use both t-tests and Mann-Whitney U tests
(non-parametric) to account for the skewed rating distribution.

### Sentiment analysis — VADER
VADER (Valence Aware Dictionary and sEntiment Reasoner) was applied to
all 10,547 text reviews. VADER was chosen for its design for user-generated
and social media text, handling informal language, punctuation emphasis,
and emoji. Sentiment was analysed at both review level and studio level,
with studio-level aggregation producing stronger signal (Spearman ρ=0.513
vs ρ=0.212 at review level).

### Topic modelling — LDA
Three LDA models were trained on different review subsets to compare
themes across sentiment groups. Hyperparameters were tuned using c_v
coherence across a grid of topic counts (k=4–10), alpha, and beta values.
Each optimal configuration was validated on a 75% corpus subset to confirm
stability — a coherence difference below 0.03 was required.

| Model | Corpus | k | alpha | beta | c_v coherence |
|-------|--------|---|-------|------|---------------|
| All reviews | 10,547 reviews | 8 | 0.5 | 0.6 | 0.517 |
| Positive (4-5★) | 9,663 reviews | 6 | 0.1 | 0.6 | 0.528 |
| Negative (1-2★) | 295 reviews | 4 | 0.1 | 0.6 | 0.317 |

Text preprocessing included standard NLTK stop words plus an extended
yoga-specific stop word list built by inspecting corpus frequency.
Bigrams and trigrams were detected and inspected before inclusion.
TF-IDF filtering removed low-value high-frequency terms.

### Classifier
A logistic regression classifier was designed but not trained to completion
due to severe class imbalance (23.4:1) and insufficient negative review
signal. The notebook documents the diagnostic process and a plan
for building a reliable classifier with an expanded multi-state dataset.
See `yoga_classifier.ipynb` for full details and future work.

---

## Tech stack

```
Python · pandas · numpy · scipy · matplotlib · seaborn
nltk · vaderSentiment · gensim · pyLDAvis · wordcloud · scikit-learn
```

---

## How to run

```bash
git clone https://github.com/gummy-brain/california-yoga-studio-reviews
cd california-yoga-studio-reviews
pip install -r requirements.txt
```

Open notebooks in Google Colab and run in order (01 → 06). Each notebook
mounts Google Drive and saves intermediate files for the next step.
The raw data files (metadata ~1–2 GB, reviews ~15–20 GB) are downloaded
directly from the UCSD dataset in Notebooks 1 and 2.

---
