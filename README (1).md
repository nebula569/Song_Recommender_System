# Song Recommender (Content-Based)

A content-based song recommendation system built with **pandas** and **scikit-learn**. Given a song title, it returns the 10 most similar songs by comparing text "tags" built from each song's artist, album, genre, and explicit-content flag using **cosine similarity**.

## How it works

1. **Load the dataset** (`spotify_songs_dataset.csv`, 50,000 songs).
2. **Select features:** `song_id`, `song_title`, `artist`, `album`, `genre`, `explicit_content`.
3. **Clean text:**
   - Remove spaces from artist names (`Sydney Clark` → `SydneyClark`) so each artist is a single token.
   - Strip periods from album and song titles.
4. **Build a `tags` column** by joining artist, album, genre, and explicit flag, then lowercasing it.
   ```
   sydneyclark what electronic yes
   ```
5. **Vectorize** the tags with `CountVectorizer` (top 5,000 features, English stop words removed).
6. **Compute similarity** between all songs using cosine similarity.
7. **Recommend:** for a chosen song, sort all other songs by similarity and return the top 10.

If several songs share the same title, the function asks for the artist name to pick the right one.

## Usage

```python
song('Company')
```

If the title is ambiguous, you'll be prompted:

```
Enter the name of the artist: LeahColeman
```

Output (top 10 similar songs):

```
Forward understand all
Nature institution allow
Option apply speak during
...
```

> **Note:** artist names are entered **without spaces** and with matching capitalization (e.g. `LeahColeman`), because spaces are removed during cleaning.

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/song-recommender.git
cd song-recommender
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the dataset

Place `spotify_songs_dataset.csv` in the project root. It is not included in this repo (see [Dataset](#dataset)).

### 4. Run the notebook

```bash
jupyter notebook song_recommender.ipynb
```

## Dataset

The notebook expects a CSV named `spotify_songs_dataset.csv` with these columns:

`song_id, song_title, artist, album, genre, release_date, duration, popularity, stream, language, explicit_content, label, composer, producer, collaboration`

Only `song_id`, `song_title`, `artist`, `album`, `genre`, and `explicit_content` are used. The data is **not included** in this repository; add your own copy and make sure you have the right to use it.

## Known limitations

- **Memory usage:** the full similarity matrix for 50,000 songs is 50,000 × 50,000 values (about 20 GB as `float64`). This will crash on most laptops. Work on a subset (e.g. `songs.sample(5000)`), use `float32`, or compute similarity for one song at a time (see below).
- **Exact title match only:** `song('company')` will not match `Company`.
- **Artist and album dominate:** because artist and album are unique-ish tokens, songs by the same artist or on the same album score highest, while genre and explicit flag add smaller signals.
- **No evaluation metric:** there is no ground truth for "similar songs" here, so quality is judged by inspection only.

## Possible improvements

- Compute similarity on demand instead of building the full matrix:
  ```python
  from sklearn.metrics.pairwise import cosine_similarity
  scores = cosine_similarity(vectors[idx].reshape(1, -1), vectors).ravel()
  top10 = scores.argsort()[::-1][1:11]
  ```
- Use a sparse matrix (drop `.toarray()`) to cut memory sharply
- Add numeric features such as `popularity`, `duration`, and release year, scaled
- Use `TfidfVectorizer` to down-weight very common tokens
- Case-insensitive and fuzzy title search
- Show artist and genre in the output, not just the title
- Wrap it in a simple web app (Streamlit or Flask)

## Project structure

```
song-recommender/
├── song_recommender.ipynb      # main notebook
├── spotify_songs_dataset.csv   # dataset (not included)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Tech stack

Python · pandas · NumPy · scikit-learn · Jupyter

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
