# Movie Recommendation System

A content-based movie recommender built with Python and Streamlit. Pick a movie you like and it shows you five similar ones, along with their posters.

It works on the TMDB 5000 dataset (about 4,800 movies) and compares movies using their genres, plot overview, keywords, cast and director.


## How it works

Each movie gets a "tags" field built from its metadata:

- genres
- the plot overview
- keywords
- the top 3 cast members
- the director

Names are joined into single tokens (`Sam Worthington` becomes `samworthington`) so that the model treats a person as one entity instead of matching on common first names. Everything is lowercased and combined into one string per movie.

These tags are converted into vectors with `CountVectorizer` (English stop words removed), and cosine similarity is calculated between every pair of movies. The resulting similarity matrix is computed once and saved, so the app only has to look up a row and sort it when you ask for recommendations:

```python
def recommend(movie, k=5):
    idx = movies[movies["title"] == movie].index[0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    return [movies.iloc[i].title for i, _ in scores[1:k + 1]]
```

The Streamlit app loads the saved data with caching, so it stays fast after the first load.

## Tech stack

- Python
- Pandas and NumPy for data processing
- Scikit-learn for `CountVectorizer` and `cosine_similarity`
- Streamlit for the web app
- TMDB API for movie posters

## Project structure

```
movie-recommender-system/
├── app.py                    # Streamlit app
├── movie_recommender.ipynb   # preprocessing and building the model
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── artifacts/
│   ├── movies.pkl
│   └── similarity.pkl        # generated locally, not in the repo
├── requirements.txt
└── README.md
```

## Running it locally

Clone the repo and install the dependencies:

```bash
git clone https://github.com/<your-username>/movie-recommender-system.git
cd movie-recommender-system
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle and put the two CSV files in `data/`.

Run the notebook to generate `movies.pkl` and `similarity.pkl`. The similarity matrix is around 180 MB, which is over GitHub's file size limit, so it isn't included in the repo and has to be generated on your machine.

The posters come from TMDB, so you'll need a free API key from [themoviedb.org](https://www.themoviedb.org/settings/api). Put it in `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_api_key_here"
```

Then start the app:

```bash
streamlit run app.py
```

It should open at `http://localhost:8501`.

## Things I'd like to add

- Try TF-IDF or sentence embeddings instead of plain word counts
- Stemming, so words like "love" and "loving" count as the same
- Mix in ratings and popularity, since right now an obscure movie can rank above a great one if the tags match
- Store only the top neighbours for each movie to make the similarity file much smaller
- Deploy it on Streamlit Community Cloud

## Credits

Data from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) on Kaggle. Posters from [TMDB](https://www.themoviedb.org/). This project uses the TMDB API but is not endorsed or certified by TMDB.

## Author

Chinthapally Purna Nanda Reddy, B.Tech Engineering Science, IIT Hyderabad
[GitHub](https://github.com/<your-username>) | [LinkedIn](https://www.linkedin.com/in/<your-profile>)
