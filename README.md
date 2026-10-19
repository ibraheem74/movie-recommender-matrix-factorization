# Movie Recommendation System (Matrix Factorization)

A collaborative filtering recommender built from scratch with NumPy. It learns hidden "taste" factors for users and movies from past ratings, then predicts what a user would rate movies they haven't seen. A Streamlit app shows the top 10 recommendations for any user.

## Dataset

[MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)

- 943 users, 1,682 movies, 100,000 ratings (scale 1-5)
- Only about 6% of the user-movie matrix is filled, so most ratings must be predicted
- Train/test split: the dataset's own `ua.base` (90,570 ratings) and `ua.test` (9,430 ratings)

## Approach

Matrix factorization with bias terms, trained by stochastic gradient descent (SGD).

Prediction for user *u* and movie *i*:

```
rating = global average + user bias + movie bias + (user factors . movie factors)
```

- **Latent factors:** each user and each movie gets a vector of 20 numbers, learned from the data
- **Bias terms:** capture generous vs harsh raters and universally loved vs disliked movies
- **Loss:** squared error plus an L2 regularization penalty to reduce overfitting
- **Training:** SGD over every training rating, repeated for several epochs

| Setting | Value |
|---|---|
| Latent factors | 20 |
| Learning rate | 0.01 |
| Regularization | 0.02 |
| Epochs | 3 |
| Initial factor values | small random numbers (0 to 0.1) |

## Results

| Metric | Value |
|---|---|
| Training RMSE | about 0.94 |
| Test RMSE | 0.972 |

The small gap between training and test error suggests the model is not badly overfitting. Always predicting the global average rating gives a noticeably higher error (about 1.1 on this dataset), so the model has learned real patterns.

## App Screenshot

![Streamlit app showing top 10 recommendations for user 20](screenshot.png)

## Observations

Top-10 lists for different users overlap strongly. User 1 and user 2 share 5 of 10 movies, and user 1 and user 43 share 7 of 10. The recommendations are mostly well-known, highly rated classics (Casablanca, Rear Window, Schindler's List, and similar). This points to movie bias dominating the predictions, with personalization from the latent factors having a smaller effect.

## Limitations

- **Cold start:** a new user or a new movie with no ratings has no learned vectors, so the model cannot score it
- **Popularity bias:** as noted above, broadly loved movies rank high for almost everyone
- **Short training:** only 3 epochs, and the hyperparameters were not tuned
- **Ratings only:** genres, user demographics, and timestamps are not used

## Future Work

- Train for more epochs and tune the learning rate, regularization, and number of factors
- Use a validation split for tuning, and keep the test set for the final score
- Add a popularity fallback for cold-start users
- Vectorize training for speed (the current loop uses `iterrows`)

## Files

| File | Purpose |
|---|---|
| `modul.py` | Loads data, trains the model, evaluates on the test set, saves the model files |
| `app.py` | Streamlit app that loads the saved model and shows top-10 recommendations |
| `*.pkl` | Saved model: user and movie factors, user and movie biases, global average |
| `requirements.txt` | Python dependencies |

## How to Run

1. Download MovieLens 100k and extract it into a folder named `ml-100k` in the project directory
2. Install the dependencies:

```
pip install -r requirements.txt
```

3. Train the model (takes a few minutes, creates the `.pkl` files):

```
python modul.py
```

4. Start the app:

```
streamlit run app.py
```

Choose a user ID (1-943), click **Recommend**, and the top 10 unseen movies appear with their predicted ratings.

## Tech Stack

Python, NumPy, pandas, joblib, Streamlit
