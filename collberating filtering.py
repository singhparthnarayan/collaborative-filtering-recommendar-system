import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import timeit

ratings = pd.read_csv(r'C:\Users\Asus\Downloads\Python\ratings.csv')
movies = pd.read_csv(r'C:\Users\Asus\Downloads\Python\movies.csv')


plt.figure(figsize=(8, 5))
plt.hist(ratings['rating'], bins=5, ec='black')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Ratings in MovieLens Dataset')
plt.show()


ap = pd.merge(
  movies, ratings,
  how="inner",
  left_on="movieId",
  right_on="movieId"
).sort_values(['rating'], ascending=False)
ap.head()

top_ten_movies = ap.groupby("title").size().sort_values(ascending=False)[:10]
plt.figure(figsize=(12, 5))
plt.barh(y= top_ten_movies.index,
         width= top_ten_movies.values)
plt.title("10 Most Rated Movies in the Data", fontsize=16)
plt.ylabel("Moive", fontsize=14)
plt.xlabel("Count", fontsize=14)
plt.show()

movies_rated = ratings.groupby("userId").size().sort_values(ascending=False)
print(f"Max movies rated by one user: {max(movies_rated)}\nMin movies rated by one user: {min(movies_rated)}")
ratings.userId.value_counts().plot.box(figsize=(15, 8))
plt.title("Number of Movies rated by a Single user", fontsize=16)
plt.show()

ratings_df = ap.pivot(
    index='userId', 
    columns='movieId', 
    values='rating'
)
ratings_df.head()

df = ratings_df.fillna(0).values
df

def train_test_split(ratings):
    
    validation = np.zeros(ratings.shape)
    train = ratings.copy()
    
    for user in np.arange(ratings.shape[0]):
        if len(ratings[user,:].nonzero()[0]) >= 35:
            val_ratings = np.random.choice(
                ratings[user, :].nonzero()[0], 
                size=15,
                replace=False
            )
            train[user, val_ratings] = 0
            validation[user, val_ratings] = ratings[user, val_ratings]
    return train, validation
    

train, val = train_test_split(df)

def rmse(prediction, actual):
    prediction = prediction[actual.nonzero()].flatten() 
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, actual))
    

class Recommender:
    
    def __init__(self, n_epochs=100, n_latent_features=20, lmbda=0.1, learning_rate=0.0001):
        self.n_epochs = n_epochs
        self.n_latent_features = n_latent_features
        self.lmbda = lmbda
        self.learning_rate = learning_rate
  
    def predictions(self, P, Q):
        return np.dot(P.T, Q)
  
    def fit(self, X_train, X_val):
        m, n = X_train.shape
        
        self.P = 3 * np.random.rand(self.n_latent_features, m)
        self.Q = 3 * np.random.rand(self.n_latent_features, n)
        
        self.train_error = []
        self.val_error = [] 
        self.J  = []
        
        users, items = X_train.nonzero()
        
        for epoch in range(self.n_epochs):
            for u, i in zip(users, items): 
                error = X_train[u, i] - self.predictions(self.P[:,u], self.Q[:,i])
                self.P[:, u] += self.learning_rate * (error * self.Q[:, i] - self.lmbda * self.P[:, u])
                self.Q[:, i] += self.learning_rate * (error * self.P[:, u] - self.lmbda * self.Q[:, i])
            cost  = np.dot(error,error.T) + self.lmbda * (np.dot(self.P[:, u],self.P[:,u].T)  + np.dot(self.Q[:, i],self.Q[:,i].T))
            self.J.append(cost)
            train_rmse = rmse(self.predictions(self.P, self.Q), X_train)
            val_rmse = rmse(self.predictions(self.P, self.Q), X_val)
            self.train_error.append(train_rmse)
            self.val_error.append(val_rmse)
        return self

        
start_time = timeit.default_timer()
recommender = Recommender().fit(train, val)
print("Run took %.2f seconds" % (timeit.default_timer() - start_time))


plt.plot(range(recommender.n_epochs), recommender.train_error, marker='o', label='Training Data');
plt.plot(range(recommender.n_epochs), recommender.val_error, marker='v', label='Validation Data');
plt.xlabel('Number of Epochs');
plt.ylabel('RMSE');
plt.legend()
plt.show()


plt.figure(figsize=(7, 5))
plt.plot(recommender.J, marker='o');
plt.xlabel('Number of Epochs');
plt.ylabel('Cost')
plt.show()


def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    user_row_number = userID - 1 
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    
    
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )

   
    
    


    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations

movies['movieId'] = movies['movieId'].apply(pd.to_numeric)

all_user_predicted_ratings = np.dot(recommender.P.T,recommender.Q)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = ratings_df.columns)

already_rated, predictions = recommend_movies(preds_df,1, movies, ratings, 10)

already_rated.head(10)

predictions