#import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

#read the data in from local files (movies and tags)
movies = pd.read_csv("./ml-latest/movies.csv")
tags = pd.read_csv("./ml-latest/tags.csv")

#get what information we need from the csv files (mainly wanted to cut out timestamp from this)
movies = movies[['movieId','title', 'genres']]
tags = tags[['movieId','tag']]

#a function to turn movies with no genres listed to NaN
def clean_genres(s):
    if s == "(no genres listed)":
        return np.nan
    return s

#joins the tags based on movieId so that they are all in one list
tags = tags.fillna('').groupby('movieId')['tag'].apply(','.join).reset_index()
#applies the clean_genres function made above
movies['genres'] = movies['genres'].apply(clean_genres)

#ensures that the movieIds and Genres are not null
tags = tags[tags['movieId'].notnull()]
movies = movies[movies['movieId'].notnull()]
movies = movies[movies['genres'].notnull()]

#converts the movieIds into integers
movies['movieId'] = movies['movieId'].astype('int')
tags['movieId'] = tags['movieId'].astype('int')

#this takes how the genres are split in the data and puts them into an array to access the data easier
movies['genres'] = movies['genres'].apply(lambda x: "".join(x.replace(" ", "").lower()).split("|"))
#merges the tags based on movieId
movies = movies.merge(tags, on="movieId")

#makes all values into 1 word rather than having a space, and then puts all the values into an array
movies['tag'] = movies['tag'].apply(lambda x: x.replace(" ", "").lower().split(","))

#creates a metadata column that contains all the tags and genres for the movie
movies['metadata'] = movies.apply(lambda x: " ".join(x["genres"]) + " ".join(x["tag"]), axis=1)

#creates a vector that counts the words and maps them based on location
vec = CountVectorizer(stop_words="english")
#puts the words into a matrix
matrix = vec.fit_transform(movies['metadata'])
#gets the cosine similarity
cos_sim = cosine_similarity(matrix, matrix)

#creates a mapping of each title to an index
mapping = pd.Series(movies.index, index=movies['title'])

def select_movie(input):
    #grabs the word closes to the input (makes search more flexible)
    mov = movies[movies['title'].str.contains(input)]
    #grabs just the title of the movie
    mov = mov['title'].values[0]
    return mov


def movie_recommender(input):
    mov_input = select_movie(input)
    print("Recommend a movie similar to: ", mov_input)
    #finds the movie based on input
    movie = mapping[mov_input]
    #gets the similarity score of the movie
    sim_score = list(enumerate(cos_sim[movie]))
    #grabs the most similar movies by reversing the list
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    #grabs the first 10 movies
    sim_score = sim_score[1:11]
    #gets the movie ids of the top 10 movies+
    indices = [i[0] for i in sim_score]
    #returns the movies with their IDs and title in a list
    return movies['title'].iloc[indices]

print(movie_recommender("Avengers"))

