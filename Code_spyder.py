#importing libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer         # Convert a collection of text documents to a matrix of token counts.
from sklearn.metrics.pairwise import cosine_similarity
import difflib 


#defining fuction to get movie from index
def get_movie_from_index(index):
    return df[df.index==index].movie_title.values[0]
    #return df.loc[index, 'movie_title']


#defining fuction to get index from movie
def get_index_from_movie(movie):
    movie_list = df['movie_title'].tolist()
    close_matches = difflib.get_close_matches(movie, movie_list, n=1)
    closest_movie = close_matches[0]
    return df[df.movie_title == closest_movie].index.values[0]


# Creating Dataframe of the dataset
df = pd.read_csv("movie_metadata.csv", encoding = 'utf-8')

df.head()

df.columns


#handling \xa0
df.movie_title[0]

df.replace(u'\xa0',u'', regex =True , inplace=True)
df.movie_title[0]


# Taking features to be used.
df['plot_keywords'].head()

features = ['director_name', 'genres', 'language', 'plot_keywords' ]


#handling nan
df[features].isnull().values.any()
df[features].isnull().sum()

df[features] = df[features].fillna('')
df[features].isnull().values.any()


#Combining features to form a string
df['combined_features'] = df['director_name'] +" "+ df['genres'] +" "+ df['language'] +' '+ df['plot_keywords']
df['combined_features'].head()


# Extracting features of data using CountVectoriser, creating count matrix.
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

cv.get_feature_names()

count_matrix.toarray()

# Computing cosine similarity based on count matrix (extracted features)
cosine_sim = cosine_similarity(count_matrix)

cosine_sim


# Taking movie from the user and finding its index
user_movie = input('Enter a movie name to know the similar movies:\n')
movie_index = get_index_from_movie(user_movie)


#finding similar movies from cosine_sim and enumerate it to form it a tuple of form (index, cosine_value) and at last listing all such tuples. 
similar_movies = list(enumerate(cosine_sim[movie_index]))

#sorting similar_movies by cosine_values i.e. tuple[1] in descending order.
similar_movies_sorted = sorted(similar_movies, key=lambda x:x[1], reverse=True)


#printing recomended movies
print("Top 50 Recomended movies for you are")
i = 1
for sim_index in similar_movies_sorted:
    print(i, ") ", get_movie_from_index(sim_index[0]))
    i+=1
    if(i == 51):
        break



