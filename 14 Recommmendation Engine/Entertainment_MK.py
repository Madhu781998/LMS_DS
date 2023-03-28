import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

enter = pd.read_csv('C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/14 Recommmendation Engine/Hands-on Material/Entertainment.csv')
enter.shape
enter.columns
enter.Category
enter.Reviews

tfidf = TfidfVectorizer(stop_words = "english")#it is a statistical measure that evaluates how relevant a word is to a document in a collection of documents

enter['Category'].isnull().sum()
enter.Reviews.isnull().sum()

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(enter.Category, enter.Reviews)
tfidf_matrix.shape


# For now we will be using cosine similarity matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
enter_index = pd.Series(enter.index, index = enter['Titles']).drop_duplicates()

enter_id = enter_index['Sabrina (1995)']
enter_id

def get_recommendations(Titles, topN) :
    
    # topN = 10
    # Getting the movie index using its title 
    enter_id = enter_index[Titles] 
    
    # Getting the pair wise similarity score for all the movies's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[enter_id]))
    
    # Sorting the cosine_similarity scores based on scores x[1]
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index
    enter_idx  =  [i[0] for i in cosine_scores_N]
    enter_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    enter_similar_show = pd.DataFrame(columns=["Titles", "Score"])
    enter_similar_show["Titles"] = enter.loc[enter_idx, "Titles"]
    enter_similar_show["Score"] = enter_scores
    enter_similar_show.reset_index(inplace = True)  
    print(enter_similar_show)
    
   # return (movie_similar_show) 
get_recommendations('Jumanji (1995)', topN = 10)
get_recommendations('Grumpier Old Men (1995)', topN = 10)    
    
    