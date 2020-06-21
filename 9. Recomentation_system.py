### RECOMENTATION SYSTEM ###
## COLLABORATIVE FILTERING ##
# Business problem :Recommend a best book based on the author, publisher and ratings.
import os
os.chdir('F:/pythonwd')

import pandas as pd
data= pd.read_excel("F:/alldatasets/books_re.xlsx")
data.shape
data.columns

# checking there have any NULL values exist in the data
data.isnull().sum()

#term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
from sklearn.feature_extraction.text import TfidfVectorizer 

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words="english")    #taking stop words from tfid vectorizer 

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(data["Book_author"],data["Publisher"])   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape 

# with the above matrix we need to find the 
# similarity score
# There are several metrics for this
# such as the euclidean, the Pearson and 
# the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity 
# between 2 books
# Cosine similarity - metric is independent of 
# magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a mapping of anime name to index number 
data_index = pd.Series(data.index,index=data['Book_title']).drop_duplicates()


data_index["Jane Doe"]

def get_data_recommendations(Book_title,topN):  
    #topN = 10
    # Getting the books index using its title 
    data_id = data_index[Book_title]
    
    # Getting the pair wise similarity score for all the books with that book
    cosine_scores = list(enumerate(cosine_sim_matrix[data_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar books 
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the books index 
    data_idx  =  [i[0] for i in cosine_scores_10]
    data_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar books and scores
    data_similar_show = pd.DataFrame(columns=["Book_title","Score"])
    data_similar_show["Book_title"] = data.loc[data_idx,"Book_title"]
    data_similar_show["Score"] = data_scores
    data_similar_show.reset_index(inplace=True)  
    data_similar_show.drop(["index"],axis=1,inplace=True)
    print (data_similar_show)
    #return (books_similar_show)

# Enter your book name and number of books to be recommended 
get_data_recommendations("Where You'll Find Me: And Other Stories",topN=5)

##################################################################################