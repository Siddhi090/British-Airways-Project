# British-Airways-Project
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
base_url = "https://www.airlinequality.com/airline-reviews/british-airways//"
pages = 37
page_size = 100

reviews = []

for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())

    print(f"   ---> {len(reviews)} total reviews")
    df = pd.DataFrame()
df["reviews"] = reviews
df
df.to_csv("BA_reviews.csv")
reviews = pd.read_csv("BA_reviews.csv")
reviews = reviews.pop('reviews')
reviews
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
    reviews = reviews.str.replace('Trip Verified', '')
reviews = reviews.str.replace('✅ ', '')
reviews = reviews.str.replace('|', '')
reviews = reviews.str.replace(r'\b(\w{1,3})\b', '')
reviews = reviews.apply(remove_punctuations)
reviews
reviews.shape
freq_words = pd.Series(''.join(reviews).lower().split()).value_counts()[:50]
freq_words
plt.figure(figsize=(10,10))
freq_words.plot.barh(x=freq_words[0], y=freq_words[1])
plt.show()
categorise = ['negative', 'positive']
num_cat = len(categorise)
num_cat
#TF -IDF Feature Generation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer

#Initialize regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')

#Create TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tf_idf_vect = TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer=tokenizer.tokenize)

#Fit and Transform text data
reviews_counts = tf_idf_vect.fit_transform(reviews)

#Check shape of count vector
reviews_counts.shape
# Import KMeans Model
from sklearn.cluster import KMeans

#Create KMeans object and fit it to the trainig data
kmeans = KMeans(n_clusters=num_cat).fit(reviews_counts)

#Get the labels using KMeans
pred_labels = kmeans.labels_
pred_labels
#Print the labels
cluster_centres = kmeans.cluster_centers_
cluster_centres
unique, counts = np.unique(pred_labels, return_counts=True)
dict(zip(unique, counts))
from sklearn import metrics

#compute DBI score
dbi = metrics.davies_bouldin_score(reviews_counts.toarray(), pred_labels)

#compute Silhoutte score
ss = metrics.silhouette_score(reviews_counts.toarray(), pred_labels, metric='euclidean')

#print the dbi and silhoutte scores
print("DBI Score:", dbi, "\nSilhoutte Score:", ss)
df_reviews = pd.DataFrame({'review': reviews, 'label': pred_labels})
df_reviews
sns.displot(df_reviews['label'], kde=True)
positive_review = df_reviews[df_reviews['label'] == 1]
positive_review
negative_review = df_reviews[df_reviews['label'] == 0]
negative_review



