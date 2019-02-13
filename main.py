import numpy as np
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath, get_tmpfile

from gensim.models import TfidfModel
import pandas as pd
from os import path
# from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

filename = path.join("data", "train_set.csv")
df = pd.read_csv(filename, sep='\t')

print("There are {} observations and {} features in this dataset. \n".format(df.shape[0], df.shape[1]))


def wordclouds():
    stopwords = set(STOPWORDS)
    stopwords.update(["said", "will", "say", "one", "now", "says", "time", "new", "first"])
    categories = df.Category.unique()
    for category in categories:
        cat = df.loc[df['Category'] == category]

        text = " ".join(article for article in cat.Content)

        # Create and generate a word cloud image:
        wordcloud = WordCloud(stopwords=stopwords).generate(text)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(category)
        plt.show()


def duplicates(similarity):
    # Create the Dictionary and Corpus
    mydict = corpora.Dictionary([simple_preprocess(line) for line in df.Content])
    corpus = [mydict.doc2bow(simple_preprocess(line)) for line in df.Content]
    print(mydict)

    # Create the TF-IDF model
    tfidf = TfidfModel(corpus)

    index_temp = get_tmpfile("index")
    index = gensim.similarities.Similarity(index_temp, tfidf[corpus], num_features=len(mydict))

    dups = pd.DataFrame(columns=['Document_ID1', 'Document_ID2', 'Similarity'])

    for idx1, sims in enumerate(index):
        for idx2, val in enumerate(sims):
            if val > similarity and idx1 < idx2:
                # print(round(idx1 / 122.66), '%', df.iloc[idx1, 2], "||", df.iloc[idx2, 2], val)
                print(round(idx1 / 122.66), '%')
                dups.loc[-1] = [df.iloc[idx1, 1], df.iloc[idx2, 1], val]
                dups.index = dups.index + 1

    dups = dups.sort_index()
    dups["Document_ID1"] = dups["Document_ID1"].astype(int)
    dups["Document_ID2"] = dups["Document_ID2"].astype(int)
    dups.to_csv(path.join("data", "duplicatePairs.csv"), index=False)
    print(dups.shape[0], "duplicates found.")


duplicates(0.7)
# wordclouds()