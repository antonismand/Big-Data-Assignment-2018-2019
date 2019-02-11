import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

filename = path.join("data", "train_set.csv")
df = pd.read_csv(filename, sep='\t')

print("There are {} observations and {} features in this dataset. \n".format(df.shape[0], df.shape[1]))

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
