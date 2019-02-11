import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

filename = path.join("data", "train_set.csv")
df = pd.read_csv(filename, sep='\t')

print("There are {} observations and {} features in this dataset. \n".format(df.shape[0], df.shape[1]))

text = " ".join(article for article in df.Content)

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()