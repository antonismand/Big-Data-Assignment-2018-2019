import numpy as np
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath, get_tmpfile

from gensim.models import TfidfModel
import pandas as pd
from os import path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn import svm
# from gensim.sklearn_api import W2VTransformer
from gensim.sklearn_api import D2VTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(path.join("data", "train_set.csv"), sep='\t')
df_test = pd.read_csv(path.join("data", "test_set.csv"), sep='\t')

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
    dups.to_csv(path.join("data", "duplicatePairs.csv"), index=False)  # TODO paradoteo sep = tab
    print(dups.shape[0], "duplicates found.")


def get_scores(true_labels, predicted_labels, scores):
    scores['Accuracy'] += accuracy_score(true_labels, predicted_labels)
    scores['Precision'] += precision_score(true_labels, predicted_labels, average='weighted')
    scores['Recall'] += recall_score(true_labels, predicted_labels, average='weighted')
    scores['F-Measure'] += f1_score(true_labels, predicted_labels, average='weighted')

    return scores


def classify(classifier, method, full=True):  # boolean full : to use the whole dataset or first 1000 observations
    kf = KFold(n_splits=10)

    X = df.Content if full else df.Content[0:2000]
    y = df.Category if full else df.Category[0:2000]

    scores = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F-Measure': 0}
    i = 0

    pipe = []
    if method == 'W2V':
        X = [simple_preprocess(line) for line in X]
        pipe.append(('vect', D2VTransformer(min_count=5)))
    elif method == 'BoW':
        pipe.append(('vect', CountVectorizer(stop_words='english')))
    elif method == 'SVD':
        pipe.append(('vect', CountVectorizer(stop_words='english')))
        pipe.append(('svd', TruncatedSVD(n_components=5, n_iter=7, random_state=42)))
    else:
        pipe.append(('tfidf', TfidfVectorizer(stop_words='english')))

    if classifier == "SVM":
        pipe.append(('clf', svm.SVC(kernel='linear')))
    elif classifier == "Random Forest":
        pipe.append(('clf', RandomForestClassifier()))
    else:
        pipe.append(('clf', MultinomialNB()))

    clf = Pipeline(pipe)
    # print(pipe)
    for train_index, test_index in kf.split(X):
        i += 1
        X_train = np.array(X)[train_index]
        X_test = np.array(X)[test_index]

        y_train = np.array(y)[train_index]
        y_test = np.array(y)[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        scores = get_scores(y_test, y_pred, scores)
        print(i, '/', 10)

    scores = {k: v / 10 for k, v in scores.items()}
    return {'Statistic Measure': classifier + '(' + method + ')', **scores}


# wordclouds()
# duplicates(0.7)
results = pd.DataFrame(columns=['Statistic Measure', 'Accuracy', 'Precision', 'Recall', 'F-Measure'])
results = results.append(classify('SVM', 'BoW', False), ignore_index=True)
results = results.append(classify('Random Forest', 'BoW', False), ignore_index=True)
results = results.append(classify('SVM', 'SVD', False), ignore_index=True)
results = results.append(classify('Random Forest', 'SVD', False), ignore_index=True)
results = results.append(classify('SVM', 'W2V', False), ignore_index=True)
results = results.append(classify('Random Forest', 'W2V', False), ignore_index=True)

results = results.append(classify('SVM', 'TF-IDF', False), ignore_index=True) # My Method
results = results.T


