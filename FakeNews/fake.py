import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#Read the data
df=pd.read_csv('news.csv')

#Get shape and head
df.shape
df.head()
df.info()
df.isnull().sum()
#Good Data There is no null values

df['label'].unique()
df['text'].unique()
df['title'].unique()
#DataFlair - Get the labels
labels=df.label
labels.head()
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

The Accuracy is greater than 92%. 
#DataFlair - Build confusion matrix
confusion=confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
import matplotlib.pyplot as plt

# Define function for plotting confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(confusion, classes=['FAKE', 'REAL'], normalize=True, title='Normalized confusion matrix')
plt.show()
# Count the number of occurrences of each label
label_counts = df['label'].value_counts()

# Plot the bar plot
plt.figure()
label_counts.plot(kind='bar', color=['blue', 'green'])
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

from wordcloud import WordCloud

# Concatenate all text data
text_data = ' '.join(df['text'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Most Frequent Words')
plt.show()

# Calculate the document lengths
document_lengths = df['text'].apply(lambda x: len(x.split()))

# Plot the histogram
plt.figure()
plt.hist(document_lengths, bins=20, color='skyblue')
plt.title('Histogram of Document Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.show()

# Calculate label proportions
label_proportions = df['label'].value_counts(normalize=True)

# Plot the pie chart
plt.figure()
plt.pie(label_proportions, labels=label_proportions.index, autopct='%1.1f%%', colors=['blue', 'green'])
plt.title('Label Proportions')
plt.show()

# Create a list of accuracy scores over iterations or time
accuracy_scores = [accuracy_score(y_test[:i], y_pred[:i]) for i in range(1, len(y_test)+1)]

# Plot the line plot
plt.figure()
plt.plot(range(1, len(y_test)+1), accuracy_scores, marker='o')
plt.title('Model Accuracy over Time')
plt.xlabel('Iterations or Time')
plt.ylabel('Accuracy')
plt.show()

# Create a dataframe with document lengths and labels
import seaborn as sns
document_lengths_df = pd.DataFrame({'Length': document_lengths, 'Label': labels})

# Plot the box plot
plt.figure()
sns.boxplot(x='Label', y='Length', data=document_lengths_df)
plt.title('Document Lengths by Label')
plt.xlabel('Label')
plt.ylabel('Number of Words')
plt.show()
