### News Classification and Visualization

This repository contains code for classifying news articles as fake or real using the Passive Aggressive Classifier algorithm. The dataset used is stored in a CSV file named 'news.csv'. 

## What is Fake News?
A type of yellow journalism, fake news encapsulates pieces of news that may be hoaxes and is generally spread through social media and other online media. This is often done to further or impose certain ideas and is often achieved with political agendas. Such news items may contain false and/or exaggerated claims, and may end up being viralized by algorithms, and users may end up in a filter bubble.

## What is a TfidfVectorizer?
- TF (Term Frequency): The number of times a word appears in a document is its Term Frequency. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.

- IDF (Inverse Document Frequency): Words that occur many times a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus.

The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.

## What is a PassiveAggressiveClassifier?
Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

Download The Dataset From here: 
`https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view`
The code performs the following steps:

1. Load and explore the dataset, including checking its shape, displaying the head, and checking for missing values.
2. Split the dataset into training and testing sets.
3. Initialize a TF-IDF vectorizer to convert text data into numerical features.
4. Fit and transform the training set and transform the test set using the TF-IDF vectorizer.
5. Initialize a Passive Aggressive Classifier and train it on the training set.
6. Predict the labels for the test set and calculate the accuracy of the classifier.
7. Build a confusion matrix to visualize the performance of the classifier.
![output](https://github.com/saqib772/Artifical-Intelligence-Projects/assets/121972215/ddf1faae-fa94-4f03-8685-a52c1bda9bd9)

8. Generate visualizations to gain insights into the data:
- Plot the distribution of labels (fake and real) using a bar plot.
![labels](https://github.com/saqib772/Artifical-Intelligence-Projects/assets/121972215/3951f02d-cc56-4f10-aa13-d4ede14b6199)

- Create a word cloud to visualize the most frequent words in the news articles.
![download](https://github.com/saqib772/Artifical-Intelligence-Projects/assets/121972215/dc779d95-95f9-4ad1-ae3a-d77d4ecea914)

- Plot a histogram of document lengths to analyze the distribution of word counts.
![word](https://github.com/saqib772/Artifical-Intelligence-Projects/assets/121972215/e40029c9-5185-4b7b-8707-c0f08b90d275)

- Display the label proportions using a pie chart.

![pop](https://github.com/saqib772/Artifical-Intelligence-Projects/assets/121972215/eeb6f168-cb6f-45c8-8e45-bede5b4d4c0a)

- Plot the model accuracy over time or iterations using a line plot.
![model](https://github.com/saqib772/Artifical-Intelligence-Projects/assets/121972215/c4ff2008-fdb4-44e9-a93b-a5413f6adcf9)

- Visualize the document lengths by label using a box plot.
![box plot](https://github.com/saqib772/Artifical-Intelligence-Projects/assets/121972215/0d3aa143-f200-40a6-8cec-22e8e94e828d)
