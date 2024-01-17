# Sentiment Analysis Model
This Natural Language Processing (NLP) code focuses on sentiment analysis by training a machine learning model using a dataset of restaurant reviews . It involves text cleaning, removing irrelevant characters and stemming words to their root form. The Bag of Words model is then employed to convert the text data into numerical features. The Naive Bayes classifier is trained on the processed data to predict whether a review is positive or negative. The code evaluates the model's performance using a confusion matrix and accuracy score on a test set. The results provide insights into the model's ability to classify sentiment in restaurant reviews.

Here's a detailed breakdown:
# Importing Libraries: 
NumPy, matplotlib, and pandas are imported for data manipulation and visualization.
# Loading Dataset: 
The code reads a dataset (Restaurant_Reviews.tsv) using pandas, considering tab ('\t') as the delimiter.
# Text Cleaning: 
The reviews undergo cleaning, which involves converting all characters to lowercase, removing non-alphabetic characters, and stemming words to their root form using the Porter stemming algorithm. Stop words (common words like "the," "and," etc.) are removed, except for the word "not". 
# Bag of Words Model: 
The CountVectorizer from scikit-learn is used to create a Bag of Words model, converting the processed text data into a numerical format suitable for machine learning. It limits the features to the top 1500 most frequent words.
# Splitting Dataset: 
The dataset is split into a training set (80%) and a test set (20%) using the train_test_split function.
# Training Naive Bayes Model: 
The Naive Bayes classifier, specifically Gaussian Naive Bayes, is chosen and trained on the training set.
# Prediction and Evaluation: 
The trained model is used to predict sentiments on the test set. The confusion matrix and accuracy score are calculated to evaluate the model's performance.
# Results Display: 
The code prints the concatenation of predicted and actual values for the test set, the confusion matrix, and the accuracy score.

# DataSet Explanation

The provided data comprises restaurant reviews along with binary indicators reflecting whether the reviewers liked the experience or not. This sentiment-labeled dataset is utilized in a Natural Language Processing (NLP) context to train a machine learning model. The goal is to develop a predictive model capable of discerning sentiments from textual reviews, distinguishing between positive and negative expressions. The dataset's reviews are processed, cleaned, and transformed into a format suitable for machine learning, laying the groundwork for training a Naive Bayes classifier. This classifier, once trained, can analyze new reviews and predict their sentiment based on the learned patterns in the data.

# Applications of this Model

The results of the above code, leveraging sentiment analysis, offer valuable insights for businesses, especially in the restaurant industry. By predicting whether customers liked or disliked their experiences, establishments can gauge overall customer satisfaction. Positive sentiments indicate successful aspects, aiding in reinforcing those strengths. On the other hand, addressing negative sentiments enables businesses to identify areas for improvement.
