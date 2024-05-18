"""
This script trains a ML model to classify disaster response messages.

It performs the following steps:
1. Loads the data from an SQLite database.
2. Splits the data into training and test sets.
3. Builds a machine learning pipeline with text processing and classification.
4. Trains the model using GridSearchCV.
5. Evaluates the model performance on the test set.
6. Saves the trained model as a pickle file.

Usage:
    python train_classifier.py <database_filepath> <model_filepath>

Example:
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
"""


import sys
import pandas as pd
import numpy as np
import re
import nltk
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def load_data(database_filepath):
    """
    Load data from SQLite database.

    Args:
        database_filepath (str): Filepath for the SQLite database.

    Returns:
        X (DataFrame): Feature DataFrame.
        Y (DataFrame): Target DataFrame.
        category_names (list): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Message', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Normalize, tokenize, remove stop words, and lemmatize text string.

    Args:
        text (str): Input text string.

    Returns:
        list: List of clean tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in
              nltk.corpus.stopwords.words("english")]
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.PorterStemmer()
    clean_tokens = [lemmatizer.lemmatize(stemmer.stem(token)).strip() for
                    token in tokens]
    return clean_tokens


class POSTagEncoder(BaseEstimator, TransformerMixin):
    """Custom transformer to extract and one-hot encode POS tag features."""

    def __init__(self):
        """Initialize POSTagEncoder object."""
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, x, y=None):
        """Fit the POSTagEncoder object."""
        pos_tags = self._extract_pos_tags(x)
        self.encoder.fit(pos_tags)
        return self

    def transform(self, x):
        """Transform the POSTagEncoder object."""
        pos_tags = self._extract_pos_tags(x)
        return self.encoder.transform(pos_tags)

    def _extract_pos_tags(self, x):
        """Extract POS tags from text."""
        pos_tags = [nltk.pos_tag(nltk.word_tokenize(text)) for text in x]
        pos_tags = [[' '.join([tag for _, tag in tags])] for tags in pos_tags]
        return pos_tags


class NEREncoder(BaseEstimator, TransformerMixin):
    """Custom transformer to extract and one-hot encode NER features."""

    def __init__(self):
        """Initialize NEREncoder object."""
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, x, y=None):
        """Fit the NEREncoder object."""
        named_entities = self._extract_named_entities(x)
        self.encoder.fit(named_entities)
        return self

    def transform(self, x):
        """Transform the NEREncoder object."""
        named_entities = self._extract_named_entities(x)
        return self.encoder.transform(named_entities)

    def _extract_named_entities(self, x):
        """Extract named entities from text."""
        named_entities = [
            nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
            for text in x]
        named_entities = [[' '.join([chunk.label() if hasattr(chunk, 'label')
                                     else '' for chunk in entity])]
                          for entity in named_entities]
        return named_entities


def build_model():
    """
    Build machine learning pipeline with GridSearchCV.

    Returns:
        GridSearchCV: GridSearchCV object with pipeline and parameters.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer(tokenizer=tokenize, token_pattern=None)),
            ('pos_tags', POSTagEncoder()),
            ('named_entities', NEREncoder())
        ])),
        ('clf',
         MultiOutputClassifier(XGBClassifier(use_label_encoder=False,
                                             eval_metric='mlogloss')))
    ])

    parameters = {
        'features__tfidf__max_df': [0.75, 1.0],
        'features__tfidf__ngram_range': [(1, 1)],
        'clf__estimator__n_estimators': [50],
        'clf__estimator__learning_rate': [0.1]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names, database_filepath):
    """
    Evaluate the model, print report, and save metrics to the database.

    Args:
        model (GridSearchCV): Trained model.
        X_test (DataFrame): Test feature data.
        Y_test (DataFrame): Test target data.
        category_names (list): List of category names.
        database_filepath (str): Filepath for the SQLite database.
    """
    Y_pred = model.predict(X_test)

    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []

    # List to store class-wise metrics
    metrics_list = []

    for i, col in enumerate(category_names):
        report = classification_report(Y_test[col], Y_pred[:, i],
                                       output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        accuracy = accuracy_score(Y_test[col], Y_pred[:, i])

        # Append class metrics to the list
        metrics_list.append({
            'class': col,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        })

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)

        print(f"Category: {col}")
        print(classification_report(Y_test[col], Y_pred[:, i]))
        print("\n")

    # Create DataFrame from the list of metrics
    class_metrics = pd.DataFrame(metrics_list)

    overall_metrics = {
        'average_accuracy': np.mean(accuracy_list),
        'average_precision': np.mean(precision_list),
        'average_recall': np.mean(recall_list),
        'average_f1_score': np.mean(f1_list)
    }

    print(f"Average Accuracy: {overall_metrics['average_accuracy']:.4f}")
    print(f"Average Precision: {overall_metrics['average_precision']:.4f}")
    print(f"Average Recall: {overall_metrics['average_recall']:.4f}")
    print(f"Average F1 Score: {overall_metrics['average_f1_score']:.4f}")

    # Save class-wise metrics to a new table in the database
    engine = create_engine(f'sqlite:///{database_filepath}')
    class_metrics.to_sql('class_metrics', engine, index=False,
                         if_exists='replace')

    # Save overall metrics to the database
    overall_metrics_df = pd.DataFrame([overall_metrics])
    overall_metrics_df.to_sql('overall_metrics', engine, index=False,
                              if_exists='replace')


def save_model(model, model_filepath):
    """
    Save the model to a pickle file.

    Args:
        model (GridSearchCV): Trained model.
        model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """Run the ML pipeline that trains the classifier."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names,
                       database_filepath)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponseData.db \
                classifier.pkl')


if __name__ == '__main__':
    main()
