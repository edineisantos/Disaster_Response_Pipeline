import sys
import os
import json
import plotly
import pandas as pd
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Image
import joblib
from sqlalchemy import create_engine

# Add the parent directory to the system path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../models')))

from train_classifier import POSTagEncoder, NEREncoder


app = Flask(__name__)

nltk.download('stopwords')

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponseData.db')
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Number of messages classified in each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Number of messages classified in each class
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    # Distribution of the number of classes per message
    class_counts = df.iloc[:, 4:].sum(axis=1)
    class_counts_distribution = class_counts.value_counts().sort_index()
    class_counts_values = list(class_counts_distribution.index)
    class_counts_freq = list(class_counts_distribution.values)

    # Generate word cloud
    stop_words = set(stopwords.words('english'))
    words = ' '.join(df['message'])
    wordcloud = WordCloud(stopwords=stop_words, 
                          max_words=100, 
                          background_color="white").generate(words)
    wordcloud_image = wordcloud.to_array()
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Number of Messages Classified in Each Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=class_counts_values,
                    y=class_counts_freq
                )
            ],
            'layout': {
                'title': 'Distribution of the Number of Classes per Message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Classes"
                }
            }
        },
        {
            'data': [
                Image(
                    z=wordcloud_image
                )
            ],
            'layout': {
                'title': 'Word Cloud of Messages',
                'xaxis': {
                    'visible': False
                },
                'yaxis': {
                    'visible': False
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()