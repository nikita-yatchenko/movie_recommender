# Movie Recommender System

The goal of this project is to build a web application
 that would recommend movies to a user utilizing various machine learning approaches.

## <u> Quick recommender system overview </u>
Recommender system is a subclass of  machine learning information filtering 
models that seek to predict users' ratings of individual items based on either 
content or collaborative filterings.
- Content filtering - model uses items' set of characteristics to 
find similarity
- Collaborative filtering - model that uses past user behavior and other user's behavior 
to pull together an estimate of what a current user might like

### Challenges



## <u> Movie Recommender Project </u> 
### Dataset
We are going to be using a popular movie dataset [MovieLens](https://grouplens.org/datasets/movielens/latest/)


To run make a single recommendation:
0) install necessary dependencies `pip install -r requirements.txt`
1) go to ../movie_recommender/src/rec_sys
2) run `python -W ignore .\knn_recommender.py --path '[path]\data' --movie_name '[movie name]' --top_n [n]`

To run on Flask or Gunicorn:

0) copy downloaded BERT model in project's root folder

1) Create virtual environment and activate it \
`virtualenv -p python3.7 env` \
[Linux] `source env/bin/activate` \
[Windows] `\env\Scripts\activate.bat`

2) Install required modules\
 `pip install Flask==0.10.1` \
 `pip install deeppavlov==0.14.0` \
 `python -m deeppavlov install squad_bert` \
 `python install itsdangerous==2.0.1` \
 `python install SpeechRecognition==3.8.1`
 [for Gunicorn]`pip install gunicorn==20.0.4`
 
3) Run \
`cd bert_app` \
[Flask]`python app.py` \
[Gunicorn]`gunicorn -b 0.0.0.0:8000 -t 300 app:app`

4) Open browser on `localhost:8000`

*P.S. To update the plots, use the function in '/bert_app/static/plots/script_for_plots_creating.py'*