from flask import Flask, render_template, request
import pickle
import subprocess
import joblib
import numpy as np
import pandas as pd
import flask
# import sklearn.ensemble.forest as forest
import sys
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor


regressor = joblib.load('iplmodel_ridge.sav')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


app = flask.Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html', val='')


@app.route('/main', methods=['GET'])  # Allow GET requests
def main():
    return render_template('main.html')


@app.route('/win_predict', methods=['POST'])
def win_predict():

    # Load the model from the file
    f = open('fit_pipe.pkl', 'rb')
    model = pickle.load(f)

    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        city = flask.request.form['city']
        Home = flask.request.form['Home']
        Away = flask.request.form['Away']
        toss_winner = flask.request.form['toss_winner']
        toss_decision = flask.request.form['toss_decision']
        venue = flask.request.form['venue']

        if toss_winner == 'Home Team':
            toss_winner = Home
        else:
            toss_winner = Away

        input_variables = pd.DataFrame([[city, Home, Away, toss_winner, toss_decision, venue]], columns=['city', 'Home', 'Away', 'toss_winner',
                                                                                                         'toss_decision', 'venue'], dtype=object)

        input_variables.Home.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
                                      'Rising Pune Supergiant', 'Royal Challengers Bangalore',
                                      'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
                                      'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
                                      'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
                                     np.arange(0, 14), inplace=True)
        input_variables.Away.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
                                      'Rising Pune Supergiant', 'Royal Challengers Bangalore',
                                      'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
                                      'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
                                      'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
                                     np.arange(0, 14), inplace=True)
        #input_variables['toss_winner'] = np.where(input_variables['toss_winner'] == 'Home Team', input_variables['Home'], input_variables['Away'])
        input_variables.toss_winner.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
                                             'Rising Pune Supergiant', 'Royal Challengers Bangalore',
                                             'Kolkata Knight Riders', 'Delhi Capitals', 'Kings XI Punjab',
                                             'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
                                             'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],
                                            np.arange(0, 14), inplace=True)
        input_variables.toss_decision.replace(
            ['bat', 'field'], [0, 1], inplace=True)
        input_variables.city.replace(['Hyderabad', 'Pune', 'Rajkot', 'Indore', 'Bangalore', 'Mumbai',
                                      'Kolkata', 'Delhi', 'Chandigarh', 'Kanpur', 'Jaipur', 'Chennai',
                                      'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
                                      'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
                                      'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi',
                                      'Visakhapatnam', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah'],
                                     np.arange(0, 30), inplace=True)
        input_variables.venue.replace(['Rajiv Gandhi International Stadium, Uppal',
                                       'Maharashtra Cricket Association Stadium',
                                       'Saurashtra Cricket Association Stadium', 'Holkar Cricket Stadium',
                                       'M Chinnaswamy Stadium', 'Wankhede Stadium', 'Eden Gardens',
                                       'Feroz Shah Kotla',
                                       'Punjab Cricket Association IS Bindra Stadium, Mohali',
                                       'Green Park', 'Punjab Cricket Association Stadium, Mohali',
                                       'Sawai Mansingh Stadium', 'MA Chidambaram Stadium, Chepauk',
                                       'Dr DY Patil Sports Academy', 'Newlands', "St George's Park",
                                       'Kingsmead', 'SuperSport Park', 'Buffalo Park',
                                       'New Wanderers Stadium', 'De Beers Diamond Oval',
                                       'OUTsurance Oval', 'Brabourne Stadium',
                                       'Sardar Patel Stadium, Motera', 'Barabati Stadium',
                                       'Vidarbha Cricket Association Stadium, Jamtha',
                                       'Himachal Pradesh Cricket Association Stadium', 'Nehru Stadium',
                                       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
                                       'Subrata Roy Sahara Stadium',
                                       'Shaheed Veer Narayan Singh International Stadium',
                                       'JSCA International Stadium Complex', 'Sheikh Zayed Stadium',
                                       'Sharjah Cricket Stadium'],
                                      np.arange(0, 34), inplace=True)
        prediction = model.predict(input_variables)
        prediction = pd.DataFrame(prediction, columns=['Winners'])
        prediction = prediction["Winners"].map({0: 'Sunrisers Hyderabad', 1: 'Mumbai Indians', 2: 'Gujarat Lions',
                                                3: 'Rising Pune Supergiant', 4: 'Royal Challengers Bangalore',
                                                5: 'Kolkata Knight Riders', 6: 'Delhi Capitals', 7: 'Kings XI Punjab',
                                                8: 'Chennai Super Kings', 9: 'Rajasthan Royals', 10: 'Deccan Chargers',
                                                11: 'Kochi Tuskers Kerala', 12: 'Pune Warriors', 13: 'Rising Pune Supergiants'})
        return flask.render_template('main.html', original_input={'city': city, 'Home': Home, 'Away': Away, 'toss_winner': toss_winner, 'toss_decision': toss_decision,
                                     'venue': venue},
                                     result=prediction[0],
                                     )


@app.route('/predict', methods=['POST'])
def predict():

    a = []

    if request.method == 'POST':

        venue = request.form['venue']
        if venue == 'ACA-VDCA Stadium, Visakhapatnam':
            a = a + [1, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Barabati Stadium, Cuttack':
            a = a + [0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Dr DY Patil Sports Academy, Mumbai':
            a = a + [0, 0, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Dubai International Cricket Stadium, Dubai':
            a = a + [0, 0, 0, 1, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Eden Gardens, Kolkata':
            a = a + [0, 0, 0, 0, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Feroz Shah Kotla, Delhi':
            a = a + [0, 0, 0, 0, 0, 1, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Himachal Pradesh Cricket Association Stadium, Dharamshala':
            a = a + [0, 0, 0, 0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Holkar Cricket Stadium, Indore':
            a = a + [0, 0, 0, 0, 0, 0, 0, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'JSCA International Stadium Complex, Ranchi':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'M Chinnaswamy Stadium, Bangalore':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'MA Chidambaram Stadium, Chepauk':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Maharashtra Cricket Association Stadium, Pune':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Punjab Cricket Association Stadium, Mohali':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        elif venue == 'Raipur International Cricket Stadium, Raipur':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

        elif venue == 'Rajiv Gandhi International Stadium, Uppal':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

        elif venue == 'Sardar Patel Stadium, Motera':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        elif venue == 'Sawai Mansingh Stadium, Jaipur':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

        elif venue == 'Sharjah Cricket Stadium, Sharjah':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

        elif venue == 'Sheikh Zayed Stadium, Abu-Dhabi':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

        elif venue == 'Wankhede Stadium, Mumbai':
            a = a + [0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            a = a + [1, 0, 0, 0, 0, 0, 0, 0]
        elif batting_team == 'Delhi Capitals':
            a = a + [0, 1, 0, 0, 0, 0, 0, 0]
        elif batting_team == 'Kings XI Punjab':
            a = a + [0, 0, 1, 0, 0, 0, 0, 0]
        elif batting_team == 'Kolkata Knight Riders':
            a = a + [0, 0, 0, 1, 0, 0, 0, 0]
        elif batting_team == 'Mumbai Indians':
            a = a + [0, 0, 0, 0, 1, 0, 0, 0]
        elif batting_team == 'Rajasthan Royals':
            a = a + [0, 0, 0, 0, 0, 1, 0, 0]
        elif batting_team == 'Royal Challengers Bangalore':
            a = a + [0, 0, 0, 0, 0, 0, 1, 0]
        elif batting_team == 'Sunrisers Hyderabad':
            a = a + [0, 0, 0, 0, 0, 0, 0, 1]

        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            a = a + [1, 0, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Delhi Capitals':
            a = a + [0, 1, 0, 0, 0, 0, 0, 0]
        elif bowling_team == 'Kings XI Punjab':
            a = a + [0, 0, 1, 0, 0, 0, 0, 0]
        elif bowling_team == 'Kolkata Knight Riders':
            a = a + [0, 0, 0, 1, 0, 0, 0, 0]
        elif bowling_team == 'Mumbai Indians':
            a = a + [0, 0, 0, 0, 1, 0, 0, 0]
        elif bowling_team == 'Rajasthan Royals':
            a = a + [0, 0, 0, 0, 0, 1, 0, 0]
        elif bowling_team == 'Royal Challengers Bangalore':
            a = a + [0, 0, 0, 0, 0, 0, 1, 0]
        elif bowling_team == 'Sunrisers Hyderabad':
            a = a + [0, 0, 0, 0, 0, 0, 0, 1]

        if batting_team == bowling_team and batting_team != 'none' and bowling_team != 'none':
            return render_template('home.html', val='Batting team and Bowling team cant be same and none of the values can\'t be empty.')

        overs = request.form['overs']
        runs = request.form['runs']
        wickets = request.form['wickets']
        runs_in_prev_5 = request.form['runs_in_prev_5']
        wickets_in_prev_5 = request.form['wickets_in_prev_5']

        if overs == '' or runs == '' or wickets == '' or runs_in_prev_5 == '' or wickets_in_prev_5 == '':
            return render_template('home.html', val='You can\'t leave any field empty!!!')

        overs = float(overs)
        runs = int(runs)
        wickets = int(wickets)
        runs_in_prev_5 = int(runs_in_prev_5)
        wickets_in_prev_5 = int(wickets_in_prev_5)

        a = np.array(a).reshape(1, -1)

        b = [runs, wickets, overs, runs_in_prev_5, wickets_in_prev_5]
        b = np.array(b).reshape(1, -1)
        b = scaler.transform(b)

        data = np.concatenate((a, b), axis=1)

        my_prediction = int(regressor.predict(data)[0])
        print(my_prediction)

        return render_template('home.html', val=f'The final score will be around {my_prediction-5} to {my_prediction+10}.')


if __name__ == '__main__':
    app.run(debug=True)
