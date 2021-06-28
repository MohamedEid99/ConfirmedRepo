import pandas as pd
from flask import Flask, request
from prophet import Prophet

app = Flask(__name__)


# http://127.0.0.1:5000/api?days=2
@app.route('/confirmed_cases', methods=['POST','GET'])
def predict():
    days = int(request.args.get('days'))

    df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/main/data/time-series-19-covid-combined.csv')
    df = df.rename(columns={'Country/Region': 'Country'}, inplace=False)
    df['Recovered'].interpolate(method='linear', direction='forward', inplace=True)
    df1 = df.drop(['Province/State'], axis='columns')

    conf = Prophet(interval_width=0.95)
    conf.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    confirmed = df1.groupby('Date').sum()['Confirmed'].reset_index()

    confirmed.columns = ['ds', 'y']
    confirmed['ds'] = pd.to_datetime(confirmed['ds'])
    conf.fit(confirmed)

    future2 = conf.make_future_dataframe(periods=days)
    forecast2 = conf.predict(future2)

    data = forecast2[['ds', 'yhat']][-days:]

    result = data.to_json(orient='records', date_format='iso')
    return result

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
