from flask import Flask, render_template, jsonify
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from app.utils.data_loader import load_stock_data
from app.core.utils.tradewise_ai import TradewiseAI

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    stock_data = load_stock_data()
    ai = TradewiseAI()
    suggestions = ai.generate_trade_suggestions(stock_data)
    
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['close'], name='Actual Price'))
    
    for suggestion in suggestions:
        fig.add_trace(go.Scatter(x=[stock_data['date'].iloc[-1]], y=[suggestion['price']], 
                                 mode='markers', name=f'{suggestion["action"]} Suggestion',
                                 marker=dict(size=10, symbol='star')))
    
    fig.update_layout(title='Stock Price: Actual vs Forecast',
                      xaxis_title='Date',
                      yaxis_title='Price')
    
    return jsonify({
        'plot': fig.to_json(),
        'suggestions': suggestions
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
