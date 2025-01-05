from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from train_model_1 import train_model_1
from train_model_2 import train_model_2

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/train-model-1', methods=['GET'])
def train_and_get_results_1():
    # Model 1'i eğit ve sonuçları al
    mse, plot_path = train_model_1()

    # plot_url'yi oluşturun
    plot_url = f"http://127.0.0.1:5000/{plot_path}"

    # Sonuçları JSON formatında döndür
    return jsonify({
        'mse': mse,
        'plot_url': plot_url
    })

@app.route('/train-model-2', methods=['GET'])
def train_and_get_results_2():
    # Model 2'yi eğit ve sonuçları al
    mse, plot_path = train_model_2()

    # plot_url'yi oluşturun
    plot_url = f"http://127.0.0.1:5000/{plot_path}"

    # Sonuçları JSON formatında döndür
    return jsonify({
        'mse': mse,
        'plot_url': plot_url
    })

@app.route('/static/<path:filename>')
def serve_image(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
