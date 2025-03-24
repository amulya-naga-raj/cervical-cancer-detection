from flask import Flask, request, render_template
from src.predict import predict_image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image = request.files['image']
    image_path = "uploads/" + image.filename
    image.save(image_path)

    result = predict_image(image_path)
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
