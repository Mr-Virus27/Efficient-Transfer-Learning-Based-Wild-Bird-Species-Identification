from flask import Flask, request, render_template_string
from src.predict import predict_file
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

INDEX_HTML = '''
<!doctype html>
<title>Bird Species Demo</title>
<h1>Upload an audio file (.wav/.mp3)</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if result %}
  <h2>Prediction: {{result}}</h2>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        f = request.files.get('file')
        if f:
            path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(path)
            # NOTE: change 'bird_vgg16_model.h5' if you saved the model with another name
            try:
                label, conf = predict_file('bird_vgg16_model.h5', path)
                result = f"{label} (confidence={conf:.2f})"
            except Exception as e:
                result = f"Error: {e}"
    return render_template_string(INDEX_HTML, result=result)

if __name__ == '__main__':
    app.run(debug=True)
