from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image
from io import BytesIO
from Detector import *

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Process the image (replace this with your actual image processing logic)
    plate_img , result = process_image(image)
    print(plate_img)
    print(result)
    if result != []:
        result= result.tolist()
        print(result) 
        processed_image_data = Image.fromarray(plate_img)

        buffered = BytesIO()
        processed_image_data.save(buffered, format="JPEG")

    # Encode processed image data as base64
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        processed_image_base64 = False
    return jsonify({'plate_image': processed_image_base64 , 'result' : result})


if __name__ == '__main__':
    app.run(debug=True)


