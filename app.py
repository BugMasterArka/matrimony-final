import os
import cv2
from flask import Flask , request, render_template, Response, redirect, url_for, flash
from werkzeug.utils import secure_filename
import glob
from model import Model
import datetime

app = Flask(__name__)

#--------------------------------------------------Load Model-------------------------------------------------------------#

vgg16 = Model()
age_model = vgg16.load_model(vgg16.age_model_path, vgg16.age_caffemodel, vgg16.age_prototxt)
gender_model = vgg16.load_model(vgg16.gender_model_path, vgg16.gender_caffemodel, vgg16.gender_prototxt)

age_given = 0
#--------------------------------------------------Dependencies-----------------------------------------------------------#

UPLOAD_FOLDER = 'static/uploads/'

gender_dict = {0:'Female', 1:'Male'}
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def giveFile():
#     list_of_files = glob.glob('C:/Users/sanja/Desktop/vgg16/static/uploads/*')
#     filepath = max(list_of_files, key=os.path.getctime)
#     return filepath

# ------------------------------------------------------Initial Routes-------------------------------------------------------#

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/openCam')
def openCam():
    return render_template('openCam.html')

# ------------------------------------------------------Image Prediction---------------------------------------------------#



def predictfile():
    list_of_files = glob.glob('D:/SDH/matromony/static/uploads/*')
    filepath = max(list_of_files, key=os.path.getctime)
    print("Filepath of uploaded image" + filepath)
    frame_bgr = cv2.imread(filepath)
    
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = vgg16.detector(frame_rgb, 1)

    for d in faces:
        left = int(0.6 * d.left())     # + 40% margin
        top = int(0.6 * d.top())       # + 40% margin
        right = int(1.4 * d.right())   # + 40% margin
        bottom = int(1.4 * d.bottom()) # + 40% margin
        face_segm = frame_rgb[top:bottom, left:right]
        age, age_confidence = vgg16.predict(age_model, face_segm, vgg16.input_height, vgg16.input_width)
        gender, gender_confidence = vgg16.predict(gender_model, face_segm, vgg16.input_height, vgg16.input_width)
        gender = gender_dict[round(gender)]
        text = [age, gender]
        # cv2.putText(frame_bgr, text, (d.left(), d.top() - 20), font, fontScale, fontColor, lineType)
        # cv2.rectangle(frame_bgr, (d.left(), d.top()), (d.right(), d.bottom()), fontColor, 2) 
    return text


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/result', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        today = datetime.date.today()
        year = today.year
        dob = request.form.get("Dob")
        curr_year = int(dob.split('-')[0])
        age_given = abs(curr_year - year)
        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # return render_template('result.html', filename=filename)
        data = predictfile()
        print(data[0])
        print(age_given)
        if(abs(age_given-data[0])>5):
            flash("Enter a more recent photo")
        else:
            flash("Login Successful")
        return render_template('index.html')
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

if __name__ == "__main__":
    app.run(debug = False, threaded = False)
    
    
    
#--------------------------------------------------Matrimony File-----------------------------------------------------#

