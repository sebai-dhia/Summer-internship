#from crypt import methods
#from fileinput import filename
#from urllib import request
from flask import Flask,render_template,request
import os
from model import OCR
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app =Flask(__name__)

BASE_PATH=os.getcwd()
UPLOAD_PATH=os.path.join(BASE_PATH,'static\\upload')



@app.route('/',methods=['POST','GET'])
def index():
    if request.method =='POST':
        upload_file= request.files['image_name']
        filename=upload_file.filename
        path_save=os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        text = OCR(path_save, filename)
        return render_template('index.html',upload=True,upload_image=filename,text=text)
    return render_template('index.html',upload=False)



@app.route('/home')
def home():
    return render_template('index.html')

if __name__ == "__main__" :
    app.run(debug=True)
