import cv2
import numpy as np
from flask import Flask, request, render_template
import io
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications import  DenseNet201
import numpy as np
from tensorflow.keras.models import load_model,Model
import pandas as pd
from base64 import b64encode


app = Flask(__name__,template_folder='templates')

caption_model=load_model('models',compile=False)

df=pd.read_csv('captions.txt')
df.head()
captions=df['caption'].tolist()

tokenizer=Tokenizer()
tokenizer.fit_on_texts(captions)
model=DenseNet201()
fe=Model(inputs=model.input,outputs=model.layers[-2].output)


def idx_to_word(integer,tokenizer):
    
    for word, index in  tokenizer.word_index.items():
        if index==integer:
            return word
    return None


app = Flask(__name__,template_folder='templates')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    file=request.files['image']
    
    img=cv2.imdecode(np.frombuffer(file.read(),np.uint8),cv2.IMREAD_UNCHANGED)
    img=cv2.resize(img,(224,224))

    img_1=img/255
    img_1=np.expand_dims(img_1,axis=0)
    feature=fe.predict(img_1,verbose=0)
    max_length=28

    in_text='startseq'
    for i in range(max_length):
        sequence=tokenizer.texts_to_sequences([in_text])[0]
        sequence=pad_sequences([sequence],max_length)
    
        y_pred=caption_model.predict([feature,sequence])
        y_pred=np.argmax(y_pred)
        
        word=idx_to_word(y_pred,tokenizer)
  
        if word is None:
            break
        in_text+=' '+word
        if word=='endseq':
            break
    
    image_=file.read()
    image_b64=b64encode(image_).decode('utf-8')
    
    return render_template('index.html',prediction=in_text,image=image_b64)

if __name__=='__main__':
    app.run(debug=True)