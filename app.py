from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field
from typing import Annotated
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


import nltk
nltk.download("stopwords")
nltk.download("punkt")

with open("etc.pkl","rb") as f:
    model=pickle.load(f)

with open("tfidf.pkl","rb") as f:
    embedding=pickle.load(f)

with open("MinMaxScaler.pkl","rb") as f:
    scaler=pickle.load(f)



stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def text_preprocessing(text):
    text = text.encode("ascii", "ignore").decode()
    df = pd.DataFrame({"text": [text]})
    df["text"] = df["text"].apply(lambda x: x.lower())
	
    def remove_html_tags(text):
        pattern=re.compile("<.*?>")
        return pattern.sub(r"",text)      
	
    df["text"]=df["text"].apply(remove_html_tags)  

    df["text"] = df["text"].apply(
        lambda x: " ".join([word for word in x.split() if word not in stop_words])
    )

    # for special-character removing
    df["text"] = df["text"].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))  


    df["text"] = df["text"].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    def stem_words(text):
        words = text.split()          # split into words
        stemmed = [ps.stem(word) for word in words]
        return " ".join(stemmed)      # return as string

	
    df["text"] = df["text"].apply(stem_words) 

    df["word_count"] = df["text"].apply(lambda x: len(x.split()))

    df=df[["text","word_count"]]

    arr = embedding.transform(df["text"]).toarray()

    arr = pd.DataFrame(arr)

    arr["word_count"] = df["word_count"]

    # IMPORTANT FIX
    arr.columns = arr.columns.astype(str)

    arr["word_count"] = scaler.transform(arr[["word_count"]])

    return arr

from fastapi import Body

THRESHOLD = 0.65 

app=FastAPI()
@app.post("/predict")
def predict_premium(inp: str = Body(..., media_type="text/plain")):
    # data is in the json formate

    refined_inp=text_preprocessing(inp)
    prediction=model.predict(refined_inp)[0]
    probabilities = model.predict_proba(refined_inp)[0]

    spam_prob = probabilities[1]      # Probability of class 1 (Spam)
    ham_prob = probabilities[0]       # Probability of class 0 (Not Spam)

    # Apply custom threshold
    if spam_prob > THRESHOLD:
        prediction = 1
    else:
        prediction = 0

    confidence = float(max(probabilities))  # highest probability

    if prediction == 1:
        return JSONResponse(status_code=200,content={"Message": f" Spam --------> with Confidence: {round(confidence * 100, 2)} %"})
    elif prediction == 0:
        return JSONResponse(status_code=200,content={"Message": f" Not Spam  --------> with Confidence: {round(confidence * 100, 2)} %"})
    else:
        return JSONResponse(status_code=500,content={"Messege":"Backend/API side error"})
    