import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def predict_disease(config):

    data = pd.read_csv("./dataset/Training.csv").dropna(axis=  1)
    X = data.iloc[:, :-1]

    encoder  = LabelEncoder()
    encoder.fit(data["prognosis"])

    # Loading the model from the saved file
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    symptoms = config["symptoms"]
    symptoms_data = X.columns.values
  
    # Creating a symptom index dictionary to encode the 
    # input symptoms into numerical form 
    symptom_index = {} 
    for index, value in enumerate(symptoms_data): 
        symptom = " ".join([i.capitalize() for i in value.split("_")]) 
        symptom_index[symptom] = index 
    
    data_dict = { 
        "symptom_index":symptom_index, 
        "predictions_classes":encoder.classes_ 
    }

    symptoms = symptoms.split(",") 
      
    # creating input data for the models 
    input_data = [0] * len(data_dict["symptom_index"]) 
    for symptom in symptoms: 
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
          
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
      
    # generating individual outputs 
    # rf_prediction = data_dict["predictions_classes"][final_rfc.predict(input_data)[0]] 
    # nb_prediction = data_dict["predictions_classes"][final_nbc.predict(input_data)[0]] 
    svm_prediction = data_dict["predictions_classes"][model.predict(input_data)[0]] 
    
    predictions = { 
        # "rf_model_prediction": rf_prediction, 
        # "naive_bayes_prediction": nb_prediction, 
        "svm_model_prediction": svm_prediction,
    }
    
    return predictions