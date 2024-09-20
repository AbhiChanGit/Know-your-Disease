from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import prediction

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

todos = {}

class Disease_Preditor(Resource):
    def get(self):
        return "Welcome to, Disease Predictor API!"
    
    def post(self):
        try:
            value = request.get_json()
            if (value):
                return {"Post Values": value}, 201
            
            return {"error": "Invalid format"}
        
        except Exception as error:
            return {'error': error}
        
class Get_Prediction_Output(Resource):
    def get(self):
        return {"error": "Invalid Method"}
    
    def post(self):
        try:
            data = request.get_json()
            predict_output = prediction.predict_disease(data)
            return predict_output
        
        except Exception as error:
            return {'error': error}
        
api.add_resource(Disease_Preditor,'/')
api.add_resource(Get_Prediction_Output,'/getPredictionOutput')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
