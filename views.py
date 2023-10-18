from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences
from LLMApp import model_training, data_preprocessing

class TrainModelView(APIView):
    
    model = None
    tokenizer = None
    training_status = "Model not trained yet."

    def post(self, request, *args, **kwargs):
        data_path = r"C:\Users\hp\Downloads\models\urlsf_subset00-7_data\0006004-f940b83d9ed9a19f39d18df04679b071.txt"
        train_data, train_labels, tokenizer = data_preprocessing.preprocess_txt_data(data_path)

        # Store the tokenizer in the class variable
        TrainModelView.tokenizer = tokenizer

        try:
            TrainModelView.model, history = model_training.train_model(train_data, train_labels)

            # Return the final epoch's accuracy
            final_accuracy = history.history['accuracy'][-1]
            TrainModelView.training_status = f"Model trained successfully with a final accuracy of {final_accuracy:.2f}."
            return Response({"message": "Model trained successfully.", 
                             "final_accuracy": final_accuracy}, 
                             status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": f"Error training model: {str(e)}"}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def get(self, request, *args, **kwargs):
        return Response({"message": TrainModelView.training_status}, status=status.HTTP_200_OK)
     
class TextGenerationView(APIView):

    def post(self, request, *args, **kwargs):
        input_text = request.data.get('input_text', None)

        if input_text is None:
            return Response({"error": "Request body must contain 'input_text'."}, 
                            status=status.HTTP_400_BAD_REQUEST)

        processed_text = self.preprocess_input_text(input_text)
        try:
            generated_text = self.generate_text(processed_text)
            return Response({"generated_text": generated_text}, 
                            status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": f"Error generating text: {str(e)}"}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def generate_text(self, processed_text):
        if TrainModelView.model is None:
            raise Exception("Model is not trained.")

        predictions = TrainModelView.model.predict(processed_text)
        threshold = 0.5
        if predictions[0][0] > threshold:
            return "Positive Class"
        else:
            return "Negative Class"

    def preprocess_input_text(self, input_text):
        if TrainModelView.tokenizer is None:
            raise Exception("Tokenizer is not initialized.")
        
        sequences = TrainModelView.tokenizer.texts_to_sequences([input_text])
        padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
        return padded_sequences
