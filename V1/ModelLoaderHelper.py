from tensorflow.keras.models import load_model, Model

class ModelLoader:
    def __init__(self, model_path):
        """
        Initializes the ModelLoader class and loads the model.
        
        :param model_path: Path to the .h5 model file.
        """
        self.model = load_model(model_path)
        
