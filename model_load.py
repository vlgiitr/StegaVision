import os
import importlib
import torch.nn as nn
import torch

MODELS_DIR = 'models'

def load_model(model_name):
    module = importlib.import_module(f'models.{model_name}')
    model_class = getattr(module, 'Make_model')
    
    class ModelWrapper(nn.Module):
        def __init__(self):
            super(ModelWrapper, self).__init__()
            self.model = model_class()
        
        def forward(self, input_S, input_C):
            return self.model(input_S, input_C)
    
    model = ModelWrapper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model

def get_available_models():
    models = []
    for file in os.listdir(MODELS_DIR):
        if file.endswith('.py') and file != '__init__.py':
            models.append(file[:-3])  
    return models

if __name__ == "__main__":
    available_models = get_available_models()
    print(f"Available models: {available_models}")
    
    for model_name in available_models:
        model = load_model(model_name)
        print(f"Loaded model: {model_name}")
