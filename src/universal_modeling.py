import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Any

class UniversalLSTM(nn.Module):
    """The architecture for the universal LSTM model."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(UniversalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_universal_model(filepath: str, input_size: int, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1) -> UniversalLSTM:
    """
    Loads a pre-trained universal model from a .pth file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UniversalLSTM(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    return model

def make_universal_prediction(model: UniversalLSTM, sequence: np.ndarray) -> Tuple[Any, Any]:
    """
    Makes a prediction using the universal LSTM model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert sequence to tensor and add batch dimension
    sequence_tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(sequence_tensor)
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability >= 0.5 else 0
        confidence = probability if prediction == 1 else 1 - probability

    return prediction, confidence
