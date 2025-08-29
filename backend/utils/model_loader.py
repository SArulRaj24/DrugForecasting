import os
import pickle
#import torch
#from myproject.models.transformer import TransformerModel  # make sure to import your class

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_model(category: str):
    """
    Load a trained forecasting model for the given ATC category.

    Returns
    -------
    model : trained model object
    transformer : optional transformer used during training (e.g. PowerTransformer)
    model_type : str ("prophet", "arima", "sarima", "transformer")
    """
    model_path = os.path.join(MODEL_DIR, f"{category}_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    # Case 1: New Transformer state_dict saving
    if isinstance(obj, dict) and obj.get("type") == "transformer" and "state_dict" in obj:
        # ⚡ you must recreate the Transformer architecture exactly as training
        model = TransformerModel(
            input_size=1,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1
        )
        model.load_state_dict(obj["state_dict"])
        model.eval()
        transformer = obj.get("transformer", None)
        return model, transformer, "transformer"

    # Case 2: Metadata dict (Prophet, ARIMA, etc.)
    if isinstance(obj, dict):
        model = obj.get("model")
        transformer = obj.get("transformer", None)
        model_type = obj.get("type", "unknown")
        return model, transformer, model_type

    # Case 3: Saved raw (older Prophet/ARIMA pickles)
    name = obj.__class__.__name__.lower()
    if "prophet" in name:
        model_type = "prophet"
    elif "sarimax" in name or "arima" in name:
        model_type = "sarima"
    else:
        model_type = "unknown"

    return obj, None, model_type