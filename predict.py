"""
    This module detects the road lanes and pavement panels, makes a refined panel representation, then classify each detected panel 
    according to their distress type: non-distressed, sealed patch, sealed crack, and thin crack.

    Usage:
        python predict.py --source agent/assets/sample.png
                                   --config [Path to configuration file, default: config.yaml]
                                   --show [To show the predicted image]
                                   --no-show [Not to show the predicted image]
                                   --cpu-only [Use CPU for prediction]
                                   --padding [Enable padding when generating patches]
"""
import argparse
import torch
from agent.utils.args_merger import merge
from agent.engine.predictor import Predictor




def predict(args):
    """
    Perform prediction using the provided arguments.

        Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Prediction result from the Predictor instance.
    """
    args = merge(args)   
    agent = Predictor(args)
    return agent.predict()
   




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This file obtains the best slabs per interval and classify its category ('Asphalt' or 'Concrete'). Also, it further classifies the subcategory of its primary category. Then, it saves the prediction in a csv file.")
    parser.add_argument('--source', required=True, help="Path to the source image.")
    parser.add_argument('--config', required=False, default="config.yaml", help="Path to configuration file.")
    parser.add_argument('--show', action='store_true', help="To show the predicted image.")
    parser.add_argument('--no-show', dest='show', action='store_false', help="Not to show the predicted image.")
    parser.set_defaults(show=True)
    parser.add_argument('--cpu-only', action='store_const', const="cpu", dest='device', help="Use CPU for prediction.")
    parser.set_defaults(device="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--padding', action='store_true', help="Enable padding when generating patches.")
    parser.set_defaults(padding=False)
    args = parser.parse_args()


    # Perform prediction
    prediction = predict(args)  
    prediction.show_masks_only()