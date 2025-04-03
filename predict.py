"""
    This module detects the road lanes and pavement panels, makes a refined panel representation, then classify each detected panel 
    according to their distress type: non-distressed, sealed patch, sealed crack, and thin crack.


    Usage:
        python predict.py --source [Path]
                                   agent/assets/sample.png

"""
import argparse
from agent.utils.args_merger import merge
from agent.engine.predictor import Predictor


def predict(args):
    args = merge(args)   
    agent = Predictor(args)
    agent.predict()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This file obtains the best slabs per interval and classify its category ('Asphalt' or 'Concrete'). Also, it further classifies the subcategory of its primary category. Then, it saves the prediction in a csv file.")
    parser.add_argument('--source', required=True, help="Path to the source image.")
    parser.add_argument('--config', required=False, default="config.yaml", help="Path to configuration file.")
    parser.add_argument('--show', action='store_true', help="To show the predicted image.")
    parser.add_argument('--no-show', dest='show', action='store_false', help="Not to show the predicted image.")
    parser.set_defaults(show=True)
    args = parser.parse_args()

    # Perform prediction
    result = predict(args)  
    

    
    
    # TODO to be converted into Results object for better access of values