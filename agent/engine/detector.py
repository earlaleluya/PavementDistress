from ultralytics import YOLO
import torch 



class YOLOPredictor:
    """
    A class for performing object detection using a YOLO model.
    
    Attributes:
        model (YOLO): The YOLO model loaded from the specified model path.
    
    Example:
        >>> predictor = YOLOPredictor("path/to/model.pt")
        >>> results = predictor.detect("path/to/image.png")
        >>> print(results)
    """

    def __init__(self, model_path, device=None):
        """
        Initializes the YOLOPredictor with the specified model path and sets up the device.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(device)


    def detect(self, input):
        """
        Performs object detection on the given input and returns the detection results.
        """
        results = self.model.predict(input)
        return results[0]