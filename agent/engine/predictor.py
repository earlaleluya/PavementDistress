from agent.engine.detector import YOLOPredictor
from agent.engine.filler import Filler
from agent.engine.classifier import ViTClassifier




class Predictor:
    """
    This class implements a multi-stage prediction pipeline for analyzing road lanes 
    and pavement panels. It orchestrates the following stages:
    1. **Stage 1**: Detects road lanes and pavement panels using YOLO models.
    2. **Stage 2**: Fills in missing predictions for road lanes and panels.
    3. **Stage 3**: Processes predictions to generate cropped RGB masks, classifies 
        them using a Vision Transformer (ViT) model, and updates the predictions with 
        classification results.

    Attributes:
         args: A configuration object containing parameters such as model paths, 
                 device information, input source, and padding for the prediction pipeline.

    Example:
        >>> args = Namespace(
        >>>     model_path_rlanes="path/to/rlane_model.pt",
        >>>     model_path_panels="path/to/panel_model.pt",
        >>>     model_path_classifier="path/to/classifier_model_folder",
        >>>     source="path/to/input/image.jpg"
        >>> )
        >>> predictor = Predictor(args)
        >>> final_prediction = predictor.predict()
    """

    def __init__(self, args):
        self.args = args 



    def predict(self):
        """
        Perform a multi-stage prediction process.

        This method executes a three-stage prediction pipeline:
        1. `stage1`: Generates initial predictions for road lanes and panels.
        2. `stage2`: Refines the predictions using the outputs from `stage1`.
        3. `stage3`: Finalizes the prediction based on the refined results.

        Returns:
            final_prediction (`Prediction` object): The final prediction result after completing all 
            three stages.
        """
        pred_rlanes, pred_panels = self.stage1() 
        prediction = self.stage2(pred_rlanes, pred_panels)
        final_prediction = self.stage3(prediction)
        return final_prediction



    def stage1(self):
        """
        Executes the first stage of the prediction pipeline, which involves 
        detecting road lanes and pavement panels in parallel using YOLO models.

        Returns:
            tuple: A tuple of `Results` objects containing two elements:
                - pred_rlanes: Predictions for road lanes.
                - pred_panels: Predictions for pavement panels.
        """
        rlane_model = YOLOPredictor(model_path=self.args.model_path_rlanes, device=self.args.device)
        panel_model = YOLOPredictor(model_path=self.args.model_path_panels, device=self.args.device)
        pred_rlanes = rlane_model.detect(input=self.args.source)
        pred_panels = panel_model.detect(input=self.args.source)
        return (pred_rlanes, pred_panels) 



    def stage2(self, pred_rlanes, pred_panels):
        """
        Executes the second stage of the prediction process by filling in the 
        predictions for road lanes and panels.

        Args:
            pred_rlanes (`Results` object): Predicted road lanes data.
            pred_panels (`Results` object): Predicted panels data.

        Returns:
            prediction (`Prediction` object): An object containing the three prediction 
                                              outputs as `Results` objects - pred_rlanes, pred_panels, and 
                                              filled prediction of panels (resolves missing panels).
        """
        filler = Filler(pred_rlanes, pred_panels, self.args)
        prediction = filler.fill()
        return prediction
    



    def stage3(self, prediction):
        """
        Processes the third stage of the prediction pipeline.

        This method performs the following steps:
        1. Generates cropped RGB masks from the prediction using the specified padding.
        2. Initializes a Vision Transformer (ViT) classifier with the provided model path and device.
        3. Classifies the cropped patches using the ViT classifier.
        4. Updates the prediction object with the classification confidence and class outputs.

        Args:
            prediction: An object containing prediction data and methods for processing.

        Returns:
            The updated prediction object with classification results.
        """
        patches = prediction.generate_cropped_rgb_masks(self.args.padding)
        classifier = ViTClassifier(self.args.model_path_classifier, device=self.args.device)
        outputs = classifier.classify(patches)
        prediction.update_conf_cls(outputs)
        return prediction