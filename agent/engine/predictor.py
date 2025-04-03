from agent.engine.detector import YOLOPredictor
from agent.engine.filler import Filler


class Predictor:
    def __init__(self, args):
        self.args = args 


    def predict(self):
        pred_rlanes, pred_panels = self.stage1() 
        # print(pred_rlanes.boxes)
        # print(pred_panels.masks)
        self.stage2(pred_rlanes, pred_panels)




    def stage1(self):
        """
        Executes the first stage of the prediction pipeline, which involves 
        detecting road lanes and pavement panels in parallel using YOLO models.

        Returns:
            tuple: A tuple of `Results` objects containing two elements:
                - pred_rlanes: Predictions for road lanes.
                - pred_panels: Predictions for pavement panels.
        """
        rlane_model = YOLOPredictor(model_path=self.args.model_path_rlanes)
        panel_model = YOLOPredictor(model_path=self.args.model_path_panels)
        pred_rlanes = rlane_model.detect(input=self.args.source)
        pred_panels = panel_model.detect(input=self.args.source)
        return (pred_rlanes, pred_panels) 



    def stage2(self, pred_rlanes, pred_panels):
        filler = Filler(pred_rlanes, pred_panels)
        filler.fill()