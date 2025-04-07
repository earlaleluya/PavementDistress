from ultralytics.engine.results import Results
import torch 
import numpy as np 
import cv2
import torch
import numpy as np


class Prediction(Results):    
    """
    This class is designed to handle and manipulate the results of object detection and segmentation tasks. 
    It provides functionality to switch between different modes of results (lanes, panels, filled_panels), visualize masks, 
    update classification outputs, and generate cropped RGB masks with or without padding. 
    
    Attributes:
        result (dict): A dictionary containing results for 'lanes', 'panels', and 'filled_panels'.
        mode (str): The current mode of the prediction ('lanes', 'panels', or 'filled_panels').
        device (torch.device): The device (CPU or GPU) used for tensor operations. 
    """

    def __init__(self, result_lanes, result_panels, result_filled_panels, mode='filled_panels', device=None):
        self.result = {'lanes': result_lanes, 'panels': result_panels, 'filled_panels': result_filled_panels}
        self.mode = mode   
        self.switch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
   
    
    
    def switch_to_lanes(self):
        """
        Switches the mode of the object to 'lanes' and triggers the switch operation.
        """
        self.mode = 'lanes'
        self.switch()


    def switch_to_panels(self):
        """
        Switches the mode of the object to 'panels' and triggers the switch operation.
        """
        self.mode = 'panels'
        self.switch()



    def switch_to_filled_panels(self):
        """
        Switches the mode of the agent to 'filled_panels' and triggers the mode switch.
        """
        self.mode = 'filled_panels'
        self.switch()


    def switch(self):
        """
        Updates the attributes of the current object based on the selected mode.

        This method assigns various attributes such as bounding boxes, masks, 
        keypoints, object names, oriented bounding boxes (OBB), original image, 
        original shape, file path, probabilities, save directory, and processing 
        speed from the `result` dictionary using the current mode.

        Attributes:
            boxes (list): Bounding boxes for detected objects.
            masks (list): Segmentation masks for detected objects.
            keypoints (list): Keypoints for detected objects.
            names (list): Names or labels of detected objects.
            obb (list): Oriented bounding boxes for detected objects.
            orig_img (ndarray): Original input image.
            orig_shape (tuple): Shape of the original input image.
            path (str): File path of the input image.
            probs (list): Probabilities or confidence scores for detected objects.
            save_dir (str): Directory where results are saved.
            speed (dict): Processing speed metrics.
        """
        self.boxes = self.result[self.mode].boxes
        self.masks = self.result[self.mode].masks
        self.keypoints = self.result[self.mode].keypoints
        self.names = self.result[self.mode].names 
        self.obb = self.result[self.mode].obb 
        self.orig_img = self.result[self.mode].orig_img 
        self.orig_shape = self.result[self.mode].orig_shape 
        self.path = self.result[self.mode].path 
        self.probs = self.result[self.mode].probs
        self.save_dir = self.result[self.mode].save_dir
        self.speed = self.result[self.mode].speed 



    def plot_masks_only(self, alpha=0.5, gamma=0):
        """
        Plots masks on the original image with specified transparency and blending.

        Args:
            alpha (float, optional): Transparency factor for blending the masks with the original image. 
                                     Defaults to 0.5.
            gamma (float, optional): Scalar added to each sum during the blending process. Defaults to 0.

        Returns:
            numpy.ndarray: The original image with overlaid masks applied.
        """
        masks_xyn = [self.get_polygon(mask.xyn[0]) for mask in self.masks]
        h, w = self.orig_shape 
        masks_xy = [np.array([[round(x * w), round(y * h)] for x, y in mask]) for mask in masks_xyn]
        mask_3d = np.zeros_like(self.orig_img, dtype=np.uint8)
        for mask_xy, box in zip(masks_xy, self.boxes):
            new_mask = np.zeros_like(self.orig_img, dtype=np.uint8)
            overlay = new_mask.copy()
            overlay = cv2.fillPoly(overlay, [mask_xy], color=self.apply_colormap(int(box.cls)))
            new_mask = cv2.addWeighted(new_mask, alpha, overlay, 1 - alpha, gamma)
            mask_3d = cv2.bitwise_or(mask_3d, new_mask)
        masked_img = cv2.addWeighted(self.orig_img.copy(), alpha, mask_3d, 1 - alpha, gamma)
        return masked_img 
    

    
    def show_masks_only(self, alpha=0.5, gamma=0):
        """
        Displays the masked image generated by the `plot_masks_only` method in a window.

        Args:
            alpha (float, optional): Transparency level for the mask overlay. 
                                     Defaults to 0.5.
            gamma (float, optional): Scalar added to each sum during image blending. 
                                     Defaults to 0.
        """
        masked_img = self.plot_masks_only(alpha, gamma)
        cv2.imshow("Prediction", masked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




    def apply_colormap(self, index):
        """
        Maps an index to a corresponding RGB color tuple based on a predefined colormap.

        Args:
            index (int): The index to map to a color. Valid indices are:
                         - 0: Green (0, 255, 0)
                         - 1: Magenta (139, 0, 139)
                         - 2: Red (139, 0, 0)
                         - 3: Dark Blue (0, 0, 139)

        Returns:
            tuple: An RGB color tuple corresponding to the given index.

        Raises:
            KeyError: If the provided index is not in the predefined colormap.
        """
        colors = {0: (0, 255, 0), 1: (139, 0, 139), 2: (139, 0, 0), 3: (0, 0, 139)}
        color = colors[index]
        return color



    def update_conf_cls(self, classification_outputs):
        """
        Updates the confidence scores and class labels for the bounding boxes based on classification outputs.

        Args:
            list of list of dict: A list where each element corresponds to an image and contains a list of
            dictionaries with classification labels and their associated probabilities, sorted in descending
            order of probability.

            Example:
                [
                    [
                        {"label": "thincrack", "score": 0.9747543334960938},
                        {"label": "sealedcrack", "score": 0.021938998252153397},
                        {"label": "sealedpatch", "score": 0.0031411931850016117},
                        {"label": "nondistressed", "score": 0.00016548742132727057},
                    ],
                    ...
                ]

        Returns:
            None: Updated in-place only.

        Notes:
            - This function requires switching to the "filled_panels" mode as the classifier operates only in this mode.
            - The `self.boxes.data` tensor is updated in-place with the new confidence scores and class labels.
            - The `self.names` dictionary is used to map class labels to their corresponding numeric class IDs.
        """
        assert self.mode != 'lanes', "Switch mode to either `panels` or `filled_panels`."
        boxes_data = self.boxes.data.clone()
        for i, output in enumerate(classification_outputs):
            max_entry = max(output, key=lambda x: x['score'])
            label = max_entry['label']
            conf = max_entry['score']
            boxes_data[i, 4] = torch.tensor(conf).to(dtype=torch.float32, device=self.device)
            cls = next(key for key, value in self.result['filled_panels'].names.items() if value == label)       
            boxes_data[i, 5] = torch.tensor(cls).to(dtype=torch.float32, device=self.device)
        self.update(boxes=boxes_data)
        self.boxes = self.result[self.mode].boxes
        self.result[self.mode].names = self.result['filled_panels'].names
        self.names = self.result[self.mode].names




    def generate_cropped_rgb_masks(self, padding):
        """
        Generates cropped RGB masks based on the specified padding option.

        This method determines whether to include padding in the cropped RGB masks
        and delegates the task to the appropriate helper method.

        Args:
            padding (bool): If True, generates cropped RGB masks with padding.
                            If False, generates cropped RGB masks without padding.

        Returns:
            List or ndarray: The generated cropped RGB masks, with or without padding,
                             depending on the input argument.
        """
        if padding:
            return self.generate_cropped_rgb_masks_with_padding()
        else:
            return self.generate_cropped_rgb_masks_without_padding()



    def get_polygon(self, mask_points):
        points = torch.tensor(mask_points)  
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        top_left = points[torch.argmin(x_coords + y_coords)].tolist()  # Minimize x + y
        top_right = points[torch.argmin(-x_coords + y_coords)].tolist()  # Minimize -x + y
        bottom_left = points[torch.argmin(x_coords - y_coords)].tolist()  # Minimize x - y
        bottom_right = points[torch.argmin(-x_coords - y_coords)].tolist()  # Minimize -x - y
        return np.array([top_left, bottom_left, bottom_right, top_right]) # counter-clockwise



    def generate_cropped_rgb_masks_without_padding(self):
        masks_xyn = [self.get_polygon(mask.xyn[0]) for mask in self.masks]
        h, w = self.orig_shape 
        masks_xy = [np.array([[round(x * w), round(y * h)] for x, y in mask]) for mask in masks_xyn]
        cropped_rgb_masks = []
        for mask_xy, box_xy in zip(masks_xy, self.boxes.xyxy):
            orig_img = self.orig_img.copy()
            mask_3d = np.zeros_like(orig_img, dtype=np.uint8)
            cv2.fillPoly(mask_3d, [mask_xy], (255, 255, 255))
            rgb_mask = cv2.bitwise_and(orig_img, mask_3d)
            x_min, y_min, x_max, y_max = torch.round(box_xy).int()
            cropped_rgb_mask = rgb_mask[y_min:y_max, x_min:x_max]
            cropped_rgb_masks.append(cropped_rgb_mask)
        return cropped_rgb_masks






    def generate_cropped_rgb_masks_with_padding(self):
        masks_xyn = [self.get_polygon(mask.xyn[0]) for mask in self.masks]
        h, w = self.orig_shape 
        masks_xy = [np.array([[round(x * w), round(y * h)] for x, y in mask]) for mask in masks_xyn]
        cropped_rgb_masks = []
        max_height, max_width = 0, 0
        # Determine the maximum height and width
        for box_xy in self.boxes.xyxy:
            x_min, y_min, x_max, y_max = torch.round(box_xy).int()
            max_height = max(max_height, y_max - y_min)
            max_width = max(max_width, x_max - x_min)
        # Pad or resize masks to the maximum dimensions
        for mask_xy, box_xy in zip(masks_xy, self.boxes.xyxy):
            orig_img = self.orig_img.copy()
            mask_3d = np.zeros_like(orig_img, dtype=np.uint8)
            cv2.fillPoly(mask_3d, [mask_xy], (255, 255, 255))
            rgb_mask = cv2.bitwise_and(orig_img, mask_3d)
            x_min, y_min, x_max, y_max = torch.round(box_xy).int()
            cropped_rgb_mask = rgb_mask[y_min:y_max, x_min:x_max]
            padded_mask = np.zeros((max_height, max_width, cropped_rgb_mask.shape[-1]), dtype=cropped_rgb_mask.dtype)        
            y_offset = (max_height - cropped_rgb_mask.shape[0]) // 2
            x_offset = (max_width - cropped_rgb_mask.shape[1]) // 2
            padded_mask[y_offset:y_offset + cropped_rgb_mask.shape[0], x_offset:x_offset + cropped_rgb_mask.shape[1], :] = cropped_rgb_mask
            cropped_rgb_masks.append(padded_mask)
        return cropped_rgb_masks





    """
    Overriding functions
    """

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        return self.result[self.mode].update(boxes, masks, probs, obb)
    
    def cpu(self):
        return self.result[self.mode].cpu()
    
    def numpy(self):
        return self.result[self.mode].numpy()
    
    def cuda(self):
        return self.result[self.mode].cuda()
    
    def to(self, *args, **kwargs):
        return self.result[self.mode].to(*args, **kwargs)
    
    def new(self):
        return self.result[self.mode].new()

    def show(self, *args, **kwargs):
        return super().show(*args, **kwargs)
        
    def plot(self, conf=True, line_width=None, font_size=None, font="Arial.ttf", pil=False, img=None, im_gpu=None, kpt_radius=5, kpt_line=True, labels=True, boxes=True, masks=True, probs=True, show=False, save=False, filename=None, color_mode="class"):
        return self.result[self.mode].plot(conf, line_width, font_size, font, pil, img, im_gpu, kpt_radius, kpt_line, labels, boxes, masks, probs, show, save, filename, color_mode)
    
    def save(self, filename=None, *args, **kwargs):
        return self.result[self.mode].save(filename, *args, **kwargs)
    
    def verbose(self):
        return self.result[self.mode].verbose()
    
    def save_txt(self, txt_file, save_conf=False):
        return self.result[self.mode].save_txt(txt_file, save_conf)
    
    def save_crop(self, save_dir, file_name=...):
        return self.result[self.mode].save_crop(save_dir, file_name)
    
    def summary(self, normalize=False, decimals=5):
        return self.result[self.mode].summary(normalize, decimals)
    
    def to_df(self, normalize=False, decimals=5):
        return self.result[self.mode].to_df(normalize, decimals)
    
    def to_json(self, normalize=False, decimals=5):
        return self.result[self.mode].to_json(normalize, decimals)
    
    def to_csv(self, normalize=False, decimals=5, *args, **kwargs):
        return self.result[self.mode].to_csv(normalize, decimals, *args, **kwargs)
    
    def to_xml(self, normalize=False, decimals=5, *args, **kwargs):
        return self.result[self.mode].to_xml(normalize, decimals, *args, **kwargs)