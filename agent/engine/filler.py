import torch
import numpy as np 
import cv2
from math import sqrt
from agent.engine.predictions import Prediction
from ultralytics.engine.results import Results
import json 
from pathlib import Path




class Filler:
    """
    The `Filler` class is responsible for processing and filling missing panels in road lane 
    and panel prediction data. It provides methods to sort, group, and resolve missing panels 
    by leveraging geometric relationships between road lanes and panels.
    
    Attributes:
        pred_rlanes (object): The predicted road lane data, containing masks and other attributes.
        pred_panels (object): The predicted panel data, containing masks and other attributes.
        image (numpy.ndarray): A copy of the original image associated with the predictions.
        args (Namespace): A namespace object containing configuration arguments.
        device (torch.device): The device (CPU or GPU) used for tensor computations.
    
    Example:
        # Assuming `pred_rlanes`, `pred_panels`, and `args` are already defined
        >>> filler = Filler(pred_rlanes, pred_panels, args)
        >>> prediction = filler.fill()
        >>> print(prediction)
        >>> prediction.show()
    """


    def __init__(self, pred_rlanes, pred_panels, args):
        self.pred_rlanes = pred_rlanes
        self.pred_panels = pred_panels
        self.image = pred_rlanes.orig_img.copy()
        self.args = args
        self.device = torch.device(args.device)
        


    def fill(self):
        """
        Fills missing panels in the prediction data by processing road lane and panel polygons.

        This method performs the following steps:
        1. Sorts the predicted road lane and panel polygons.
        2. Extracts road lane curves from the sorted road lane polygons.
        3. Groups the panel polygons by their corresponding road lane curves.
        4. Resolves missing panels within the grouped panel polygons.
        5. Creates a result object containing the filled panel polygons and road lane polygons.

        Returns:
            Prediction (`Results` object): An object containing the original road lane and panel predictions,
                                           along with the filled panel predictions.
        """
        polygons_rlanes = self.get_sort_polygons(self.pred_rlanes)
        polygons_panels = self.get_sort_polygons(self.pred_panels)
        lane_curves = self.get_road_lane_curves(polygons_rlanes)
        grouped_polygons_panels = self.group_by_curves(polygons_panels, lane_curves)      
        # grouped_polygons_panels = self.debug_test(grouped_polygons_panels) # TODO remove
        filled_polygons_panels = self.resolve_missing_panels(grouped_polygons_panels)
        result_filled_panels = self.create_panels_result_obj(filled_polygons_panels, polygons_rlanes)
        prediction = Prediction(
            result_lanes=self.pred_rlanes, 
            result_panels=self.pred_panels,
            result_filled_panels=result_filled_panels,
            device=self.device
        )
        return prediction



    def create_panels_result_obj(self, filled_polygons_panels, polygons_rlanes):
        """
        Creates a `Results` object for the filled panels using the provided polygons.

        Args:
            filled_polygons_panels (numpy.ndarray): A 4D array of shape 
                (n_lanes, max_n_panels, 4, 2) representing the polygons for each panel.
            polygons_rlanes (list of numpy.ndarray): A list of polygons representing 
                road lanes.

        Returns:
            Results: A `Results` object containing the updated boxes and masks data.
        """
        filled_panels = Results(
            orig_img=self.pred_panels.orig_img,
            path=self.pred_panels.path,
            names=self.get_classifier_id2label(),
        )
        masks_data = self.create_masks_data(filled_polygons_panels, polygons_rlanes)
        filled_panels.update(masks=masks_data)
        boxes_data = self.create_boxes_data(filled_panels.masks.xy)
        filled_panels.update(boxes_data)
        return filled_panels



    def get_classifier_id2label(self):
        """
        Retrieves the mapping of classifier IDs to labels from the configuration file.

        Returns:
            dict: A dictionary where keys are classifier IDs (as strings) and values are
            their corresponding labels.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the configuration file is not a valid JSON.
            KeyError: If the 'id2label' key is missing in the configuration file.
        """
        vit_cfg_file = Path(self.args.model_path_classifier) / 'config.json'
        with open(vit_cfg_file, 'r') as f:
            vit_cfg = json.load(f)
        return {int(k): v for k, v in vit_cfg['id2label'].items()}
        


    def create_boxes_data(self, masks_xy):
        """
        Generates a tensor containing bounding box data for a given list of masks.

        Args:
            masks_xy (list of torch.Tensor): A list of tensors where each tensor represents 
                the coordinates of a mask. The list contains `n_lanes * max_n_panels` number of masks.

        Returns:
            torch.Tensor: A tensor of shape (len(masks_xy), 6) where each row contains the bounding 
                box data for a corresponding mask. The first four columns represent the bounding box 
                coordinates, and the remaining columns (for conf, cls, respectively) are initialized to zero.
        """
        boxes_data = torch.zeros((len(masks_xy), 6), device=self.device, dtype=torch.float32)
        for i, mask_xy in enumerate(masks_xy):
            box = torch.from_numpy(np.array(self.polygon2box(mask_xy))).to(device=self.device, dtype=torch.float32)
            box_data = box.view(1, 4)
            boxes_data[i, 0:4] = box_data
        return boxes_data
        



    def create_masks_data(self, filled_polygons_panels, polygons_rlanes): 
        """
        Creates mask data for filled polygons and road lanes.

        Args:
            filled_polygons_panels (numpy.ndarray): A 4D array of shape 
                (n_lanes, max_n_panels, n_points, n_coords) representing the 
                polygons for each panel.
            polygons_rlanes (list of numpy.ndarray): A list of polygons representing 
                road lanes.

        Returns:
            torch.Tensor: A 3D tensor of shape (n_lanes * max_n_panels, img_height, img_width) 
                containing the mask data for each panel.
        """
        img_height, img_width = self.pred_rlanes.orig_shape   
        n_lanes, max_n_panels, n_points, n_coords = filled_polygons_panels.shape        
        masks_data = torch.zeros((n_lanes * max_n_panels, img_height, img_width), device=self.device, dtype=torch.float32)
        # Loop for each panel
        reshaped_polygons = filled_polygons_panels.reshape(n_lanes * max_n_panels, n_points, n_coords)
        for i, polygon in enumerate(reshaped_polygons):
            # Create mask for road lane
            mask_rlane = np.zeros((img_height, img_width), dtype=np.uint8)
            polygon_rlane = polygons_rlanes[i // max_n_panels]
            mask_rlane = cv2.fillPoly(mask_rlane, [polygon_rlane], 255)            
            # Create mask for panel
            mask_panel = np.zeros((img_height, img_width), dtype=np.uint8)
            box = np.array(self.polygon2box(polygon)).reshape(-1) 
            x_min, y_min, x_max, y_max = np.round(box).astype(np.int32)
            box_width = (x_max - x_min)
            x1, x2 = max(0, x_min - box_width), max(img_width, x_max + box_width)
            mask_panel = cv2.rectangle(mask_panel, (x1, y_min), (x2, y_max), 255, thickness=-1)                 
            # Combine masks by intersection
            mask = cv2.bitwise_and(mask_rlane, mask_panel)        
            mask_data = mask.astype(np.float32)  / 255.0
            masks_data[i] = torch.from_numpy(mask_data).to(device=self.device, dtype=torch.float32)        
        return masks_data
    
         



    def debug_test(self, panels):
        """
        Filters out specific panel indices from a 2D array of panels and returns a ragged array.

        Args:
            panels (numpy.ndarray): A 2D array representing panels, where the shape is (n_lanes, n_panels).

        Returns:
            list: A ragged array (list of numpy arrays) where each sub-array corresponds to a lane
                  with specific panel indices removed. The sub-arrays are of dtype `object`.

        Notes:
            The indices to be removed are hardcoded as:
            [(2, 4), (3, 4), (4, 4), (3, 3), (3, 5)].
        """
        indices_to_remove = [(2,4), (3,4),(4,4),(3,3), (3,5)]
        ragged_array = []
        n_lanes, n_panels = panels.shape[:2]  
        for i in range(n_lanes):
            lane_panels = []
            for j in range(n_panels):
                if (i, j) not in indices_to_remove:
                    lane_panels.append(panels[i, j])
            ragged_array.append(np.array(lane_panels, dtype=object)) 
        return ragged_array



    def debug_show(self, panels):
        """
        Visualizes the given panels by overlaying them on the image and displaying the result.

        This method takes a set of panels, reshapes them into polygons, and fills them with green color
        on a copy of the image. Each frame is displayed one at a time, and the user can press any key 
        to proceed to the next frame.

        Args:
            panels (numpy.ndarray): A numpy array of shape (N, 8) representing the coordinates of the 
                                    panels. Each panel is expected to have 4 points (x, y) flattened 
                                    into a single array.
        """
        canvas = self.image.copy()
        reshaped_polygons = panels.reshape(-1, 4, 2)
        for poly in reshaped_polygons:
            cv2.fillPoly(canvas, [poly], (0,255,0))
            cv2.imshow("frame", canvas)
            cv2.waitKey(0)
        cv2.destroyAllWindows()




    def resolve_missing_panels(self, panels_with_missing):
        """
        Resolves missing panels in a set of lanes by getting the neighbors' coordinates 
        and ensuring all lanes have the same number of panels.

        Args:
            panels_with_missing (list of list of ndarray): A list where each element 
                represents a lane, and each lane contains a list of panels. Each panel 
                is represented as a 4x2 ndarray of integer coordinates.

        Returns:
            ndarray: A 4D array of shape (n_lanes, max_n_panels, 4, 2) where `n_lanes` 
                is the number of lanes, `max_n_panels` is the maximum number of panels 
                in any lane, and each panel is represented as a 4x2 ndarray of integer 
                coordinates. Missing panels are filled with the neighbors' coordinates.
        """
        n_panels_per_lane = [len(panels) for panels in panels_with_missing]
        max_n_panels = max(n_panels_per_lane)
        n_lanes = len(panels_with_missing)
        ref_idx = np.argmax(n_panels_per_lane) 
        result_panels = np.zeros((n_lanes, max_n_panels, 4, 2), dtype=np.int32)
        for i, n_panels in enumerate(n_panels_per_lane):
            if n_panels==max_n_panels:  # Copy directly the complete group of panels
                result_panels[i] = panels_with_missing[i]
                ref_idx = i
            else:   # Insert empty polygons for all missing panels at their appropriate positions
                result_panels[i] = self.insert_empty_polygons(panels_with_missing[i], panels_with_missing[ref_idx])
        # Fill missing panels based on the neighbors' coordinates
        result_panels, force_resolve = self.fill_missing_panels(result_panels)
        if force_resolve:
            result_panels, force_resolve = self.fill_missing_panels(result_panels, force_resolve)
        return result_panels



    # TODO update docstring
    def fill_missing_panels(self, panels, force_resolve=False):
        """
        Fills missing panels in a 4D numpy array based on the coordinates of 
        neighboring panels.

        A panel is considered missing if all its values are zeros. The function 
        checks the neighboring panels (top, bottom, left, right) to gather 
        coordinate points and uses these points to estimate the missing panel's 
        coordinates.

        Args:
            panels (np.ndarray): A 4D array of shape (rows, cols, 4, 2), where each panel 
            is represented by four corner points in 2D space (x, y). Missing panels are 
            represented by all-zero entries.

        Returns:
            panels (np.ndarray): The updated 4D array with missing panels filled based on 
            neighboring panels' coordinates.

        Notes:
        - If a missing panel has no valid neighbors, it remains unchanged.
        - Neighboring panels are checked in the following order: top, bottom, left, right.
        """
        def is_valid_top_neighbor(panels, i, j, rows, cols):
            if (i in list(range(rows))) and (j in list(range(1, cols))):
                return not np.all(panels[i, j - 1] == 0)
            return False 
        def is_valid_bottom_neighbor(panels, i, j, rows, cols):
            if (i in list(range(rows))) and (j in list(range(cols - 1))):
                return not np.all(panels[i, j + 1] == 0)
            return False 
        def is_valid_left_neighbor(panels, i, j, rows, cols):
            if (i in list(range(1, rows))) and (j in list(range(cols))):
                return not np.all(panels[i - 1, j] == 0)
            return False
        def is_valid_right_neighbor(panels, i, j, rows, cols):
            if (i in list(range(rows - 1))) and (j in list(range(cols))):
                return not np.all(panels[i + 1, j] == 0)
            return False
        rows, cols, _, _ = panels.shape
        non_empty = True
        for i in range(rows):
            for j in range(cols):
                if np.all(panels[i, j] == 0):  # Check if the entire panel is missing (all zeros)
                    non_empty = False
                    top_left_points = []
                    bot_left_points = []
                    bot_right_points = []
                    top_right_points = []
                    # Check neighbors and add specific points accordingly
                    if is_valid_top_neighbor(panels, i, j, rows, cols): 
                        top_left_points.append(panels[i, j - 1][1])     # top neighbor's bottom left 
                        top_right_points.append(panels[i, j - 1][2])    # top neighbor's bottom right
                    if is_valid_bottom_neighbor(panels, i, j, rows, cols):  
                        bot_left_points.append(panels[i, j + 1][0])     # bottom neighbor's top left
                        bot_right_points.append(panels[i, j + 1][3])    # bottom neighbor's top right
                    if is_valid_left_neighbor(panels, i, j, rows, cols):
                        bot_left_points.append(panels[i - 1, j][2])    # left neighbor's bottom right
                        top_left_points.append(panels[i - 1, j][3])    # left neighbor's top right
                    if is_valid_right_neighbor(panels, i, j, rows, cols):  # Right neighbor
                        top_right_points.append(panels[i + 1, j][0])     # right neighbor's top left
                        bot_right_points.append(panels[i + 1, j][1])     # right neighbor's bottom left
                    # Get the point from top_left_points where it has the minimum x-coordinate and y-coordinate
                    if top_left_points:
                        top_left_points = np.array(top_left_points)
                        x_min = np.min(top_left_points[:, 0])  
                        y_min = np.min(top_left_points[:, 1])  
                        top_left_points = np.array([[x_min, y_min]])
                    # Get the point from bot_left_points where it has the minimum x-coordinate and maximum y-coordinate
                    if bot_left_points:
                        bot_left_points = np.array(bot_left_points)
                        x_min = np.min(bot_left_points[:, 0])  
                        y_max = np.max(bot_left_points[:, 1])  
                        bot_left_points = np.array([[x_min, y_max]])
                    # Get the point from bot_right_points where it has the maximum x-coordinate and maximum y-coordinate
                    if bot_right_points:    
                        bot_right_points = np.array(bot_right_points)   
                        x_max = np.max(bot_right_points[:, 0])  
                        y_max = np.max(bot_right_points[:, 1])  
                        bot_right_points = np.array([[x_max, y_max]])
                    # Get the point from top_right_points where it has the maximum x-coordinate and minimum y-coordinate    
                    if top_right_points:
                        top_right_points = np.array(top_right_points)
                        x_max = np.max(top_right_points[:, 0])  
                        y_min = np.min(top_right_points[:, 1])  
                        top_right_points = np.array([[x_max, y_min]])
                    # Concatenate the points to form the polygon
                    if len(top_left_points) == 1 and len(bot_left_points) == 1 and len(bot_right_points) == 1 and len(top_right_points) == 1:
                        panels[i, j] = np.vstack([top_left_points, bot_left_points, bot_right_points, top_right_points])
                    elif force_resolve:
                        # TODO debug this part: @debug_test(), set indices_to_remove = [(2,4), (3,4),(4,4),(3,3), (3,5), (0,7), (2,5)] 
                        valid_points = [arr for arr in [top_left_points, bot_left_points, bot_right_points, top_right_points] if len(arr) > 0]
                        if valid_points:  # Ensure there are valid points
                            points = np.vstack(valid_points)
                            [[x_min, y_min], [y_min, y_max]] = self.polygon2box(points)
                            panels[i, j] = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
        # Do recursion if missing panels are found in between     
        if non_empty:
            return panels, False
        else:
            try:
                return self.fill_missing_panels(panels)
            except RecursionError:
                return panels, True



    def insert_empty_polygons(self, col_panels_with_missing, ref_col_panels, dim_thresh=0.2):
        """
        Inserts empty polygons into a collection of panels with missing polygons 
        based on a reference collection of panels.

        This function aligns polygons from `col_panels_with_missing` to the 
        corresponding positions in `ref_col_panels` by comparing their vertical 
        positions (y-coordinates). If a match is found within a specified 
        threshold, the polygon is inserted into the result array at the 
        corresponding index. Empty slots in the result array remain as zeros.

        Args:
            col_panels_with_missing (list of ndarray): A list of polygons (each 
                represented as a 4x2 array of coordinates) with some missing 
                polygons.
            ref_col_panels (list of ndarray): A reference list of polygons 
                (each represented as a 4x2 array of coordinates) used to determine 
                the positions of missing polygons.
            dim_thresh (float, optional): A threshold value for determining 
                whether two polygons are vertically aligned based on their 
                y-coordinates. Defaults to 0.2.

        Returns:
            ndarray: A 3D array of shape (max_n_panels, 4, 2) where `max_n_panels` 
                is the number of polygons in `ref_col_panels`. The array contains the 
                aligned polygons from `col_panels_with_missing`, with zeros for 
                missing polygons.
        """
        max_n_panels = len(ref_col_panels)
        col_result_panels = np.zeros((max_n_panels, 4, 2), dtype=np.int32)
        for i, polygon in enumerate(col_panels_with_missing):
            [[_, y_min], [_, y_max]] = self.polygon2box(polygon)    
            for j, ref_polygon in enumerate(ref_col_panels):
                [[_, ref_y_min], [_, _]] = self.polygon2box(ref_polygon)    
                if abs(y_min - ref_y_min) < abs(y_max - y_min) * dim_thresh:
                    col_result_panels[j] = col_panels_with_missing[i]
                    break
        return col_result_panels




    def polygon2box(self, polygon):
        """
        Converts a polygon represented as a set of points into a bounding box.

        Args:
            polygon (numpy.ndarray): A 2D array of shape (N, 2) where N is the number
                of points in the polygon. Each row represents a point as [x, y].

        Returns:
            list: A list containing two points that define the bounding box:
                [[x_min, y_min], [x_max, y_max]].
        """
        x_min, y_min = np.min(polygon, axis=0)  
        x_max, y_max = np.max(polygon, axis=0)  
        return [[x_min, y_min], [x_max, y_max]]



    def group_by_curves(self, polygons_panels, lane_curves):
        """
        Groups and sorts panel polygons into lanes based on their association with lane curves.

        Args:
            polygons_panels (np.ndarray): An array of polygons representing panels.
            lane_curves (np.ndarray): A list of lane curves used to determine lane associations.

        Returns:
            np.ndarray: A 2D `ragged` array where each row corresponds to a lane, and each lane contains
                        polygons sorted from top to bottom.
        """
        lane_indices = self.get_lane_indices(polygons_panels, lane_curves)
        sorted_polygons_panels = self.sort_by_curves(polygons_panels, lane_indices)
        n_polygons_per_lanes = [np.count_nonzero(lane_indices == n) for n in range(len(lane_curves))]
        split_arrays = np.split(sorted_polygons_panels, np.cumsum(n_polygons_per_lanes)[:-1])
        sorted_panels = np.array(split_arrays, dtype=object)
        sorted_panels = np.array([self.sort_polygons_by_y(col) for col in sorted_panels], dtype=object) 
        return sorted_panels




    def sort_polygons_by_y(self, polygons):
        """
        Sorts a list of polygons based on the y-coordinate of their first vertex.

        Args:
            polygons (list of numpy.ndarray): A list of polygons, where each polygon 
                is represented as a numpy array of vertices. 

        Returns:
            list of numpy.ndarray: The input list of polygons sorted in ascending 
            order by the y-coordinate of their first vertex.
        """
        return sorted(polygons, key=lambda poly: poly[0, 1])  




    def sort_by_curves(self, polygons_panels, lane_indices):
        """
        Sorts a list of polygon panels based on their corresponding lane curves.

        Args:
            polygons_panels (list of numpy.ndarray): A collection of polygon panels to be sorted.
            lane_indices (numpy.ndarray): A collection of lane indices used for sorting.

        Returns:
            numpy.ndarray: The sorted polygon panels based on the lane curves.
        """
        sorted_indices = np.argsort(lane_indices)
        sorted_polygons_panels = np.array(polygons_panels)[sorted_indices]
        return sorted_polygons_panels


    
    def get_lane_indices(self, polygons_panels, lane_curves):
        """
        Determines the indices of the closest lane curves for each panel in a set of polygon panels.

        Args:
            polygons_panels (list of ndarray): A list of polygon panels, where each panel is represented 
                                               as an array of points.
            lane_curves (numpy.ndarray): An array of lane curves, where each curve is represented as an 
                                         array of points.

        Returns:
            numpy.ndarray: An array of indices, where each index corresponds to the lane curve that is closest 
                           to the respective polygon panel.
        """
        indices = []
        for panel in polygons_panels:
            distance_per_panel = np.array([self.distance_from_slope(curve, panel[0]) for curve in lane_curves])
            lane_idx = np.argmin(distance_per_panel) 
            indices.append(lane_idx)
        return np.array(indices)
    


    def distance_from_slope(self, curve, top_left_point):
        """
        Calculate the perpendicular distance from a point to a line represented by its slope.

        Args:
            curve (tuple): A tuple (a, b, c) representing the coefficients of the line equation ax + by + c = 0.
            top_left_point (tuple): A tuple (x, y) representing the coordinates of the point.

        Returns:
            float: The perpendicular distance from the point to the line.
        """
        a, b, c = curve
        x, y = top_left_point
        numerator = abs(a * x + b * y + c)
        denum =  sqrt(a * a + b * b)
        return (numerator / denum)



    def get_road_lane_curves(self, polygons):
        """
        Computes the coefficients of the linear equations representing the curves 
        (lines) for a list of road lane polygons.

        Args:
            polygons (list of 2D arrays): A list of polygons, where each polygon 
                is represented as a 2D array of four points 
                [top_left, bottom_left, bottom_right, top_right]. 
                Each point contains the (x, y) coordinates.

        Returns:
            numpy.ndarray: A 2D array where each row contains the coefficients 
                           [a, b, c] of the line equation for a polygon.
        """
        curves = []
        for polygon in polygons:
            x_tl, y_tl = polygon[0]
            x_bl, y_bl = polygon[1]
            a =  y_bl - y_tl
            b =  x_tl - x_bl
            c = (x_bl * y_tl) - (x_tl * y_bl)
            curves.append([a, b, c])
        return np.array(curves)



    def get_sort_polygons(self, predictions):
        """
        Extracts and sorts polygons from the given predictions.

        Args:
            predictions (object): An object containing prediction data, 
                where `predictions.masks` is expected to be an iterable of masks, 
                and each mask has an `xy` attribute containing polygon coordinates.

        Returns:
            list: A list of polygons sorted based on their proximity to a reference point.
        """
        polygons = [self.get_polygon(mask.xy[0]) for mask in predictions.masks]
        polygons = self.sort(polygons, ref_point=(0, self.image.shape[1]))
        return polygons 
    

    
    def get_polygon(self, mask_points):
        """
        Computes the four corners of a polygon (top-left, top-right, bottom-left, bottom-right)
        from a set of mask points and returns them in a counter-clockwise order.

        Args:
            mask_points (torch.tensor): A tensor array of 2D points representing the mask.
                                        Each point is expected to be in the format [x, y].

        Returns:
            numpy.ndarray: A 2D array containing the four corner points of the polygon in 
                           counter-clockwise order: [top-left, bottom-left, bottom-right, top-right].
                           Each point is represented as [x, y].
        """
        points = torch.tensor(mask_points)  
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        top_left = points[torch.argmin(x_coords + y_coords)].int().tolist()  # Minimize x + y
        top_right = points[torch.argmin(-x_coords + y_coords)].int().tolist()  # Minimize -x + y
        bottom_left = points[torch.argmin(x_coords - y_coords)].int().tolist()  # Minimize x - y
        bottom_right = points[torch.argmin(-x_coords - y_coords)].int().tolist()  # Minimize -x - y
        return np.array([top_left, bottom_left, bottom_right, top_right]) # counter-clockwise
    




    def sort(self, polygons, ref_point):
        """
        Sorts a list of polygons based on their distance to a reference point.

        Args:
            polygons (list of numpy.ndarray): A list of polygons, where each polygon is 
                represented as a 2D numpy array of shape (N, 2), with N being the number 
                of vertices. Each row in the array represents a vertex as [x, y].
            ref_point (tuple): A tuple (x, y) representing the reference point.

        Returns:
            list of numpy.ndarray: The input list of polygons sorted in ascending order 
            of their centroid's distance to the reference point.
        """
        def distance_to_center(polygon):
            polygon_centroid_x = np.mean(polygon[:, 0])
            polygon_centroid_y = np.mean(polygon[:, 1])
            distance = np.sqrt((polygon_centroid_x - ref_point[0])**2 + (polygon_centroid_y - ref_point[1])**2)
            return distance
        sorted_polygons = sorted(polygons, key=distance_to_center)
        return sorted_polygons

