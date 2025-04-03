import torch
import numpy as np 
from math import sqrt



class Filler:
    def __init__(self, pred_rlanes, pred_panels):
        self.pred_rlanes = pred_rlanes
        self.pred_panels = pred_panels
        self.image = pred_rlanes.orig_img.copy()



    def fill(self):
        polygons_rlanes = self.get_sort_polygons(self.pred_rlanes)
        polygons_panels = self.get_sort_polygons(self.pred_panels)
        lane_curves = self.get_road_lane_curves(polygons_rlanes)
        grouped_polygons_panels = self.group_by_curves(polygons_panels, lane_curves)
        filled_polygons_panels = self.resolve_missing_panels(grouped_polygons_panels)
        print(filled_polygons_panels.shape)




    def restructure_result(self):
        pass 




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
        result_panels = self.fill_missing_panels(result_panels)
        return result_panels



    def fill_missing_panels(self, panels):
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
        for i in range(rows):
            for j in range(cols):
                if np.all(panels[i, j] == 0):  # Check if the entire panel is missing (all zeros)
                    selected_points = []
                    # Check neighbors and add specific points accordingly
                    if is_valid_top_neighbor(panels, i, j, rows, cols): 
                        selected_points.append(panels[i, j - 1][[1, 2]]) # Bottom left and bottom right
                    if is_valid_bottom_neighbor(panels, i, j, rows, cols):  
                        selected_points.append(panels[i, j + 1][[0, 3]]) # Top left and top right
                    if is_valid_left_neighbor(panels, i, j, rows, cols): 
                        selected_points.append(panels[i - 1, j][[2, 3]]) # Bottom right and top right
                    if is_valid_right_neighbor(panels, i, j, rows, cols):  # Right neighbor
                        selected_points.append(panels[i + 1, j][[0, 1]]) # Top left and bottom left
                    # Select only one entry if multiple points were added on each side
                    if selected_points:
                        selected_points = np.vstack(selected_points)
                        x_coords = selected_points[:, 0]
                        y_coords = selected_points[:, 1]
                        x_min, y_min = np.min(x_coords), np.min(y_coords)
                        x_max, y_max = np.max(x_coords), np.max(y_coords)
                        panels[i, j] = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
        return panels



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

