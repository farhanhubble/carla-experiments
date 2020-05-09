from queue import Queue
from sklearn.cluster import KMeans

import cv2
import numpy as np


class SimpleLaneDector(object):
    """ Simple, straight-line edge detection based on Canny edge detection, 
    Hough transform, clustering and smoothing over frames.
    TODO: Document the ROI application better.
    TODO: More speed gains, can parallellize parts of the pipeline over 
    two halves of the image. 
    TODO: Try RANSAC for 
    https://bit.ly/ransac-kalman

    TODO: Temporal blurring 
    TODO: Guard against abrupt changes in lane position.
    TODO: Vanishing point stabilization.
    TODO: How can better SLAM help?
    TODO: how can depth (Lidar/Stereo vision) help?
    TODO: How can motion and optical flow help?
    TODO: How can semantic segmentation help?
    """

    def __init__(self):
        self.enable_edges = False
        self.enable_ROI = False


        self.q_left = Queue(maxsize=3)
        self.q_right = Queue(maxsize=3)


    def toggle_edge_detection(self):
        self.enable_edges = not self.enable_edges


    def toggle_ROI(self):
        self.enable_ROI = not self.enable_ROI


    def cart2pol(self, x1,y1,x2,y2):
        """Find polar representation given end points of a line segment.
        Equation of the line: y-y1 = ((y2-y1)/(x2-x1))*(x-x1)
        Compare with the standard equation: (y1-y2)*x + (x2-x1)*y - y1(x2-x1)+x1(y2-y1) = 0 cf. Ax+By+C=0
        Normal distance from origin: rho = (A.0 + B.0 +C)/sqrt(A**2+B**2) = (-y1*(x2-x1)+x1*(y2-y1))/sqrt((y2-y1)**2 + (x2-x1)**2)
        Slope of the normal: theta = pi/2-arctan((y2-y1)/(x2-x1))
        """
        rho = (-y1*(x2-x1)+x1*(y2-y1))/np.sqrt((y2-y1)**2+(x2-x1)**2)
        theta = np.arctan((y2-y1)/(x2-x1)) - np.pi/2 if x2 != x1 else 0 

        return rho,theta


    def normalize_params(self, rho, theta, w, h):
        """Normalize parameters of a line in to the range [0,1]
        """
        normalized_theta = (theta + np.pi) / np.pi
        normalized_rho = rho/(np.sqrt(w**2+h**2)) + 1

        return normalized_rho, normalized_theta


    def cluster_edges(self,lines,w,h):
        cluster_ids = []
        x_center = w//2

        for [x1, _, x2, _] in lines:
            if x1 <= x_center and x2 <= x_center:
                cluster_ids.append(0)
            elif x1 > x_center and x2 > x_center:
                cluster_ids.append(1)

            else:
                if abs(x1 - x_center) > abs(x2 - x_center):
                    if x2 <= x_center:
                        cluster_ids.append(0)
                    else:
                        cluster_ids.append(1)

                else:
                    if x1 <= x_center:
                        cluster_ids.append(0)
                    else:
                        cluster_ids.append(1)

        return cluster_ids 



    def filter_by_slope(self, edges, min_slope, max_slope):

        assert min_slope <= max_slope,\
            print(f'min_slope {min_slope} is not <= max_slope {max_slope}')

        def _get_theta(edge):
            _, theta = self.cart2pol(*edge)
            return theta

        if edges is not None:
            # print('slopes:',list(slopes))
            # print(f'min slope {min_slope} max slope {max_slope}')
            return list(filter(lambda e : min_slope <= _get_theta(e) <= max_slope, edges))




    def fit_curve(self, lines):
        flattend_list_lines = [z for line_and_cluster in lines for line, cluster_id in line_and_cluster for z in line]
        list_xs = flattend_list_lines[0: len(flattend_list_lines): 2]
        list_ys = flattend_list_lines[1: len(flattend_list_lines): 2]

        # Interpolate all the points
        coeffs = np.polyfit(list_xs, list_ys, deg=1)

        return coeffs


    def get_hough_lines(self,*args,**kwargs):
        lines =  cv2.HoughLinesP(*args, **kwargs)
        if list:
            lines = [line for [line] in lines]
        return lines
    """
    TODO: Remove global queues.
    TODO: 
    """

    def detect_lanes(self,img):
        img_lines = np.zeros_like(img)

        # Region of interest is enabled
        if self.enable_ROI:
            height, width = img.shape[0], img.shape[1]
            center_y = int(width // 2)
            extent_bottom = (center_y-int(width*0.45), center_y+int(width*0.45))
            extent_top    = (center_y-int(width*0.08), center_y+int(width*0.08))


            ROI_xs = [extent_top[0], extent_bottom[0], extent_bottom[1], extent_top[1]]
            ROI_ys = [int(0.6*height), height, height, int(0.6*height)]
            ROI_segment_starts = list(zip(ROI_xs, ROI_ys))
            ROI_segment_ends   = ROI_segment_starts[1:] + [ROI_segment_starts[0]]
            ROI_segments = list(zip(ROI_segment_starts, ROI_segment_ends))

            ROI_mask = np.zeros_like(img)
            cv2.fillPoly(ROI_mask, np.array([ROI_segment_starts], dtype=np.int32), color=[1, 1, 1])

            img_with_ROI = cv2.copyTo(img, mask=ROI_mask)

            img_low_pass = cv2.GaussianBlur(img_with_ROI, (3, 3), 0)
            img_canny = cv2.Canny(img_low_pass, 100, 200)

            for [start, end] in ROI_segments:
                cv2.line(img_canny, start, end, (0, 0, 0), 3)

            lines = self.get_hough_lines(image=img_canny, rho=3,theta=np.pi/36.0,
                    lines=None, threshold=20, minLineLength=10, maxLineGap=3)

            # Remove horizontal lines
            theta_threshold = np.pi/3
            filtered_lines = self.filter_by_slope(lines, -np.pi, -np.pi+theta_threshold)
            filtered_lines += self.filter_by_slope(lines, 0-theta_threshold, 0)

            lines = filtered_lines
            _colors= [[0,0,255],       # red
            [0,255,0]]               # orange

            if lines is not None:
                cluster_ids = self.cluster_edges(lines, width, height)

            for [x1, y1, x2, y2],cluster_id in zip(lines,cluster_ids):
                cv2.line(img_lines, (x1, y1), (x2, y2), _colors[cluster_id], 2)

            left_lines = []
            right_lines = []

            for index in range(len(cluster_ids)):
                if cluster_ids[index] == 0:
                    left_lines.append((lines[index], cluster_ids[index]))
                else:
                    right_lines.append((lines[index], cluster_ids[index]))

            if not self.q_right.full():
                self.q_right.put(right_lines)
            else:
                self.q_right.get()
                self.q_right.put(right_lines)

            if not self.q_left.full():
                self.q_left.put(left_lines)
            else:
                self.q_left.get()
                self.q_left.put(left_lines)

            left_edges  = list(self.q_left.queue)
            right_edges = list(self.q_right.queue)

            coeffs_left_lines = self.fit_curve(left_edges)
            coeffs_right_lines = self.fit_curve(right_edges)

            for coeffs in [coeffs_left_lines, coeffs_right_lines]:
                line = np.poly1d(coeffs)
                y_min = line(0)
                y_max = line(width)
                cv2.line(img_lines, (0, int(y_min)), (width, int(y_max)), (255, 255, 0), 2)

            for [start, end] in ROI_segments:
                cv2.line(img_lines, start, end, (255, 0, 0), 5)

            img_lines = cv2.copyTo(img_lines, mask=ROI_mask)

            res = cv2.addWeighted(img, 1, img_lines, 1, 0)

        else:
            img_low_pass = cv2.GaussianBlur(img, (3, 3), 0)
            img_canny = cv2.Canny(img_low_pass, 100, 200)
            lines = self.get_hough_lines(image=img_canny, rho=3, theta=np.pi / 36.0,
                                    lines=None, threshold=20, minLineLength=3, maxLineGap=3)
            for [x1, y1, x2, y2] in lines:
                cv2.line(img_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)


            res = cv2.addWeighted(img, 1, img_lines, 1, 0)

        if lines is not None:
            print('Plotting ', len(lines), ' lines.')
        return res
