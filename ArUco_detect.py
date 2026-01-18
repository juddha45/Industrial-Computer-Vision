import cv2
import cv2.aruco as aruco
import numpy as np
from picamera.array import PiRGBArray # type: ignore
from picamera import PiCamera # type: ignore
import time

class ScrewPresenceInspector:
    def __init__(self, marker_dict_type=aruco.DICT_6X6_50):
        self.dictionary = aruco.getPredefinedDictionary(marker_dict_type)
        self.parameters = aruco.DetectorParameters_create() 
        self.marker_unit = 1.0 # Standard marker scale 
        self.parts_db = {
            0: [(2.25, 0.4), (1.0, 2.7), (0.0, 3.5)],
            1: [(-1.45, -0.3), (-1.4, 0.9), (1.3, -1.35), (1.3, 2.1)]
        }

    def get_transform_matrix(self, corners):
        marker_pts = np.array([
            [0, 0], 
            [self.marker_unit, 0],
            [self.marker_unit, self.marker_unit], 
            [0, self.marker_unit] ], dtype="float32") # ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])
        
        matrix, _ = cv2.findHomography(marker_pts, corners) 
        return matrix

    def check_presence_pro(self, gray_frame, px, py, inner_r=12, outer_r=22):
        inner_mask = np.zeros(gray_frame.shape, dtype="uint8")
        outer_mask = np.zeros(gray_frame.shape, dtype="uint8")
        
        cv2.circle(inner_mask, (px, py), inner_r, 255, -1)
        cv2.circle(outer_mask, (px, py), outer_r, 255, -1)
        cv2.circle(outer_mask, (px, py), inner_r + 2, 0, -1)
        
        screw_val = cv2.mean(gray_frame, mask=inner_mask)[0]
        paper_val = cv2.mean(gray_frame, mask=outer_mask)[0]
        
        if paper_val == 0: return False
        return (screw_val / paper_val) < 0.75 # Local contrast check 

    def start_inspection(self):
         
        with PiCamera() as camera:
            camera.resolution = (640, 480) 
            camera.framerate = 10
            raw_capture = PiRGBArray(camera, size=(640, 480)) # buffer to store raw pixel data: 640*480*3(RGB) bytes
            time.sleep(0.5)

            for frame_obj in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
                image = frame_obj.array
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

                if ids is not None:
                    for i, m_id in enumerate(ids.flatten()):
                        if m_id in self.parts_db:
                            m_corners = corners[i][0]
                            h_mat = self.get_transform_matrix( )
                            
                            for rel_x, rel_y in self.parts_db[m_id]:
                                target = np.array([[[rel_x, rel_y]]], dtype="float32")
                                screen_pt = cv2.perspectiveTransform(target, h_mat)[0][0]
                                sx = int(screen_pt[0])
                                sy = int(screen_pt[1])

                                if 0 <= sx < 640 and 0 <= sy < 480:
                                    is_present = self.check_presence_pro(gray, sx, sy)
                                    color = (0, 255, 0) if is_present else (0, 0, 255) 
                                    cv2.circle(image, (sx, sy), 20, color, 2)

                
                cv2.imshow("Presence Check", image)
                
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                raw_capture.truncate(0)

if __name__ == "__main__":
    inspector = ScrewPresenceInspector()
    inspector.start_inspection() 