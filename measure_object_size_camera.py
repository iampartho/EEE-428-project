import cv2
from object_detector import *
import numpy as np


#tolerance
tolerance =  1.63
# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)


# Load Object Detector
detector = HomogeneousBgDetector()

ip_filename = "3D Object - Input.mp4"
output_filename = "test.avi"

# Load Cap
cap = cv2.VideoCapture(ip_filename)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, 30, (320, 576))

while True:
    success, img = cap.read()

    if not success:
        print("Reached end of the feed !!! Exiting")
        break

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if corners:

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20 + tolerance
        #print(pixel_cm_ratio)

        contours = detector.detect_objects(img)


        # Draw objects boundaries
        for cnt in contours:
            # get area from the contour
            cnt_area = cv2.contourArea(cnt)
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            rect_area = w*h

            # difference in contour area and rect area
            diff_area = abs(cnt_area-rect_area)

            #print("\nDifference in area ", diff_area)


            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = round(w / pixel_cm_ratio , 1)
            object_height = round(h / pixel_cm_ratio,1)

            diff = abs(object_height - object_width)

            if diff_area >= 4000 and ip_filename.find("3D") == -1 :
                cv2.putText(img, "Diameter {} cm".format(object_width), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            else:
                # Display rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.polylines(img, [box], True, (255, 0, 0), 2)
                cv2.putText(img, "Width {} cm".format(object_width), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(img, "Height {} cm".format(object_height), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)



    cv2.imshow("Image", img)
    out.write(img)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
out.release()
cv2.destroyAllWindows()