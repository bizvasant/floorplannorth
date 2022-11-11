import cv2
import numpy as np

class img_contour:
    def __init__(self):
        pass

    def find_contour(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        cv2.bitwise_not(thresh, thresh)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        box = max(cnts, key=cv2.contourArea)
        left = tuple(box[box[:, :, 0].argmin()][0])
        right = tuple(box[box[:, :, 0].argmax()][0])
        return box, thresh

    def find_center(img, contour, thresh):
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 6)
#         cv2.drawContours(img, [contour], -1, (0,0,255), 6)
        mask = np.zeros(thresh.shape[:2], np.uint8)
        cv2.fillPoly(mask, pts=[np.asarray(contour)], color=(1))
        M = cv2.moments(mask, binaryImage=True)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return x, y, w, h, cx, cy

    def draw_contour(img, x, y, w, h, cx, cy):
        cv2.putText(img, "Q1", (cx + 30, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 4)
        cv2.putText(img, "Q2", (cx - 70, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 4)
        cv2.putText(img, "Q3", (cx - 70, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 4)
        cv2.putText(img, "Q4", (cx + 30, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 4)
#         cv2.line(img, (x, cy), (x + w, cy), (0, 0, 255), 3)
#         cv2.line(img, (cx, y), (cx, (y + h)), (0, 0, 255), 3)
        return img