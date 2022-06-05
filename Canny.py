import os
import cv2


def pre():
    img = cv2.imread("sample/image/1.png", 0)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(blur, 180, 250)
    cv2.imshow("gray", img)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyWindow()


if __name__ == '__main__':
    pre()
