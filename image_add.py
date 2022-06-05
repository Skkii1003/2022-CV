import cv2

img1 = cv2.imread("output/enet/1.png")
img2 = cv2.imread("output/enet/1_out.png")
img1 = cv2.resize(img1, (1024, 512))


re = cv2.addWeighted(img1, 0.6, img2, 0.4, 0, dtype=32)

cv2.imwrite("output/enet/1_co.png",re)
cv2.imshow("re", re)
cv2.waitKey(0)
cv2.destroyAllWindows()
