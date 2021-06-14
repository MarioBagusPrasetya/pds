import cv2 as cv
import imutils as im

image = cv.imread('C://Users//ASUS//PycharmProjects//pythonProject1//Plate//D3426ZCP.jpg')
image = im.resize(image, width=500)
cv.imshow("Gambar Asli", image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gambar Grayscale", gray)

blur = cv.bilateralFilter(gray, 11, 17, 17)
cv.imshow("Bilateral Filter", blur)

edgeDet = cv.Canny(blur, 170, 200)
cv.imshow("Deteksi Tepi", edgeDet)

(cnts, _) = cv.findContours(edgeDet.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:30]

NumberPlateCnt = None

count = 0
for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

cv.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv.imshow("Plat Nomer Yang Terdeteksi", image)

cv.waitKey(0)
cv.destroyAllWindows()