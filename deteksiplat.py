import cv2
import sys
import os
import pytesseract
import re
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"


def proyeksi_vertical(img):
    blurred = cv2.GaussianBlur(img.copy(), (5, 5), 0)
    gray = cv2.cvtColor(blurred.copy(), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray.copy(), (450, 145))
    ret, bw = cv2.threshold(resized.copy(), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = bw / 255.
    bw_data = np.asarray(bw)
    pvertical = np.sum(bw_data, axis=1)
    return pvertical


# Template untuk proyeksi vertikal
pv_template = proyeksi_vertical(cv2.imread(
    "templates/plate/template.jpg", cv2.IMREAD_ANYCOLOR))

# Input gambar yg ingin dideteksi
nama_file = sys.argv[1]
image = cv2.imread(nama_file)

src = image.copy()
blurred = image.copy()
# Filtering gaussian blur
for i in range(10):
    blurred = cv2.GaussianBlur(image, (5, 5), 0.5)

# Conversi image BGR2GRAY
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Image binerisasi menggunakan adaptive thresholding
bw = cv2.adaptiveThreshold(
    rgb, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 10)

# Operasi dilasi
bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

# Ekstraksi kontur
contours, hierarchy = cv2.findContours(
    bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

slices = []
img_slices = image.copy()
idx = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    ras = format(w / h, '.2f')
    # Pilih kontur dengan ukuran dan rasio tertentu
    if 30 <= h and (100 <= w <= 400) and (2.7 <= float(ras) <= 4):
        idx = idx + 1
        cv2.rectangle(image, (x, y), (x + w, y + h),
                      (0, 0, 255), thickness=1)
        cv2.putText(image, "{}x{}".format(w, h), (x, int(
            y + (h / 2))), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv2.putText(image, "{}".format(ras), (x + int(w / 2), y + h + 13), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255))
        crop = img_slices[y:y - 3 + h + 6, x:x - 3 + w + 6]
        slices.append(crop)

result = None
max_value = sys.float_info.max
for sl in slices:
    pv_numpy = proyeksi_vertical(sl.copy())
    rs_sum = cv2.sumElems(cv2.absdiff(pv_template, pv_numpy))
    if rs_sum[0] <= max_value:
        max_value = rs_sum[0]
        result = sl

grayImage = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(
    grayImage, 127, 255, cv2.THRESH_BINARY)

text = pytesseract.image_to_string(result, config='--psm 13')
text2 = pytesseract.image_to_string(blackAndWhiteImage, config='--psm 7')

platindo = re.compile("^[a-zA-z]{1,2}\s?\d{1,4}\s?[a-zA-Z]{1,3}$")
if(platindo.match(text)):
    text3 = text
elif(platindo.match(text2)):
    text3 = text2
else:
    text3 = "PLAT TIDAK TERDETEKSI"

print("HASIL DETEKSI PLAT RGB : "+text)
print("HASIL DETEKSI PLAT BINER : "+text2)
print("Hasil Akhir : "+text3)


cv2.imshow('Gambar Asli', src)
cv2.imshow('Plat', result)
cv2.waitKey()
cv2.destroyAllWindows()
