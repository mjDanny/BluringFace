import cv2
import matplotlib.pyplot as plt
import numpy as np

def blur_face(img, bboxes):
    # создаем маску для лица
    face_mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    for box in bboxes:
        x, y, width, height = box
        # рисуем овал на маске, чтобы затемнить все пиксели лица
        cv2.ellipse(face_mask, (int(x + width/2), int(y + height/2)), (int(width/2), int(height/1)), 0, 0, 360, (0, 0, 0), -1)
        face = img[y:y + height, x:x + width]
        eyes = classifier_eye.detectMultiScale(face)
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            # рисуем круги вокруг глаз, чтобы сделать их видимыми
            cv2.circle(face_mask, (x + int(x_eye + w_eye/2), y + int(y_eye + h_eye/2)), int(w_eye/2), (255, 255, 255), -1)
    # применяем гауссов размытие ко всему изображению
    blurred_img = cv2.GaussianBlur(img, (99, 99), 30)
    # заменяем все пиксели лица, кроме глаз, размытыми пикселями
    result = np.where(face_mask[:,:,None] == 0, blurred_img, img)
    return result

# загружаем изображение
img = cv2.imread('./jason.jpg')
# копирование переменной
img2 = img.copy()
# Загрузка каскада Хаара для поиска лиц
classifier = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_frontalface_default.xml")
# загрузка каскада Хаара для поиска глаз
classifier_eye = cv2.CascadeClassifier(cv2.data.haarcascades+ "haarcascade_eye.xml")
# выполнение распознавания лиц
bboxes = classifier.detectMultiScale(img, scaleFactor=2, minNeighbors=3, minSize=(50, 50))
# заблуривание лиц, кроме глаз
result_img = blur_face(img, bboxes)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.xaxis.set_ticks([])
ax1.yaxis.set_ticks([])
ax1.set_title('Исходное изображение')

ax2.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
ax2.xaxis.set_ticks([])
ax2.yaxis.set_ticks([])
ax2.set_title('Распознанные лица')

plt.show()
