import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(5000)

block_cipher = None

rrr = 0
lll = 0
rcount = 0
lcount = 0
a = 0
b = 0
allall = 0
time = 0

xtr = []
ytr = []

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('models/2018_12_17_22_58_35.h5')
model.summary()

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

# main
cap = cv2.VideoCapture(0)

while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    #cv2.imshow('l', eye_img_l)
    #cv2.imshow('r', eye_img_r)


    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)
   
    # visualize
   
   
    state_l = 'O' if pred_l >= 0.1 else '-'
    state_ll = '%.1f' if pred_l >= 0.1 else '%.1f'
    if pred_l > 0.1: a = 1
    if a == lcount: a = 0
    else: lll += 1
    if pred_l > 0.1: lcount = 1
    else: lcout = 0
   
    state_r = 'O' if pred_r >= 0.1 else '-'
    state_rr = '%.1f' if pred_r >= 0.1 else '%.1f'
    if pred_r > 0.1: a = 1
    if a == rcount: a = 0
    else: rrr += 1
    if pred_r > 0.1: rcount = 1
    else: rcout = 0
   
    #수정
    if pred_l > 0.1 and pred_r > 0.1: a = 1
    if a == lcount or a == rcount: a = 0
    else: allall += 1
    if pred_l > 0.1 and pred_r > 0.1: rcount = 1; lcount = 1
    else: lcout = 0; rcount = 0
    #수정
   
    state_ll = state_ll % pred_l
    state_rr = state_rr % pred_r
   


    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
   
    cv2.putText(img, state_ll, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(img, state_rr, (150,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
   
    cv2.putText(img, str(lll), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(img, str(rrr), (150,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
   
    cv2.putText(img, str(allall), (80,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    time += 0.1
    xtr.append(time)
    ytr.append(allall)
    plt.plot(xtr,ytr)
    plt.show()
   
  cv2.namedWindow('result', cv2.WINDOW_NORMAL)
  cv2.imshow('result', img)
 
  if cv2.waitKey(1) == 32:
    cv2.destroyWindow("result")
    cap.release()
    break 
