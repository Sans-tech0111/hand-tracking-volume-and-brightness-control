import cv2
from matplotlib.pyplot import draw
import mediapipe as mp
import time
import numpy as np
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minV = volRange[0]
maxV = volRange[1]

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_tracking_confidence=0.3)
mpDraw = mp.solutions.drawing_utils
pTime=0

while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    lmlist = []

    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(img,lms,mpHands.HAND_CONNECTIONS)
        for id,lm in  enumerate(lms.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            lmlist.append([id,cx,cy])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    if len(lmlist)!=0:
                x1, y1 = lmlist[4][1],lmlist[4][2]
                x2, y2 = lmlist[8][1],lmlist[8][2]

                cx,cy = (x1+x2)//2,(y1+y2)//2

                cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
                cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)

                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
                cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)

                length = math.hypot(x2-x1,y2-y1)
                cv2.putText(img,f'{int(length)}',(200,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

                vol = np.interp(length,[20,200],[minV,maxV])
                volume.SetMasterVolumeLevel(vol, None)

                if length<50:
                    cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)

    cv2.putText(img,f'FPS:{int(fps)}',(20,60),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
    

    cv2.imshow("Image",img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
