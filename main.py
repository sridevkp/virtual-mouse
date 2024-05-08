import cv2 as cv
import time
import mediapipe as mp
import autopy
# import pyautogui

MOUSE_SENSITIVITY = 1
MIN_DIST_SQUARED = pow(100,2)

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) 

index = None

def to_screen( pos, s ):
    w, h = s
    return ( int(pos.x *w), int(pos.y *h))

def dist_squared( a, b, s ):
    dx = a.x - b.x
    dy = a.y - b.y
    w, h = s
    return pow(dx*w,2)+pow(dy*h,2)

ptime = 0
while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    fps = 1/( time.time() -ptime )
    ptime = time.time()
    if not ret : continue 

    results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    if results.multi_hand_landmarks :
        for hand_landmarks in results.multi_hand_landmarks:
            i = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            t = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            cv.circle( frame, to_screen( i, (w, h) ), 5, (255,0,0), 2 )
            cv.circle( frame, to_screen( t, (w, h) ), 5, (255,0,0), 2 )
            
            if not index : index = i
            if dist_squared( i, t, (w, h) ) < MIN_DIST_SQUARED : 
                cv.line( frame,  to_screen( i, (w, h) ), to_screen( t, (w, h) ), ( 0, 0, 255 ), 2)
                dx = i.x - index.x
                dy = i.y - index.y
                autopy.mouse.smooth_move(dx*MOUSE_SENSITIVITY, dy*MOUSE_SENSITIVITY)
            index = i

    cv.putText( frame , str(round(fps)), ( 0, h ), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, ( 255, 0, 0 ), 2)
    cv.imshow( "main", frame )

    if cv.waitKey(1) == ord("q"):
        break


cv.destroyAllWindows()

