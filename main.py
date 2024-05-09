import cv2 as cv
import time
import mediapipe as mp
from utils import CoolDown
import pyautogui

MOUSE_SENSITIVITY = 1
MIN_DIST_SQUARED = pow(100,2)

fps = 0

cap = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) 

index = None

@CoolDown( 1 )
def update_fps( _fps ):
    global fps
    fps = _fps

def to_screen( pos, s ):
    h, w = s
    x, y = pos 
    return ( int(x *w), int(y *h))

def dist_squared( a, b, s ):
    dx = a.x - b.x
    dy = a.y - b.y
    h, w = s
    return pow(dx*w,2)+pow(dy*h,2)

ptime = 0
while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    cur_fps = 1/( time.time() -ptime )
    update_fps( cur_fps )
    ptime = time.time()
    if not ret : continue 

    results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    if results.multi_hand_landmarks :
        for hand_landmarks in results.multi_hand_landmarks:
            i = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            t = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            cv.circle( frame, to_screen( ( i.x, i.y ), frame.shape[:2] ), 5, (255,0,0), 2 )
            cv.circle( frame, to_screen( ( t.x, t.y ), frame.shape[:2] ), 5, (255,0,0), 2 )
            
            if index == None : index = i
            if dist_squared( i, t, (w, h) ) < MIN_DIST_SQUARED : 
                cv.line( frame,  to_screen( ( i.x, i.y ), frame.shape[:2] ), to_screen(( t.x, t.y ), frame.shape[:2] ), ( 0, 0, 255 ), 2)
                
                dx = index.x -i.x
                dy = index.y -i.y
                
                pyautogui.move( *to_screen(( dx, -dy ), pyautogui.size()) )
                # # autopy.mouse.smooth_move(dx*MOUSE_SENSITIVITY, dy*MOUSE_SENSITIVITY)
            index = i
    
    cv.putText( frame , str(round(fps)), ( 0, h ), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, ( 255, 0, 0 ), 2)
    cv.imshow( "main", frame )

    if cv.waitKey(1) == ord("q"):
        break


cv.destroyAllWindows()

