import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

cap = cv2.VideoCapture('assets/asl_hand.25fps.mp4')


with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:


    while True:

        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        res = hands.process(frame)
        frame.flags.writeable = True

        if res.multi_hand_landmarks:
            for hand_landmark in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, landmark_list=hand_landmark, connections= mp_hand.HAND_CONNECTIONS)

        cv2.imshow('Frame', frame[:,:,::-1])

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
