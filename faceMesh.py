import cv2
import mediapipe as mp



mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness= 1, circle_radius=1)

cap = cv2.VideoCapture('assets/face1.mp4')


with mp_face_mesh.FaceMesh( max_num_faces=1, min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as face_mesh:




    while True:

        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame.flags.writeable = False
        
        mesh = face_mesh.process(frame)

        # print(mesh)
        frame.flags.writeable = True

        if mesh.multi_face_landmarks:
            for face_landmark in mesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, landmark_list= face_landmark, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec= drawing_spec, connection_drawing_spec=drawing_spec)

        cv2.imshow('Face Mesh', frame[:,:,::-1])

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()