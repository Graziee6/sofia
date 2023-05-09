import cv2

# Read the input image
live_stream=cv2.VideoCapture(0)
if not live_stream.isOpened():
    print("Err opening the webcam")

#Setting the video frame size
live_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
live_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret,frame=live_stream.read()
    if ret:
        cv2.imshow("Myself", frame)
          
        # Convert into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces and crop the faces
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            faces = frame[y:y + h, x:x + w]
            cv2.imshow("face",faces)
            cv2.imwrite(f'face{i+1}.jpg', faces)
            
        if cv2.waitKey(25) & 0xFF==ord("q"):
            break
    else:
        break

cv2.destroyAllWindows()
live_stream.release()