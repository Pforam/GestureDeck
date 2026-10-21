import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open camera")
    exit()

print("Camera opened successfully!")
print("Press Q to close the camera.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    cv2.imshow("GestureDeck - Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()