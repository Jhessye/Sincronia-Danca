import importlib

cv2 = importlib.import_module("cv2")


video = cv2.VideoCapture("videos/blackpink_danca.mp4")

while True:
    conseguiu_ler, frame = video.read()

    if not conseguiu_ler:
        break

    cv2.imshow("Meu Video", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()