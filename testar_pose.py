import cv2
from ultralytics import YOLO

# 1. Carrega o modelo leve de detecção de pose (baixa na 1ª execução)
modelo = YOLO("yolov8n-pose.pt")

video = cv2.VideoCapture("videos/blackpink_danca.mp4")

while True:
    conseguiu_ler, frame = video.read()

    if not conseguiu_ler:
        break

    # 3. Roda a detecção de pose no frame atual
    resultados = modelo(frame, verbose=False)

    # 4. Pega o primeiro resultado e desenha os pontos/esqueleto no frame
    frame_com_pose = resultados[0].plot()

    # 5. Mostra o resultado na tela
    cv2.imshow("Pose dos Dancarinos", frame_com_pose)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()