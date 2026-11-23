import cv2
from ultralytics import YOLO

modelo = YOLO("yolov8n-pose.pt")
video = cv2.VideoCapture("videos/blackpink_danca.mp4")

numero_frame = 0

while True:
    conseguiu_ler, frame = video.read()
    if not conseguiu_ler:
        break

    numero_frame += 1
    resultados = modelo(frame, verbose=False)

    # Pega as coordenadas (x, y) de todas as pessoas detectadas no frame
    # Formato: [num_pessoas, 17, 2]
    pontos_xy = resultados[0].keypoints.xy

    qtd_pessoas = pontos_xy.shape[0]

    # Imprime a cada 15 frames para não encher o terminal rápido demais
    if numero_frame % 15 == 0:
        print(f"--- Frame {numero_frame} | Pessoas detectadas: {qtd_pessoas} ---")
        for i in range(qtd_pessoas):
            # Ponto 8 = cotovelo direito (x, y)
            x, y = pontos_xy[i][8]
            print(f"  Pessoa {i + 1} -> Cotovelo dir (x={x:.1f}, y={y:.1f})")

    cv2.imshow("Ver Pontos", resultados[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()