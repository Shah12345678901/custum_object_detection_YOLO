import torch
import cv2
import time
from torch.multiprocessing import Pool, Process, set_start_method

trained_model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=trained_model_path, force_reload=True)

def detectObject(video,name):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        pTime = time.time()
        ret, img = cap.read()
        cTime = time.time()
        fps = str(int(1 / (cTime - pTime)))
        if img is None:
            break
        else:
            results = model(img)
            labels = results.xyxyn[0][:, -1].cpu().numpy()
            cord = results.xyxyn[0][:, :-1].cpu().numpy()
            n = len(labels)
            x_shape, y_shape = img.shape[1], img.shape[0]
            for i in range(n):
                row = cord[i]
                # If score is less than 0.3 we avoid making a prediction.
                if row[4] < 0.4:
                    continue
                x1 = int(row[0] * x_shape)
                y1 = int(row[1] * y_shape)
                x2 = int(row[2] * x_shape)
                y2 = int(row[3] * y_shape)
                bgr = (0, 255, 0)  # color of the box
                classes = model.names  # Get the name of label index
                label_font = cv2.FONT_HERSHEY_COMPLEX  # Font for the label.
                cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)  # Plot the boxes
                cv2.putText(img, classes[int(labels[i])], (x1, y1), label_font, 2, bgr, 2)
                cv2.putText(img, f'FPS={fps}', (8, 70), label_font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        img = cv2.resize(img, (500, 500))
        cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
if __name__ == '__main__':
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    Videos = ['Test1.mp4', 'Test2.mp4']
    for i in Videos:
        process = Process(target=detectObject, args=(i, str(i)))
        process.start()
