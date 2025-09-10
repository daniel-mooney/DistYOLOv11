from distyolo import DistYOLO
from distyolo.results import ResultsDistribution
import cv2

def main() -> None:
    model = DistYOLO("yolo11s.pt")
    win = "YOLOv11 Test Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)

    preds: ResultsDistribution = model.predict()

    disp = preds[0].plot()  # BGR with preds
    print(preds[0])

    overlay = disp.copy()
    text = f""
    cv2.putText(overlay, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(win, overlay)
    k = cv2.waitKey(0) & 0xFF

    if k in (ord("q"), 27):  # q or ESC
        return
    elif k in (ord("n"), 83):  # n or Right arrow
        pass
    elif k in (ord("p"), 81):  # p or Left arrow
        pass
    elif k == ord("g"):
        show_gt = not show_gt
    
    
if __name__ == "__main__":
    main()