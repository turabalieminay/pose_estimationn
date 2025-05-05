import os
import cv2
import numpy as np
import math
from ultralytics import YOLO
from sort import Sort

# ——— AYARLAR ———
VIDEO_PATH                   = r"C:/Users/aytur/Desktop/pose_estimstion/media3.mp4"
OUTPUT_PATH                  = r"C:/Users/aytur/Desktop/pose_estimstion/media1_result_sort.mp4"
MODEL_PATH                   = "yolov8n-pose.pt"
CONF_THRESH                  = 0.5

# Profil ve eşikler
LYING_TORSO_ANGLE_THRESH     = 60      # Gövde ekseni, yere dik eksene göre bu açıdan büyükse “yerde yatma”.
SQUAT_KNEE_ANGLE_THRESH      = 90      # Diz açısı bu değerden küçükse “çömelme”.
RUN_SPEED_THRESH_SIDE        = 200.0   # Yalnızca yan görünümde kişinin merkez hızı (px/s) bu eşiğin üzerindeyse koşma adayı.
RUN_AREA_THRESH_FRONT        = 10000   # Önden/arkadan koşma tespiti için bounding box alan değişim hızı.
RUN_KNEE_ANGLE_THRESH        = 140     # Koşmada diz bükülme açısı max 140°’den küçük olmalı.
VERTICAL_TORSO_THRESH        = 20      # Gövde açısının dik kabul edilebilmesi için 20° altında olması gerekiyor.
STAND_KNEE_ANGLE_THRESH      = 150     # Ayakta durma tespiti için hız, diz açısı ve gövde açısı eşikleri.
STAND_SPEED_THRESH           = 50.0    # 
SHOULDER_WIDTH_RATIO_SIDE    = 0.3     # Yan görünüm mü değil mi kararında omuz genişliğinin kutu genişliğine oranı eşiği.

MIN_STABLE_FRAMES            = 3       # yeni duruma geçmeden önce kaç kare onay
PANEL_WIDTH                  = 500

# Fonksiyon: PostureDetector sınıfını başlatır ve durum sözlüğünü oluşturur
class PostureDetector:
    # Fonksiyon: Nesne oluşturulurken çağrılır, başlangıç durumlarını ayarlar
    def __init__(self):
        self.states = {}

    # Fonksiyon: Görünüm tipini omuz genişliği oranına göre belirler
    def _view_type(self, l_sh, r_sh, bbox_w):
        shoulder_w = abs(r_sh[0] - l_sh[0])
        ratio = shoulder_w / (bbox_w + 1e-6)
        return "side" if ratio < SHOULDER_WIDTH_RATIO_SIDE else "frontal"

    # Fonksiyon: Gözlem parametrelerine göre aday durumu kararlaştırır
    def _decide_candidate(self,
            view, speed, area_speed,
            torso_ang, avg_knee):
        # Koşma tespiti yan görünüm için hız, gövde diklik ve diz açısına bakar
        if view == "side":
            if speed > RUN_SPEED_THRESH_SIDE \
               and torso_ang < VERTICAL_TORSO_THRESH \
               and avg_knee < RUN_KNEE_ANGLE_THRESH:
                return "kosma"
        else:
            # Koşma tespiti önden görünüm için alan hızı, gövde diklik ve diz açısını kontrol eder
            if area_speed > RUN_AREA_THRESH_FRONT \
               and torso_ang < VERTICAL_TORSO_THRESH \
               and avg_knee < RUN_KNEE_ANGLE_THRESH:
                return "kosma"

        # Yatar tespiti gövde açısı eşiğinin üzerinde kontrol
        if torso_ang > LYING_TORSO_ANGLE_THRESH:
            return "yerde yatma"
        # Çömelme tespiti diz eşiğini aşan açılara bakar
        if avg_knee < SQUAT_KNEE_ANGLE_THRESH:
            return "comelme"
        # Ayakta durma tespiti hız, diz ve gövde dikliğine göre kontrol
        if (speed < STAND_SPEED_THRESH
            and avg_knee > STAND_KNEE_ANGLE_THRESH
            and torso_ang < VERTICAL_TORSO_THRESH):
            return "ayakta durma"
        # Belirsiz durum
        return None

    # Fonksiyon: Takip ID'si için yeni durumu günceller ve stabil hale getirir
    def update(self, tid,
               l_sh, r_sh, bbox_w,
               speed, area_speed,
               torso_ang, avg_knee):
        view = self._view_type(l_sh, r_sh, bbox_w)
        cand = self._decide_candidate(view,
                                      speed, area_speed,
                                      torso_ang, avg_knee)
        # İlk karede yoksa varsayılan durumu ayarla
        if tid not in self.states:
            init = cand or "ayakta durma"
            self.states[tid] = {"state": init, "counter": 1}
            return init

        st = self.states[tid]
        # Eğer candidate yok ya da aynı durum gelmişse sayaç artır
        if cand is None or cand == st["state"]:
            st["counter"] = min(st["counter"] + 1, MIN_STABLE_FRAMES)
            return st["state"]

        # Farklı candidate geldiyse sayacı artır ve yeterli tekrar varsa geçiş yap
        st["counter"] += 1
        if st["counter"] >= MIN_STABLE_FRAMES:
            st["state"]   = cand
            st["counter"] = 1
        return st["state"]

# Fonksiyon: Üç nokta arasındaki açıyı hesaplar
def angle(a, b, c):
    ba, bc = a - b, c - b
    cos_v = np.dot(ba, bc) / ((np.linalg.norm(ba)*np.linalg.norm(bc)) + 1e-6)
    return abs(math.degrees(math.acos(np.clip(cos_v, -1, 1))))

# Fonksiyon: Omuz ve kalça noktalarından gövde açısı ve omuz merkezini hesaplar
def torso_angle(kp):
    l_sh, r_sh = kp[5][:2], kp[6][:2]
    l_hp, r_hp = kp[11][:2], kp[12][:2]
    mid_sh = (l_sh + r_sh) / 2
    mid_hp = (l_hp + r_hp) / 2
    v = mid_sh - mid_hp
    cos_v = np.dot(v, [0, -1]) / (np.linalg.norm(v) + 1e-6)
    return abs(math.degrees(math.acos(np.clip(cos_v, -1, 1)))), mid_sh.astype(int)

# Fonksiyon: Video işleme, tespit, çizim ve çıktı kaydetme döngüsü
def main():
    # Video kaynağını açmayı dener
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Video açılamadı:", VIDEO_PATH)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt           = 1.0 / fps
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"▶️ Toplam kare: {total_frames}, FPS: {fps:.2f}")

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w + PANEL_WIDTH, h)
    )

    # Model ve takip nesnelerini oluştur
    model        = YOLO(MODEL_PATH, verbose=False)
    sort_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    detector     = PostureDetector()
    durations    = {}
    prev_centers = {}
    prev_areas   = {}

    cv2.namedWindow("Takip", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Takip", min(w+PANEL_WIDTH, 1200), min(h,800))

    frame_idx = 0
    # Sonsuza dek kareleri işle
    while True:
        # Video karelerini sırayla oku
        ret, frame = cap.read()
        if not ret:
            # Bittiğinde döngüden çık
            break
        frame_idx += 1
        print(f"● İşlenen kare: {frame_idx}/{total_frames}", end="\r")

        # YOLO ile poz tespiti
        res    = model(frame, verbose=False)[0]
        kps    = res.keypoints.data.cpu().numpy()
        bboxes = res.boxes.xyxy.cpu().numpy()
        confs  = res.boxes.conf.cpu().numpy().reshape(-1,1)
        dets   = np.hstack((bboxes, confs))

        # SORT takibi güncelle ve sonuçları al
        tracks = sort_tracker.update(dets)
        annotated = frame.copy()

        # Her takip edilen kişi için işlemler
        for x1, y1, x2, y2, tid in tracks:
            tid = int(tid)
            # Kutu çizimi
            cv2.rectangle(annotated,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (255, 0, 0), 2)

            # Hız hesaplamak için merkez noktalarını kullan
            cx, cy = (x1+x2)/2, (y1+y2)/2
            center = np.array([cx, cy])
            if tid in prev_centers:
                # Önceki pozisyona göre hızı hesapla
                speed = np.linalg.norm(center - prev_centers[tid]) / dt
            else:
                speed = 0.0
            prev_centers[tid] = center

            # Alan değişim hızını hesapla
            area = (x2-x1)*(y2-y1)
            if tid in prev_areas:
                # Önceki alana göre alan değişimini hesapla
                area_speed = abs(area - prev_areas[tid]) / dt
            else:
                area_speed = 0.0
            prev_areas[tid] = area

            # En yakın keypoint setini eşleştir
            centers_kp = np.column_stack((
                (bboxes[:,0]+bboxes[:,2])/2,
                (bboxes[:,1]+bboxes[:,3])/2
            ))
            idx = np.argmin(np.linalg.norm(centers_kp - [cx, cy], axis=1))
            kp  = kps[idx]

            # Açıları hesapla: gövde ve diz açısı
            torso_ang, _ = torso_angle(kp)
            l_k = angle(kp[11][:2], kp[13][:2], kp[15][:2])
            r_k = angle(kp[12][:2], kp[14][:2], kp[16][:2])
            avg_knee = (l_k + r_k) / 2

            # Durum tespiti yap
            label = detector.update(tid,
                                    kp[5][:2], kp[6][:2],
                                    x2-x1,
                                    speed, area_speed,
                                    torso_ang, avg_knee)
            if not label:
                # Geçerli durum yoksa sonraki kişiye geç
                continue

            # Süreleri biriktir
            durations.setdefault(tid, {})
            durations[tid][label] = durations[tid].get(label, 0.0) + dt

            # Durumu yazdır
            text = f"ID{tid} {label}"
            cv2.putText(annotated, text,
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

        # Sağ paneli çiz ve süreleri göster
        panel = np.ones((h, PANEL_WIDTH, 3), np.uint8) * 255
        y0 = 30
        cv2.putText(panel, "Sureler (s)", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        y = y0 + 30
        for tid, acts in durations.items():
            cv2.putText(panel, f"ID{tid}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            y += 25
            for act, sec in acts.items():
                cv2.putText(panel,
                            f"{act:12}: {sec:5.1f}",
                            (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,0,0), 1)
                y += 25
            y += 15
            if y > h - 30:
                break

        # Sonuçları birleştir, göster ve kaydet
        combined = np.hstack([annotated, panel])
        out.write(combined)
        cv2.imshow("Takip", combined)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            # Kullanıcı 'q' tuşuna bastığında çıkılır
            break

    print("\n✅ İşlem tamamlandı. Çıktı:", OUTPUT_PATH)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
