import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import gdown
import os
import re
from ultralytics import YOLO

# ─── Cấu hình trang ───
st.set_page_config(
    page_title='Nhận Diện Phương Tiện & Biển Số',
    page_icon='🚗',
    layout='centered'
)

st.title('🚗 Hệ Thống Nhận Diện Phương Tiện & Biển Số')
st.markdown('''
Hệ thống tự động nhận diện:
- 🚘 **Loại xe** (ô tô, xe máy, xe buýt, xe tải)
- 🎨 **Màu xe**
- 🔢 **Biển số xe**
- 📏 **Khoảng cách ước tính**

*Ứng dụng: Ghi nhận thông tin phương tiện khi xảy ra va chạm giao thông.*
''')

# ============================================================
# 1. LOAD MODEL (chạy 1 lần lúc khởi động app)
# ============================================================
MODEL_PATH = 'best.pt'
if not os.path.exists(MODEL_PATH):
    # Thay FILE_ID bằng ID thật của file best.pt trên Google Drive
    # (lấy từ link share: drive.google.com/file/d/FILE_ID/view)
    gdown.download(id='1QDoeKQBMa9O9riapJhxdmzxMozFfPIPL', output=MODEL_PATH, quiet=False)
plate_model = YOLO(MODEL_PATH)
vehicle_model = YOLO('yolov8n.pt')             # model COCO gốc, có sẵn, không cần train
ocr_reader = easyocr.Reader(['en'], gpu=False) # đổi gpu=True nếu máy có GPU

# Các class xe trong COCO (chỉ giữ lại loại liên quan)
VEHICLE_CLASSES = {2: 'Car', 3: 'Motorbike', 5: 'Bus', 7: 'Truck'}

# Dịch tên loại xe sang tiếng Việt để hiển thị
VEHICLE_VI = {
    'Car': 'Ô tô',
    'Motorbike': 'Xe máy',
    'Bus': 'Xe buýt',
    'Truck': 'Xe tải',
    'Không xác định': 'Không xác định',
}


# ============================================================
# 2. HÀM PHỤ TRỢ — khớp biển số với đúng xe chứa nó
# ============================================================
def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2, (y1+y2)/2)

def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def match_plate_to_vehicle(plate_box, vehicle_boxes):
    """Tìm xe nào chứa biển số này (biển số nằm trong bbox xe)"""
    center = bbox_center(plate_box)
    for v_box, v_cls, v_conf in vehicle_boxes:
        if point_in_box(center, v_box):
            return v_cls, v_conf
    return None, None


# ============================================================
# 3. MÀU XE — HSV, không cần train
# ============================================================
def detect_vehicle_color(vehicle_crop):
    """Xác định màu chủ đạo của xe bằng HSV"""
    hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    avg_h = np.median(h[s > 40]) if np.any(s > 40) else np.median(h)
    avg_s = np.median(s)
    avg_v = np.median(v)

    if avg_v < 50:
        return 'Đen'
    if avg_s < 40 and avg_v > 180:
        return 'Trắng'
    if avg_s < 40:
        return 'Xám/Bạc'

    color_ranges = [
        (0, 10, 'Đỏ'), (10, 25, 'Cam'), (25, 35, 'Vàng'),
        (35, 85, 'Xanh lá'), (85, 130, 'Xanh dương'),
        (130, 160, 'Tím'), (160, 180, 'Đỏ'),
    ]
    for lo, hi, name in color_ranges:
        if lo <= avg_h < hi:
            return name
    return 'Không xác định'


# ============================================================
# 4. OCR — tiền xử lý + EasyOCR + hậu xử lý (đã tối ưu từ trước)
# ============================================================
def preprocess_plate(plate_img):
    h, w = plate_img.shape[:2]
    if h < 100:
        scale = 100 / h
        plate_img = cv2.resize(plate_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(denoised)

def clean_plate_text(raw_text):
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return text

def read_plate_ocr(plate_crop):
    """Đọc ký tự biển số, trả về (text, confidence)"""
    enhanced = preprocess_plate(plate_crop)
    results = ocr_reader.readtext(
        enhanced,
        allowlist='0123456789ABCDEFGHKLMNPSTUVXYZ.-',
        text_threshold=0.6, low_text=0.3, mag_ratio=2.0,
    )
    if not results:
        return '', 0.0
    best = max(results, key=lambda r: r[2])
    text = clean_plate_text(best[1])
    conf = best[2]
    return text, conf


# ============================================================
# 5. KHOẢNG CÁCH — công thức pinhole camera, không cần train
# ============================================================
FOCAL_LENGTH_PX = 800  # cần hiệu chỉnh lại bằng ảnh mẫu khoảng cách đã biết

def estimate_distance(plate_box):
    """Ước tính khoảng cách dựa trên chiều rộng BIỂN SỐ — chính xác hơn, dùng khi thấy được biển."""
    x1, y1, x2, y2 = plate_box
    plate_width_px = x2 - x1
    if plate_width_px <= 0:
        return None
    real_width_mm = 440  # biển số ô tô VN chuẩn ~440mm
    distance_m = (real_width_mm * FOCAL_LENGTH_PX) / plate_width_px / 1000
    return round(distance_m, 2)


# Chiều rộng thực tế trung bình theo loại xe (mm) — dùng khi KHÔNG thấy biển số
# (chỉ mang tính ước lượng thô, sai số lớn hơn cách tính theo biển số vì mỗi xe rộng khác nhau)
REAL_VEHICLE_WIDTH_MM = {
    'Car': 1800, 'Motorbike': 700, 'Bus': 2500, 'Truck': 2500,
}

def estimate_distance_vehicle(v_box, v_cls):
    """Ước tính khoảng cách dựa trên chiều rộng cả XE — dùng dự phòng khi không detect được biển số
    (ví dụ xe bị che biển, hoặc góc chụp không thấy biển)."""
    x1, y1, x2, y2 = v_box
    width_px = x2 - x1
    if width_px <= 0:
        return None
    real_width_mm = REAL_VEHICLE_WIDTH_MM.get(v_cls, 1800)
    distance_m = (real_width_mm * FOCAL_LENGTH_PX) / width_px / 1000
    return round(distance_m, 2)


# ============================================================
# 6. HÀM CHÍNH — ghép toàn bộ pipeline
# ============================================================
def detect_vehicle_and_plate(frame, conf_threshold=0.25):
    results = []
    result_img = frame.copy()

    vehicle_res = vehicle_model.predict(frame, conf=conf_threshold, verbose=False)[0]
    vehicle_boxes = []
    for box in vehicle_res.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            xyxy = box.xyxy[0].cpu().numpy()
            v_cls = VEHICLE_CLASSES[cls_id]
            vehicle_boxes.append((xyxy, v_cls, float(box.conf[0])))

            # Vẽ khung xe + GỌI TÊN loại xe ngay trên khung
            vx1, vy1, vx2, vy2 = map(int, xyxy)
            cv2.rectangle(result_img, (vx1, vy1), (vx2, vy2), (255, 200, 0), 2)
            label = VEHICLE_VI.get(v_cls, v_cls)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_img, (vx1, max(0, vy1 - th - 10)),
                          (vx1 + tw + 6, vy1), (255, 200, 0), -1)
            cv2.putText(result_img, label, (vx1 + 3, max(15, vy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    plate_res = plate_model.predict(frame, conf=conf_threshold, verbose=False)[0]
    for box in plate_res.boxes:
        plate_box = box.xyxy[0].cpu().numpy()
        plate_conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, plate_box)
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        plate_text, ocr_conf = read_plate_ocr(plate_crop)
        distance = estimate_distance(plate_box)  # ưu tiên tính theo biển số (chính xác hơn)

        v_type, v_conf = match_plate_to_vehicle(plate_box, vehicle_boxes)
        v_color = 'N/A'
        if v_type:
            for v_box, v_cls, _ in vehicle_boxes:
                if v_cls == v_type:
                    vx1, vy1, vx2, vy2 = map(int, v_box)
                    v_crop = frame[vy1:vy2, vx1:vx2]
                    if v_crop.size > 0:
                        v_color = detect_vehicle_color(v_crop)
                    # Nếu vì lý do nào đó không tính được khoảng cách theo biển số,
                    # dùng khoảng cách theo chiều rộng xe làm dự phòng
                    if distance is None:
                        distance = estimate_distance_vehicle(v_box, v_cls)
                    break

        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if plate_text:
            cv2.putText(result_img, plate_text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        results.append({
            'plate_box': plate_box,
            'plate_conf': plate_conf,
            'plate_text': plate_text,
            'ocr_conf': ocr_conf,
            'plate_crop': plate_crop,
            'vehicle_type': v_type or 'Không xác định',
            'vehicle_conf': v_conf or 0.0,
            'vehicle_color': v_color,
            'distance_m': distance,
        })

    # Với những xe KHÔNG khớp được biển số nào (bị che biển / góc chụp không thấy biển),
    # vẫn ghi nhận xe đó vào kết quả, dùng khoảng cách ước tính theo chiều rộng xe
    matched_vehicle_types = {r['vehicle_type'] for r in results}
    for v_box, v_cls, v_conf in vehicle_boxes:
        if v_cls not in matched_vehicle_types:
            vx1, vy1, vx2, vy2 = map(int, v_box)
            v_crop = frame[vy1:vy2, vx1:vx2]
            v_color = detect_vehicle_color(v_crop) if v_crop.size > 0 else 'N/A'
            results.append({
                'plate_box': None,
                'plate_conf': 0.0,
                'plate_text': '',
                'ocr_conf': 0.0,
                'plate_crop': None,
                'vehicle_type': v_cls,
                'vehicle_conf': v_conf,
                'vehicle_color': v_color,
                'distance_m': estimate_distance_vehicle(v_box, v_cls),
            })

    return result_img, results, vehicle_boxes


# ─── UI ───
st.sidebar.header('⚙️ Cài đặt')
conf_threshold = st.sidebar.slider(
    'Ngưỡng confidence', 0.1, 0.9, 0.25, 0.05,
    help='Càng cao càng ít detect nhưng chính xác hơn'
)
st.sidebar.markdown('---')
st.sidebar.markdown('**Về hệ thống:**')
st.sidebar.markdown('- Model: YOLOv8n')
st.sidebar.markdown('- mAP50: **96.4%**')
st.sidebar.markdown('- Dataset: 10,127 ảnh biển số VN')
st.sidebar.markdown('- 📏 Khoảng cách: ưu tiên tính theo biển số, '
                     'dự phòng theo chiều rộng xe nếu không thấy biển')

tab1, tab2 = st.tabs(['📷 Ảnh', '🎬 Video'])

with tab1:
    uploaded = st.file_uploader(
        'Upload ảnh xe', type=['jpg','jpeg','png'],
        key='img_upload'
    )
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner('Đang phân tích...'):
            result_img, detections, vehicles = detect_vehicle_and_plate(img_bgr, conf_threshold)

        st.image(result_img, caption='Kết quả', use_column_width=True)

        if not detections and not vehicles:
            st.warning('Không phát hiện phương tiện nào trong ảnh.')
        else:
            st.success(f'Phát hiện {len(vehicles)} phương tiện, '
                       f'{sum(1 for d in detections if d["plate_text"])} biển số đọc được!')

            for i, d in enumerate(detections):
                dist_str = f'{d["distance_m"]}m' if d['distance_m'] is not None else 'N/A'
                with st.expander(
                    f'🚘 Phương tiện #{i+1} — '
                    f'{VEHICLE_VI[d["vehicle_type"]]} | '
                    f'{d["vehicle_color"]} | '
                    f'Biển: {d["plate_text"] or "Không đọc được"} | '
                    f'📏 {dist_str}'
                ):
                    col1, col2, col3 = st.columns(3)
                    col1.metric('Loại xe',    VEHICLE_VI[d['vehicle_type']])
                    col1.metric('Màu xe',     d['vehicle_color'])
                    col2.metric('Biển số',    d['plate_text'] or 'N/A')
                    col2.metric('Độ tin cậy detect', f"{d['plate_conf']:.1%}")
                    col3.metric('Khoảng cách ước tính', dist_str)

                    if d['plate_crop'] is not None and d['plate_crop'].size > 0:
                        crop_rgb = cv2.cvtColor(
                            d['plate_crop'], cv2.COLOR_BGR2RGB
                        )
                        st.image(crop_rgb, caption='Ảnh biển số (crop)',
                                 width=300)

            if detections:
                st.markdown('---')
                st.markdown('### 📋 Báo cáo nhanh')
                for i, d in enumerate(detections):
                    plate_str = d['plate_text'] if d['plate_text'] else 'Không đọc được'
                    dist_str = f'{d["distance_m"]}m' if d['distance_m'] is not None else 'N/A'
                    st.markdown(
                        f"**Xe #{i+1}:** "
                        f"{VEHICLE_VI[d['vehicle_type']]} | "
                        f"Màu {d['vehicle_color']} | "
                        f"Biển số: `{plate_str}` | "
                        f"Khoảng cách: {dist_str}"
                    )

with tab2:
    uploaded_vid = st.file_uploader(
        'Upload video', type=['mp4','avi','mov'],
        key='vid_upload'
    )
    if uploaded_vid:
        tmp_path = f'/tmp/{uploaded_vid.name}'
        with open(tmp_path, 'wb') as f:
            f.write(uploaded_vid.read())

        cap      = cv2.VideoCapture(tmp_path)
        total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 25

        st.info(f'Video: {total_fr} frames | {fps:.0f} FPS | '
                f'Xử lý mỗi 10 frames')

        frame_ph  = st.empty()
        progress  = st.progress(0)
        log_ph    = st.empty()
        all_info  = {}  # plate_text -> {type, color, count, distance_m}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 10 != 0:
                continue

            result_frame, dets, _ = detect_vehicle_and_plate(frame, conf_threshold)
            for d in dets:
                key = d['plate_text'] or f'unknown_{frame_idx}'
                if key not in all_info:
                    all_info[key] = {
                        'type':  VEHICLE_VI[d['vehicle_type']],
                        'color': d['vehicle_color'],
                        'count': 1,
                        'distance_m': d['distance_m'],
                    }
                else:
                    all_info[key]['count'] += 1
                    all_info[key]['distance_m'] = d['distance_m']  # cập nhật khoảng cách mới nhất

            frame_ph.image(result_frame,
                           caption=f'Frame {frame_idx}/{total_fr}',
                           use_column_width=True)
            progress.progress(min(frame_idx/total_fr, 1.0))

            if all_info:
                log_text = '**Phương tiện đã ghi nhận:**\n'
                for plate, info in all_info.items():
                    dist_str = f'{info["distance_m"]}m' if info['distance_m'] is not None else 'N/A'
                    log_text += (f"- {info['type']} | "
                                 f"Màu {info['color']} | "
                                 f"Biển: `{plate}` | "
                                 f"📏 {dist_str} "
                                 f"({info['count']} lần)\n")
                log_ph.markdown(log_text)

        cap.release()
        st.success('Xử lý xong!')
        st.markdown('### 📋 Tổng kết')
        for plate, info in all_info.items():
            dist_str = f'{info["distance_m"]}m' if info['distance_m'] is not None else 'N/A'
            st.markdown(
                f"🚘 **{info['type']}** | "
                f"Màu {info['color']} | "
                f"Biển: `{plate}` | "
                f"Khoảng cách: {dist_str}"
            )

st.markdown('---')
st.caption('Model: YOLOv8n | mAP50: 96.4% | Dataset: 10,127 ảnh biển số VN')
