import streamlit as st
import cv2
import numpy as np
from PIL import Image
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

*Ứng dụng: Ghi nhận thông tin phương tiện khi xảy ra va chạm giao thông.*
''')

# ─── Load model ───
@st.cache_resource
def load_models():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        with st.spinner('Đang tải model lần đầu...'):
            gdown.download(
                'https://drive.google.com/uc?id=1cV3XdX9FP-hIqhmOBg6QIgnYp7FbZwzt',
                model_path, quiet=False
            )
    plate_model   = YOLO(model_path)
    vehicle_model = YOLO('yolov8n.pt')
    return plate_model, vehicle_model

plate_model, vehicle_model = load_models()
VEHICLE_CLASSES = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
VEHICLE_VI = {
    'car': 'Ô tô', 'motorcycle': 'Xe máy',
    'bus': 'Xe buýt', 'truck': 'Xe tải', 'unknown': 'Không rõ'
}

# ─── OCR ───
@st.cache_resource
def load_ocr():
    import easyocr
    return easyocr.Reader(['en'], verbose=False)

ocr_reader = load_ocr()

def preprocess_plate(crop_bgr):
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray    = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, None, fx=2, fy=2,
                         interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def fix_plate(text):
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9\-]', '', text)
    text = (text.replace('O','0').replace('Q','0')
                .replace('I','1').replace('L','1')
                .replace('S','5').replace('Z','2')
                .replace('B','8').replace('G','6'))
    text = re.sub(r'^(\d{2})([A-Z])', r'\1-\2', text)
    return text

def run_ocr(crop_bgr):
    processed = preprocess_plate(crop_bgr)
    result    = ocr_reader.readtext(processed, detail=1, paragraph=False)
    if not result:
        return '', 0.0
    result = sorted(result, key=lambda r: (r[0][0][1], r[0][0][0]))
    text   = ''.join([r[1] for r in result])
    conf   = max([r[2] for r in result])
    return fix_plate(text), round(conf, 4)

# ─── Nhận diện màu xe ───
def detect_color(region_bgr):
    """Nhận diện màu xe từ vùng ảnh xe bằng HSV."""
    if region_bgr is None or region_bgr.size == 0:
        return 'Không rõ'

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    h   = hsv[:,:,0]
    s   = hsv[:,:,1]
    v   = hsv[:,:,2]

    # Lọc bỏ vùng quá tối hoặc quá sáng (kính, gương, bóng)
    mask = (s > 30) & (v > 40) & (v < 240)
    if mask.sum() < 100:
        # Ảnh xám/trắng/đen — phân biệt qua V
        mean_v = v.mean()
        if mean_v > 180: return 'Trắng'
        if mean_v < 60:  return 'Đen'
        return 'Xám/Bạc'

    h_filtered = h[mask]
    mean_h     = float(np.median(h_filtered))
    mean_s     = float(s[mask].mean())
    mean_v     = float(v[mask].mean())

    # Xám/bạc — saturation thấp
    if mean_s < 40:
        if mean_v > 180: return 'Trắng'
        if mean_v < 70:  return 'Đen'
        return 'Xám/Bạc'

    # Phân loại theo Hue
    if mean_h < 10 or mean_h > 160:   return 'Đỏ'
    if 10  <= mean_h < 25:             return 'Cam'
    if 25  <= mean_h < 35:             return 'Vàng'
    if 35  <= mean_h < 85:             return 'Xanh lá'
    if 85  <= mean_h < 130:            return 'Xanh dương'
    if 130 <= mean_h < 160:            return 'Tím'
    return 'Không rõ'

# ─── Helper ───
def box_contains_center(outer, inner):
    ox1,oy1,ox2,oy2 = outer
    ix1,iy1,ix2,iy2 = inner
    cx, cy = (ix1+ix2)/2, (iy1+iy2)/2
    return ox1<=cx<=ox2 and oy1<=cy<=oy2

# ─── Hàm detect chính ───
def detect(img_bgr, conf=0.25):
    h_img, w_img = img_bgr.shape[:2]

    # Detect xe
    vehicles = []
    for r in vehicle_model.predict(img_bgr, conf=conf, verbose=False):
        if r.boxes is None: continue
        for b in r.boxes:
            cid = int(b.cls[0].item())
            if cid in VEHICLE_CLASSES:
                x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
                # Crop vùng thân xe (bỏ 30% dưới để tránh lấy đường)
                y_crop = y1 + int((y2-y1)*0.1)
                y_crop_end = y1 + int((y2-y1)*0.7)
                region = img_bgr[y_crop:y_crop_end, x1:x2]
                color  = detect_color(region)
                vehicles.append({
                    'box':   (x1,y1,x2,y2),
                    'type':  VEHICLE_CLASSES[cid],
                    'color': color,
                    'conf':  float(b.conf[0].item())
                })

    # Detect biển số
    detections = []
    for r in plate_model.predict(img_bgr, conf=conf, verbose=False):
        if r.boxes is None: continue
        for b in r.boxes:
            px1,py1,px2,py2 = map(int, b.xyxy[0].cpu().numpy())
            plate_conf = float(b.conf[0].item())

            # Ghép với xe
            matched = None
            for v in vehicles:
                if box_contains_center(v['box'],(px1,py1,px2,py2)):
                    matched = v
                    break

            # OCR
            pad  = 4
            crop = img_bgr[max(0,py1-pad):min(h_img,py2+pad),
                           max(0,px1-pad):min(w_img,px2+pad)]
            if crop.size == 0: continue
            plate_text, ocr_conf = run_ocr(crop)

            detections.append({
                'plate_box':    (px1,py1,px2,py2),
                'plate_text':   plate_text,
                'plate_conf':   plate_conf,
                'ocr_conf':     ocr_conf,
                'vehicle_type': matched['type']  if matched else 'unknown',
                'vehicle_color':matched['color'] if matched else 'Không rõ',
                'plate_crop':   crop,
            })

    # Vẽ kết quả
    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    for v in vehicles:
        x1,y1,x2,y2 = v['box']
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,200,0),2)
        label = f"{VEHICLE_VI[v['type']]} | {v['color']}"
        cv2.putText(vis, label, (x1, max(y1-10,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,220,0), 2)
    for d in detections:
        x1,y1,x2,y2 = d['plate_box']
        cv2.rectangle(vis,(x1,y1),(x2,y2),(255,50,50),2)
        label = d['plate_text'] if d['plate_text'] else '?'
        cv2.putText(vis, label, (x1, min(y2+22,vis.shape[0]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,50,50), 2)

    return vis, detections, vehicles

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
            result_img, detections, vehicles = detect(img_bgr, conf_threshold)

        st.image(result_img, caption='Kết quả', use_column_width=True)

        if not detections and not vehicles:
            st.warning('Không phát hiện phương tiện nào trong ảnh.')
        else:
            # Hiện xe không có biển
            plates_matched = set()
            for d in detections:
                plates_matched.add(id(d))

            st.success(f'Phát hiện {len(vehicles)} phương tiện, '
                       f'{len(detections)} biển số!')

            for i, d in enumerate(detections):
                with st.expander(
                    f'🚘 Phương tiện #{i+1} — '
                    f'{VEHICLE_VI[d["vehicle_type"]]} | '
                    f'{d["vehicle_color"]} | '
                    f'Biển: {d["plate_text"] or "Không đọc được"}'
                ):
                    col1, col2 = st.columns(2)
                    col1.metric('Loại xe',    VEHICLE_VI[d['vehicle_type']])
                    col1.metric('Màu xe',     d['vehicle_color'])
                    col2.metric('Biển số',    d['plate_text'] or 'N/A')
                    col2.metric('Độ tin cậy detect', f"{d['plate_conf']:.1%}")

                    if d['plate_crop'] is not None and d['plate_crop'].size > 0:
                        crop_rgb = cv2.cvtColor(
                            d['plate_crop'], cv2.COLOR_BGR2RGB
                        )
                        st.image(crop_rgb, caption='Ảnh biển số (crop)',
                                 width=300)

            # Tóm tắt báo cáo
            if detections:
                st.markdown('---')
                st.markdown('### 📋 Báo cáo nhanh')
                for i, d in enumerate(detections):
                    plate_str = d['plate_text'] if d['plate_text'] else 'Không đọc được'
                    st.markdown(
                        f"**Xe #{i+1}:** "
                        f"{VEHICLE_VI[d['vehicle_type']]} | "
                        f"Màu {d['vehicle_color']} | "
                        f"Biển số: `{plate_str}`"
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
        all_info  = {}  # plate_text -> {type, color, count}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % 10 != 0: continue

            result_frame, dets, _ = detect(frame, conf_threshold)
            for d in dets:
                key = d['plate_text'] or f'unknown_{frame_idx}'
                if key not in all_info:
                    all_info[key] = {
                        'type':  VEHICLE_VI[d['vehicle_type']],
                        'color': d['vehicle_color'],
                        'count': 1
                    }
                else:
                    all_info[key]['count'] += 1

            frame_ph.image(result_frame,
                           caption=f'Frame {frame_idx}/{total_fr}',
                           use_column_width=True)
            progress.progress(min(frame_idx/total_fr, 1.0))

            # Cập nhật log
            if all_info:
                log_text = '**Phương tiện đã ghi nhận:**\n'
                for plate, info in all_info.items():
                    log_text += (f"- {info['type']} | "
                                 f"Màu {info['color']} | "
                                 f"Biển: `{plate}` "
                                 f"({info['count']} lần)\n")
                log_ph.markdown(log_text)

        cap.release()
        st.success('Xử lý xong!')
        st.markdown('### 📋 Tổng kết')
        for plate, info in all_info.items():
            st.markdown(
                f"🚘 **{info['type']}** | "
                f"Màu {info['color']} | "
                f"Biển: `{plate}`"
            )

st.markdown('---')
st.caption('Model: YOLOv8n | mAP50: 96.4% | Dataset: 10,127 ảnh biển số VN')
