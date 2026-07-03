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
    page_title='Nhận Diện Biển Số Xe',
    page_icon='🚗',
    layout='centered'
)

st.title('🚗 Hệ Thống Nhận Diện Biển Số Xe Việt Nam')
st.markdown('Upload ảnh hoặc video — hệ thống sẽ tự động detect biển số và loại xe.')

# ─── Load model (cache để không tải lại mỗi lần) ───
@st.cache_resource
def load_models():
    # Download best.pt từ Google Drive nếu chưa có
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        with st.spinner('Đang tải model lần đầu (chỉ mất 1 lần)...'):
            gdown.download(
                'https://drive.google.com/uc?id=1cV3XdX9FP-hIqhmOBg6QIgnYp7FbZwzt',
                model_path, quiet=False
            )
    plate_model   = YOLO(model_path)
    vehicle_model = YOLO('yolov8n.pt')
    return plate_model, vehicle_model

plate_model, vehicle_model = load_models()

VEHICLE_CLASSES = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}

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

def box_contains_center(outer, inner):
    ox1,oy1,ox2,oy2 = outer
    ix1,iy1,ix2,iy2 = inner
    cx, cy = (ix1+ix2)/2, (iy1+iy2)/2
    return ox1<=cx<=ox2 and oy1<=cy<=oy2

# ─── Hàm detect chính ───
def detect(img_bgr, conf=0.25):
    # Detect xe
    vehicles = []
    for r in vehicle_model.predict(img_bgr, conf=conf, verbose=False):
        if r.boxes is None: continue
        for b in r.boxes:
            cid = int(b.cls[0].item())
            if cid in VEHICLE_CLASSES:
                x1,y1,x2,y2 = map(int, b.xyxy[0].cpu().numpy())
                vehicles.append({
                    'box': (x1,y1,x2,y2),
                    'type': VEHICLE_CLASSES[cid]
                })

    # Detect biển số
    detections = []
    for r in plate_model.predict(img_bgr, conf=conf, verbose=False):
        if r.boxes is None: continue
        for b in r.boxes:
            px1,py1,px2,py2 = map(int, b.xyxy[0].cpu().numpy())
            plate_conf = float(b.conf[0].item())

            v_type = 'unknown'
            for v in vehicles:
                if box_contains_center(v['box'],(px1,py1,px2,py2)):
                    v_type = v['type']
                    break

            h,w = img_bgr.shape[:2]
            pad  = 4
            crop = img_bgr[max(0,py1-pad):min(h,py2+pad),
                           max(0,px1-pad):min(w,px2+pad)]
            if crop.size == 0: continue

            plate_text, ocr_conf = run_ocr(crop)

            detections.append({
                'plate_box':   (px1,py1,px2,py2),
                'plate_text':  plate_text,
                'plate_conf':  plate_conf,
                'ocr_conf':    ocr_conf,
                'vehicle_type': v_type,
            })

    # Vẽ kết quả
    vis = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    for v in vehicles:
        x1,y1,x2,y2 = v['box']
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,200,0),2)
        cv2.putText(vis, v['type'], (x1, max(y1-10,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)
    for d in detections:
        x1,y1,x2,y2 = d['plate_box']
        cv2.rectangle(vis,(x1,y1),(x2,y2),(255,50,50),2)
        label = f"{d['vehicle_type']}: {d['plate_text']}"
        cv2.putText(vis, label, (x1, min(y2+22, vis.shape[0]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,50,50), 2)

    return vis, detections

# ─── UI ───
st.sidebar.header('⚙️ Cài đặt')
conf_threshold = st.sidebar.slider(
    'Ngưỡng confidence', 0.1, 0.9, 0.25, 0.05
)

tab1, tab2 = st.tabs(['📷 Ảnh', '🎬 Video'])

with tab1:
    uploaded = st.file_uploader(
        'Upload ảnh', type=['jpg','jpeg','png'],
        key='img_upload'
    )
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner('Đang phân tích...'):
            result_img, detections = detect(img_bgr, conf_threshold)

        st.image(result_img, caption='Kết quả detect', use_column_width=True)

        if not detections:
            st.warning('Không tìm thấy biển số trong ảnh.')
        else:
            st.success(f'Tìm thấy {len(detections)} biển số!')
            for i, d in enumerate(detections):
                with st.expander(f'Biển số #{i+1} — {d["plate_text"] or "Không đọc được"}'):
                    col1, col2 = st.columns(2)
                    col1.metric('Loại xe',       d['vehicle_type'])
                    col1.metric('Biển số (OCR)',  d['plate_text'] or 'N/A')
                    col2.metric('Detect conf',   f"{d['plate_conf']:.1%}")
                    col2.metric('OCR conf',      f"{d['ocr_conf']:.1%}")

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

        st.info(f'Video: {total_fr} frames, {fps:.0f} FPS — xử lý mỗi 10 frame')

        frame_placeholder = st.empty()
        progress          = st.progress(0)
        all_plates        = set()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            # Chỉ xử lý mỗi 10 frame cho nhanh
            if frame_idx % 10 != 0:
                continue

            result_frame, dets = detect(frame, conf_threshold)
            for d in dets:
                if d['plate_text']:
                    all_plates.add(d['plate_text'])

            frame_placeholder.image(
                result_frame,
                caption=f'Frame {frame_idx}/{total_fr}',
                use_column_width=True
            )
            progress.progress(min(frame_idx/total_fr, 1.0))

        cap.release()
        st.success(f'Xong! Tổng biển số unique: {len(all_plates)}')
        if all_plates:
            st.write('Các biển số đã detect được:')
            for p in sorted(all_plates):
                st.code(p)

st.markdown('---')
st.caption('Model: YOLOv8n trained trên dataset biển số Việt Nam | mAP50: 96.4%')
