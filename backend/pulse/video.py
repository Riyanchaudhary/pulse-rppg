"""Sample video frames, extract facial ROI RGB means, and assemble measured results."""
from __future__ import annotations
import time
import cv2
import numpy as np
from .signals import pos, chrom, bandpass, spectral_metrics, signal_quality, roi_rank

# MediaPipe Face Mesh landmark polygons (forehead, left/right cheek, nose).
POLYGONS = {
 "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 340, 346, 347, 348, 349, 350, 451, 452, 453, 464, 417, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12],
 "left_cheek": [50, 101, 118, 117, 123, 147, 187, 207, 206, 203, 129, 49],
 "right_cheek": [280, 330, 347, 346, 352, 376, 411, 427, 426, 423, 358, 279],
 "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0]
}

def analyze_video(path: str, sample_fps=15, max_seconds=180, progress=None):
    started=time.time(); cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError("The uploaded file is not a readable video")
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); duration=count/fps if fps>0 else 0
    if not fps or duration<=0: cap.release(); raise ValueError("Video has no usable frame rate or duration")
    if duration>max_seconds: cap.release(); raise ValueError(f"Video exceeds the {max_seconds}s duration limit")
    try:
        import mediapipe as mp
        mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=5,refine_landmarks=False,min_detection_confidence=.5,min_tracking_confidence=.5)
    except Exception as e:
        cap.release(); raise RuntimeError("MediaPipe Face Mesh is unavailable; install backend requirements") from e
    stride=max(1,round(fps/sample_fps)); timestamps=[]; rgb={k:[] for k in POLYGONS}; vis={k:[] for k in POLYGONS}; overlays=[]; faces=0; frame_i=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if frame_i % stride: frame_i+=1; continue
            t=frame_i/fps; h,w=frame.shape[:2]; result=mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); timestamps.append(t)
            landmarks=None
            if result.multi_face_landmarks:
                faces+=1
                # Deterministic primary face: largest landmark bounding box.
                lm=max(result.multi_face_landmarks,key=lambda f:(max(p.x for p in f.landmark)-min(p.x for p in f.landmark))*(max(p.y for p in f.landmark)-min(p.y for p in f.landmark))).landmark
                landmarks=lm
            overlay=frame.copy()
            for name,ids in POLYGONS.items():
                values=[]; visible=0.
                if landmarks:
                    pts=np.array([[int(np.clip(landmarks[i].x*w,0,w-1)),int(np.clip(landmarks[i].y*h,0,h-1))] for i in ids],np.int32)
                    mask=np.zeros((h,w),np.uint8); cv2.fillPoly(mask,[pts],255)
                    # Only central skin polygon pixels; reject very small or absent masks.
                    pixels=frame[mask>0]
                    if len(pixels)>=20: values=np.mean(pixels,axis=0)[::-1].tolist(); visible=1.
                    cv2.polylines(overlay,[pts],True,(110,145,116),2)
                rgb[name].append(values if values else [np.nan]*3); vis[name].append(visible)
            if landmarks: overlays.append(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
            frame_i+=1
            if progress and frame_i%max(1,stride*10)==0: progress(min(85,int(10+75*frame_i/max(1,count))))
    finally: cap.release(); mesh.close()
    if not timestamps: raise ValueError("No frames could be sampled")
    effective=fps/stride; regions=[]; signals={}
    for name,rows in rgb.items():
        arr=np.array(rows,float); valid=np.isfinite(arr).all(axis=1); missing=1-float(valid.mean()); clean=lambda a: np.where(np.isfinite(a),a,0).tolist()
        rgb_json=arr.tolist(); rgb_json=[[None if not np.isfinite(v) else float(v) for v in row] for row in rgb_json]
        signals[name]={"rgb":rgb_json,"timestamps":timestamps,"visibility":vis[name]}
        if valid.sum()<max(12,int(effective*4)):
            p=c=np.full(len(arr),np.nan); pm=cm={"heart_rate_bpm":None,"dominant_frequency_hz":None,"snr_db":None,"peak_prominence":None,"psd_frequency_hz":[],"psd":[]}; quality=0.
        else:
            # Missing frames remain NaN in stored measurements. For transform computation only, interpolate gaps;
            # long gaps lower coverage/continuity and the original RGB data is never overwritten.
            interp=np.column_stack([np.interp(np.arange(len(arr)),np.flatnonzero(valid),arr[valid,j]) for j in range(3)])
            p=bandpass(pos(interp,effective),effective); c=bandpass(chrom(interp),effective)
            pm=spectral_metrics(p,effective); cm=spectral_metrics(c,effective)
            quality=signal_quality(pm,missing,float(np.mean(vis[name])),float(max(0,1-missing*1.5)))
        signals[name].update(pos=clean(p),chrom=clean(c),pos_filtered=clean(p),chrom_filtered=clean(c))
        hr=[v for v in (pm["heart_rate_bpm"],cm["heart_rate_bpm"]) if v is not None]
        regions.append({"roi":name,"pos":pm,"chrom":cm,"snr_db":pm.get("snr_db"),"heart_rate_bpm":float(np.mean(hr)) if hr else None,"quality":quality,"missing_fraction":missing,"visibility":float(np.mean(vis[name]))})
    chosen=roi_rank(regions)
    return {"metadata":{"filename":path.split("/")[-1].split("\\")[-1],"duration":duration,"fps":fps,"width":width,"height":height,"total_frames":count,"analysed_frames":len(timestamps),"face_coverage":faces/max(1,len(timestamps)),"processing_seconds":time.time()-started,"sample_fps":effective},"regions":regions,"selected_roi":chosen["roi"] if chosen else None,"interpretation":"ANALYSIS ONLY — CLASSIFIER NOT TRAINED","signals":signals,"timestamps":timestamps,"overlay_frame":overlays[0].tolist() if overlays else None,"pipeline_stages":["metadata","face_detection","roi_rgb_extraction","pos_chrom","quality_assessment"],"warnings":["rPPG is supporting evidence and cannot independently prove authenticity or manipulation."]}
