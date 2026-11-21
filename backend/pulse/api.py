from __future__ import annotations
import os, uuid, json, shutil, threading, time
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .video import analyze_video
from .database import save as save_analysis, list_history

ROOT=Path(os.getenv("PULSE_DATA_DIR", Path(__file__).resolve().parents[1]/"data")); UPLOADS=ROOT/"uploads"; RESULTS=ROOT/"results"
for p in (UPLOADS,RESULTS): p.mkdir(parents=True,exist_ok=True)
LIMIT=int(os.getenv("PULSE_MAX_UPLOAD_MB","500"))*1024*1024
ALLOWED={".mp4",".mov",".avi",".mkv"}
LOCK=threading.Lock(); JOBS={}
app=FastAPI(title="PULSE",version="0.1.0",description="Physiological video signal analysis. No classifier is installed by default.")
app.add_middleware(CORSMiddleware,allow_origins=os.getenv("PULSE_CORS_ORIGINS","http://localhost:5173").split(","),allow_methods=["*"],allow_headers=["*"])

def persist(job_id, data):
    result_path=RESULTS/f"{job_id}.json"
    result_path.write_text(json.dumps(data,allow_nan=False),encoding="utf-8")
    save_analysis(data,result_path)
    with LOCK: JOBS[job_id].update(status="complete",progress=100,stage="complete",result=data)

def run_job(job_id,path):
    with LOCK: JOBS[job_id].update(status="processing",stage="starting")
    try:
        data=analyze_video(str(path),progress=lambda p: JOBS[job_id].update(progress=p,stage="analysing frames"))
        data["id"]=job_id; data["created_at"]=JOBS[job_id]["created_at"]; persist(job_id,data)
    except Exception as exc:
        with LOCK: JOBS[job_id].update(status="failed",stage="failed",error=str(exc))
    finally:
        try: path.unlink(missing_ok=True)
        except Exception: pass

@app.get("/api/health")
def health(): return {"status":"ok","classifier":{"trained":False,"version":None},"storage":"local","time":datetime.now(timezone.utc).isoformat()}

@app.post("/api/analyze/video",status_code=202)
async def upload_video(background: BackgroundTasks, video: UploadFile=File(...)):
    suffix=Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED: raise HTTPException(415,"Unsupported video format")
    jid=str(uuid.uuid4()); path=UPLOADS/f"{jid}{suffix}"; size=0
    try:
        with path.open("wb") as out:
            while chunk:=await video.read(1024*1024):
                size+=len(chunk)
                if size>LIMIT: raise HTTPException(413,"Video exceeds configured upload size")
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True); raise
    if size==0: path.unlink(missing_ok=True); raise HTTPException(400,"Empty upload")
    now=datetime.now(timezone.utc).isoformat()
    with LOCK: JOBS[jid]={"id":jid,"filename":Path(video.filename).name,"status":"queued","progress":0,"stage":"queued","created_at":now}
    background.add_task(run_job,jid,path)
    return {"id":jid,"status":"queued","filename":Path(video.filename).name}

def getjob(jid):
    if jid in JOBS: return JOBS[jid]
    f=RESULTS/f"{jid}.json"
    if f.exists():
        data=json.loads(f.read_text(encoding="utf-8")); JOBS[jid]={"id":jid,"status":"complete","progress":100,"stage":"complete","result":data,"created_at":data.get("created_at")}; return JOBS[jid]
    raise HTTPException(404,"Analysis not found")

@app.get("/api/analyze/{analysis_id}/status")
def status(analysis_id):
    j=getjob(analysis_id); return {k:j.get(k) for k in ("id","filename","status","progress","stage","error","created_at")}

@app.get("/api/analyze/{analysis_id}/results")
def results(analysis_id):
    j=getjob(analysis_id)
    if j["status"]!="complete": raise HTTPException(409,"Analysis is not complete")
    data=j["result"]; return {k:v for k,v in data.items() if k!="signals" and k!="overlay_frame"}

@app.get("/api/analyze/{analysis_id}/signals")
def signals(analysis_id):
    j=getjob(analysis_id)
    if j["status"]!="complete": raise HTTPException(409,"Analysis is not complete")
    return {"timestamps":j["result"]["timestamps"],"regions":j["result"]["signals"],"overlay_frame":j["result"].get("overlay_frame")}

@app.get("/api/analyze/{analysis_id}/report")
def report(analysis_id, format="json"):
    j=getjob(analysis_id)
    if j["status"]!="complete": raise HTTPException(409,"Analysis is not complete")
    if format=="json": return FileResponse(RESULTS/f"{analysis_id}.json",media_type="application/json",filename=f"pulse-{analysis_id}.json")
    if format!="pdf": raise HTTPException(400,"format must be json or pdf")
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen.canvas import Canvas
    out=RESULTS/f"{analysis_id}.pdf"; d=j["result"]; c=Canvas(str(out),pagesize=letter); y=750
    c.setFont("Helvetica-Bold",18); c.drawString(54,y,"PULSE | Physiological Signal Analysis"); y-=32
    c.setFont("Helvetica",10)
    lines=[f"File: {d['metadata']['filename']}",f"Duration: {d['metadata']['duration']:.2f}s  |  FPS: {d['metadata']['fps']:.2f}  |  Resolution: {d['metadata']['width']} x {d['metadata']['height']}",f"Face detection coverage: {d['metadata']['face_coverage']:.1%}",f"Selected ROI: {d['selected_roi'] or 'INSUFFICIENT SIGNAL QUALITY'}",d['interpretation'],"", "ROI measurements (heart rates shown only when measurable):"]
    for r in d["regions"]: lines.append(f"{r['roi']}: POS {fmt(r['pos'].get('heart_rate_bpm'))} BPM; CHROM {fmt(r['chrom'].get('heart_rate_bpm'))} BPM; SNR {fmt(r.get('snr_db'))} dB; quality {r['quality']:.2f}; missing {r['missing_fraction']:.1%}")
    lines += ["", "Method: Face Mesh landmark regions; POS and CHROM chrominance transforms; Butterworth band-pass; Welch PSD.","Interpretation: rPPG is supporting evidence and cannot independently prove authenticity or manipulation.","No trained, validated classifier is installed. This document reports physiological measurements only."]
    for line in lines:
        c.drawString(54,y,line[:105]); y-=18
        if y<55: c.showPage(); y=750; c.setFont("Helvetica",10)
    c.save(); return FileResponse(out,media_type="application/pdf",filename=f"pulse-{analysis_id}.pdf")

def fmt(v): return "N/A" if v is None else f"{v:.2f}"

@app.get("/api/history")
def history():
    return list_history()

@app.post("/api/roi/compare")
def compare(payload:dict):
    rows=payload.get("regions",[]); threshold=float(payload.get("threshold",.45))
    if not 0<=threshold<=1: raise HTTPException(400,"threshold must be between 0 and 1")
    from .signals import roi_rank
    selected=roi_rank(rows,threshold); return {"regions":rows,"selected_roi":selected.get("roi") if selected else None}

@app.get("/api/classifier/status")
def classifier_status(): return {"trained":False,"version":None,"message":"ANALYSIS ONLY — CLASSIFIER NOT TRAINED"}

@app.post("/api/classifier/train")
def train_classifier(): raise HTTPException(501,"Training data ingestion and validated model training are not available in this release")
