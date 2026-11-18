"""Annotrix API — AI-assisted image annotation.

Run locally:  uvicorn app.main:app --reload   (from the backend/ folder)
"""
from __future__ import annotations

import io
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from PIL import ImageOps
from pydantic import BaseModel, Field
from sqlmodel import Session, func, select

from . import config, ml
from .db import Annotation, Image, Project, WorkSession, get_session, utcnow
from .export import export_zip

app = FastAPI(title="Annotrix", version="0.1.0")


# --------------------------------------------------------------------------- schemas


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    labels: list[str] = Field(default_factory=list)
    mode: str = "assisted"


class ProjectPatch(BaseModel):
    name: Optional[str] = None
    labels: Optional[list[str]] = None
    mode: Optional[str] = None


class AnnotationIn(BaseModel):
    label: str
    x: float
    y: float
    w: float
    h: float
    polygon: Optional[list[list[float]]] = None
    source: str = "manual"
    score: Optional[float] = None
    edited: bool = False


class SessionIn(BaseModel):
    active_seconds: float = 0
    n_actions: int = 0
    n_drawn: int = 0
    n_accepted: int = 0
    n_adjusted: int = 0
    n_rejected: int = 0
    n_relabelled: int = 0
    n_ai_calls: int = 0


class SaveIn(BaseModel):
    annotations: list[AnnotationIn]
    done: bool = True
    session: Optional[SessionIn] = None


class DetectIn(BaseModel):
    conf: float = Field(0.25, ge=0.01, le=0.99)
    only_project_labels: bool = True


class SegmentIn(BaseModel):
    points: Optional[list[list[float]]] = None  # [[x, y], ...]
    point_labels: Optional[list[int]] = None  # 1 = object, 0 = background
    box: Optional[list[float]] = None  # [x1, y1, x2, y2]


class DetectorIn(BaseModel):
    weights: str


# --------------------------------------------------------------------------- helpers


def _project(s: Session, pid: int) -> Project:
    p = s.get(Project, pid)
    if not p:
        raise HTTPException(404, "project not found")
    return p


def _image(s: Session, iid: int) -> Image:
    im = s.get(Image, iid)
    if not im:
        raise HTTPException(404, "image not found")
    return im


def _clean_labels(labels: list[str]) -> list[str]:
    out: list[str] = []
    for l in labels:
        l = l.strip()
        if l and l not in out:
            out.append(l)
    return out


def _image_dict(im: Image, n_ann: int = 0) -> dict:
    d = im.model_dump()
    d["n_annotations"] = n_ann
    d["url"] = f"/api/images/{im.id}/file"
    return d


# --------------------------------------------------------------------------- system


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": ml.resolve_device(),
        "detector": config.DETECTOR_WEIGHTS,
        "segmenter": config.SEGMENTER_WEIGHTS,
        "detector_loaded": ml._detector is not None,
        "segmenter_loaded": ml._segmenter is not None,
    }


@app.get("/api/models/detector")
def detector_info():
    d = ml.detector()
    return {"weights": d.weights, "classes": list(d.names.values())}


@app.post("/api/models/detector")
def set_detector(body: DetectorIn):
    try:
        d = ml.set_detector_weights(body.weights)
    except Exception as e:  # bad path / incompatible file
        raise HTTPException(400, f"could not load detector: {e}")
    return {"weights": d.weights, "classes": list(d.names.values())}


# --------------------------------------------------------------------------- projects


@app.get("/api/projects")
def list_projects(s: Session = Depends(get_session)):
    out = []
    for p in s.exec(select(Project).order_by(Project.created_at.desc())).all():
        n_img = s.exec(select(func.count()).select_from(Image).where(Image.project_id == p.id)).one()
        n_done = s.exec(
            select(func.count()).select_from(Image).where(Image.project_id == p.id, Image.status == "done")
        ).one()
        out.append({**p.model_dump(), "n_images": n_img, "n_done": n_done})
    return out


@app.post("/api/projects", status_code=201)
def create_project(body: ProjectIn, s: Session = Depends(get_session)):
    if body.mode not in ("assisted", "manual"):
        raise HTTPException(422, "mode must be 'assisted' or 'manual'")
    p = Project(name=body.name.strip(), labels=_clean_labels(body.labels), mode=body.mode)
    s.add(p)
    s.commit()
    s.refresh(p)
    return p


@app.get("/api/projects/{pid}")
def get_project(pid: int, s: Session = Depends(get_session)):
    return _project(s, pid)


@app.patch("/api/projects/{pid}")
def patch_project(pid: int, body: ProjectPatch, s: Session = Depends(get_session)):
    p = _project(s, pid)
    if body.name is not None:
        p.name = body.name.strip()
    if body.labels is not None:
        p.labels = _clean_labels(body.labels)
    if body.mode is not None:
        if body.mode not in ("assisted", "manual"):
            raise HTTPException(422, "mode must be 'assisted' or 'manual'")
        p.mode = body.mode
    s.add(p)
    s.commit()
    s.refresh(p)
    return p


@app.delete("/api/projects/{pid}", status_code=204)
def delete_project(pid: int, s: Session = Depends(get_session)):
    p = _project(s, pid)
    images = s.exec(select(Image).where(Image.project_id == pid)).all()
    for im in images:
        for a in s.exec(select(Annotation).where(Annotation.image_id == im.id)).all():
            s.delete(a)
        for w in s.exec(select(WorkSession).where(WorkSession.image_id == im.id)).all():
            s.delete(w)
        (config.IMAGES_DIR / im.stored_name).unlink(missing_ok=True)
        s.delete(im)
    s.delete(p)
    s.commit()
    return Response(status_code=204)


# --------------------------------------------------------------------------- images


@app.post("/api/projects/{pid}/images", status_code=201)
async def upload_images(pid: int, files: list[UploadFile] = File(...), s: Session = Depends(get_session)):
    _project(s, pid)
    config.ensure_dirs()
    created, skipped = [], []
    for f in files:
        name = Path(f.filename or "image").name
        if Path(name).suffix.lower() not in config.ALLOWED_EXT:
            skipped.append({"file": name, "reason": "unsupported type"})
            continue
        raw = await f.read()
        try:
            pil = PILImage.open(io.BytesIO(raw))
            pil = ImageOps.exif_transpose(pil)  # bake in camera rotation so pixels match coordinates
        except Exception:
            skipped.append({"file": name, "reason": "not a readable image"})
            continue
        if pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        stored = f"{uuid.uuid4().hex}.png" if pil.mode == "RGBA" else f"{uuid.uuid4().hex}.jpg"
        if stored.endswith(".jpg"):
            pil.convert("RGB").save(config.IMAGES_DIR / stored, quality=95)
        else:
            pil.save(config.IMAGES_DIR / stored)
        im = Image(project_id=pid, filename=name, stored_name=stored, width=pil.width, height=pil.height)
        s.add(im)
        s.commit()
        s.refresh(im)
        created.append(_image_dict(im))
    return {"created": created, "skipped": skipped}


@app.get("/api/projects/{pid}/images")
def list_images(pid: int, s: Session = Depends(get_session)):
    _project(s, pid)
    counts = dict(
        s.exec(
            select(Annotation.image_id, func.count())
            .join(Image, Image.id == Annotation.image_id)
            .where(Image.project_id == pid)
            .group_by(Annotation.image_id)
        ).all()
    )
    imgs = s.exec(select(Image).where(Image.project_id == pid).order_by(Image.id)).all()
    return [_image_dict(im, counts.get(im.id, 0)) for im in imgs]


@app.get("/api/images/{iid}")
def get_image(iid: int, s: Session = Depends(get_session)):
    im = _image(s, iid)
    ids = s.exec(select(Image.id).where(Image.project_id == im.project_id).order_by(Image.id)).all()
    i = ids.index(im.id)
    return {
        **_image_dict(im),
        "prev_id": ids[i - 1] if i > 0 else None,
        "next_id": ids[i + 1] if i + 1 < len(ids) else None,
        "position": i + 1,
        "total": len(ids),
    }


@app.get("/api/images/{iid}/file")
def image_file(iid: int, s: Session = Depends(get_session)):
    im = _image(s, iid)
    return FileResponse(config.IMAGES_DIR / im.stored_name)


@app.delete("/api/images/{iid}", status_code=204)
def delete_image(iid: int, s: Session = Depends(get_session)):
    im = _image(s, iid)
    for a in s.exec(select(Annotation).where(Annotation.image_id == iid)).all():
        s.delete(a)
    for w in s.exec(select(WorkSession).where(WorkSession.image_id == iid)).all():
        s.delete(w)
    (config.IMAGES_DIR / im.stored_name).unlink(missing_ok=True)
    s.delete(im)
    s.commit()
    return Response(status_code=204)


# --------------------------------------------------------------------------- annotations


@app.get("/api/images/{iid}/annotations")
def get_annotations(iid: int, s: Session = Depends(get_session)):
    _image(s, iid)
    return s.exec(select(Annotation).where(Annotation.image_id == iid).order_by(Annotation.id)).all()


@app.put("/api/images/{iid}/annotations")
def save_annotations(iid: int, body: SaveIn, s: Session = Depends(get_session)):
    """Replace the image's annotations with the editor state and log the work session."""
    im = _image(s, iid)
    project = _project(s, im.project_id)
    for a in s.exec(select(Annotation).where(Annotation.image_id == iid)).all():
        s.delete(a)
    saved = []
    for a in body.annotations:
        # clamp to the image so exports are always valid
        x1, y1 = max(0.0, a.x), max(0.0, a.y)
        x2, y2 = min(float(im.width), a.x + a.w), min(float(im.height), a.y + a.h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        row = Annotation(
            image_id=iid,
            label=a.label.strip() or "object",
            x=x1,
            y=y1,
            w=x2 - x1,
            h=y2 - y1,
            polygon=a.polygon,
            source=a.source,
            score=a.score,
            edited=a.edited,
            updated_at=utcnow(),
        )
        s.add(row)
        saved.append(row)
    im.status = "done" if body.done else "new"
    s.add(im)
    if body.session is not None:
        s.add(
            WorkSession(
                image_id=iid,
                project_id=im.project_id,
                mode=project.mode,
                n_final=len(saved),
                **body.session.model_dump(),
            )
        )
    s.commit()
    for r in saved:
        s.refresh(r)
    return {"saved": len(saved), "status": im.status, "annotations": saved}


# --------------------------------------------------------------------------- AI assistance


@app.post("/api/images/{iid}/ai/detect")
def ai_detect(iid: int, body: DetectIn = DetectIn(), s: Session = Depends(get_session)):
    """Propose boxes for the whole image. Suggestions are not stored: the user
    accepts, edits or rejects them in the editor and the result is saved normally."""
    im = _image(s, iid)
    project = _project(s, im.project_id)
    if project.mode == "manual":
        raise HTTPException(409, "AI assistance is disabled in manual mode")
    labels = project.labels if (body.only_project_labels and project.labels) else None
    t0 = time.perf_counter()
    with ml.infer_lock:
        dets = ml.detector().predict(config.IMAGES_DIR / im.stored_name, conf=body.conf, labels=labels)
    return {
        "model": ml.detector().weights,
        "ms": round((time.perf_counter() - t0) * 1000),
        "suggestions": [d.__dict__ for d in dets],
    }


@app.post("/api/images/{iid}/ai/segment")
def ai_segment(iid: int, body: SegmentIn, s: Session = Depends(get_session)):
    """Click-to-segment / box-to-mask with the promptable segmenter."""
    im = _image(s, iid)
    project = _project(s, im.project_id)
    if project.mode == "manual":
        raise HTTPException(409, "AI assistance is disabled in manual mode")
    if not body.points and not body.box:
        raise HTTPException(422, "give at least one point or a box")
    t0 = time.perf_counter()
    with ml.infer_lock:
        r = ml.segmenter().segment(
            config.IMAGES_DIR / im.stored_name, body.points, body.point_labels, body.box
        )
    if r is None:
        raise HTTPException(422, "no object found at that prompt")
    return {"model": ml.segmenter().weights, "ms": round((time.perf_counter() - t0) * 1000), **r.__dict__}


# --------------------------------------------------------------------------- export & stats


@app.get("/api/projects/{pid}/export")
def export(
    pid: int,
    format: str = Query("coco", pattern="^(coco|yolo|yolo-seg)$"),
    only_done: bool = False,
    s: Session = Depends(get_session),
):
    p = _project(s, pid)
    images = s.exec(select(Image).where(Image.project_id == pid).order_by(Image.id)).all()
    anns: dict[int, list[Annotation]] = {}
    for a in s.exec(
        select(Annotation).join(Image, Image.id == Annotation.image_id).where(Image.project_id == pid)
    ).all():
        anns.setdefault(a.image_id, []).append(a)
    data = export_zip(p, images, anns, format, only_done)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in p.name) or "project"
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe}_{format}.zip"'},
    )


@app.get("/api/projects/{pid}/sessions")
def sessions(pid: int, s: Session = Depends(get_session)):
    """Raw work-session log, used by eval/effort.py."""
    _project(s, pid)
    rows = s.exec(select(WorkSession).where(WorkSession.project_id == pid).order_by(WorkSession.id)).all()
    names = {im.id: im.filename for im in s.exec(select(Image).where(Image.project_id == pid)).all()}
    return [{**r.model_dump(), "filename": names.get(r.image_id)} for r in rows]


@app.get("/api/projects/{pid}/stats")
def stats(pid: int, s: Session = Depends(get_session)):
    """Per-mode summary for the dashboard: seconds and actions per image."""
    _project(s, pid)
    rows = s.exec(select(WorkSession).where(WorkSession.project_id == pid)).all()
    # sum all sessions per image (an image may be revisited)
    per_image: dict[tuple[str, int], dict] = {}
    for r in rows:
        k = (r.mode, r.image_id)
        d = per_image.setdefault(k, {"seconds": 0.0, "actions": 0, "objects": 0, "accepted": 0, "adjusted": 0, "drawn": 0})
        d["seconds"] += r.active_seconds
        d["actions"] += r.n_actions
        d["objects"] = r.n_final
        d["accepted"] += r.n_accepted
        d["adjusted"] += r.n_adjusted
        d["drawn"] += r.n_drawn
    out: dict[str, dict] = {}
    for (mode, _), d in per_image.items():
        m = out.setdefault(mode, {"images": 0, "seconds": 0.0, "actions": 0, "objects": 0, "accepted": 0, "adjusted": 0, "drawn": 0})
        m["images"] += 1
        for k in ("seconds", "actions", "objects", "accepted", "adjusted", "drawn"):
            m[k] += d[k]
    for m in out.values():
        n = max(m["images"], 1)
        objs = max(m["objects"], 1)
        m["seconds_per_image"] = round(m["seconds"] / n, 2)
        m["seconds_per_object"] = round(m["seconds"] / objs, 2)
        m["actions_per_image"] = round(m["actions"] / n, 2)
    return out


# --------------------------------------------------------------------------- frontend

if (config.STATIC_DIR / "index.html").is_file():
    app.mount("/assets", StaticFiles(directory=config.STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = config.STATIC_DIR / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(config.STATIC_DIR / "index.html")
