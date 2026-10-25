import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Circle, Group, Image as KImage, Layer, Line, Rect, Stage, Transformer } from "react-konva";
import type Konva from "konva";
import { Ann, api, ImageInfo, labelColor, Project, SessionStats, Source } from "../api";

// --------------------------------------------------------------------------- types

type Tool = "select" | "box" | "segment";

interface Shape extends Ann {
  key: string;
  status: "kept" | "suggested";
}

interface PromptPoint {
  x: number;
  y: number;
  l: 0 | 1; // 1 = object, 0 = background
}

let keySeq = 0;
const newKey = () => `s${++keySeq}`;

const IDLE_CUTOFF_S = 30; // time with no input is not counted as work

function emptyCounters(): SessionStats {
  return {
    active_seconds: 0,
    n_actions: 0,
    n_drawn: 0,
    n_accepted: 0,
    n_adjusted: 0,
    n_rejected: 0,
    n_relabelled: 0,
    n_ai_calls: 0,
  };
}

function readPref<T>(k: string, d: T): T {
  try {
    const v = localStorage.getItem(k);
    return v === null ? d : (JSON.parse(v) as T);
  } catch {
    return d;
  }
}
function writePref(k: string, v: unknown) {
  try {
    localStorage.setItem(k, JSON.stringify(v));
  } catch {
    /* storage unavailable */
  }
}

// Move a polygon along with its bounding box (old box -> new box).
function remapPolygon(poly: number[][] | null | undefined, from: Ann, to: Ann): number[][] | null | undefined {
  if (!poly || from.w <= 0 || from.h <= 0) return poly;
  return poly.map(([x, y]) => [to.x + ((x - from.x) / from.w) * to.w, to.y + ((y - from.y) / from.h) * to.h]);
}

// --------------------------------------------------------------------------- component

export default function Annotator() {
  const iid = Number(useParams().iid);
  const nav = useNavigate();

  const [info, setInfo] = useState<ImageInfo | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [shapes, setShapes] = useState<Shape[]>([]);
  const [history, setHistory] = useState<Shape[][]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [label, setLabel] = useState<string>("");
  const [dirty, setDirty] = useState(false);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [conf, setConf] = useState<number>(() => readPref("la.conf", 0.3));
  const [autoSuggest, setAutoSuggest] = useState<boolean>(() => readPref("la.auto", true));
  const [showMasks, setShowMasks] = useState(true);

  // segment tool state
  const [prompt, setPrompt] = useState<PromptPoint[]>([]);
  const [preview, setPreview] = useState<{ polygon: number[][]; x: number; y: number; w: number; h: number; score: number | null } | null>(null);

  // view transform: screen = image * scale + offset
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);

  // drawing / panning
  const drawStart = useRef<{ x: number; y: number } | null>(null);
  const [draft, setDraft] = useState<{ x: number; y: number; w: number; h: number } | null>(null);
  const panStart = useRef<{ px: number; py: number; vx: number; vy: number } | null>(null);
  const [spaceDown, setSpaceDown] = useState(false);

  // effort counters + active timer
  const counters = useRef<SessionStats>(emptyCounters());
  const lastInput = useRef<number>(Date.now());
  const lastTick = useRef<number>(Date.now());
  const [elapsed, setElapsed] = useState(0);

  const assisted = project?.mode === "assisted";
  const labels = project?.labels ?? [];

  // ------------------------------------------------------------------ helpers

  const bump = (k: keyof SessionStats, n = 1) => {
    (counters.current[k] as number) += n;
  };
  const touch = () => {
    lastInput.current = Date.now();
  };

  const commit = useCallback(
    (next: Shape[] | ((prev: Shape[]) => Shape[])) => {
      setShapes((prev) => {
        const value = typeof next === "function" ? (next as (p: Shape[]) => Shape[])(prev) : next;
        setHistory((h) => [...h.slice(-49), prev]);
        return value;
      });
      setDirty(true);
      bump("n_actions");
      touch();
    },
    [],
  );

  const toImage = (sx: number, sy: number) => ({ x: (sx - view.x) / view.scale, y: (sy - view.y) / view.scale });
  const clampToImage = (p: { x: number; y: number }) =>
    info ? { x: Math.min(Math.max(p.x, 0), info.width), y: Math.min(Math.max(p.y, 0), info.height) } : p;

  const fit = useCallback(() => {
    if (!info) return;
    const pad = 24;
    const s = Math.min((size.w - pad * 2) / info.width, (size.h - pad * 2) / info.height);
    setView({ scale: s, x: (size.w - info.width * s) / 2, y: (size.h - info.height * s) / 2 });
  }, [info, size]);

  // ------------------------------------------------------------------ loading

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setStatus("");
      setImg(null);
      setPrompt([]);
      setPreview(null);
      setSelected(null);
      setHistory([]);
      counters.current = emptyCounters();
      lastTick.current = Date.now();
      lastInput.current = Date.now();
      setElapsed(0);
      const i = await api.image(iid);
      const [p, anns] = await Promise.all([api.project(i.project_id), api.annotations(iid)]);
      if (cancelled) return;
      setInfo(i);
      setProject(p);
      setLabel((l) => (p.labels.includes(l) ? l : p.labels[0] ?? "object"));
      const loaded: Shape[] = anns.map((a) => ({ ...a, key: newKey(), status: "kept" }));
      setShapes(loaded);
      setDirty(false);
      const el = new window.Image();
      el.onload = () => !cancelled && setImg(el);
      el.src = i.url;
      if (p.mode === "assisted" && readPref("la.auto", true) && loaded.length === 0) {
        runDetect(i.id, true);
      }
    })().catch((e) => setStatus(e.message));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [iid]);

  useEffect(() => {
    fit();
  }, [fit, img]);

  // active-time clock
  useEffect(() => {
    const t = setInterval(() => {
      const now = Date.now();
      const dt = (now - lastTick.current) / 1000;
      lastTick.current = now;
      if (document.visibilityState === "visible" && (now - lastInput.current) / 1000 < IDLE_CUTOFF_S) {
        counters.current.active_seconds += dt;
        setElapsed(counters.current.active_seconds);
      }
    }, 500);
    return () => clearInterval(t);
  }, []);

  // keep the transformer attached to the selected box
  useEffect(() => {
    const tr = trRef.current;
    const stage = stageRef.current;
    if (!tr || !stage) return;
    const node = selected && tool === "select" ? stage.findOne(`#${selected}`) : null;
    tr.nodes(node ? [node] : []);
    tr.getLayer()?.batchDraw();
  }, [selected, shapes, tool]);

  // ------------------------------------------------------------------ AI calls

  async function runDetect(imageId = iid, auto = false) {
    setBusy(true);
    setStatus(auto ? "Running detector…" : "Detecting…");
    try {
      const r = await api.detect(imageId, readPref("la.conf", conf));
      if (!auto) bump("n_ai_calls");
      setShapes((prev) => {
        // replace earlier unaccepted suggestions, keep everything the user kept
        const kept = prev.filter((s) => s.status === "kept");
        const sugg: Shape[] = r.suggestions.map((s) => ({
          key: newKey(),
          label: s.label,
          x: s.x,
          y: s.y,
          w: s.w,
          h: s.h,
          score: s.score,
          source: "ai_detect" as Source,
          status: "suggested",
          edited: false,
        }));
        return [...kept, ...sugg];
      });
      setStatus(
        r.suggestions.length
          ? `${r.suggestions.length} suggestion(s) in ${r.ms} ms — A accepts all, click one to review`
          : `No objects found above ${Math.round(conf * 100)}% confidence (${r.ms} ms)`,
      );
    } catch (e) {
      setStatus(`Detector: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  async function runSegment(points: PromptPoint[], box?: number[]) {
    if (!points.length && !box) {
      setPreview(null);
      return;
    }
    setBusy(true);
    bump("n_ai_calls");
    try {
      const r = await api.segment(iid, {
        points: points.length ? points.map((p) => [p.x, p.y]) : undefined,
        point_labels: points.length ? points.map((p) => p.l) : undefined,
        box,
      });
      setPreview(r);
      setStatus(`Mask ready (${r.ms} ms) — Enter to keep, click to add (+) / shift-click to remove (−)`);
    } catch (e) {
      setPreview(null);
      setStatus(`Segmenter: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  function keepPreview() {
    if (!preview) return;
    commit((prev) => [
      ...prev,
      {
        key: newKey(),
        label,
        x: preview.x,
        y: preview.y,
        w: preview.w,
        h: preview.h,
        polygon: preview.polygon,
        score: preview.score,
        source: "ai_segment",
        status: "kept",
        edited: false,
      },
    ]);
    bump("n_drawn");
    setPrompt([]);
    setPreview(null);
    setStatus("Object added.");
  }

  async function refineSelected() {
    const s = shapes.find((x) => x.key === selected);
    if (!s) return;
    setBusy(true);
    bump("n_ai_calls");
    try {
      const r = await api.segment(iid, { box: [s.x, s.y, s.x + s.w, s.y + s.h] });
      commit((prev) =>
        prev.map((x) =>
          x.key === s.key
            ? {
                ...x,
                polygon: r.polygon,
                x: r.x,
                y: r.y,
                w: r.w,
                h: r.h,
                status: "kept",
                edited: x.source !== "manual" ? true : x.edited,
                source: x.source === "manual" ? "ai_segment" : x.source,
              }
            : x,
        ),
      );
      setStatus(`Mask fitted to box (${r.ms} ms)`);
    } catch (e) {
      setStatus(`Segmenter: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  // ------------------------------------------------------------------ edits

  function acceptAll() {
    const n = shapes.filter((s) => s.status === "suggested").length;
    if (!n) return;
    commit((prev) => prev.map((s) => (s.status === "suggested" ? { ...s, status: "kept" } : s)));
    setStatus(`Accepted ${n} suggestion(s).`);
  }

  function accept(key: string) {
    commit((prev) => prev.map((s) => (s.key === key ? { ...s, status: "kept" } : s)));
  }

  function remove(key: string | null) {
    if (!key) return;
    const s = shapes.find((x) => x.key === key);
    if (!s) return;
    if (s.source === "ai_detect" && !s.edited) bump("n_rejected");
    commit((prev) => prev.filter((x) => x.key !== key));
    setSelected(null);
  }

  function relabel(key: string, newLabel: string) {
    const s = shapes.find((x) => x.key === key);
    if (!s || s.label === newLabel) return;
    bump("n_relabelled");
    commit((prev) =>
      prev.map((x) =>
        x.key === key ? { ...x, label: newLabel, status: "kept", edited: x.source !== "manual" ? true : x.edited } : x,
      ),
    );
  }

  function updateBox(key: string, box: { x: number; y: number; w: number; h: number }) {
    commit((prev) =>
      prev.map((s) => {
        if (s.key !== key) return s;
        const next = { ...s, ...box };
        return {
          ...next,
          polygon: remapPolygon(s.polygon, s, next),
          status: "kept",
          edited: s.source !== "manual" ? true : s.edited,
        };
      }),
    );
  }

  function undo() {
    setHistory((h) => {
      if (!h.length) return h;
      setShapes(h[h.length - 1]);
      setDirty(true);
      return h.slice(0, -1);
    });
  }

  // ------------------------------------------------------------------ saving / navigation

  async function save(done = true): Promise<boolean> {
    if (!info) return false;
    const kept = shapes.filter((s) => s.status === "kept");
    const leftover = shapes.filter((s) => s.status === "suggested").length;
    const c = counters.current;
    const session: SessionStats = {
      ...c,
      active_seconds: Math.round(c.active_seconds * 10) / 10,
      n_rejected: c.n_rejected + leftover,
      n_accepted: kept.filter((s) => s.source === "ai_detect" && !s.edited).length,
      n_adjusted: kept.filter((s) => s.source === "ai_detect" && s.edited).length,
    };
    setBusy(true);
    try {
      const anns: Ann[] = kept.map(({ label, x, y, w, h, polygon, source, score, edited }) => ({
        label, x, y, w, h, polygon, source, score, edited,
      }));
      const r = await api.save(info.id, anns, done, session);
      counters.current = emptyCounters();
      setElapsed(0);
      setDirty(false);
      setStatus(`Saved ${r.saved} object(s).`);
      return true;
    } catch (e) {
      setStatus(`Save failed: ${(e as Error).message}`);
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function go(id: number | null | undefined, saveFirst: boolean) {
    if (saveFirst && !(await save(true))) return;
    if (!saveFirst && dirty && !confirm("Discard unsaved changes on this image?")) return;
    if (id) nav(`/annotate/${id}`);
    else if (info) nav(`/projects/${info.project_id}`);
  }

  // ------------------------------------------------------------------ keyboard

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement;
      if (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA") return;
      touch();
      if (e.code === "Space") {
        setSpaceDown(e.type === "keydown");
        e.preventDefault();
        return;
      }
      if (e.type !== "keydown") return;
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === "z") return e.preventDefault(), undo();
      if ((e.ctrlKey || e.metaKey) && k === "s") return e.preventDefault(), void save(true);
      if (k === "v") setTool("select");
      else if (k === "b") setTool("box");
      else if (k === "s" && assisted) setTool("segment");
      else if (k === "d" && assisted) runDetect();
      else if (k === "a" && assisted) acceptAll();
      else if (k === "r" && assisted && selected) refineSelected();
      else if (k === "f") fit();
      else if (k === "m") setShowMasks((m) => !m);
      else if (k === "delete" || k === "backspace") remove(selected);
      else if (k === "escape") {
        setPrompt([]);
        setPreview(null);
        setDraft(null);
        drawStart.current = null;
        setSelected(null);
      } else if (k === "enter") {
        if (preview) keepPreview();
        else void go(info?.next_id, true);
      } else if (k === "arrowright") void go(info?.next_id, dirty);
      else if (k === "arrowleft") void go(info?.prev_id, dirty);
      else if (/^[1-9]$/.test(k)) {
        const l = labels[Number(k) - 1];
        if (l) {
          setLabel(l);
          if (selected) relabel(selected, l);
        }
      }
    }
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  });

  // ------------------------------------------------------------------ pointer handling

  function pointer() {
    const p = stageRef.current?.getPointerPosition();
    return p ? toImage(p.x, p.y) : null;
  }

  function onWheel(e: Konva.KonvaEventObject<WheelEvent>) {
    e.evt.preventDefault();
    const p = stageRef.current?.getPointerPosition();
    if (!p) return;
    const factor = e.evt.deltaY < 0 ? 1.12 : 1 / 1.12;
    const scale = Math.min(Math.max(view.scale * factor, 0.05), 40);
    const ip = toImage(p.x, p.y);
    setView({ scale, x: p.x - ip.x * scale, y: p.y - ip.y * scale });
  }

  function onDown(e: Konva.KonvaEventObject<MouseEvent>) {
    touch();
    const p = stageRef.current?.getPointerPosition();
    if (!p) return;
    const onEmpty = e.target === e.target.getStage() || e.target.name() === "image";
    if (e.evt.button === 1 || spaceDown || (tool === "select" && onEmpty && e.evt.button === 0)) {
      panStart.current = { px: p.x, py: p.y, vx: view.x, vy: view.y };
      if (tool === "select" && onEmpty) setSelected(null);
      return;
    }
    if (e.evt.button !== 0) return;
    if (tool === "box" || tool === "segment") {
      const ip = clampToImage(toImage(p.x, p.y));
      drawStart.current = ip;
      setDraft({ x: ip.x, y: ip.y, w: 0, h: 0 });
    }
  }

  function onMove() {
    const p = stageRef.current?.getPointerPosition();
    if (!p) return;
    if (panStart.current) {
      const s = panStart.current;
      setView((v) => ({ ...v, x: s.vx + p.x - s.px, y: s.vy + p.y - s.py }));
      return;
    }
    if (drawStart.current) {
      const a = drawStart.current;
      const b = clampToImage(toImage(p.x, p.y));
      setDraft({ x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(b.x - a.x), h: Math.abs(b.y - a.y) });
    }
  }

  function onUp(e: Konva.KonvaEventObject<MouseEvent>) {
    if (panStart.current) {
      panStart.current = null;
      return;
    }
    if (!drawStart.current) return;
    const d = draft;
    drawStart.current = null;
    setDraft(null);
    const minSize = 4 / view.scale;
    if (tool === "box") {
      if (d && d.w > minSize && d.h > minSize) {
        const key = newKey();
        commit((prev) => [...prev, { key, label, ...d, source: "manual", status: "kept", edited: false }]);
        bump("n_drawn");
        setSelected(key);
      }
    } else if (tool === "segment") {
      if (d && d.w > minSize && d.h > minSize) {
        // dragged a box: box prompt (+ any points already placed)
        void runSegment(prompt, [d.x, d.y, d.x + d.w, d.y + d.h]);
      } else {
        const ip = pointer();
        if (!ip || !info || ip.x < 0 || ip.y < 0 || ip.x > info.width || ip.y > info.height) return;
        const neg = e.evt.shiftKey || e.evt.altKey;
        const pts = [...prompt, { x: ip.x, y: ip.y, l: (neg ? 0 : 1) as 0 | 1 }];
        setPrompt(pts);
        void runSegment(pts);
      }
    }
  }

  // ------------------------------------------------------------------ render

  const sw = 2 / view.scale;
  const selShape = shapes.find((s) => s.key === selected) ?? null;
  const nSuggested = shapes.filter((s) => s.status === "suggested").length;
  const nKept = shapes.length - nSuggested;
  const cursor =
    spaceDown || panStart.current ? "grab" : tool === "box" ? "crosshair" : tool === "segment" ? "cell" : "default";

  const sorted = useMemo(
    () => [...shapes].sort((a, b) => (a.status === b.status ? 0 : a.status === "suggested" ? -1 : 1)),
    [shapes],
  );

  const fmtTime = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  return (
    <div className="editor">
      <header className="editor-top">
        <div className="row gap">
          {info && (
            <Link to={`/projects/${info.project_id}`} className="btn ghost" onClick={(e) => {
              if (dirty && !confirm("Leave without saving?")) e.preventDefault();
            }}>
              ← {project?.name}
            </Link>
          )}
          <span className="mono muted">
            {info ? `${info.position}/${info.total} · ${info.filename} · ${info.width}×${info.height}` : "…"}
          </span>
          {project && <span className={`pill ${project.mode}`}>{project.mode}</span>}
        </div>
        <div className="row gap">
          <span className="mono timer" title="Active annotation time on this image (idle time not counted)">
            ⏱ {fmtTime(elapsed)}
          </span>
          <button className="btn ghost" disabled={!info?.prev_id} onClick={() => go(info?.prev_id, dirty)}>←</button>
          <button className="btn" onClick={() => save(true)} disabled={busy}>Save</button>
          <button className="btn primary" onClick={() => go(info?.next_id, true)} disabled={busy}>
            Save &amp; next ⏎
          </button>
        </div>
      </header>

      <div className="editor-body">
        <aside className="toolbar">
          <ToolBtn active={tool === "select"} onClick={() => setTool("select")} k="V" title="Select / move / resize">⬚</ToolBtn>
          <ToolBtn active={tool === "box"} onClick={() => setTool("box")} k="B" title="Draw box">▭</ToolBtn>
          {assisted && (
            <>
              <ToolBtn active={tool === "segment"} onClick={() => setTool("segment")} k="S" title="Smart segment: click object (shift-click = background), or drag a box">✦</ToolBtn>
              <div className="tb-sep" />
              <ToolBtn onClick={() => runDetect()} k="D" title="Run detector (AI suggestions)" disabled={busy}>⚡</ToolBtn>
              <ToolBtn onClick={acceptAll} k="A" title="Accept all suggestions" disabled={!nSuggested}>✓</ToolBtn>
              <ToolBtn onClick={refineSelected} k="R" title="Fit mask to selected box" disabled={!selShape || busy}>◍</ToolBtn>
            </>
          )}
          <div className="tb-sep" />
          <ToolBtn onClick={undo} k="^Z" title="Undo" disabled={!history.length}>↶</ToolBtn>
          <ToolBtn onClick={fit} k="F" title="Fit image">⤢</ToolBtn>
          <ToolBtn active={showMasks} onClick={() => setShowMasks((m) => !m)} k="M" title="Show masks">◐</ToolBtn>
        </aside>

        <div className="canvas-wrap" ref={wrapRef} style={{ cursor }} onContextMenu={(e) => e.preventDefault()}>
          <Stage
            ref={stageRef}
            width={size.w}
            height={size.h}
            onWheel={onWheel}
            onMouseDown={onDown}
            onMouseMove={onMove}
            onMouseUp={onUp}
          >
            <Layer>
              <Group x={view.x} y={view.y} scaleX={view.scale} scaleY={view.scale}>
                {img && <KImage image={img} name="image" />}
                {sorted.map((s) => {
                  const color = labelColor(s.label, labels);
                  const sugg = s.status === "suggested";
                  const isSel = s.key === selected;
                  return (
                    <Group key={s.key}>
                      {showMasks && s.polygon && s.polygon.length > 2 && (
                        <Line
                          points={s.polygon.flat()}
                          closed
                          fill={color + (isSel ? "66" : "40")}
                          stroke={color}
                          strokeWidth={sw * 0.75}
                          listening={false}
                        />
                      )}
                      <Rect
                        id={s.key}
                        x={s.x}
                        y={s.y}
                        width={s.w}
                        height={s.h}
                        stroke={color}
                        strokeWidth={isSel ? sw * 1.5 : sw}
                        dash={sugg ? [8 / view.scale, 5 / view.scale] : undefined}
                        opacity={sugg ? 0.85 : 1}
                        fill={isSel ? color + "14" : "transparent"}
                        draggable={tool === "select" && isSel}
                        listening={tool === "select"}
                        onMouseDown={(e) => {
                          if (tool !== "select") return;
                          e.cancelBubble = true;
                          setSelected(s.key);
                          setLabel(s.label);
                        }}
                        onDblClick={() => sugg && accept(s.key)}
                        onDragEnd={(e) => updateBox(s.key, { x: e.target.x(), y: e.target.y(), w: s.w, h: s.h })}
                        onTransformEnd={(e) => {
                          const n = e.target;
                          const w = Math.max(2, n.width() * n.scaleX());
                          const h = Math.max(2, n.height() * n.scaleY());
                          n.scaleX(1);
                          n.scaleY(1);
                          updateBox(s.key, { x: n.x(), y: n.y(), w, h });
                        }}
                      />
                    </Group>
                  );
                })}
                {draft && (
                  <Rect
                    x={draft.x}
                    y={draft.y}
                    width={draft.w}
                    height={draft.h}
                    stroke={tool === "segment" ? "#ffffff" : labelColor(label, labels)}
                    strokeWidth={sw}
                    dash={[6 / view.scale, 4 / view.scale]}
                    listening={false}
                  />
                )}
                {preview && (
                  <Line
                    points={preview.polygon.flat()}
                    closed
                    fill={labelColor(label, labels) + "55"}
                    stroke="#ffffff"
                    strokeWidth={sw}
                    dash={[6 / view.scale, 4 / view.scale]}
                    listening={false}
                  />
                )}
                {prompt.map((p, i) => (
                  <Circle
                    key={i}
                    x={p.x}
                    y={p.y}
                    radius={6 / view.scale}
                    fill={p.l ? "#2ec4b6" : "#ee4266"}
                    stroke="#fff"
                    strokeWidth={sw}
                    listening={false}
                  />
                ))}
              </Group>
              <Transformer
                ref={trRef}
                rotateEnabled={false}
                keepRatio={false}
                ignoreStroke
                anchorSize={9}
                borderStroke="#ffffff"
                anchorStroke="#ffffff"
                anchorFill="#1b1f24"
              />
            </Layer>
          </Stage>
          {busy && <div className="busy mono">working…</div>}
        </div>

        <aside className="side">
          <section>
            <h3>Label</h3>
            <div className="labels">
              {labels.map((l, i) => (
                <button
                  key={l}
                  className={`label-btn ${l === label ? "on" : ""}`}
                  style={{ ["--c" as string]: labelColor(l, labels) }}
                  onClick={() => {
                    setLabel(l);
                    if (selected) relabel(selected, l);
                  }}
                >
                  <span className="kbd">{i + 1}</span> {l}
                </button>
              ))}
            </div>
          </section>

          {assisted && (
            <section>
              <h3>AI assistance</h3>
              <label className="slider">
                <span>Detector confidence <b className="mono">{Math.round(conf * 100)}%</b></span>
                <input
                  type="range"
                  min={0.05}
                  max={0.9}
                  step={0.05}
                  value={conf}
                  onChange={(e) => {
                    setConf(Number(e.target.value));
                    writePref("la.conf", Number(e.target.value));
                  }}
                />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={autoSuggest}
                  onChange={(e) => {
                    setAutoSuggest(e.target.checked);
                    writePref("la.auto", e.target.checked);
                  }}
                />
                Suggest automatically on new images
              </label>
              {prompt.length > 0 && (
                <div className="row gap">
                  <button className="btn primary small" onClick={keepPreview} disabled={!preview}>Keep mask ⏎</button>
                  <button className="btn small" onClick={() => { setPrompt([]); setPreview(null); }}>Clear</button>
                </div>
              )}
            </section>
          )}

          <section className="objects">
            <h3>
              Objects <span className="muted mono">{nKept}</span>
              {nSuggested > 0 && <span className="pill suggested">{nSuggested} suggested</span>}
            </h3>
            {nSuggested > 0 && (
              <button className="btn small primary wide" onClick={acceptAll}>Accept all suggestions (A)</button>
            )}
            <ul className="obj-list">
              {sorted.map((s) => (
                <li
                  key={s.key}
                  className={`${s.key === selected ? "sel" : ""} ${s.status}`}
                  onClick={() => { setTool("select"); setSelected(s.key); setLabel(s.label); }}
                >
                  <span className="dot" style={{ background: labelColor(s.label, labels) }} />
                  <select
                    value={s.label}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => relabel(s.key, e.target.value)}
                  >
                    {[...new Set([...labels, s.label])].map((l) => <option key={l}>{l}</option>)}
                  </select>
                  <span className="src mono" title={s.source}>
                    {s.source === "manual" ? "hand" : s.source === "ai_detect" ? "det" : "sam"}
                    {s.score != null && s.source === "ai_detect" ? ` ${Math.round(s.score * 100)}` : ""}
                    {s.edited ? "*" : ""}
                  </span>
                  {s.status === "suggested" && (
                    <button className="icon ok" title="Accept" onClick={(e) => { e.stopPropagation(); accept(s.key); }}>✓</button>
                  )}
                  <button className="icon del" title="Delete" onClick={(e) => { e.stopPropagation(); remove(s.key); }}>✕</button>
                </li>
              ))}
            </ul>
          </section>

          <section className="help muted small">
            <p><b>Wheel</b> zoom · <b>Space/drag</b> pan · <b>1–9</b> label · <b>Del</b> delete · <b>Esc</b> cancel · <b>←/→</b> images</p>
          </section>
        </aside>
      </div>

      <footer className="statusbar mono">
        <span>{status || (tool === "segment" ? "Click an object to segment it; shift-click marks background; drag for a box prompt" : tool === "box" ? "Drag to draw a box" : "Click a box to select; drag to move; handles to resize")}</span>
        <span>{dirty ? "● unsaved" : "saved"}</span>
      </footer>
    </div>
  );
}

function ToolBtn(props: { active?: boolean; onClick: () => void; k: string; title: string; disabled?: boolean; children: React.ReactNode }) {
  return (
    <button
      className={`tool ${props.active ? "on" : ""}`}
      onClick={props.onClick}
      title={`${props.title} (${props.k})`}
      disabled={props.disabled}
    >
      <span className="tool-ico">{props.children}</span>
      <span className="tool-k">{props.k}</span>
    </button>
  );
}
