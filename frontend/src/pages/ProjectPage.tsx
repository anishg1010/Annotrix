import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, ImageInfo, Project } from "../api";

export default function ProjectPage() {
  const pid = Number(useParams().pid);
  const nav = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [stats, setStats] = useState<Record<string, Record<string, number>>>({});
  const [busy, setBusy] = useState("");
  const [msg, setMsg] = useState("");
  const [drag, setDrag] = useState(false);
  const [labelsText, setLabelsText] = useState("");
  const [format, setFormat] = useState("coco");
  const [onlyDone, setOnlyDone] = useState(true);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    const [p, ims, st] = await Promise.all([api.project(pid), api.images(pid), api.stats(pid)]);
    setProject(p);
    setLabelsText(p.labels.join(", "));
    setImages(ims);
    setStats(st);
  }, [pid]);

  useEffect(() => {
    load().catch((e) => setMsg(e.message));
  }, [load]);

  async function upload(files: File[]) {
    if (!files.length) return;
    setBusy(`Uploading ${files.length} file(s)…`);
    setMsg("");
    try {
      // upload in chunks so large folders don't hit request limits
      let created = 0;
      const skipped: string[] = [];
      for (let i = 0; i < files.length; i += 25) {
        const r = await api.upload(pid, files.slice(i, i + 25));
        created += r.created.length;
        skipped.push(...r.skipped.map((s) => `${s.file} (${s.reason})`));
        setBusy(`Uploading… ${Math.min(i + 25, files.length)}/${files.length}`);
      }
      setMsg(`Added ${created} image(s).` + (skipped.length ? ` Skipped: ${skipped.join(", ")}` : ""));
      await load();
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function saveLabels() {
    const p = await api.updateProject(pid, {
      labels: labelsText.split(",").map((s) => s.trim()).filter(Boolean),
    });
    setProject(p);
    setMsg("Labels updated.");
  }

  async function toggleMode() {
    if (!project) return;
    const p = await api.updateProject(pid, { mode: project.mode === "assisted" ? "manual" : "assisted" });
    setProject(p);
  }

  async function remove() {
    if (!project || !confirm(`Delete project "${project.name}" and all its images?`)) return;
    await api.deleteProject(pid);
    nav("/");
  }

  if (!project) return <main className="page"><p className="muted">{msg || "Loading…"}</p></main>;

  const done = images.filter((i) => i.status === "done").length;
  const firstOpen = images.find((i) => i.status !== "done") ?? images[0];

  return (
    <main className="page">
      <section className="panel">
        <div className="row between">
          <div>
            <Link to="/" className="muted small">← all projects</Link>
            <h1>
              {project.name} <span className={`pill ${project.mode}`}>{project.mode}</span>
            </h1>
            <p className="muted mono">{done}/{images.length} images done</p>
          </div>
          <div className="row gap">
            {firstOpen && (
              <Link className="btn primary" to={`/annotate/${firstOpen.id}`}>
                {done ? "Continue annotating" : "Start annotating"} →
              </Link>
            )}
          </div>
        </div>
        <div className="progress"><div style={{ width: `${images.length ? (100 * done) / images.length : 0}%` }} /></div>
      </section>

      <section
        className={`panel dropzone ${drag ? "drag" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); upload(Array.from(e.dataTransfer.files)); }}
        onClick={() => fileInput.current?.click()}
      >
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => { upload(Array.from(e.target.files ?? [])); e.target.value = ""; }}
        />
        <p><b>Drop images here</b> or click to choose files</p>
        <p className="muted small">JPG, PNG, BMP, WEBP, TIFF</p>
        {busy && <p className="mono">{busy}</p>}
      </section>
      {msg && <p className="note">{msg}</p>}

      <section className="panel">
        <h2>Images</h2>
        {images.length === 0 ? (
          <p className="muted">No images yet.</p>
        ) : (
          <div className="grid">
            {images.map((im) => (
              <Link key={im.id} to={`/annotate/${im.id}`} className={`thumb ${im.status}`}>
                <img src={im.url} loading="lazy" alt={im.filename} />
                <span className="thumb-meta mono">
                  {im.status === "done" ? "✓ " : ""}{im.n_annotations} · {im.filename}
                </span>
              </Link>
            ))}
          </div>
        )}
      </section>

      <div className="two-col">
        <section className="panel">
          <h2>Export</h2>
          <div className="form">
            <label>
              Format
              <select value={format} onChange={(e) => setFormat(e.target.value)}>
                <option value="coco">COCO JSON (boxes + polygons)</option>
                <option value="yolo">YOLO detection (txt)</option>
                <option value="yolo-seg">YOLO segmentation (txt)</option>
              </select>
            </label>
            <label className="check">
              <input type="checkbox" checked={onlyDone} onChange={(e) => setOnlyDone(e.target.checked)} />
              Only images marked done
            </label>
            <a className="btn" href={api.exportUrl(pid, format, onlyDone)}>Download .zip</a>
          </div>
        </section>

        <section className="panel">
          <h2>Effort log</h2>
          {Object.keys(stats).length === 0 ? (
            <p className="muted">Time and actions are recorded each time you save an image.</p>
          ) : (
            <table className="table mono">
              <thead>
                <tr><th>mode</th><th>images</th><th>s / image</th><th>s / object</th><th>actions / image</th></tr>
              </thead>
              <tbody>
                {Object.entries(stats).map(([m, s]) => (
                  <tr key={m}>
                    <td>{m}</td><td>{s.images}</td><td>{s.seconds_per_image}</td>
                    <td>{s.seconds_per_object}</td><td>{s.actions_per_image}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>

      <section className="panel">
        <h2>Settings</h2>
        <div className="form">
          <label>
            Labels
            <input value={labelsText} onChange={(e) => setLabelsText(e.target.value)} />
          </label>
          <div className="row gap">
            <button className="btn" onClick={saveLabels}>Save labels</button>
            <button className="btn" onClick={toggleMode}>
              Switch to {project.mode === "assisted" ? "manual" : "assisted"} mode
            </button>
            <button className="btn danger" onClick={remove}>Delete project</button>
          </div>
        </div>
      </section>
    </main>
  );
}
