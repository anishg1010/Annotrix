import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, Mode, Project } from "../api";

export default function Projects() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [name, setName] = useState("");
  const [labels, setLabels] = useState("buffalo, elephant, rhino, zebra");
  const [mode, setMode] = useState<Mode>("assisted");
  const [error, setError] = useState("");
  const [health, setHealth] = useState<{ device: string; detector: string; segmenter: string } | null>(null);

  const load = () => api.projects().then(setProjects).catch((e) => setError(String(e.message)));
  useEffect(() => {
    load();
    api.health().then(setHealth).catch(() => {});
  }, []);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    try {
      await api.createProject({
        name: name.trim(),
        labels: labels.split(",").map((s) => s.trim()).filter(Boolean),
        mode,
      });
      setName("");
      load();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <main className="page">
      <section className="panel">
        <h1>Projects</h1>
        <p className="muted">
          A project is a set of images with one label list. Use <b>assisted</b> mode for normal work; create a
          second project in <b>manual</b> mode (AI tools hidden) to measure how much time the AI saves.
        </p>
        {projects === null ? (
          <p className="muted">Loading…</p>
        ) : projects.length === 0 ? (
          <p className="muted">No projects yet. Create one below.</p>
        ) : (
          <ul className="cards">
            {projects.map((p) => (
              <li key={p.id}>
                <Link to={`/projects/${p.id}`} className="card">
                  <div className="card-title">
                    {p.name} <span className={`pill ${p.mode}`}>{p.mode}</span>
                  </div>
                  <div className="muted mono">
                    {p.n_done ?? 0}/{p.n_images ?? 0} images done
                  </div>
                  <div className="chips">
                    {p.labels.map((l) => (
                      <span key={l} className="chip">{l}</span>
                    ))}
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel">
        <h2>New project</h2>
        <form onSubmit={create} className="form">
          <label>
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} required placeholder="Wildlife — assisted" />
          </label>
          <label>
            Labels (comma separated)
            <input value={labels} onChange={(e) => setLabels(e.target.value)} />
            <span className="small">
              Any words work: labels the trained detector knows use it; any other label (e.g. "car, person,
              backpack") is detected zero-shot by an open-vocabulary model from its name.
            </span>
          </label>
          <label>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
              <option value="assisted">assisted — AI suggestions and click-to-segment</option>
              <option value="manual">manual — baseline, no AI</option>
            </select>
          </label>
          <button className="btn primary" type="submit">Create project</button>
          {error && <p className="error">{error}</p>}
        </form>
      </section>

      {health && (
        <p className="muted mono small footer-info">
          device {health.device} · detector {health.detector} · segmenter {health.segmenter}
        </p>
      )}
    </main>
  );
}
