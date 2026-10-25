// Thin typed client for the LabelAssist API.

export type Mode = "assisted" | "manual";

export interface Project {
  id: number;
  name: string;
  labels: string[];
  mode: Mode;
  created_at: string;
  n_images?: number;
  n_done?: number;
}

export interface ImageInfo {
  id: number;
  project_id: number;
  filename: string;
  width: number;
  height: number;
  status: "new" | "done";
  url: string;
  n_annotations: number;
  prev_id?: number | null;
  next_id?: number | null;
  position?: number;
  total?: number;
}

export type Source = "manual" | "ai_detect" | "ai_segment";

export interface Ann {
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  polygon?: number[][] | null;
  source: Source;
  score?: number | null;
  edited?: boolean;
}

export interface Suggestion {
  label: string;
  score: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SegResult {
  polygon: number[][];
  x: number;
  y: number;
  w: number;
  h: number;
  score: number | null;
  ms: number;
}

export interface SessionStats {
  active_seconds: number;
  n_actions: number;
  n_drawn: number;
  n_accepted: number;
  n_adjusted: number;
  n_rejected: number;
  n_relabelled: number;
  n_ai_calls: number;
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  if (r.status === 204) return undefined as T;
  return r.json();
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => req<{ device: string; detector: string; segmenter: string }>("/api/health"),
  projects: () => req<Project[]>("/api/projects"),
  project: (id: number) => req<Project>(`/api/projects/${id}`),
  createProject: (p: { name: string; labels: string[]; mode: Mode }) =>
    req<Project>("/api/projects", json("POST", p)),
  updateProject: (id: number, p: Partial<Pick<Project, "name" | "labels" | "mode">>) =>
    req<Project>(`/api/projects/${id}`, json("PATCH", p)),
  deleteProject: (id: number) => req<void>(`/api/projects/${id}`, { method: "DELETE" }),
  images: (pid: number) => req<ImageInfo[]>(`/api/projects/${pid}/images`),
  image: (id: number) => req<ImageInfo>(`/api/images/${id}`),
  deleteImage: (id: number) => req<void>(`/api/images/${id}`, { method: "DELETE" }),
  upload: (pid: number, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return req<{ created: ImageInfo[]; skipped: { file: string; reason: string }[] }>(
      `/api/projects/${pid}/images`,
      { method: "POST", body: fd },
    );
  },
  annotations: (iid: number) => req<(Ann & { id: number })[]>(`/api/images/${iid}/annotations`),
  save: (iid: number, annotations: Ann[], done: boolean, session?: SessionStats) =>
    req<{ saved: number }>(`/api/images/${iid}/annotations`, json("PUT", { annotations, done, session })),
  detect: (iid: number, conf: number) =>
    req<{ suggestions: Suggestion[]; ms: number; model: string }>(
      `/api/images/${iid}/ai/detect`,
      json("POST", { conf }),
    ),
  segment: (iid: number, body: { points?: number[][]; point_labels?: number[]; box?: number[] }) =>
    req<SegResult>(`/api/images/${iid}/ai/segment`, json("POST", body)),
  stats: (pid: number) => req<Record<string, Record<string, number>>>(`/api/projects/${pid}/stats`),
  exportUrl: (pid: number, format: string, onlyDone: boolean) =>
    `/api/projects/${pid}/export?format=${format}&only_done=${onlyDone}`,
};

// A stable colour per label, picked from a palette that reads on photos.
const PALETTE = ["#ff6b35", "#2ec4b6", "#ffd23f", "#ee4266", "#3a86ff", "#8ac926", "#c77dff", "#ff9f1c", "#06d6a0", "#f15bb5"];
export function labelColor(label: string, labels: string[]): string {
  let i = labels.indexOf(label);
  if (i < 0) {
    i = 0;
    for (const ch of label) i = (i * 31 + ch.charCodeAt(0)) >>> 0;
  }
  return PALETTE[i % PALETTE.length];
}
