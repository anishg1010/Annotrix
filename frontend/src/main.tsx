import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import "./styles.css";
import Projects from "./pages/Projects";
import ProjectPage from "./pages/ProjectPage";
import Annotator from "./pages/Annotator";

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-mark">◩</span> LabelAssist
        </Link>
        <span className="tagline">AI-assisted image annotation</span>
      </header>
      {children}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Shell><Projects /></Shell>} />
        <Route path="/projects/:pid" element={<Shell><ProjectPage /></Shell>} />
        <Route path="/annotate/:iid" element={<Annotator />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
