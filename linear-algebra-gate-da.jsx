import React, { useState, useMemo, useRef, useEffect } from "react";
import { Search, ChevronRight, ChevronDown, Check, X, Lightbulb, AlertTriangle, BookOpen, Menu, RotateCcw } from "lucide-react";

/* ============================== MATH ENGINE ============================== */
const nf = (x, d = 3) => {
  if (!isFinite(x)) return "∞";
  if (Math.abs(x) < 1e-9) x = 0;
  let s = Number(x).toFixed(d);
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
  return s === "-0" ? "0" : s;
};
const cloneM = (M) => M.map((r) => r.slice());
const tp = (M) => M[0].map((_, j) => M.map((r) => r[j]));
const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
const mmul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const mvec = (A, x) => A.map((r) => r.reduce((s, v, j) => s + v * x[j], 0));

function rref(A) {
  const R = cloneM(A), m = R.length, n = R[0].length, pivots = [], log = [];
  let row = 0;
  for (let col = 0; col < n && row < m; col++) {
    let best = row;
    for (let i = row; i < m; i++) if (Math.abs(R[i][col]) > Math.abs(R[best][col])) best = i;
    if (Math.abs(R[best][col]) < 1e-10) { log.push(`Column ${col + 1}: koi pivot nahi → free column`); continue; }
    if (best !== row) { const t = R[best]; R[best] = R[row]; R[row] = t; log.push(`R${row + 1} ↔ R${best + 1}`); }
    const p = R[row][col];
    if (Math.abs(p - 1) > 1e-10) { for (let j = 0; j < n; j++) R[row][j] /= p; log.push(`R${row + 1} ÷ ${nf(p)} → pivot = 1`); }
    for (let i = 0; i < m; i++) {
      if (i !== row && Math.abs(R[i][col]) > 1e-10) {
        const f = R[i][col];
        for (let j = 0; j < n; j++) R[i][j] -= f * R[row][j];
        log.push(`R${i + 1} − (${nf(f)})·R${row + 1}`);
      }
    }
    pivots.push(col); row++;
  }
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) if (Math.abs(R[i][j]) < 1e-10) R[i][j] = 0;
  return { R, pivots, rank: pivots.length, log };
}
function nullBasis(A) {
  const { R, pivots } = rref(A), n = A[0].length;
  const free = []; for (let j = 0; j < n; j++) if (!pivots.includes(j)) free.push(j);
  return free.map((f) => {
    const v = new Array(n).fill(0); v[f] = 1;
    pivots.forEach((pc, pi) => { v[pc] = -R[pi][f]; });
    return v;
  });
}
function luFact(A) {
  const n = A.length, U = cloneM(A), L = eye(n), P = eye(n), log = [];
  let swaps = 0;
  for (let k = 0; k < n; k++) {
    let best = k;
    for (let i = k; i < n; i++) if (Math.abs(U[i][k]) > Math.abs(U[best][k])) best = i;
    if (Math.abs(U[best][k]) < 1e-10) { log.push(`Column ${k + 1}: pivot = 0 → singular`); continue; }
    if (best !== k) {
      [U[k], U[best]] = [U[best], U[k]]; [P[k], P[best]] = [P[best], P[k]];
      for (let j = 0; j < k; j++) [L[k][j], L[best][j]] = [L[best][j], L[k][j]];
      swaps++; log.push(`Row swap R${k + 1} ↔ R${best + 1} (P mein record)`);
    }
    for (let i = k + 1; i < n; i++) {
      const f = U[i][k] / U[k][k];
      if (Math.abs(f) < 1e-12) continue;
      L[i][k] = f;
      for (let j = k; j < n; j++) U[i][j] -= f * U[k][j];
      log.push(`m${i + 1}${k + 1} = ${nf(f)} → R${i + 1} − m·R${k + 1}`);
    }
  }
  return { L, U, P, swaps, log };
}
function detOf(A) {
  const n = A.length;
  if (n === 1) return A[0][0];
  const { U, swaps } = luFact(A);
  let d = Math.pow(-1, swaps);
  for (let i = 0; i < n; i++) d *= U[i][i];
  return Math.abs(d) < 1e-9 ? 0 : d;
}
function invOf(A) {
  const n = A.length, M = A.map((r, i) => r.concat(eye(n)[i])), { R } = rref(M);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (Math.abs(R[i][j] - (i === j ? 1 : 0)) > 1e-8) return null;
  return R.map((r) => r.slice(n));
}
function eig2(a, b, c, d) {
  const tr = a + d, det = a * d - b * c, disc = tr * tr - 4 * det;
  if (disc < -1e-9) return { kind: "complex", tr, det, re: tr / 2, im: Math.sqrt(-disc) / 2 };
  const s = Math.sqrt(Math.max(disc, 0)), l1 = (tr + s) / 2, l2 = (tr - s) / 2;
  const vecFor = (l) => {
    const cand = [[-b, a - l], [d - l, -c]];
    let best = [1, 0], bn = 0;
    cand.forEach((v) => { const n2 = Math.hypot(v[0], v[1]); if (n2 > bn) { bn = n2; best = v; } });
    if (bn < 1e-9) return [1, 0];
    return [best[0] / bn, best[1] / bn];
  };
  if (Math.abs(disc) < 1e-9) {
    const z = Math.abs(a - l1) + Math.abs(b) + Math.abs(c) + Math.abs(d - l1);
    if (z < 1e-9) return { kind: "repeated", tr, det, l1, l2, v1: [1, 0], v2: [0, 1] };
    const v = vecFor(l1);
    return { kind: "defective", tr, det, l1, l2, v1: v, v2: v };
  }
  return { kind: "distinct", tr, det, l1, l2, v1: vecFor(l1), v2: vecFor(l2) };
}
/* symmetric 2x2 SVD-ish helper: singular values of any 2x2 via eig of AᵀA */
function svd2(A) {
  const At = tp(A), AtA = mmul(At, A);
  const e = eig2(AtA[0][0], AtA[0][1], AtA[1][0], AtA[1][1]);
  const l1 = Math.max(e.l1 ?? 0, 0), l2 = Math.max(e.l2 ?? 0, 0);
  const s1 = Math.sqrt(l1), s2 = Math.sqrt(l2);
  const v1 = e.v1 || [1, 0], v2 = e.v2 || [0, 1];
  const u1 = s1 > 1e-9 ? mvec(A, v1).map((x) => x / s1) : [1, 0];
  const u2 = s2 > 1e-9 ? mvec(A, v2).map((x) => x / s2) : [-u1[1], u1[0]];
  return { s1, s2, v1, v2, u1, u2 };
}

/* ============================== MARKUP ============================== */
/* inline: **bold**  `mono`  __hi__ (accent)  */
function Inline({ text }) {
  const parts = String(text).split(/(\*\*[^*]+\*\*|`[^`]+`|__[^_]+__)/g);
  return (
    <>
      {parts.map((p, i) => {
        if (p.startsWith("**")) return <b key={i} className="font-semibold text-stone-900">{p.slice(2, -2)}</b>;
        if (p.startsWith("`")) return <code key={i} className="font-mono text-sm bg-stone-100 text-stone-800 px-1 py-0.5 rounded">{p.slice(1, -1)}</code>;
        if (p.startsWith("__")) return <span key={i} className="text-indigo-700 font-medium">{p.slice(2, -2)}</span>;
        return <span key={i}>{p}</span>;
      })}
    </>
  );
}

function Block({ b }) {
  if (b.t === "p") return <p className="text-stone-700 leading-relaxed mb-3"><Inline text={b.x} /></p>;
  if (b.t === "list")
    return (
      <ul className="mb-3 space-y-1.5">
        {b.x.map((li, i) => (
          <li key={i} className="flex gap-2 text-stone-700 leading-relaxed">
            <span className="text-indigo-400 mt-1.5 shrink-0">▪</span>
            <span><Inline text={li} /></span>
          </li>
        ))}
      </ul>
    );
  if (b.t === "num")
    return (
      <ol className="mb-3 space-y-1.5">
        {b.x.map((li, i) => (
          <li key={i} className="flex gap-2.5 text-stone-700 leading-relaxed">
            <span className="shrink-0 w-5 h-5 mt-0.5 rounded-full bg-indigo-100 text-indigo-700 text-xs font-semibold flex items-center justify-center">{i + 1}</span>
            <span><Inline text={li} /></span>
          </li>
        ))}
      </ol>
    );
  if (b.t === "f")
    return (
      <div className="mb-3 rounded-lg border border-indigo-200 bg-indigo-50 px-4 py-3">
        {b.label && <div className="text-xs font-semibold uppercase tracking-wide text-indigo-500 mb-1.5">{b.label}</div>}
        <div className="font-mono text-base text-indigo-950 whitespace-pre-wrap break-words">{b.x}</div>
        {b.note && <div className="text-sm text-indigo-700 mt-2"><Inline text={b.note} /></div>}
      </div>
    );
  if (b.t === "table")
    return (
      <div className="mb-3 overflow-x-auto rounded-lg border border-stone-200">
        <table className="w-full text-sm">
          <thead className="bg-stone-100">
            <tr>{b.head.map((h, i) => <th key={i} className="text-left px-3 py-2 font-semibold text-stone-700 border-b border-stone-200">{h}</th>)}</tr>
          </thead>
          <tbody>
            {b.rows.map((r, i) => (
              <tr key={i} className={i % 2 ? "bg-stone-50" : "bg-white"}>
                {r.map((c, j) => <td key={j} className="px-3 py-2 align-top text-stone-700 border-b border-stone-100"><Inline text={c} /></td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  if (b.t === "trap")
    return (
      <div className="mb-3 rounded-lg border-l-4 border-rose-400 bg-rose-50 px-4 py-3">
        <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-rose-600 mb-1">
          <AlertTriangle size={13} /> {b.label || "Exam trap"}
        </div>
        <div className="text-stone-700 text-sm leading-relaxed"><Inline text={b.x} /></div>
      </div>
    );
  if (b.t === "tip")
    return (
      <div className="mb-3 rounded-lg border-l-4 border-emerald-400 bg-emerald-50 px-4 py-3">
        <div className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-emerald-700 mb-1">
          <Lightbulb size={13} /> {b.label || "Shortcut"}
        </div>
        <div className="text-stone-700 text-sm leading-relaxed"><Inline text={b.x} /></div>
      </div>
    );
  return null;
}
const Blocks = ({ list }) => <>{(list || []).map((b, i) => <Block key={i} b={b} />)}</>;

/* ============================== UI PRIMITIVES ============================== */
function Mtx({ M, cls = "" }) {
  return (
    <span className={"inline-flex items-stretch mx-1 align-middle " + cls}>
      <span className="w-1.5 border-y-2 border-l-2 border-stone-400 rounded-l-sm" />
      <span className="px-1.5 py-1">
        <span className="inline-grid gap-x-3 gap-y-0.5" style={{ gridTemplateColumns: `repeat(${M[0].length}, minmax(0,auto))` }}>
          {M.flatMap((r, i) => r.map((v, j) => (
            <span key={i + "-" + j} className="text-center font-mono text-sm tabular-nums text-stone-800">
              {typeof v === "string" ? v : nf(v)}
            </span>
          )))}
        </span>
      </span>
      <span className="w-1.5 border-y-2 border-r-2 border-stone-400 rounded-r-sm" />
    </span>
  );
}

function Example({ ex }) {
  const [open, setOpen] = useState(0);
  if (!ex) return null;
  return (
    <div className="mt-6 rounded-xl border border-amber-300 bg-amber-50 overflow-hidden">
      <div className="px-4 py-2.5 bg-amber-100 border-b border-amber-300 flex items-center gap-2">
        <BookOpen size={15} className="text-amber-700" />
        <span className="font-semibold text-amber-900 text-sm">Solved example — step by step</span>
      </div>
      <div className="px-4 py-3">
        <div className="text-stone-800 mb-3 leading-relaxed"><Inline text={ex.problem} /></div>
        <div className="space-y-2">
          {ex.steps.map((s, i) => {
            const shown = i <= open;
            return (
              <div key={i} className={"rounded-lg border transition-colors " + (shown ? "border-amber-300 bg-white" : "border-stone-200 bg-stone-50")}>
                <button
                  onClick={() => setOpen(shown && open === i ? i - 1 : i)}
                  className="w-full flex items-start gap-2.5 px-3 py-2 text-left"
                >
                  <span className={"shrink-0 w-5 h-5 mt-0.5 rounded-full text-xs font-bold flex items-center justify-center " + (shown ? "bg-amber-500 text-white" : "bg-stone-300 text-stone-600")}>{i + 1}</span>
                  <span className={"font-medium text-sm " + (shown ? "text-stone-900" : "text-stone-500")}>{s.title}</span>
                </button>
                {shown && s.detail && (
                  <div className="px-3 pb-3 pl-10 text-sm text-stone-700 leading-relaxed whitespace-pre-wrap font-mono">{s.detail}</div>
                )}
              </div>
            );
          })}
        </div>
        {open >= ex.steps.length - 1 && (
          <div className="mt-3 rounded-lg bg-emerald-100 border border-emerald-300 px-3 py-2.5">
            <span className="text-xs font-bold uppercase tracking-wide text-emerald-700">Answer</span>
            <div className="text-stone-800 font-medium mt-0.5"><Inline text={ex.answer} /></div>
          </div>
        )}
        {open < ex.steps.length - 1 && (
          <button onClick={() => setOpen(ex.steps.length - 1)} className="mt-3 text-xs font-medium text-amber-800 hover:text-amber-950 underline">
            saare steps kholo
          </button>
        )}
      </div>
    </div>
  );
}

function Quiz({ q }) {
  const [pick, setPick] = useState(null);
  if (!q) return null;
  return (
    <div className="mt-6 rounded-xl border border-stone-300 bg-white p-4">
      <div className="text-xs font-bold uppercase tracking-wide text-stone-500 mb-2">Quick check</div>
      <div className="text-stone-800 mb-3"><Inline text={q.q} /></div>
      <div className="space-y-1.5">
        {q.opts.map((o, i) => {
          const state = pick === null ? "idle" : i === q.a ? "right" : i === pick ? "wrong" : "idle";
          return (
            <button key={i} onClick={() => setPick(i)} disabled={pick !== null}
              className={"w-full text-left px-3 py-2 rounded-lg border text-sm flex items-center gap-2 " +
                (state === "right" ? "border-emerald-400 bg-emerald-50 text-emerald-900"
                  : state === "wrong" ? "border-rose-400 bg-rose-50 text-rose-900"
                  : "border-stone-200 bg-stone-50 text-stone-700 hover:border-stone-400")}>
              {state === "right" && <Check size={15} />}
              {state === "wrong" && <X size={15} />}
              <span><Inline text={o} /></span>
            </button>
          );
        })}
      </div>
      {pick !== null && (
        <div className="mt-3 text-sm text-stone-700 bg-stone-100 rounded-lg px-3 py-2"><Inline text={q.why} /></div>
      )}
    </div>
  );
}

/* small controls */
function Slider({ label, value, min, max, step, onChange }) {
  return (
    <label className="block mb-2">
      <span className="flex justify-between text-xs text-stone-600 mb-1">
        <span>{label}</span><span className="font-mono font-semibold text-stone-900">{nf(value, 2)}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-indigo-600" />
    </label>
  );
}
function MatInput({ M, onChange, label }) {
  return (
    <div className="mb-3">
      {label && <div className="text-xs font-semibold text-stone-500 mb-1">{label}</div>}
      <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${M[0].length}, 3.25rem)` }}>
        {M.map((r, i) => r.map((v, j) => (
          <input key={i + "-" + j} value={v} inputMode="decimal"
            onChange={(e) => {
              const N = cloneM(M);
              N[i][j] = e.target.value === "" || e.target.value === "-" ? e.target.value : (isNaN(parseFloat(e.target.value)) ? 0 : parseFloat(e.target.value));
              onChange(N);
            }}
            className="w-full px-1 py-1.5 text-center font-mono text-sm border border-stone-300 rounded bg-white text-stone-800 focus:border-indigo-500 focus:outline-none" />
        )))}
      </div>
    </div>
  );
}
const numM = (M) => M.map((r) => r.map((v) => (typeof v === "number" ? v : parseFloat(v) || 0)));
function Btn({ children, onClick, active }) {
  return (
    <button onClick={onClick}
      className={"px-2.5 py-1 rounded-md text-xs font-medium border " +
        (active ? "bg-indigo-600 border-indigo-600 text-white" : "bg-white border-stone-300 text-stone-700 hover:border-stone-500")}>
      {children}
    </button>
  );
}
function Panel({ title, hint, children, side }) {
  return (
    <div className="mt-6 rounded-xl border border-stone-300 bg-white overflow-hidden">
      <div className="px-4 py-2.5 border-b border-stone-200 bg-stone-50">
        <div className="font-semibold text-sm text-stone-800">{title}</div>
        {hint && <div className="text-xs text-stone-500 mt-0.5">{hint}</div>}
      </div>
      <div className="grid md:grid-cols-5">
        <div className="md:col-span-3 p-3 bg-stone-50 border-b md:border-b-0 md:border-r border-stone-200">{children}</div>
        <div className="md:col-span-2 p-3 text-sm">{side}</div>
      </div>
    </div>
  );
}

/* ============================== JACOBI + FULL SVD ============================== */
function jacobiEig(Sin) {
  const n = Sin.length, S = cloneM(Sin);
  let V = eye(n);
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += S[i][j] * S[i][j];
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(S[p][q]) < 1e-14) continue;
      const theta = (S[q][q] - S[p][p]) / (2 * S[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const skp = S[k][p], skq = S[k][q];
        S[k][p] = c * skp - s * skq; S[k][q] = s * skp + c * skq;
      }
      for (let k = 0; k < n; k++) {
        const spk = S[p][k], sqk = S[q][k];
        S[p][k] = c * spk - s * sqk; S[q][k] = s * spk + c * sqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => S[b][b] - S[a][a]);
  return { values: idx.map((i) => S[i][i]), vectors: idx.map((i) => V.map((r) => r[i])) };
}
function svdFull(A) {
  const At = tp(A), AtA = mmul(At, A);
  const { values, vectors } = jacobiEig(AtA);
  const sig = values.map((v) => Math.sqrt(Math.max(v, 0)));
  const U = sig.map((s, i) => (s > 1e-9 ? mvec(A, vectors[i]).map((x) => x / s) : null));
  return { sig, V: vectors, U };
}

/* ============================== SVG HELPERS ============================== */
const mk = (range, w = 380, h = 300) => {
  const s = Math.min(w, h) / (2 * range);
  return { X: (x) => w / 2 + x * s, Y: (y) => h / 2 - y * s, s, w, h, range };
};
function Grid({ g }) {
  const lines = [];
  for (let v = -Math.ceil(g.range); v <= Math.ceil(g.range); v++) {
    lines.push(<line key={"v" + v} x1={g.X(v)} y1={0} x2={g.X(v)} y2={g.h} stroke="#e7e5e4" strokeWidth="1" />);
    lines.push(<line key={"h" + v} x1={0} y1={g.Y(v)} x2={g.w} y2={g.Y(v)} stroke="#e7e5e4" strokeWidth="1" />);
  }
  return <g>{lines}</g>;
}
function Axes({ g }) {
  return (
    <g>
      <line x1={0} y1={g.Y(0)} x2={g.w} y2={g.Y(0)} stroke="#a8a29e" strokeWidth="1.4" />
      <line x1={g.X(0)} y1={0} x2={g.X(0)} y2={g.h} stroke="#a8a29e" strokeWidth="1.4" />
    </g>
  );
}
function Arrow({ g, from = [0, 0], to, color, label, width = 2.4, dash }) {
  const x1 = g.X(from[0]), y1 = g.Y(from[1]), x2 = g.X(to[0]), y2 = g.Y(to[1]);
  const a = Math.atan2(y2 - y1, x2 - x1), L = Math.hypot(x2 - x1, y2 - y1);
  const hd = Math.min(10, L * 0.4);
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2 - Math.cos(a) * hd * 0.7} y2={y2 - Math.sin(a) * hd * 0.7}
        stroke={color} strokeWidth={width} strokeLinecap="round" strokeDasharray={dash} />
      {L > 3 && (
        <polygon points={`${x2},${y2} ${x2 - Math.cos(a - 0.42) * hd},${y2 - Math.sin(a - 0.42) * hd} ${x2 - Math.cos(a + 0.42) * hd},${y2 - Math.sin(a + 0.42) * hd}`} fill={color} />
      )}
      {label && <text x={x2 + 7 * Math.cos(a - 0.6)} y={y2 + 7 * Math.sin(a - 0.6) - 3} fill={color} fontSize="12" fontWeight="600">{label}</text>}
    </g>
  );
}
function FullLine({ g, dir, color, through = [0, 0], dash = "5 4" }) {
  const n = Math.hypot(dir[0], dir[1]) || 1, L = g.range * 3;
  const d = [dir[0] / n, dir[1] / n];
  return <line x1={g.X(through[0] - d[0] * L)} y1={g.Y(through[1] - d[1] * L)} x2={g.X(through[0] + d[0] * L)} y2={g.Y(through[1] + d[1] * L)} stroke={color} strokeWidth="1.6" strokeDasharray={dash} />;
}
const CO = { u: "#d97706", v: "#0891b2", b: "#e11d48", p: "#059669", e: "#ea580c", eig: "#7c3aed", g2: "#4f46e5" };
const KV = ({ k, v, c }) => (
  <div className="flex justify-between gap-2 py-0.5 border-b border-stone-100 last:border-0">
    <span className="text-stone-500">{k}</span>
    <span className="font-mono font-medium tabular-nums" style={{ color: c || "#1c1917" }}>{v}</span>
  </div>
);

/* ============================== VISUAL TOOLS ============================== */
function VSpan() {
  const [u, setU] = useState([2, 1]), [v, setV] = useState([-1, 2]), [c, setC] = useState(1), [d, setD] = useState(1);
  const g = mk(5);
  const cross = u[0] * v[1] - u[1] * v[0];
  const r = [c * u[0] + d * v[0], c * u[1] + d * v[1]];
  return (
    <Panel title="Linear combination c·u + d·v — span kaisa dikhta hai" hint="Sliders ghumao. Cross product zero ho to span sirf ek line hai."
      side={
        <div>
          <Slider label="c (u ka multiple)" value={c} min={-3} max={3} step={0.1} onChange={setC} />
          <Slider label="d (v ka multiple)" value={d} min={-3} max={3} step={0.1} onChange={setD} />
          <div className="flex gap-1.5 my-2">
            <Btn onClick={() => { setU([2, 1]); setV([-1, 2]); }}>Independent</Btn>
            <Btn onClick={() => { setU([2, 1]); setV([-4, -2]); }}>Dependent</Btn>
          </div>
          <div className="mt-2 space-y-0">
            <KV k="u × v (2D cross)" v={nf(cross)} />
            <KV k="c·u + d·v" v={`(${nf(r[0], 2)}, ${nf(r[1], 2)})`} c={CO.b} />
          </div>
          <div className={"mt-2 text-sm font-medium " + (Math.abs(cross) < 1e-6 ? "text-rose-600" : "text-emerald-700")}>
            {Math.abs(cross) < 1e-6 ? "Span = ek LINE (dependent)" : "Span = poora R² (independent)"}
          </div>
        </div>
      }>
      <svg viewBox={`0 0 ${g.w} ${g.h}`} className="w-full">
        <Grid g={g} />
        {Math.abs(cross) < 1e-6
          ? <FullLine g={g} dir={Math.hypot(u[0], u[1]) > 0 ? u : v} color="#c4b5fd" dash="6 4" />
          : <rect x="0" y="0" width={g.w} height={g.h} fill="#eef2ff" opacity="0.55" />}
        <Axes g={g} />
        <Arrow g={g} to={[c * u[0], c * u[1]]} color="#fbbf24" width={5} />
        <Arrow g={g} from={[c * u[0], c * u[1]]} to={r} color="#67e8f9" width={5} />
        <Arrow g={g} to={u} color={CO.u} label="u" />
        <Arrow g={g} to={v} color={CO.v} label="v" />
        <Arrow g={g} to={r} color={CO.b} label="cu+dv" />
      </svg>
    </Panel>
  );
}

function VRowCol() {
  const [A, setA] = useState([[2, -1], [-1, 2]]);
  const [b, setB] = useState([0, 3]);
  const [mode, setMode] = useState("row");
  const N = numM(A), g = mk(4.5);
  const det = N[0][0] * N[1][1] - N[0][1] * N[1][0];
  const x = Math.abs(det) < 1e-9 ? null : [(b[0] * N[1][1] - N[0][1] * b[1]) / det, (N[0][0] * b[1] - b[0] * N[1][0]) / det];
  const c1 = [N[0][0], N[1][0]], c2 = [N[0][1], N[1][1]];
  return (
    <Panel title="Row picture vs Column picture" hint="Ek hi system, do tarah ki tasveer — GATE mein dono poochte hain"
      side={
        <div>
          <MatInput M={A} onChange={setA} label="Matrix A" />
          <MatInput M={[[b[0]], [b[1]]]} onChange={(M) => setB([parseFloat(M[0][0]) || 0, parseFloat(M[1][0]) || 0])} label="Vector b" />
          <div className="flex gap-1.5 mb-2">
            <Btn active={mode === "row"} onClick={() => setMode("row")}>Row picture</Btn>
            <Btn active={mode === "col"} onClick={() => setMode("col")}>Column picture</Btn>
          </div>
          <KV k="det A" v={nf(det)} />
          <KV k="solution x" v={x ? `(${nf(x[0], 2)}, ${nf(x[1], 2)})` : "nahi (singular)"} c={x ? CO.p : CO.b} />
          <p className="text-xs text-stone-500 mt-2">
            {mode === "row" ? "Har row ek line hai; lines ka intersection = solution." : "x·col₁ + y·col₂ = b — columns ko kitna kheenchna hai?"}
          </p>
        </div>
      }>
      <svg viewBox={`0 0 ${g.w} ${g.h}`} className="w-full">
        <Grid g={g} /><Axes g={g} />
        {mode === "row" ? (
          <g>
            {[0, 1].map((i) => {
              const a = N[i][0], cc = N[i][1];
              if (Math.abs(cc) > 1e-9) {
                const y1 = (b[i] + 20 * a) / cc, y2 = (b[i] - 20 * a) / cc;
                return <line key={i} x1={g.X(-20)} y1={g.Y(y1)} x2={g.X(20)} y2={g.Y(y2)} stroke={i ? CO.v : CO.u} strokeWidth="2" />;
              }
              if (Math.abs(a) > 1e-9) return <line key={i} x1={g.X(b[i] / a)} y1={0} x2={g.X(b[i] / a)} y2={g.h} stroke={i ? CO.v : CO.u} strokeWidth="2" />;
              return null;
            })}
            {x && <circle cx={g.X(x[0])} cy={g.Y(x[1])} r="5" fill={CO.b} />}
          </g>
        ) : (
          <g>
            <Arrow g={g} to={c1} color="#fcd34d" label="col₁" />
            <Arrow g={g} to={c2} color="#a5f3fc" label="col₂" />
            {x && <Arrow g={g} to={[x[0] * c1[0], x[0] * c1[1]]} color={CO.u} label={nf(x[0], 2) + "·col₁"} />}
            {x && <Arrow g={g} from={[x[0] * c1[0], x[0] * c1[1]]} to={[x[0] * c1[0] + x[1] * c2[0], x[0] * c1[1] + x[1] * c2[1]]} color={CO.v} label={nf(x[1], 2) + "·col₂"} />}
            <Arrow g={g} to={b} color={CO.b} label="b" />
          </g>
        )}
      </svg>
    </Panel>
  );
}

function VElim() {
  const [A, setA] = useState([[2, 1, 1, 5], [4, -6, 0, -2], [-2, 7, 2, 9]]);
  const [step, setStep] = useState(0);
  const N = numM(A);
  const log = useMemo(() => {
    const M = cloneM(N), n = 3, out = [{ t: "Shuruaat — augmented [A | b]", M: cloneM(M) }];
    let swaps = 0;
    for (let k = 0; k < n; k++) {
      let best = k;
      for (let i = k; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[best][k])) best = i;
      if (Math.abs(M[best][k]) < 1e-10) { out.push({ t: `Column ${k + 1}: pivot 0 — singular matrix`, M: cloneM(M), bad: true }); continue; }
      if (best !== k) { [M[k], M[best]] = [M[best], M[k]]; swaps++; out.push({ t: `Row swap R${k + 1} ↔ R${best + 1}`, M: cloneM(M) }); }
      out.push({ t: `Pivot ${k + 1} = ${nf(M[k][k])}`, M: cloneM(M) });
      for (let i = k + 1; i < n; i++) {
        if (Math.abs(M[i][k]) < 1e-12) continue;
        const f = M[i][k] / M[k][k];
        for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
        out.push({ t: `m${i + 1}${k + 1} = ${nf(f)} → R${i + 1} − (${nf(f)})·R${k + 1}`, M: cloneM(M) });
      }
    }
    const U = M.map((r) => r.slice(0, 3)), c = M.map((r) => r[3]);
    let piv = 0; for (let i = 0; i < 3; i++) if (Math.abs(U[i][i]) > 1e-9) piv++;
    let det = Math.pow(-1, swaps); for (let i = 0; i < 3; i++) det *= U[i][i];
    let x = null;
    if (piv === 3) { x = [0, 0, 0]; for (let i = 2; i >= 0; i--) { let s = c[i]; for (let j = i + 1; j < 3; j++) s -= U[i][j] * x[j]; x[i] = s / U[i][i]; } }
    return { out, piv, det, x };
  }, [JSON.stringify(N)]);
  const i = Math.min(step, log.out.length - 1), cur = log.out[i];
  return (
    <Panel title="Gaussian elimination — har step alag se" hint="[A | b] ko U tak le jao, phir back-substitution"
      side={
        <div>
          <MatInput M={A} onChange={(M) => { setA(M); setStep(0); }} label="Augmented [A | b]" />
          <div className="flex gap-1.5 mb-2 flex-wrap">
            <Btn onClick={() => { setA([[1, 2, 3, 6], [2, 4, 6, 12], [1, 1, 1, 3]]); setStep(0); }}>Singular</Btn>
            <Btn onClick={() => { setA([[0, 2, 1, 4], [1, 1, 1, 3], [2, 1, 3, 7]]); setStep(0); }}>Swap chahiye</Btn>
          </div>
          <KV k="pivots = rank" v={String(log.piv)} />
          <KV k="det A" v={nf(log.det)} />
          <KV k="solution" v={log.x ? `(${log.x.map((t) => nf(t, 2)).join(", ")})` : "unique nahi"} c={log.x ? CO.p : CO.b} />
          <p className="text-xs text-stone-500 mt-2">Ek hi elimination se rank, det, invertibility aur solution — chaaron mil jaate hain.</p>
        </div>
      }>
      <div>
        <div className="flex gap-1.5 mb-3 items-center">
          <Btn onClick={() => setStep(Math.max(0, i - 1))}>← Peeche</Btn>
          <Btn active onClick={() => setStep(Math.min(log.out.length - 1, i + 1))}>Aage →</Btn>
          <Btn onClick={() => setStep(0)}>Reset</Btn>
          <span className="text-xs text-stone-500 ml-1">Step {i + 1} / {log.out.length}</span>
        </div>
        <div className="bg-white rounded-lg border border-stone-200 p-3 mb-2">
          <Mtx M={cur.M.map((r) => r.map((v) => nf(v, 2)))} />
        </div>
        <div className={"text-sm font-medium mb-2 " + (cur.bad ? "text-rose-600" : "text-indigo-700")}>{cur.t}</div>
        <div className="max-h-32 overflow-y-auto text-xs space-y-1">
          {log.out.map((s, k) => (
            <div key={k} className={k <= i ? "text-stone-700" : "text-stone-400"}>
              <span className="font-semibold">{k + 1}.</span> {s.t}
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function VRref() {
  const [A, setA] = useState([[1, 2, 2, 2], [2, 4, 6, 8], [3, 6, 8, 10]]);
  const N = numM(A);
  const { R, pivots, rank } = rref(N);
  const nb = nullBasis(N);
  const m = N.length, n = N[0].length;
  const free = []; for (let j = 0; j < n; j++) if (!pivots.includes(j)) free.push(j + 1);
  return (
    <Panel title="RREF Lab — rank, pivot/free columns, null space basis" hint="Apni matrix daalo. Yeh hi tool har GATE numerical mein chahiye."
      side={
        <div className="space-y-0">
          <KV k="size m × n" v={`${m} × ${n}`} />
          <KV k="rank r" v={String(rank)} c={CO.g2} />
          <KV k="pivot columns" v={pivots.map((p) => p + 1).join(", ") || "—"} />
          <KV k="free columns" v={free.join(", ") || "koi nahi"} />
          <KV k="nullity = n − r" v={String(n - rank)} />
          <KV k="dim C(A) = r" v={String(rank)} />
          <KV k="dim N(Aᵀ) = m − r" v={String(m - rank)} />
          <div className="mt-3">
            <div className="text-xs font-semibold text-stone-500 mb-1">Special solutions (N(A) ka basis)</div>
            {nb.length === 0 ? <div className="text-sm text-stone-600">sirf x = 0 (columns independent)</div>
              : nb.map((v, i) => <div key={i} className="font-mono text-sm text-emerald-700">({v.map((t) => nf(t, 2)).join(", ")})</div>)}
          </div>
          <div className="mt-3">
            <div className="text-xs font-semibold text-stone-500 mb-1">C(A) ka basis = A ke ORIGINAL pivot columns</div>
            {pivots.map((p) => <div key={p} className="font-mono text-sm text-indigo-700">col{p + 1} = ({N.map((r) => nf(r[p], 2)).join(", ")})</div>)}
          </div>
        </div>
      }>
      <div>
        <MatInput M={A} onChange={setA} label="Matrix A (3 × 4)" />
        <div className="flex gap-1.5 mb-3 flex-wrap">
          <Btn onClick={() => setA([[1, 2, 2, 2], [2, 4, 6, 8], [3, 6, 8, 10]])}>rank 2</Btn>
          <Btn onClick={() => setA([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4]])}>full row rank</Btn>
          <Btn onClick={() => setA([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]])}>rank 1</Btn>
        </div>
        <div className="bg-white rounded-lg border border-stone-200 p-3">
          <div className="text-xs font-semibold text-stone-500 mb-1">RREF (R)</div>
          <Mtx M={R.map((r) => r.map((v) => nf(v, 2)))} />
        </div>
      </div>
    </Panel>
  );
}

function VLU() {
  const [A, setA] = useState([[2, 1, 1], [4, -6, 0], [-2, 7, 2]]);
  const N = numM(A), { L, U, P, swaps, log } = luFact(N);
  const isI = P.every((r, i) => r.every((v, j) => v === (i === j ? 1 : 0)));
  const piv = [U[0][0], U[1][1], U[2][2]];
  return (
    <Panel title="A = LU factorisation" hint="L mein multipliers seedhe bhar jaate hain — dubara likhne ki zarurat nahi"
      side={
        <div>
          <KV k="row swaps" v={String(swaps)} />
          <KV k="pivots" v={piv.map((v) => nf(v, 2)).join(", ")} />
          <KV k="det A = ±∏pivots" v={nf(Math.pow(-1, swaps) * piv.reduce((a, b) => a * b, 1))} />
          <div className="mt-3 max-h-32 overflow-y-auto text-xs space-y-1 text-stone-600">
            {log.map((s, i) => <div key={i}><b>{i + 1}.</b> {s}</div>)}
          </div>
        </div>
      }>
      <div>
        <MatInput M={A} onChange={setA} label="Matrix A (3 × 3)" />
        <div className="flex gap-1.5 mb-3 flex-wrap">
          <Btn onClick={() => setA([[1, 2, 1], [3, 8, 1], [0, 4, 1]])}>no swap</Btn>
          <Btn onClick={() => setA([[0, 1, 2], [1, 3, 1], [2, 1, 0]])}>swap chahiye</Btn>
          <Btn onClick={() => setA([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])}>symmetric</Btn>
        </div>
        <div className="bg-white rounded-lg border border-stone-200 p-3 overflow-x-auto">
          <div className="text-xs font-semibold text-stone-500 mb-1">{isI ? "A = L · U" : "P · A = L · U (row swaps lage)"}</div>
          <div className="flex items-center flex-wrap gap-1">
            {!isI && <><Mtx M={P} /><span className="text-stone-400">·</span></>}
            <Mtx M={N} /><span className="text-stone-500 mx-1">=</span>
            <Mtx M={L.map((r) => r.map((v) => nf(v, 2)))} />
            <Mtx M={U.map((r) => r.map((v) => nf(v, 2)))} />
          </div>
        </div>
      </div>
    </Panel>
  );
}

function VTransform() {
  const [A, setA] = useState([[1, 0.5], [0, 1]]);
  const g = mk(3.2);
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const e = eig2(A[0][0], A[0][1], A[1][0], A[1][1]);
  const P = [[0, 0], [A[0][0], A[1][0]], [A[0][0] + A[0][1], A[1][0] + A[1][1]], [A[0][1], A[1][1]]];
  return (
    <Panel title="Matrix = space par action (det = area factor)" hint="Unit square kya ban jaata hai — wahi A ka asli chehra hai"
      side={
        <div>
          {[["a", 0, 0], ["b", 0, 1], ["c", 1, 0], ["d", 1, 1]].map(([lb, i, j]) => (
            <Slider key={lb} label={lb} value={A[i][j]} min={-3} max={3} step={0.1}
              onChange={(v) => { const M = cloneM(A); M[i][j] = v; setA(M); }} />
          ))}
          <div className="flex gap-1.5 my-2 flex-wrap">
            <Btn onClick={() => setA([[0.707, -0.707], [0.707, 0.707]])}>Rotation</Btn>
            <Btn onClick={() => setA([[1, 1], [0, 1]])}>Shear</Btn>
            <Btn onClick={() => setA([[2, 1], [4, 2]])}>Singular</Btn>
          </div>
          <KV k="det A" v={nf(det)} c={Math.abs(det) < 1e-6 ? CO.b : CO.p} />
          <KV k="trace" v={nf(A[0][0] + A[1][1])} />
          <KV k="area factor" v={nf(Math.abs(det)) + "×"} />
          <p className={"text-xs mt-2 " + (Math.abs(det) < 1e-6 ? "text-rose-600" : "text-stone-500")}>
            {Math.abs(det) < 1e-6 ? "Singular — square ek line par gir gaya, ek dimension gum." : det < 0 ? "det < 0 → orientation flip ho gaya." : "Invertible — poora plane bacha hai."}
          </p>
        </div>
      }>
      <svg viewBox={`0 0 ${g.w} ${g.h}`} className="w-full">
        <Grid g={g} /><Axes g={g} />
        <polygon points={P.map((p) => `${g.X(p[0])},${g.Y(p[1])}`).join(" ")} fill="#c7d2fe" fillOpacity="0.55" stroke={CO.g2} strokeWidth="1.6" />
        <polygon points={[[0, 0], [1, 0], [1, 1], [0, 1]].map((p) => `${g.X(p[0])},${g.Y(p[1])}`).join(" ")} fill="none" stroke="#a8a29e" strokeWidth="1.2" strokeDasharray="4 3" />
        {e.kind !== "complex" && <FullLine g={g} dir={e.v1} color="#ddd6fe" />}
        <Arrow g={g} to={[A[0][0], A[1][0]]} color={CO.u} label="A·i" />
        <Arrow g={g} to={[A[0][1], A[1][1]]} color={CO.v} label="A·j" />
      </svg>
    </Panel>
  );
}

function VProject() {
  const [a, setA] = useState([3, 1]), [b, setB] = useState([1, 3]);
  const g = mk(4);
  const aa = a[0] * a[0] + a[1] * a[1], ab = a[0] * b[0] + a[1] * b[1];
  const xh = ab / (aa || 1), p = [xh * a[0], xh * a[1]], e = [b[0] - p[0], b[1] - p[1]];
  const Pm = [[a[0] * a[0] / aa, a[0] * a[1] / aa], [a[1] * a[0] / aa, a[1] * a[1] / aa]];
  return (
    <Panel title="Projection onto a line — p, e aur P matrix" hint="e hamesha a ke perpendicular: aᵀe = 0"
      side={
        <div>
          <Slider label="a₁" value={a[0]} min={-4} max={4} step={0.1} onChange={(v) => setA([v, a[1]])} />
          <Slider label="a₂" value={a[1]} min={-4} max={4} step={0.1} onChange={(v) => setA([a[0], v])} />
          <Slider label="b₁" value={b[0]} min={-4} max={4} step={0.1} onChange={(v) => setB([v, b[1]])} />
          <Slider label="b₂" value={b[1]} min={-4} max={4} step={0.1} onChange={(v) => setB([b[0], v])} />
          <KV k="x̂ = aᵀb / aᵀa" v={nf(xh)} />
          <KV k="p = x̂a" v={`(${nf(p[0], 2)}, ${nf(p[1], 2)})`} c={CO.p} />
          <KV k="e = b − p" v={`(${nf(e[0], 2)}, ${nf(e[1], 2)})`} c={CO.e} />
          <KV k="aᵀe" v={nf(a[0] * e[0] + a[1] * e[1], 3)} c={CO.p} />
          <div className="mt-2 bg-white border border-stone-200 rounded-lg p-2">
            <div className="text-xs text-stone-500 mb-1">P = aaᵀ / aᵀa (rank 1)</div>
            <Mtx M={Pm.map((r) => r.map((v) => nf(v, 2)))} />
          </div>
        </div>
      }>
      <svg viewBox={`0 0 ${g.w} ${g.h}`} className="w-full">
        <Grid g={g} /><Axes g={g} />
        <FullLine g={g} dir={a} color="#fcd34d" />
        <line x1={g.X(b[0])} y1={g.Y(b[1])} x2={g.X(p[0])} y2={g.Y(p[1])} stroke={CO.e} strokeWidth="2" strokeDasharray="5 4" />
        <Arrow g={g} to={a} color={CO.u} label="a" />
        <Arrow g={g} to={b} color={CO.b} label="b" />
        <Arrow g={g} to={p} color={CO.p} label="p" />
        <circle cx={g.X(p[0])} cy={g.Y(p[1])} r="4" fill={CO.p} />
      </svg>
    </Panel>
  );
}

function VEigen() {
  const [A, setA] = useState([[2, 1], [1, 2]]);
  const g = mk(3.6);
  const e = eig2(A[0][0], A[0][1], A[1][0], A[1][1]);
  const pts = [];
  for (let i = 0; i <= 72; i++) {
    const t = (i / 72) * Math.PI * 2, cx = 1.5 * Math.cos(t), cy = 1.5 * Math.sin(t);
    pts.push([A[0][0] * cx + A[0][1] * cy, A[1][0] * cx + A[1][1] * cy]);
  }
  return (
    <Panel title="Ax = λx — eigen-directions dekho" hint="Neela circle = saare unit x; laal ellipse = unka image Ax"
      side={
        <div>
          {[["a", 0, 0], ["b", 0, 1], ["c", 1, 0], ["d", 1, 1]].map(([lb, i, j]) => (
            <Slider key={lb} label={lb} value={A[i][j]} min={-3} max={3} step={0.1}
              onChange={(v) => { const M = cloneM(A); M[i][j] = v; setA(M); }} />
          ))}
          <div className="flex gap-1.5 my-2 flex-wrap">
            <Btn onClick={() => setA([[2, 1], [1, 2]])}>Symmetric</Btn>
            <Btn onClick={() => setA([[2, 1], [0, 2]])}>Defective</Btn>
            <Btn onClick={() => setA([[0, -1], [1, 0]])}>Complex</Btn>
          </div>
          <KV k="trace = Σλ" v={nf(e.tr)} />
          <KV k="det = Πλ" v={nf(e.det)} />
          {e.kind === "complex"
            ? <div className="text-sm text-rose-600 mt-2">λ = {nf(e.re, 2)} ± {nf(e.im, 2)}i — koi real eigenvector nahi (pure rotation).</div>
            : <>
              <KV k="λ₁" v={nf(e.l1, 3)} c={CO.eig} />
              <KV k="λ₂" v={nf(e.l2, 3)} c={CO.e} />
              <div className={"text-sm mt-2 " + (e.kind === "defective" ? "text-rose-600" : "text-emerald-700")}>
                {e.kind === "defective" ? "Repeated λ par GM = 1 → DEFECTIVE, diagonalize nahi hoga"
                  : e.kind === "repeated" ? "Repeated λ, GM = AM = 2 → diagonalizable"
                    : "Distinct λ → hamesha diagonalizable"}
              </div>
            </>}
        </div>
      }>
      <svg viewBox={`0 0 ${g.w} ${g.h}`} className="w-full">
        <Grid g={g} /><Axes g={g} />
        <circle cx={g.X(0)} cy={g.Y(0)} r={1.5 * g.s} fill="none" stroke="#93c5fd" strokeWidth="1.5" />
        <polyline points={pts.map((p) => `${g.X(p[0])},${g.Y(p[1])}`).join(" ")} fill="none" stroke="#fda4af" strokeWidth="2" />
        {e.kind !== "complex" && <FullLine g={g} dir={e.v1} color={CO.eig} />}
        {e.kind !== "complex" && e.kind !== "defective" && <FullLine g={g} dir={e.v2} color={CO.e} />}
        {e.kind !== "complex" && <Arrow g={g} to={[e.v1[0] * 1.5, e.v1[1] * 1.5]} color={CO.eig} label="x₁" />}
        {e.kind !== "complex" && <Arrow g={g} to={[e.v1[0] * 1.5 * e.l1, e.v1[1] * 1.5 * e.l1]} color={CO.eig} width={1.6} dash="4 3" label="λ₁x₁" />}
      </svg>
    </Panel>
  );
}

function VQuad() {
  const [A, setA] = useState([[2, 1], [1, 3]]);
  const g = mk(3);
  const e = eig2(A[0][0], A[0][1], A[1][0], A[1][1]);
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const p1 = A[0][0], p2 = Math.abs(p1) > 1e-9 ? det / p1 : NaN;
  const kind = e.l1 > 1e-9 && e.l2 > 1e-9 ? "Positive definite — upward bowl"
    : e.l1 < -1e-9 && e.l2 < -1e-9 ? "Negative definite — inverted bowl"
      : e.l1 >= -1e-9 && e.l2 >= -1e-9 ? "Positive semi-definite — valley (ek direction flat)"
        : e.l1 <= 1e-9 && e.l2 <= 1e-9 ? "Negative semi-definite" : "Indefinite — saddle point";
  const curve = (c) => {
    const out = [];
    for (let i = 0; i <= 240; i++) {
      const th = (i / 240) * Math.PI * 2, dx = Math.cos(th), dy = Math.sin(th);
      const q = A[0][0] * dx * dx + 2 * A[0][1] * dx * dy + A[1][1] * dy * dy;
      if (q * c <= 1e-6) { out.push(null); continue; }
      const r = Math.sqrt(c / q);
      out.push(r > g.range * 1.5 ? null : [dx * r, dy * r]);
    }
    const segs = []; let cur = [];
    out.forEach((p) => { if (!p) { if (cur.length > 1) segs.push(cur); cur = []; } else cur.push(p); });
    if (cur.length > 1) segs.push(cur);
    return segs;
  };
  return (
    <Panel title="Quadratic form xᵀAx — bowl ya saddle?" hint="Green curve: xᵀAx = +1 · Red curve: xᵀAx = −1"
      side={
        <div>
          <Slider label="a" value={A[0][0]} min={-4} max={4} step={0.1} onChange={(v) => setA([[v, A[0][1]], [A[1][0], A[1][1]]])} />
          <Slider label="b (dono corners)" value={A[0][1]} min={-4} max={4} step={0.1} onChange={(v) => setA([[A[0][0], v], [v, A[1][1]]])} />
          <Slider label="c" value={A[1][1]} min={-4} max={4} step={0.1} onChange={(v) => setA([[A[0][0], A[0][1]], [A[1][0], v]])} />
          <div className="flex gap-1.5 my-2 flex-wrap">
            <Btn onClick={() => setA([[2, 1], [1, 3]])}>PD</Btn>
            <Btn onClick={() => setA([[1, 1], [1, 1]])}>PSD</Btn>
            <Btn onClick={() => setA([[1, 2], [2, 1]])}>Indefinite</Btn>
          </div>
          <div className="font-medium text-sm text-stone-800 mb-1">{kind}</div>
          <KV k="Test 1 · eigenvalues" v={`${nf(e.l1, 2)}, ${nf(e.l2, 2)}`} />
          <KV k="Test 2 · pivots" v={`${nf(p1, 2)}, ${nf(p2, 2)}`} />
          <KV k="Test 3 · minors" v={`${nf(A[0][0], 2)}, ${nf(det, 2)}`} />
          <p className="text-xs text-stone-500 mt-2">Teenon test hamesha ek hi jawab dete hain. xᵀAx = {nf(A[0][0])}x² + {nf(2 * A[0][1])}xy + {nf(A[1][1])}y²</p>
        </div>
      }>
      <svg viewBox={`0 0 ${g.w} ${g.h}`} className="w-full">
        <Grid g={g} /><Axes g={g} />
        {curve(1).map((sg, i) => <polyline key={"a" + i} points={sg.map((p) => `${g.X(p[0])},${g.Y(p[1])}`).join(" ")} fill="none" stroke="#059669" strokeWidth="2.2" />)}
        {curve(-1).map((sg, i) => <polyline key={"b" + i} points={sg.map((p) => `${g.X(p[0])},${g.Y(p[1])}`).join(" ")} fill="none" stroke="#e11d48" strokeWidth="2.2" />)}
        {curve(4).map((sg, i) => <polyline key={"c" + i} points={sg.map((p) => `${g.X(p[0])},${g.Y(p[1])}`).join(" ")} fill="none" stroke="#a7f3d0" strokeWidth="1.6" />)}
        {e.kind !== "complex" && <><FullLine g={g} dir={e.v1} color="#c4b5fd" /><FullLine g={g} dir={e.v2} color="#c4b5fd" /></>}
      </svg>
    </Panel>
  );
}

function VSVD() {
  const [A, setA] = useState([[3, 1], [1, 2]]);
  const g = mk(4.5);
  const { s1, s2, v1, v2, u1, u2 } = svd2(A);
  const pts = [];
  for (let i = 0; i <= 96; i++) {
    const t = (i / 96) * Math.PI * 2, x = Math.cos(t), y = Math.sin(t);
    pts.push([A[0][0] * x + A[0][1] * y, A[1][0] * x + A[1][1] * y]);
  }
  return (
    <Panel title="SVD ka geometry — circle → ellipse" hint="v₁, v₂ = input ke perpendicular axes; σ₁u₁, σ₂u₂ = output ke axes"
      side={
        <div>
          {[["a", 0, 0], ["b", 0, 1], ["c", 1, 0], ["d", 1, 1]].map(([lb, i, j]) => (
            <Slider key={lb} label={lb} value={A[i][j]} min={-4} max={4} step={0.1}
              onChange={(v) => { const M = cloneM(A); M[i][j] = v; setA(M); }} />
          ))}
          <KV k="σ₁ (largest)" v={nf(s1, 3)} c={CO.b} />
          <KV k="σ₂" v={nf(s2, 3)} c={CO.v} />
          <KV k="rank" v={String((s1 > 1e-6 ? 1 : 0) + (s2 > 1e-6 ? 1 : 0))} />
          <KV k="|det A| = σ₁σ₂" v={nf(s1 * s2, 3)} />
          <KV k="‖A‖₂ = σ₁" v={nf(s1, 3)} />
          <KV k="‖A‖F = √(σ₁²+σ₂²)" v={nf(Math.sqrt(s1 * s1 + s2 * s2), 3)} />
          <KV k="cond = σ₁/σ₂" v={s2 > 1e-9 ? nf(s1 / s2, 2) : "∞"} />
          <p className="text-xs text-stone-500 mt-2">Av₁ = σ₁u₁ aur Av₂ = σ₂u₂ — yahi poori SVD ka dil hai.</p>
        </div>
      }>
      <svg viewBox={`0 0 ${g.w} ${g.h}`} className="w-full">
        <Grid g={g} /><Axes g={g} />
        <circle cx={g.X(0)} cy={g.Y(0)} r={g.s} fill="none" stroke="#93c5fd" strokeWidth="1.5" />
        <polyline points={pts.map((p) => `${g.X(p[0])},${g.Y(p[1])}`).join(" ")} fill="none" stroke="#fb923c" strokeWidth="2" />
        <Arrow g={g} to={v1} color="#2563eb" label="v₁" width={2} />
        <Arrow g={g} to={v2} color="#60a5fa" label="v₂" width={2} />
        <Arrow g={g} to={[u1[0] * s1, u1[1] * s1]} color={CO.b} label="σ₁u₁" />
        <Arrow g={g} to={[u2[0] * s2, u2[1] * s2]} color={CO.v} label="σ₂u₂" />
      </svg>
    </Panel>
  );
}

function VRankK() {
  const base = useMemo(() => {
    const n = 12, M = [];
    for (let i = 0; i < n; i++) { const r = []; for (let j = 0; j < n; j++) r.push(0.5 + 0.5 * Math.sin(i / 2.2) * Math.cos(j / 1.8) + 0.25 * Math.sin((i + j) / 3)); M.push(r); }
    return M;
  }, []);
  const [k, setK] = useState(1);
  const { sig, V, U } = useMemo(() => svdFull(base), [base]);
  const approx = useMemo(() => {
    const n = base.length, out = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let t = 0; t < k; t++) {
      if (!U[t] || sig[t] < 1e-9) continue;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i][j] += sig[t] * U[t][i] * V[t][j];
    }
    return out;
  }, [k, base]);
  const errF = Math.sqrt(sig.slice(k).reduce((s, v) => s + v * v, 0));
  const cell = (v) => { const c = Math.max(0, Math.min(1, v)); const g2 = Math.round(255 * c); return `rgb(${g2},${g2},${g2})`; };
  const G = ({ M }) => (
    <svg viewBox="0 0 120 120" className="w-full max-w-xs border border-stone-300 rounded">
      {M.map((r, i) => r.map((v, j) => <rect key={i + "-" + j} x={j * 10} y={i * 10} width="10" height="10" fill={cell(v)} />))}
    </svg>
  );
  return (
    <Panel title="Rank-k approximation (Eckart–Young live)" hint="k badhao — top-k singular values kitni jaldi poori tasveer bana dete hain"
      side={
        <div>
          <Slider label={"k (rank)"} value={k} min={1} max={12} step={1} onChange={(v) => setK(Math.round(v))} />
          <KV k="σ values (top 6)" v={sig.slice(0, 6).map((s) => nf(s, 2)).join(", ")} />
          <KV k="‖A − A_k‖₂ = σ_{k+1}" v={nf(sig[k] || 0, 3)} c={CO.b} />
          <KV k="‖A − A_k‖F" v={nf(errF, 3)} c={CO.e} />
          <KV k="storage saved" v={`${nf(100 * (1 - (k * (12 + 12 + 1)) / 144), 0)}%`} />
          <p className="text-xs text-stone-500 mt-2">Eckart–Young: koi bhi rank-k matrix B is A_k se behtar nahi ho sakti — Frobenius, spectral aur nuclear teenon norms mein.</p>
        </div>
      }>
      <div className="flex gap-4 items-start flex-wrap">
        <div><div className="text-xs text-stone-500 mb-1">Original (rank {sig.filter((s) => s > 1e-6).length})</div><G M={base} /></div>
        <div><div className="text-xs text-stone-500 mb-1">Rank-{k} approximation</div><G M={approx} /></div>
      </div>
    </Panel>
  );
}

function VBlock() {
  const [r, setR] = useState(3), [c, setC] = useState(2);
  const n = 5, cellSize = 34;
  return (
    <Panel title="Partition (block) matrix — edge-to-edge cuts" hint="Cut lines poori width/height cover karni chahiye"
      side={
        <div>
          <Slider label="row cut ke baad" value={r} min={1} max={4} step={1} onChange={(v) => setR(Math.round(v))} />
          <Slider label="column cut ke baad" value={c} min={1} max={4} step={1} onChange={(v) => setC(Math.round(v))} />
          <KV k="A block" v={`${r} × ${c}`} />
          <KV k="B block" v={`${r} × ${n - c}`} />
          <KV k="C block" v={`${n - r} × ${c}`} />
          <KV k="D block" v={`${n - r} × ${n - c}`} />
          <p className="text-xs text-stone-500 mt-2">
            det formula ke liye A aur D dono <b>square</b> hone chahiye — yaani r = c. Abhi {r === c ? "✓ square hain" : "✗ square nahi hain, det formula lagega hi nahi"}.
          </p>
        </div>
      }>
      <svg viewBox={`0 0 ${n * cellSize + 20} ${n * cellSize + 20}`} className="w-full max-w-sm">
        {Array.from({ length: n }).map((_, i) => Array.from({ length: n }).map((_, j) => {
          const inA = i < r && j < c, inB = i < r && j >= c, inC = i >= r && j < c;
          const fill = inA ? "#c7d2fe" : inB ? "#fed7aa" : inC ? "#bbf7d0" : "#fbcfe8";
          return <rect key={i + "-" + j} x={10 + j * cellSize} y={10 + i * cellSize} width={cellSize - 2} height={cellSize - 2} fill={fill} stroke="#d6d3d1" />;
        }))}
        <line x1={10} y1={10 + r * cellSize - 1} x2={10 + n * cellSize - 2} y2={10 + r * cellSize - 1} stroke="#dc2626" strokeWidth="2.5" strokeDasharray="6 4" />
        <line x1={10 + c * cellSize - 1} y1={10} x2={10 + c * cellSize - 1} y2={10 + n * cellSize - 2} stroke="#dc2626" strokeWidth="2.5" strokeDasharray="6 4" />
        <text x={10 + (c * cellSize) / 2 - 6} y={16 + (r * cellSize) / 2} fontSize="18" fontWeight="700" fill="#3730a3">A</text>
        <text x={10 + c * cellSize + ((n - c) * cellSize) / 2 - 6} y={16 + (r * cellSize) / 2} fontSize="18" fontWeight="700" fill="#9a3412">B</text>
        <text x={10 + (c * cellSize) / 2 - 6} y={16 + r * cellSize + ((n - r) * cellSize) / 2} fontSize="18" fontWeight="700" fill="#166534">C</text>
        <text x={10 + c * cellSize + ((n - c) * cellSize) / 2 - 6} y={16 + r * cellSize + ((n - r) * cellSize) / 2} fontSize="18" fontWeight="700" fill="#9d174d">D</text>
      </svg>
    </Panel>
  );
}

function VComplexity() {
  const [n, setN] = useState(10);
  const fwd = (2 * n ** 3 + 3 * n ** 2 - 5 * n) / 3, back = n ** 2;
  const g = { w: 360, h: 220 };
  const maxN = 30, maxY = (2 * maxN ** 3 + 3 * maxN ** 2 - 5 * maxN) / 3;
  const path = (f) => Array.from({ length: maxN }, (_, i) => i + 1).map((x) => `${20 + ((x - 1) / (maxN - 1)) * (g.w - 40)},${g.h - 20 - (f(x) / maxY) * (g.h - 40)}`).join(" ");
  return (
    <Panel title="Ax = b solve karne ki cost" hint="Forward elimination ~⅔n³ dominate karti hai; back substitution ~n² almost free"
      side={
        <div>
          <Slider label="n (matrix size)" value={n} min={2} max={30} step={1} onChange={(v) => setN(Math.round(v))} />
          <KV k="Forward elimination" v={nf(fwd, 0) + " ops"} c={CO.b} />
          <KV k="Back substitution" v={nf(back, 0) + " ops"} c={CO.v} />
          <KV k="Total" v={nf((2 * n ** 3 + 6 * n ** 2 - 5 * n) / 3, 0) + " ops"} />
          <KV k="back / forward" v={nf((100 * back) / fwd, 1) + "%"} />
          <p className="text-xs text-stone-500 mt-2">n badhne par back substitution ka hissa 0 ki taraf jaata hai — isliye complexity O(n³) likhte hain.</p>
        </div>
      }>
      <svg viewBox={`0 0 ${g.w} ${g.h}`} className="w-full">
        <line x1="20" y1={g.h - 20} x2={g.w - 20} y2={g.h - 20} stroke="#a8a29e" />
        <line x1="20" y1="20" x2="20" y2={g.h - 20} stroke="#a8a29e" />
        <polyline points={path((x) => (2 * x ** 3 + 3 * x ** 2 - 5 * x) / 3)} fill="none" stroke={CO.b} strokeWidth="2.5" />
        <polyline points={path((x) => x ** 2)} fill="none" stroke={CO.v} strokeWidth="2.5" />
        <circle cx={20 + ((n - 1) / (maxN - 1)) * (g.w - 40)} cy={g.h - 20 - (fwd / maxY) * (g.h - 40)} r="4" fill={CO.b} />
        <text x={g.w - 120} y="34" fontSize="12" fill={CO.b} fontWeight="600">forward ≈ ⅔n³</text>
        <text x={g.w - 120} y="50" fontSize="12" fill={CO.v} fontWeight="600">back ≈ n²</text>
        <text x="24" y={g.h - 6} fontSize="11" fill="#78716c">n = 1</text>
        <text x={g.w - 50} y={g.h - 6} fontSize="11" fill="#78716c">n = 30</text>
      </svg>
    </Panel>
  );
}

const VISUALS = { span: VSpan, rowcol: VRowCol, elim: VElim, rref: VRref, lu: VLU, transform: VTransform, project: VProject, eigen: VEigen, quad: VQuad, svd: VSVD, rankk: VRankK, block: VBlock, complexity: VComplexity };

/* ============================== CONTENT ============================== */
const CH1 = {
  id: "ch1", n: "01", title: "Basics & System of Linear Equations",
  blurb: "Vectors, span, independence, AX = B, elimination, rank aur solution ki teen possibilities.",
  topics: [
    {
      id: "vectors", title: "Vectors in Rⁿ", body: [
        { t: "p", x: "Vector ek **ordered list** hai n real numbers ki. R² mein 2 entries, R³ mein 3, Rⁿ mein n. Geometrically yeh origin se ek arrow hai." },
        { t: "f", label: "Notation", x: "u = [3, −1]ᵀ  ∈ R²        A = [2, 3, 4]ᵀ  ∈ R³", note: "Course bhar vectors **column** form mein likhte hain. Transpose isliye lagta hai ki line mein likh sakein." },
        { t: "list", x: [
          "**Scalar multiplication:** c·u vector ko stretch (|c|>1), shrink (|c|<1) ya flip (c<0) karta hai.",
          "Ek non-zero vector u ke saare multiples c·u milkar **origin se guzarti ek line** banate hain.",
          "**Length:** ‖u‖ = √(u₁² + u₂² + ... + uₙ²) = √(uᵀu)",
          "**Unit vector:** û = u / ‖u‖ — direction wahi, length 1.",
        ]},
        { t: "f", label: "Dot product", x: "u · v = uᵀv = u₁v₁ + u₂v₂ + ... = ‖u‖‖v‖ cos θ", note: "uᵀv = 0 ⟺ **orthogonal**. Yeh baat projection aur four subspaces ki poori buniyad hai." },
      ],
      ex: {
        problem: "u = [3, −1]ᵀ aur v = [1, 2]ᵀ ke liye: (a) 2u − v, (b) ‖u‖, (c) u aur v ke beech ka angle nikalo.",
        steps: [
          { title: "2u − v component-wise nikalo", detail: "2u = [6, −2]ᵀ\n2u − v = [6 − 1, −2 − 2]ᵀ = [5, −4]ᵀ" },
          { title: "Length formula lagao", detail: "‖u‖ = √(3² + (−1)²) = √(9 + 1) = √10 ≈ 3.162" },
          { title: "Dot product nikalo", detail: "u·v = (3)(1) + (−1)(2) = 3 − 2 = 1" },
          { title: "cos θ formula se angle", detail: "‖v‖ = √(1 + 4) = √5\ncos θ = 1 / (√10 · √5) = 1/√50 = 0.1414\nθ = cos⁻¹(0.1414) ≈ 81.87°" },
        ],
        answer: "2u − v = [5, −4]ᵀ · ‖u‖ = √10 ≈ 3.162 · θ ≈ 81.87°",
      },
      quiz: { q: "Agar u·v = 0 aur dono non-zero hain, to?", opts: ["u = v", "u aur v perpendicular hain", "u = −v", "dono unit vectors hain"], a: 1, why: "cos θ = 0 ⟹ θ = 90°. Yahi orthogonality ki definition hai." },
    },
    {
      id: "span", title: "Linear Combination & Span", vis: "span", body: [
        { t: "f", label: "Linear combination", x: "y = a₁v₁ + a₂v₂ + ... + a_p v_p", note: "Scalars a₁...a_p ko har possible value do — jo saare y ban sakte hain unka set = **Span{v₁,...,v_p}**." },
        { t: "p", x: "Span hamesha ek **subspace** hota hai aur hamesha origin se guzarta hai (saare a = 0 rakhne se y = 0 milta hai)." },
        { t: "table", head: ["Vectors", "Condition", "Span ka shape"], rows: [
          ["1 vector u in Rⁿ", "u ≠ 0", "Origin se guzarti ek line"],
          ["2 vectors u, v in R²", "v = c·u", "Ek line"],
          ["2 vectors u, v in R²", "v ≠ c·u", "Poora R²"],
          ["2 vectors u, v in R³", "v = c·u", "R³ mein ek line"],
          ["2 vectors u, v in R³", "v ≠ c·u", "R³ mein ek plane"],
          ["3 vectors in R³", "saare independent", "Poora R³"],
          ["3 vectors in R³", "ek doosron ka combo", "Ek plane"],
        ]},
        { t: "tip", x: "Jo vector pehle se Span mein hai, usko add karne se span **nahi badalta** — wo koi nayi dimension contribute nahi karta." },
      ],
      ex: {
        problem: "Kya b = [7, 4]ᵀ, vectors v₁ = [1, 2]ᵀ aur v₂ = [3, 1]ᵀ ke span mein hai? Agar haan to combination likho.",
        steps: [
          { title: "Sawaal ko AX = b mein badlo", detail: "b = x₁v₁ + x₂v₂ poochna hi Ax = b poochna hai\nA = [v₁ v₂] = [[1, 3], [2, 1]],  b = [7, 4]ᵀ" },
          { title: "det A check karo", detail: "det = (1)(1) − (3)(2) = 1 − 6 = −5 ≠ 0\n⟹ columns independent ⟹ span = poora R² ⟹ b zaroor andar hai" },
          { title: "Solve karo (elimination)", detail: "[1 3 | 7]\n[2 1 | 4]\nR2 − 2R1: [0 −5 | −10] ⟹ x₂ = 2" },
          { title: "Back-substitute", detail: "x₁ + 3(2) = 7 ⟹ x₁ = 1" },
        ],
        answer: "Haan — b = 1·v₁ + 2·v₂. Kyunki v₁, v₂ poora R² span karte hain, **har** b andar hoga.",
      },
      quiz: { q: "R³ mein 2 independent vectors ka span kya hai?", opts: ["Ek line", "Ek plane", "Poora R³", "Sirf origin"], a: 1, why: "2 independent directions = 2-dimensional subspace = plane (origin se guzarta hua)." },
    },
    {
      id: "independence", title: "Linear Independence & Dependence", body: [
        { t: "f", label: "Definition", x: "c₁v₁ + c₂v₂ + ... + cₙvₙ = 0  sirf tab jab saare cᵢ = 0  ⟹ INDEPENDENT", note: "Matrix bhaasha mein: N(A) = {0}, yaani Ax = 0 ka sirf trivial solution." },
        { t: "trap", label: "2 vs 3+ vectors ka farq", x: "**Exactly 2 vectors** dependent tabhi hain jab ek doosre ka scalar multiple ho (v₂ = c·v₁). **3 ya zyada** vectors ke liye yeh galat test hai — koi bhi ek vector baaki ka **general combination** (v₃ = a·v₁ + b·v₂) ho to bhi dependent hain, chahe koi do vector aapas mein multiple na hon." },
        { t: "list", x: [
          "Agar vectors ki ginti > dimension (n > m), to wo **hamesha dependent** hain — bina calculation ke.",
          "**Pivot columns** hamesha linearly independent hote hain.",
          "**Free variable ke columns** hamesha dependent hote hain (wo pivot columns ka combination hain).",
          "Koi bhi set jisme zero vector ho, wo dependent hai.",
        ]},
      ],
      ex: {
        problem: "v₁ = [1, 2, 3]ᵀ, v₂ = [2, 4, 6]ᵀ, v₃ = [1, 0, 1]ᵀ — independent hain ya dependent? Dependent hain to relation likho.",
        steps: [
          { title: "Pehle scalar-multiple check (sabse sasta)", detail: "v₂ = 2·v₁ ? → [2,4,6] = 2·[1,2,3] ✓ HAAN" },
          { title: "Turant conclusion", detail: "Ek pair hi dependent mil gaya, isliye poora set dependent hai.\nAur aage calculation ki zarurat nahi." },
          { title: "Relation likho", detail: "2v₁ − v₂ + 0·v₃ = 0\nYaani c = (2, −1, 0) ek non-trivial solution hai Ax = 0 ka." },
          { title: "Rank se cross-check", detail: "A = [[1,2,1],[2,4,0],[3,6,1]]\nR2 − 2R1 → [0,0,−2];  R3 − 3R1 → [0,0,−2]\nR3 − R2 → [0,0,0] ⟹ rank = 2 < 3 ⟹ dependent ✓" },
        ],
        answer: "Dependent. Relation: **2v₁ − v₂ = 0**. Rank = 2, nullity = 1.",
      },
      quiz: { q: "4 vectors R³ mein diye gaye. Independence ke baare mein kya keh sakte ho?", opts: ["Independent ho sakte hain", "Hamesha dependent", "Depends on values", "Hamesha independent"], a: 1, why: "n = 4 > m = 3. Rank zyada se zyada 3 ho sakta hai, isliye kam se kam 1 free variable — dependent guaranteed." },
    },
    {
      id: "system", title: "System of Linear Equations — AX = B", vis: "rowcol", body: [
        { t: "f", label: "Linear equation", x: "a₁x₁ + a₂x₂ + ... + aₙxₙ = b", note: "Har variable mein **degree 1** — na x², na x·y, na √x. Warna linear nahi." },
        { t: "p", x: "Matrix form **AX = B**: A = coefficient matrix, X = unknown vector, B = RHS. Solve karne ke liye **augmented matrix [A | B]** banake row-reduce karte hain." },
        { t: "table", head: ["Outcome", "2D geometry", "Consistency"], rows: [
          ["Unique solution", "Lines ek point par milti hain", "CONSISTENT"],
          ["No solution", "Lines parallel hain", "INCONSISTENT"],
          ["Infinite solutions", "Lines bilkul coincide karti hain", "CONSISTENT"],
        ]},
        { t: "p", x: "3 variables mein har equation ek **plane** hai R³ mein: teen planes ek point par mil sakte hain (unique), ek line par (infinite), ya kahin nahi (no solution)." },
        { t: "tip", label: "Column picture — sabse important", x: "AX = B ka solution hai **⟺ B, A ke columns ka linear combination hai ⟺ B ∈ C(A)**. Poore course ka yehi central idea hai." },
      ],
      ex: {
        problem: "Solve karo: 2x − y = 0 aur −x + 2y = 3. Dono row aur column picture mein interpret karo.",
        steps: [
          { title: "Augmented matrix banao", detail: "[ 2  −1 | 0 ]\n[ −1  2 | 3 ]" },
          { title: "Elimination (R2 + ½R1)", detail: "m = −1/2 → R2 − (−1/2)R1 = R2 + ½R1\n[ 2  −1 | 0 ]\n[ 0  1.5 | 3 ]" },
          { title: "Back-substitution", detail: "1.5y = 3 ⟹ y = 2\n2x − 2 = 0 ⟹ x = 1" },
          { title: "Column picture mein padho", detail: "1·[2, −1]ᵀ + 2·[−1, 2]ᵀ = [2 − 2, −1 + 4]ᵀ = [0, 3]ᵀ = b ✓\nYaani column 1 ko 1 guna aur column 2 ko 2 guna kheencho." },
        ],
        answer: "x = 1, y = 2. Row picture: do lines (1,2) par milti hain. Column picture: 1·c₁ + 2·c₂ = b.",
      },
      quiz: { q: "3 planes R³ mein triangle jaisa prism banate hain (koi common point nahi). System?", opts: ["Unique solution", "Infinite solutions", "No solution", "Trivial solution"], a: 2, why: "Koi bhi point teenon planes par nahi hai ⟹ inconsistent ⟹ rank(A) < rank[A|B]." },
    },
    {
      id: "elimination", title: "Gaussian Elimination — REF & RREF", vis: "elim", body: [
        { t: "num", x: [
          "**[A | B]** likho — augmented matrix.",
          "Pehla non-zero element = **pivot**. Multiplier m = a_ij / pivot nikalo.",
          "Rᵢ ← Rᵢ − m·Rⱼ karke pivot ke neeche zero banao.",
          "Agla column repeat karo → **REF (U)** milta hai.",
          "**Back-substitution** se X nikalo.",
        ]},
        { t: "table", head: ["REF (Row Echelon Form)", "RREF (Reduced REF)"], rows: [
          ["Pivots koi bhi non-zero number", "Saare pivots = 1"],
          ["Sirf pivot ke **neeche** zeros", "Pivot ke upar **aur** neeche dono zeros"],
          ["**NOT unique** — operation order pe depend", "**Hamesha unique**"],
        ]},
        { t: "list", x: [
          "**Pivot variable** — pivot column ka variable. Hamesha linearly independent.",
          "**Free variable** — non-pivot column ka variable. Koi bhi real value le sakta hai.",
          "**Free variables ki ginti = n − rank(A)**",
        ]},
        { t: "trap", x: "Pivot position par 0 aa gaya: agar **neeche non-zero** hai to rows swap karo (temporary failure, kaam chalu). Agar poora column neeche bhi zero hai to wo column **free variable column** hai (permanent failure — matrix singular)." },
        { t: "trap", label: "Sabse zyada poocha jaane wala", x: "Agar augmented matrix [A|B] ka **last column pivot column ban jaaye**, to system **inconsistent** hai. Jaise row [0 0 0 | 5] ka matlab hai 0 = 5 — impossible." },
      ],
      ex: {
        problem: "Elimination se solve karo: 2x + 4y − 2z = 2, 4x + 9y − 3z = 8, −2x − 2y + 7z = 10.",
        steps: [
          { title: "Augmented matrix", detail: "[  2   4  −2 |  2 ]\n[  4   9  −3 |  8 ]\n[ −2  −2   7 | 10 ]" },
          { title: "Column 1 clear karo (pivot = 2)", detail: "m₂₁ = 4/2 = 2 → R2 − 2R1 = [0, 1, 1 | 4]\nm₃₁ = −2/2 = −1 → R3 + R1 = [0, 2, 5 | 12]" },
          { title: "Column 2 clear karo (pivot = 1)", detail: "m₃₂ = 2/1 = 2 → R3 − 2R2 = [0, 0, 3 | 4]" },
          { title: "REF (U) mil gaya", detail: "[ 2  4  −2 | 2 ]\n[ 0  1   1 | 4 ]\n[ 0  0   3 | 4 ]\nPivots = 2, 1, 3 ⟹ rank = 3, det = 2·1·3 = 6" },
          { title: "Back-substitution (neeche se upar)", detail: "3z = 4 ⟹ z = 4/3\ny + 4/3 = 4 ⟹ y = 8/3\n2x + 4(8/3) − 2(4/3) = 2 ⟹ 2x + 32/3 − 8/3 = 2 ⟹ 2x = 2 − 8 = −6 ⟹ x = −3" },
        ],
        answer: "x = −3, y = 8/3, z = 4/3. Rank = 3, det A = 6 (pivots ka product).",
      },
      quiz: { q: "REF mein ek poori zero row aa gayi. Iska matlab?", opts: ["Matrix invertible hai", "Original A ka wo row baaki rows ka linear combination tha", "System inconsistent hai", "Determinant 1 hai"], a: 1, why: "Zero row banne ka matlab hai wo row dependent thi. Inconsistency tabhi hai jab uske saamne B side non-zero ho." },
    },
    {
      id: "rank", title: "Rank & Column Space", vis: "rref", body: [
        { t: "f", label: "Rank ki teen equivalent definitions", x: "rank(A) = independent columns ki ginti\n         = independent rows ki ginti\n         = pivots ki ginti\n         = REF mein non-zero rows", note: "**Row rank = column rank** — hamesha, har matrix ke liye. Yeh linear algebra ka sabse sundar result hai." },
        { t: "f", label: "Column space", x: "C(A) = Span{a₁, a₂, ..., aₙ} ⊆ Rᵐ,   dim C(A) = r", note: "AX = b solvable hai ⟺ b ∈ C(A)." },
        { t: "list", x: [
          "**rank(A + B) ≤ rank(A) + rank(B)** — bilkul 0 tak bhi ja sakta hai.",
          "**rank(AB) ≤ min(rank A, rank B)**",
          "rank(A) = rank(Aᵀ) = rank(AᵀA) = rank(AAᵀ)",
          "C(A) mein pehle se maujood column add karne se C(A) **nahi badalta**.",
        ]},
        { t: "tip", x: "C(A) ka basis chahiye? **A ke ORIGINAL pivot columns** lo — RREF ke columns kabhi mat lo. Row operations column space ko badal dete hain, row space ko nahi." },
      ],
      ex: {
        problem: "A = [[1, 2, 2, 2], [2, 4, 6, 8], [3, 6, 8, 10]] ka rank aur C(A) ka basis nikalo.",
        steps: [
          { title: "Elimination — column 1", detail: "R2 − 2R1 = [0, 0, 2, 4]\nR3 − 3R1 = [0, 0, 2, 4]" },
          { title: "Column 2 dekho", detail: "Column 2 mein pivot ke neeche sab zero — koi pivot nahi ⟹ **free column**" },
          { title: "Column 3 se aage", detail: "R3 − R2 = [0, 0, 0, 0]\nREF:\n[1 2 2 2]\n[0 0 2 4]\n[0 0 0 0]" },
          { title: "Pivots ginno", detail: "Pivot columns = 1 aur 3 ⟹ **rank = 2**\nFree columns = 2 aur 4 ⟹ nullity = 4 − 2 = 2" },
          { title: "C(A) ka basis — ORIGINAL A ke columns", detail: "col1 = [1, 2, 3]ᵀ,  col3 = [2, 6, 8]ᵀ\ndim C(A) = 2, aur yeh R³ mein ek plane hai." },
        ],
        answer: "rank = 2. C(A) ka basis = {[1,2,3]ᵀ, [2,6,8]ᵀ}. Nullity = 2.",
      },
      quiz: { q: "A ek 5×7 matrix hai jiska rank 4 hai. dim N(A) kitni hai?", opts: ["1", "3", "2", "4"], a: 1, why: "Rank-nullity: r + nullity = n = 7 ⟹ nullity = 7 − 4 = 3." },
    },
    {
      id: "matmul", title: "Matrix Multiplication — teen tareeke", body: [
        { t: "f", label: "Size rule", x: "A(m×n) · B(n×p) = AB(m×p)", note: "**Inner dimensions match** karni chahiye; result outer dimensions leta hai. Multiplications = m·n·p, additions = m·(n−1)·p." },
        { t: "table", head: ["Method", "Rule", "Kab useful"], rows: [
          ["I — Dot product", "(AB)ᵢⱼ = row i of A · col j of B", "Numbers nikaalne ke liye"],
          ["II — Row method", "Row i of AB = B ke rows ka weighted combination, weights A ke row i se", "Row space, elimination"],
          ["III — Column method", "Col j of AB = A ke columns ka weighted combination, weights B ke col j se", "Column space samajhne ke liye"],
          ["IV — Outer product", "AB = Σₖ (A ka colₖ)(B ka rowₖ)", "Rank-1 pieces, LU, SVD"],
        ]},
        { t: "table", head: ["Observation", "Kya batata hai"], rows: [
          ["B mein b₁ = b₂", "AB ke pehle do columns bhi equal honge"],
          ["B ka col3 = b₁ + b₂", "AB ka col3 = Ab₁ + Ab₂"],
          ["B ka col2 = 0", "AB ka col2 = 0"],
          ["AB ka last col = 0", "A ke columns dependent hain (Ax = 0 ka non-trivial solution)"],
          ["B ke cols dependent", "AB ke cols bhi dependent"],
        ]},
        { t: "trap", label: "Critical non-properties", x: "**AB ≠ BA** (commutative nahi). **AB = AC se B = C nahi aata** (cancellation fail). **AB = 0 se A = 0 ya B = 0 nahi aata.** Lekin A(BC) = (AB)C associative hamesha sahi hai." },
      ],
      ex: {
        problem: "A = [[1, 2], [3, 4]], B = [[0, 1], [1, 0]]. AB aur BA nikaal kar dikhao ki wo alag hain, aur column method se AB verify karo.",
        steps: [
          { title: "AB — dot product method", detail: "(AB)₁₁ = 1(0) + 2(1) = 2,  (AB)₁₂ = 1(1) + 2(0) = 1\n(AB)₂₁ = 3(0) + 4(1) = 4,  (AB)₂₂ = 3(1) + 4(0) = 3\nAB = [[2, 1], [4, 3]]" },
          { title: "BA — dot product method", detail: "(BA)₁₁ = 0(1) + 1(3) = 3,  (BA)₁₂ = 0(2) + 1(4) = 4\n(BA)₂₁ = 1(1) + 0(3) = 1,  (BA)₂₂ = 1(2) + 0(4) = 2\nBA = [[3, 4], [1, 2]]" },
          { title: "Column method se AB check", detail: "AB ka col1 = A · (B ka col1) = A·[0,1]ᵀ = 0·[1,3]ᵀ + 1·[2,4]ᵀ = [2,4]ᵀ ✓\nAB ka col2 = A·[1,0]ᵀ = 1·[1,3]ᵀ + 0·[2,4]ᵀ = [1,3]ᵀ ✓" },
          { title: "Interpret karo", detail: "B ek permutation matrix hai. B ko **right** se multiply karne se A ke COLUMNS swap hue.\nB ko **left** se multiply karne se A ke ROWS swap hote." },
        ],
        answer: "AB = [[2,1],[4,3]] ≠ BA = [[3,4],[1,2]]. Row interchange = left multiplication (PA), column interchange = right multiplication (AP).",
      },
      quiz: { q: "Permutation matrix P se rows swap karne ke liye kya karna hoga?", opts: ["A · P", "P · A", "Pᵀ · Aᵀ", "A · Pᵀ"], a: 1, why: "Row operations LEFT multiplication se hote hain (P·A); column operations RIGHT multiplication se (A·P)." },
    },
    {
      id: "solutions", title: "Solutions of AX = B — consistency & uniqueness", body: [
        { t: "num", x: [
          "**Rows check karo → consistency**: solution exist karta hai ya nahi?",
          "**Columns check karo → uniqueness**: kitne solutions hain?",
        ]},
        { t: "table", head: ["Rows (consistency)", "Columns (uniqueness)", "Solution type"], rows: [
          ["Saare rows LI", "Saare cols LI — koi free var nahi", "**Unique solution**"],
          ["Saare rows LI", "Kuch cols LD — free vars hain", "**Infinite solutions**"],
          ["Kuch rows LD; restriction B par bhi apply", "—", "**Infinite solutions**"],
          ["Kuch rows LD; restriction B par apply NAHI", "—", "**No solution**"],
        ]},
        { t: "f", label: "Rank test (sabse tez tareeka)", x: "rank(A) < rank[A|B]  ⟹  NO SOLUTION\nrank(A) = rank[A|B] = n  ⟹  UNIQUE\nrank(A) = rank[A|B] < n  ⟹  INFINITE (n − r free variables)" },
        { t: "list", x: [
          "AX = b **har** b ∈ Rᵐ ke liye solvable hai ⟺ A ke columns Rᵐ span karte hain ⟺ **har row mein pivot** ⟺ rank = m.",
          "Solution **hamesha unique** hai ⟺ A ke columns independent ⟺ rank = n ⟺ N(A) = {0}.",
        ]},
        { t: "table", head: ["m = n (square)", "m > n (tall)", "n > m (wide)"], rows: [
          ["Full rank (r = n): har b ke liye unique solution. r < n: kabhi unique nahi.", "Free variable nahi ho sakta agar saare n cols LI hon. Par Rᵐ bhar nahi sakta — har b ke liye solvable nahi.", "**ALWAYS** free variables (kam se kam n − m). AX = 0 ka hamesha non-trivial solution."],
        ]},
      ],
      ex: {
        problem: "k ki kis value ke liye system solvable hai? x + 2y + 3z = 1, 2x + 4y + 6z = 2, x + y + z = k.",
        steps: [
          { title: "Augmented matrix banao", detail: "[ 1  2  3 | 1 ]\n[ 2  4  6 | 2 ]\n[ 1  1  1 | k ]" },
          { title: "Eliminate karo", detail: "R2 − 2R1 = [0, 0, 0 | 0]  ← poori zero row, dono taraf\nR3 − R1 = [0, −1, −2 | k − 1]" },
          { title: "Ranks compare karo", detail: "REF:\n[1  2  3 | 1]\n[0 −1 −2 | k−1]\n[0  0  0 | 0]\nrank(A) = 2, rank[A|B] = 2 — **har k ke liye barabar**" },
          { title: "Conclusion", detail: "Zero row ke saamne B side bhi 0 hai (k se independent), isliye system **har k ke liye consistent** hai.\nn = 3, r = 2 ⟹ 1 free variable ⟹ infinite solutions." },
          { title: "Agar R2 = [2 4 6 | 5] hota", detail: "Tab R2 − 2R1 = [0,0,0 | 3] ⟹ 0 = 3 ⟹ inconsistent, koi k bachaa nahi sakta." },
        ],
        answer: "Har real k ke liye system consistent hai, aur hamesha **infinite solutions** (rank 2 < n = 3).",
      },
      quiz: { q: "A ek 3×5 matrix hai. AX = 0 ke baare mein kya pakka keh sakte ho?", opts: ["Sirf trivial solution", "Non-trivial solution zaroor hoga", "Koi solution nahi", "Depends on entries"], a: 1, why: "n = 5 > m = 3, isliye rank ≤ 3 aur free variables ≥ 2. Wide matrix ka homogeneous system hamesha non-trivial solution deta hai." },
    },
    {
      id: "homogeneous", title: "Homogeneous System AX = 0", body: [
        { t: "p", x: "**AX = 0 hamesha consistent hai** — X = 0 (trivial solution) hamesha kaam karta hai. Sawaal sirf yeh hai ki koi aur solution bhi hai ya nahi." },
        { t: "f", label: "Do equivalences", x: "Non-trivial solution ⟺ kam se kam 1 free variable ⟺ columns dependent\nSirf trivial solution ⟺ koi free variable nahi ⟺ columns independent" },
        { t: "table", head: ["Koi free variable nahi", "k free variables"], rows: [
          ["AX = 0 ka sirf X = 0. Columns LI. Kisi b ke liye AX = b ka unique solution.", "AX = 0 ka solution = Span{v₁,...,v_k} — ek **k-dimensional subspace**. AX = b ke infinite solutions."],
        ]},
        { t: "tip", x: "AX = 0 ka solution set **hamesha origin se guzarta hai**. AX = b (b ≠ 0) ka solution set **kabhi origin se nahi guzarta** — wo null space ka parallel translate hai." },
      ],
      ex: {
        problem: "A = [[1, 2, 2, 2], [2, 4, 6, 8], [3, 6, 8, 10]] ke liye AX = 0 ka complete solution nikalo.",
        steps: [
          { title: "RREF tak le jao", detail: "REF: [1 2 2 2; 0 0 2 4; 0 0 0 0]\nR2 ÷ 2: [0 0 1 2]\nR1 − 2R2: [1 2 0 −2]\nRREF R = [1 2 0 −2; 0 0 1 2; 0 0 0 0]" },
          { title: "Pivot aur free variables pehchano", detail: "Pivot columns: 1, 3 ⟹ pivot variables x₁, x₃\nFree columns: 2, 4 ⟹ free variables x₂, x₄\nNumber of special solutions = n − r = 4 − 2 = 2" },
          { title: "Equations likho", detail: "Row1: x₁ + 2x₂ − 2x₄ = 0 ⟹ x₁ = −2x₂ + 2x₄\nRow2: x₃ + 2x₄ = 0 ⟹ x₃ = −2x₄" },
          { title: "Special solution 1: x₂ = 1, x₄ = 0", detail: "x₁ = −2, x₃ = 0\ns₁ = [−2, 1, 0, 0]ᵀ" },
          { title: "Special solution 2: x₂ = 0, x₄ = 1", detail: "x₁ = 2, x₃ = −2\ns₂ = [2, 0, −2, 1]ᵀ" },
          { title: "Verify karo", detail: "A·s₁ = [1(−2)+2(1), 2(−2)+4(1), 3(−2)+6(1)]ᵀ = [0, 0, 0]ᵀ ✓" },
        ],
        answer: "N(A) = Span{[−2,1,0,0]ᵀ, [2,0,−2,1]ᵀ}, dimension 2. Complete solution: X = c₁s₁ + c₂s₂.",
      },
      quiz: { q: "Ek invertible 4×4 matrix A ke liye AX = 0 ka solution set?", opts: ["Ek line", "Ek plane", "Sirf {0}", "Poora R⁴"], a: 2, why: "Invertible ⟹ rank = 4 = n ⟹ nullity = 0 ⟹ N(A) = {0}, sirf trivial solution." },
    },
    {
      id: "general", title: "General Solution X = Xp + Xn", body: [
        { t: "f", label: "Theorem", x: "AXp = b  aur  AXn = 0   ⟹   X = Xp + Xn  poora solution set hai", note: "Proof: A(Xp + Xn) = AXp + AXn = b + 0 = b ✓" },
        { t: "table", head: ["Free variables", "Solution ka roop"], rows: [
          ["1", "X = Xp + t·v"],
          ["2", "X = Xp + s·v₁ + t·v₂"],
          ["k", "X = Xp + t₁v₁ + t₂v₂ + ... + t_k v_k"],
        ]},
        { t: "num", x: [
          "**Xp nikalne ke liye:** saare free variables = 0 rakho, pivot variables solve karo.",
          "**Xn nikalne ke liye:** ek free variable = 1 (baaki 0) rakho, solve karo — har free variable ke liye repeat.",
        ]},
        { t: "tip", x: "Geometrically: AX = 0 ka solution set origin se guzarta hai; AX = b ka solution set **wahi shape hai par Xp se shift ho gaya**. Isliye nullspace line hai to solution set bhi line hoga (bas origin se nahi guzregi)." },
      ],
      ex: {
        problem: "AX = b solve karo jahan A = [[1, 3, 0, 2], [0, 0, 1, 4], [1, 3, 1, 6]] aur b = [1, 6, 7]ᵀ.",
        steps: [
          { title: "Augmented matrix reduce karo", detail: "[1 3 0 2 | 1]\n[0 0 1 4 | 6]\n[1 3 1 6 | 7]\nR3 − R1: [0 0 1 4 | 6]\nR3 − R2: [0 0 0 0 | 0]  ← consistent ✓" },
          { title: "RREF", detail: "R = [1 3 0 2 | 1]\n    [0 0 1 4 | 6]\n    [0 0 0 0 | 0]\nPivot cols: 1, 3.  Free: x₂, x₄. rank r = 2, n = 4" },
          { title: "Xp — free variables = 0 rakho", detail: "x₂ = 0, x₄ = 0\nRow1: x₁ = 1\nRow2: x₃ = 6\nXp = [1, 0, 6, 0]ᵀ" },
          { title: "Xn₁ — x₂ = 1, x₄ = 0", detail: "x₁ + 3(1) = 0 ⟹ x₁ = −3\nx₃ = 0\ns₁ = [−3, 1, 0, 0]ᵀ" },
          { title: "Xn₂ — x₂ = 0, x₄ = 1", detail: "x₁ + 2(1) = 0 ⟹ x₁ = −2\nx₃ + 4(1) = 0 ⟹ x₃ = −4\ns₂ = [−2, 0, −4, 1]ᵀ" },
          { title: "Complete solution jodo", detail: "X = [1, 0, 6, 0]ᵀ + s[−3, 1, 0, 0]ᵀ + t[−2, 0, −4, 1]ᵀ" },
        ],
        answer: "X = [1,0,6,0]ᵀ + s·[−3,1,0,0]ᵀ + t·[−2,0,−4,1]ᵀ — ek 2-dimensional plane jo Xp se shift hai.",
      },
      quiz: { q: "X₁ aur X₂ dono AX = b solve karte hain. X₁ − X₂ kahan hoga?", opts: ["C(A) mein", "N(A) mein", "Row space mein", "Kahin nahi"], a: 1, why: "A(X₁ − X₂) = b − b = 0, isliye difference hamesha null space mein hota hai." },
    },
  ],
};

const CH2 = {
  id: "ch2", n: "02", title: "Matrix Algebra & LU Decomposition",
  blurb: "Operations, transpose, inverse, Gauss-Jordan, A = LU / PA = LU / LDU, aur invertibility ki saari equivalences.",
  topics: [
    {
      id: "ops", title: "Matrix Operations & Properties", vis: "transform", body: [
        { t: "p", x: "Addition tabhi possible hai jab **dimensions bilkul same** hon — element-wise hota hai. Alag size par A + C **undefined** hai." },
        { t: "table", head: ["Property", "Statement"], rows: [
          ["Associative", "A(BC) = (AB)C"],
          ["Left distributive", "A(B + C) = AB + AC"],
          ["Right distributive", "(B + C)A = BA + CA"],
          ["Scalar", "γ(AB) = (γA)B = A(γB)"],
          ["Identity", "Iₘ·A = A = A·Iₙ"],
        ]},
        { t: "trap", label: "Critical non-properties", x: "**AB ≠ BA** general mein. **AB = AC ⇏ B = C.** **AB = 0 ⇏ A = 0 ya B = 0** — jaise A = [[0,1],[0,0]] ke liye A² = 0 par A ≠ 0." },
        { t: "p", x: "Geometrically matrix ek **transformation** hai: A ke columns batate hain ki basis vectors i aur j kahan pahunche. Unit square ka naya area = |det A|." },
      ],
      ex: {
        problem: "A = [[1, 1], [0, 1]] aur B = [[1, 0], [1, 1]]. AB, BA nikaalo aur dikhao ki AB ≠ BA. Phir A^n ka pattern nikalo.",
        steps: [
          { title: "AB nikalo", detail: "AB = [[1·1 + 1·1, 1·0 + 1·1], [0·1 + 1·1, 0·0 + 1·1]]\n   = [[2, 1], [1, 1]]" },
          { title: "BA nikalo", detail: "BA = [[1·1 + 0·0, 1·1 + 0·1], [1·1 + 1·0, 1·1 + 1·1]]\n   = [[1, 1], [1, 2]]" },
          { title: "Compare", detail: "AB = [[2,1],[1,1]] ≠ [[1,1],[1,2]] = BA ✓\n(Dono ka det = 1 aur trace = 3 hai, par matrices alag hain!)" },
          { title: "A² nikaal kar pattern dekho", detail: "A² = [[1,1],[0,1]]·[[1,1],[0,1]] = [[1, 2], [0, 1]]\nA³ = [[1, 3], [0, 1]]" },
          { title: "Generalise", detail: "A^n = [[1, n], [0, 1]]\nYeh shear matrix hai — har power aur zyada shear karti hai. Note: eigenvalue 1 repeated hai par A defective hai." },
        ],
        answer: "AB = [[2,1],[1,1]], BA = [[1,1],[1,2]] — alag hain. A^n = [[1, n], [0, 1]].",
      },
      quiz: { q: "AB = 0 diya hai aur A ≠ 0. Kya conclude kar sakte ho?", opts: ["B = 0 zaroori hai", "B ke columns N(A) mein hain", "A invertible hai", "AB = BA"], a: 1, why: "AB ka har column = A·(B ka column) = 0, isliye B ka har column A ke null space mein hai." },
    },
    {
      id: "transpose", title: "Transpose", body: [
        { t: "f", label: "Definition", x: "A = [[a, b], [c, d]]  ⟹  Aᵀ = [[a, c], [b, d]]", note: "m×n matrix ka transpose n×m hota hai — rows columns ban jaate hain." },
        { t: "table", head: ["Rule", "Formula"], rows: [
          ["(a)", "(Aᵀ)ᵀ = A"],
          ["(b)", "(A + B)ᵀ = Aᵀ + Bᵀ"],
          ["(c)", "(γA)ᵀ = γAᵀ"],
          ["(d) ★", "**(AB)ᵀ = BᵀAᵀ** — order ULTA ho jaata hai"],
        ]},
        { t: "tip", label: "Socks and shoes rule", x: "(AB)ᵀ = BᵀAᵀ aur (AB)⁻¹ = B⁻¹A⁻¹ — dono mein order reverse. Mozey pehle pehnte ho, jute baad mein; utaarte waqt jute pehle." },
        { t: "table", head: ["Type", "Condition", "Khaas baat"], rows: [
          ["Symmetric", "Aᵀ = A", "Real eigenvalues, orthogonal eigenvectors"],
          ["Skew-symmetric", "Aᵀ = −A", "Diagonal saare 0; λ = 0 ya pure imaginary"],
          ["Orthogonal", "QᵀQ = I", "Q⁻¹ = Qᵀ, length preserve, |λ| = 1"],
        ]},
      ],
      ex: {
        problem: "A = [[1, 2, 3], [4, 5, 6]] aur B = [[1, 0], [0, 1], [1, 1]]. (AB)ᵀ nikalo do tareeke se aur verify karo.",
        steps: [
          { title: "AB nikalo (2×3 · 3×2 = 2×2)", detail: "(AB)₁₁ = 1(1)+2(0)+3(1) = 4\n(AB)₁₂ = 1(0)+2(1)+3(1) = 5\n(AB)₂₁ = 4(1)+5(0)+6(1) = 10\n(AB)₂₂ = 4(0)+5(1)+6(1) = 11\nAB = [[4, 5], [10, 11]]" },
          { title: "Seedha transpose lo", detail: "(AB)ᵀ = [[4, 10], [5, 11]]" },
          { title: "Ab BᵀAᵀ se karo", detail: "Bᵀ = [[1, 0, 1], [0, 1, 1]]  (2×3)\nAᵀ = [[1, 4], [2, 5], [3, 6]]  (3×2)" },
          { title: "BᵀAᵀ multiply karo", detail: "(BᵀAᵀ)₁₁ = 1(1)+0(2)+1(3) = 4\n(BᵀAᵀ)₁₂ = 1(4)+0(5)+1(6) = 10\n(BᵀAᵀ)₂₁ = 0(1)+1(2)+1(3) = 5\n(BᵀAᵀ)₂₂ = 0(4)+1(5)+1(6) = 11\nBᵀAᵀ = [[4, 10], [5, 11]] ✓" },
          { title: "Galat tareeka bhi dekho", detail: "AᵀBᵀ = (3×2)(2×3) = 3×3 matrix — size hi match nahi karti! Isliye order reverse karna zaroori hai." },
        ],
        answer: "(AB)ᵀ = BᵀAᵀ = [[4, 10], [5, 11]]. AᵀBᵀ to define hi nahi hota yahan.",
      },
      quiz: { q: "A symmetric hai aur invertible. A⁻¹ ke baare mein?", opts: ["Skew-symmetric", "Bhi symmetric", "Orthogonal", "Kuch nahi keh sakte"], a: 1, why: "(A⁻¹)ᵀ = (Aᵀ)⁻¹ = A⁻¹ kyunki Aᵀ = A. Isliye A⁻¹ bhi symmetric hai." },
    },
    {
      id: "inverse", title: "Inverse of a Matrix & Gauss-Jordan", body: [
        { t: "f", label: "Definition", x: "AA⁻¹ = I  aur  A⁻¹A = I  (dono hone chahiye)" },
        { t: "f", label: "2×2 formula", x: "A = [[a, b], [c, d]]  ⟹  A⁻¹ = (1/(ad − bc)) · [[d, −b], [−c, a]]", note: "Diagonal swap, off-diagonal ke sign badlo, det se divide karo." },
        { t: "table", head: ["#", "Statement", "Note"], rows: [
          ["1", "Inverse exist karta hai ⟺ elimination se **n pivots** milein", "Row exchanges allowed hain"],
          ["2", "Ek matrix ke do alag inverse nahi ho sakte", "BA = I aur AC = I ⟹ B = C"],
          ["3", "Ax = 0 ka non-trivial solution ⟹ A NOT invertible", "Non-trivial solution ↔ singular"],
          ["4", "A invertible ⟹ Ax = b ka unique solution x = A⁻¹b", "Har b ∈ Rⁿ ke liye"],
          ["5", "2×2 invertible ⟺ ad − bc ≠ 0", "ad − bc = determinant"],
          ["6", "Diagonal matrix invertible ⟺ koi diagonal entry 0 na ho", "Inverse = 1/dᵢ diagonal par"],
        ]},
        { t: "table", head: ["Algebraic property", "Formula"], rows: [
          ["(a)", "(A⁻¹)⁻¹ = A"],
          ["(b) ★", "**(AB)⁻¹ = B⁻¹A⁻¹** — order reverse"],
          ["(c)", "(Aᵀ)⁻¹ = (A⁻¹)ᵀ"],
        ]},
        { t: "table", head: ["Case", "Verdict", "Reason"], rows: [
          ["Diagonally dominant: |aᵢᵢ| > Σ|aᵢⱼ|", "**Always invertible**", "Diagonal baaki row par haavi"],
          ["Square with a zero row", "NOT invertible", "det = 0"],
          ["Row = other rows ka combination", "NOT invertible", "Rows dependent ⟹ det = 0"],
          ["Product of pivots", "= det(A)", "Elimination aur det ka connection"],
        ]},
        { t: "trap", x: "Diagonal dominance **sufficient** hai, **necessary** nahi. B = [[2,1,1],[1,2,1],[1,1,2]] mein 2 = 1 + 1 (strict nahi) — dominant nahi hai, phir bhi invertible hai." },
      ],
      ex: {
        problem: "Gauss-Jordan se A = [[2, 1, 1], [4, −6, 0], [−2, 7, 2]] ka inverse nikalo.",
        steps: [
          { title: "[A | I] likho", detail: "[  2   1  1 | 1 0 0 ]\n[  4  −6  0 | 0 1 0 ]\n[ −2   7  2 | 0 0 1 ]" },
          { title: "Forward elimination — column 1", detail: "R2 − 2R1: [0, −8, −2 | −2, 1, 0]\nR3 + R1:  [0,  8,  3 |  1, 0, 1]" },
          { title: "Column 2", detail: "R3 + R2: [0, 0, 1 | −1, 1, 1]\nAb REF mil gaya, pivots = 2, −8, 1 ⟹ det = 2(−8)(1) = −16 ≠ 0 ✓" },
          { title: "Backward (Jordan step) — column 3 clear", detail: "R2 + 2R3: [0, −8, 0 | −4, 3, 2]\nR1 − R3:  [2, 1, 0 | 2, −1, −1]" },
          { title: "Column 2 clear", detail: "R2 ÷ (−8): [0, 1, 0 | 0.5, −0.375, −0.25]\nR1 − R2: [2, 0, 0 | 1.5, −0.625, −0.75]" },
          { title: "Pivots ko 1 banao", detail: "R1 ÷ 2: [1, 0, 0 | 0.75, −0.3125, −0.375]\nA⁻¹ = [[0.75, −0.3125, −0.375], [0.5, −0.375, −0.25], [−1, 1, 1]]" },
          { title: "Verify (ek entry)", detail: "(AA⁻¹)₁₁ = 2(0.75) + 1(0.5) + 1(−1) = 1.5 + 0.5 − 1 = 1 ✓" },
        ],
        answer: "A⁻¹ = [[0.75, −0.3125, −0.375], [0.5, −0.375, −0.25], [−1, 1, 1]], det A = −16.",
      },
      quiz: { q: "A 3×5 matrix hai aur CA = I₅. Yeh possible hai?", opts: ["Haan", "Nahi — A ke columns dependent hain", "Sirf agar rank 3 ho", "Sirf agar A symmetric ho"], a: 1, why: "Columns (5) > rows (3) ⟹ columns dependent ⟹ Ax = 0 ka non-trivial solution ⟹ left inverse impossible." },
    },
    {
      id: "lu", title: "LU Factorisation — A = LU, PA = LU, LDU", vis: "lu", body: [
        { t: "f", label: "Core idea", x: "A = L · U   (jab koi row exchange na lage)", note: "L = lower triangular, **diagonal par 1s**, multipliers mᵢⱼ neeche stored. U = upper triangular, **pivots diagonal par**." },
        { t: "list", x: [
          "Row op Rᵢ → Rᵢ − m·Rⱼ ke liye multiplier m = aᵢⱼ/aⱼⱼ, jo L ke position (i, j) par **positive sign (+m)** ke saath baith jaata hai.",
          "Row exchanges lage to decomposition **PA = LU** ban jaata hai (P = permutation matrix).",
          "**A = LDU:** U ko aur factor karo — D = pivots ka diagonal matrix, U′ = unit upper triangular (1s diagonal par).",
          "**UL factorisation:** elimination bottom-up karo — U mein 1s diagonal par, L mein pivots.",
          "Symmetric A ke liye A = LDLᵀ; positive definite ke liye A = RᵀR (Cholesky).",
        ]},
        { t: "f", label: "Ax = b solve karna — do step", x: "A = LU ⟹ LUx = b.  Ux = c rakho:\nStep 1: Lc = b solve karo  → forward substitution\nStep 2: Ux = c solve karo  → back substitution", note: "Fayda: naye b ke liye elimination dobara nahi karni padti — sirf do triangular solves." },
      ],
      ex: {
        problem: "A = [[1, 2, 1], [3, 8, 1], [0, 4, 1]] ka LU decomposition nikalo, phir b = [2, 12, 2]ᵀ ke liye Ax = b solve karo.",
        steps: [
          { title: "Column 1 eliminate karo", detail: "m₂₁ = 3/1 = 3 → R2 − 3R1 = [0, 2, −2]\nm₃₁ = 0/1 = 0 → R3 unchanged = [0, 4, 1]" },
          { title: "Column 2 eliminate karo", detail: "m₃₂ = 4/2 = 2 → R3 − 2R2 = [0, 0, 5]" },
          { title: "U likho (pivots diagonal par)", detail: "U = [[1, 2, 1], [0, 2, −2], [0, 0, 5]]\nPivots = 1, 2, 5 ⟹ det A = 1·2·5 = 10" },
          { title: "L likho (multipliers seedhe daalo)", detail: "L = [[1, 0, 0], [3, 1, 0], [0, 2, 1]]\nm₂₁ = 3 position (2,1) par, m₃₂ = 2 position (3,2) par, m₃₁ = 0 position (3,1) par" },
          { title: "Step 1 — Lc = b (forward)", detail: "c₁ = 2\n3c₁ + c₂ = 12 ⟹ 6 + c₂ = 12 ⟹ c₂ = 6\n2c₂ + c₃ = 2 ⟹ 12 + c₃ = 2 ⟹ c₃ = −10" },
          { title: "Step 2 — Ux = c (backward)", detail: "5x₃ = −10 ⟹ x₃ = −2\n2x₂ − 2(−2) = 6 ⟹ 2x₂ = 2 ⟹ x₂ = 1\nx₁ + 2(1) + 1(−2) = 2 ⟹ x₁ = 2" },
        ],
        answer: "L = [[1,0,0],[3,1,0],[0,2,1]], U = [[1,2,1],[0,2,−2],[0,0,5]], x = [2, 1, −2]ᵀ.",
      },
      quiz: { q: "A = LU mein det(A) kaise nikalte hain?", opts: ["det L × det U", "U ke diagonal ka product", "Dono sahi hain (det L = 1)", "L ke diagonal ka product"], a: 2, why: "det L = 1 (unit lower triangular), isliye det A = 1 × (U ke pivots ka product)." },
    },
    {
      id: "equiv", title: "Invertibility — saari equivalences (MUST READ)", body: [
        { t: "p", x: "Ek n×n matrix A ke liye neeche wali **saari statements ek hi baat** hain. Agar ek sach hai to sab sach; ek galat hai to sab galat. GATE mein yahi ek page se bahut saare questions ban jaate hain." },
        { t: "f", label: "The equivalence list", x: "A invertible\n⟺ elimination se n pivots milte hain (koi zero pivot nahi)\n⟺ Ax = 0 ka sirf trivial solution x = 0\n⟺ A ke columns linearly independent\n⟺ A ke rows linearly independent\n⟺ det(A) ≠ 0\n⟺ rank(A) = n (full rank)\n⟺ N(A) = {0}, nullity = 0\n⟺ C(A) = Rⁿ\n⟺ Ax = b ka har b ∈ Rⁿ ke liye unique solution\n⟺ koi eigenvalue 0 nahi hai\n⟺ A⁻¹ exist karta hai" },
        { t: "trap", x: "**Invertible ≠ Diagonalizable.** Invertibility eigenvalues ke baare mein hai (λ = 0 hai ya nahi). Diagonalizability eigenvectors ke baare mein hai (kitne independent hain). [[2,1],[0,2]] invertible hai par diagonalizable nahi; [[0,0],[0,0]] diagonalizable hai par invertible nahi." },
        { t: "list", x: [
          "**CA = Iₙ** ⟹ A aur C ke columns independent ⟹ Ax = 0 ka sirf trivial solution.",
          "Columns > rows ⟹ columns dependent ⟹ Ax = 0 ka non-trivial solution ⟹ left inverse impossible.",
          "**AD = Iₘ** ⟹ A ke rows independent ⟹ Ax = b har b ∈ Rᵐ ke liye solvable.",
          "Rows > columns ⟹ A ke liye AD = I possible hi nahi.",
        ]},
      ],
      ex: {
        problem: "k ki kis value ke liye A = [[1, 2, 3], [2, k, 6], [3, 6, 9]] singular hai? Us k par rank aur nullity nikalo.",
        steps: [
          { title: "det = 0 condition lagao (cofactor along row 1)", detail: "det = 1·(k·9 − 6·6) − 2·(2·9 − 6·3) + 3·(2·6 − k·3)\n    = 1·(9k − 36) − 2·(18 − 18) + 3·(12 − 3k)" },
          { title: "Simplify karo", detail: "= 9k − 36 − 0 + 36 − 9k\n= 0" },
          { title: "Result interpret karo", detail: "det = 0 **har k ke liye**! Kyunki row3 = 3 × row1 hamesha hai:\nrow1 = [1,2,3], row3 = [3,6,9] = 3·row1\nRows dependent ⟹ singular, k chahe kuch bhi ho." },
          { title: "Rank nikalo (general k)", detail: "R3 − 3R1 = [0, 0, 0]\nR2 − 2R1 = [0, k − 4, 0]\nAgar k ≠ 4: rank = 2, nullity = 1\nAgar k = 4: R2 bhi zero ⟹ rank = 1, nullity = 2" },
        ],
        answer: "A **har** k ke liye singular hai (row3 = 3·row1). k ≠ 4 → rank 2, nullity 1. k = 4 → rank 1, nullity 2.",
      },
      quiz: { q: "det(A) = 0 diya hai. Kya NAHI keh sakte?", opts: ["A singular hai", "Ax = 0 ka non-trivial solution hai", "Ax = b ka koi solution nahi hai", "Kam se kam ek eigenvalue 0 hai"], a: 2, why: "Singular matrix ke liye Ax = b ya to **no solution** ya **infinite solutions** deta hai — b ke upar depend karta hai. Zero solutions guarantee nahi." },
    },
  ],
};

const CH3 = {
  id: "ch3", n: "03", title: "Vector Spaces & Four Fundamental Subspaces",
  blurb: "Subspace test, C(A), N(A), basis, rank-nullity, subspaces par operations, aur the big picture.",
  topics: [
    {
      id: "vs", title: "Vector Space & Subspace", body: [
        { t: "p", x: "Vector space V ek aisa set hai jo **vector addition** aur **scalar multiplication** ke under closed hai — koi bhi linear combination lo, wo set ke andar hi rehta hai." },
        { t: "f", label: "Formal rules", x: "1. Closed under addition:  u + v ∈ V\n2. Closed under scalar mult:  cu ∈ V", note: "c = 0 le sakte hain, isliye **har vector space mein zero vector hona hi hai** (0 ∈ V)." },
        { t: "f", label: "Subspace test — sirf 3 cheez", x: "1. 0 ∈ H\n2. u + v ∈ H  (addition ke under closed)\n3. cu ∈ H  (scalar mult ke under closed)", note: "**Ultra-fast check:** teenon ko ek mein milao — cu + dv ∈ H har scalar c, d ke liye." },
        { t: "table", head: ["R² ke subspaces", "R³ ke subspaces"], rows: [
          ["{0} only", "{0} only"],
          ["Origin se guzarti line", "Origin se guzarti line"],
          ["Poora R²", "Origin se guzarta plane"],
          ["—", "Poora R³"],
        ]},
        { t: "trap", x: "Origin se **na** guzarne wala plane ya line subspace **nahi** hai — 0 andar nahi hai, aur do points ka sum bhi bahar chala jaata hai. Yahi Ax = b (b ≠ 0) ka solution set hai — wo affine set hai, subspace nahi." },
        { t: "p", x: "**Matrix spaces:** saare m×n matrices ka set M(m×n) bhi vector space hai. Subspace check karte waqt sirf addition aur scalar multiplication dekhte hain — **matrix multiplication se koi lena-dena nahi**. Determinant ya invertibility jaise properties non-linear hain aur subspace rules tod deti hain." },
      ],
      ex: {
        problem: "Check karo kaunsa subspace hai: (a) H = {(x, y) : x + 2y = 0}, (b) K = {(x, y) : x + 2y = 3}, (c) S = {2×2 matrices with det = 0}.",
        steps: [
          { title: "(a) zero check", detail: "(0, 0): 0 + 2(0) = 0 ✓ andar hai" },
          { title: "(a) closure check", detail: "u = (x₁,y₁), v = (x₂,y₂) dono andar\n(x₁+x₂) + 2(y₁+y₂) = (x₁+2y₁) + (x₂+2y₂) = 0 + 0 = 0 ✓\ncu: cx + 2cy = c(x + 2y) = 0 ✓\n⟹ H **subspace hai** (origin se guzarti line, dim 1)" },
          { title: "(b) zero check", detail: "(0,0): 0 + 2(0) = 0 ≠ 3 ✗\n⟹ K subspace **nahi** hai — pehla hi test fail" },
          { title: "(c) counterexample dhoondo", detail: "A = [[1,0],[0,0]], det = 0 ✓ andar\nB = [[0,0],[0,1]], det = 0 ✓ andar\nA + B = [[1,0],[0,1]] = I, det = 1 ≠ 0 ✗ bahar" },
          { title: "(c) conclusion", detail: "Addition ke under closed nahi ⟹ S subspace **nahi** hai.\nDeterminant ek non-linear function hai, isliye aisa hota hai." },
        ],
        answer: "(a) Subspace ✓ (b) Nahi — 0 andar nahi (c) Nahi — det=0 wale matrices addition ke under closed nahi.",
      },
      quiz: { q: "Kaunsa R³ ka subspace NAHI hai?", opts: ["{(x,y,z) : x + y + z = 0}", "{(x,y,z) : x = y = z}", "{(x,y,z) : x + y + z = 1}", "{(0,0,0)}"], a: 2, why: "= 1 wala plane origin se nahi guzarta, isliye zero vector andar nahi hai." },
    },
    {
      id: "colnull", title: "Column Space C(A) & Null Space N(A)", vis: "rref", body: [
        { t: "f", label: "Column space", x: "C(A) = Span{a₁, a₂, ..., aₙ} ⊆ Rᵐ", note: "Kyunki har column mein m entries hain. **Ax = b solvable ⟺ b ∈ C(A)**." },
        { t: "f", label: "Null space", x: "N(A) = {x ∈ Rⁿ : Ax = 0} ⊆ Rⁿ", note: "Kyunki x ko n entries chahiye. Yeh un saare inputs ka set hai jinhe A zero par crush kar deta hai." },
        { t: "tip", label: "Basis nikalne ka shortcut", x: "Pehle A ko REF/RREF karo, phir:\n**C(A) ka basis** = R ke pivot columns ke corresponding **ORIGINAL A ke columns** (R ke columns kabhi nahi!).\n**N(A) ka basis** = Rx = 0 solve karke special solutions — free variables ke saath jude vectors." },
        { t: "trap", x: "C(A) ka basis nikaalte waqt RREF ke columns use karna sabse common galti hai. Row operations **column space badal dete hain** (par row space aur null space ko nahi chhedte)." },
      ],
      ex: {
        problem: "A = [[1, 2, 3, 1], [1, 1, 2, 1], [1, 2, 3, 1]] ke liye C(A) aur N(A) dono ke basis nikalo.",
        steps: [
          { title: "Elimination", detail: "R2 − R1 = [0, −1, −1, 0]\nR3 − R1 = [0, 0, 0, 0]\nREF: [1 2 3 1; 0 −1 −1 0; 0 0 0 0]" },
          { title: "RREF banao", detail: "R2 × (−1): [0, 1, 1, 0]\nR1 − 2R2: [1, 0, 1, 1]\nR = [1 0 1 1; 0 1 1 0; 0 0 0 0]" },
          { title: "Pivot / free identify karo", detail: "Pivot columns: 1, 2 ⟹ rank r = 2\nFree columns: 3, 4 ⟹ nullity = 4 − 2 = 2" },
          { title: "C(A) ka basis — ORIGINAL columns", detail: "col1 = [1, 1, 1]ᵀ,  col2 = [2, 1, 2]ᵀ\ndim C(A) = 2 → R³ mein ek plane" },
          { title: "N(A) — x₃ = 1, x₄ = 0", detail: "x₁ + x₃ + x₄ = 0 ⟹ x₁ = −1\nx₂ + x₃ = 0 ⟹ x₂ = −1\ns₁ = [−1, −1, 1, 0]ᵀ" },
          { title: "N(A) — x₃ = 0, x₄ = 1", detail: "x₁ = −1, x₂ = 0\ns₂ = [−1, 0, 0, 1]ᵀ" },
        ],
        answer: "C(A) basis = {[1,1,1]ᵀ, [2,1,2]ᵀ}. N(A) basis = {[−1,−1,1,0]ᵀ, [−1,0,0,1]ᵀ}. r = 2, nullity = 2.",
      },
      quiz: { q: "A ek 4×6 matrix hai jiska rank 3 hai. C(A) kahan rehta hai aur uski dimension?", opts: ["R⁶ mein, dim 3", "R⁴ mein, dim 3", "R⁴ mein, dim 4", "R⁶ mein, dim 6"], a: 1, why: "C(A) ⊆ Rᵐ = R⁴ (har column mein 4 entries), aur dim = rank = 3." },
    },
    {
      id: "subops", title: "Operations on Subspaces — ∩, ∪, +, ⊕", body: [
        { t: "table", head: ["Operation", "Subspace?", "Kyun"], rows: [
          ["**H ∩ K** (intersection)", "**HAMESHA ✓**", "u, v dono H aur K mein hain ⟹ u+v dono mein hai (dono closed hain)"],
          ["**H ∪ K** (union)", "**KABHI NAHI ✗** (jab tak ek doosre ke andar na ho)", "h ∈ H aur k ∈ K lo — h + k dono se bahar nikal jaata hai"],
          ["**H + K** (vector sum)", "**HAMESHA ✓**", "Saare possible h + k ka set — linear combination rule satisfy karta hai"],
        ]},
        { t: "f", label: "Direct sum", x: "U + W ko **U ⊕ W** kehte hain agar  U ∩ W = {0}", note: "Tab har vector v ∈ U ⊕ W ka **unique decomposition** hai: v = u + w ek hi tarah se." },
        { t: "f", label: "Dimension formula", x: "dim(H + K) = dim H + dim K − dim(H ∩ K)", note: "Direct sum mein dim(H ∩ K) = 0, isliye dimensions seedhe jud jaati hain." },
        { t: "f", label: "Column spaces", x: "C(A) + C(B) = C([A  B])", note: "Redundant columns add karne se span nahi badalta, isliye A aur [A A] ka column space bilkul same hai." },
      ],
      ex: {
        problem: "R³ mein U = plane {z = 0} aur W = line span{[1,1,1]ᵀ}. U ∩ W, U + W nikalo aur batao ki direct sum hai ya nahi.",
        steps: [
          { title: "U ∩ W nikalo", detail: "W ka general element: t[1,1,1]ᵀ = (t, t, t)\nU mein hone ke liye z = 0 ⟹ t = 0\n⟹ U ∩ W = {(0,0,0)} = {0}" },
          { title: "Direct sum check", detail: "Intersection sirf zero vector hai ⟹ **U ⊕ W** direct sum hai ✓" },
          { title: "Dimension formula lagao", detail: "dim U = 2, dim W = 1, dim(U ∩ W) = 0\ndim(U + W) = 2 + 1 − 0 = 3" },
          { title: "U + W identify karo", detail: "R³ ka 3-dimensional subspace = poora R³\n⟹ U ⊕ W = R³" },
          { title: "Unique decomposition dikhao", detail: "v = (2, 5, 3) lo. w = 3[1,1,1]ᵀ = (3,3,3) (z match karne ke liye)\nu = v − w = (−1, 2, 0) ✓ z = 0, isliye U mein\nSirf ek hi tarika — yahi direct sum ki khoobi hai." },
        ],
        answer: "U ∩ W = {0}, U ⊕ W = R³ (dim 3). Har vector ka unique split hota hai.",
      },
      quiz: { q: "R³ mein do alag planes (dono origin se guzarte) ka intersection kya hoga?", opts: ["Sirf origin", "Ek line", "Ek plane", "Poora R³"], a: 1, why: "dim(H∩K) = 2 + 2 − dim(H+K) = 4 − 3 = 1. Do alag planes hamesha ek line par milte hain." },
    },
    {
      id: "ranknullity", title: "Rank–Nullity Theorem", body: [
        { t: "f", label: "The theorem", x: "rank(A) + nullity(A) = n   (n = number of COLUMNS)", note: "Columns do type ke hote hain: pivot columns (rank) aur free variable columns (nullity). Total = n." },
        { t: "list", x: [
          "**rank(A) = dim C(A) = dim C(Aᵀ)** — row rank aur column rank barabar.",
          "**nullity(A) = dim N(A) = n − r** = free variables ki ginti.",
          "Left nullspace: **dim N(Aᵀ) = m − r**.",
          "Rank ki do aur definitions: REF mein non-zero rows ki ginti; ya (total columns − free variable columns).",
        ]},
        { t: "trap", x: "Formula mein **n = columns** hai, rows nahi. 4×7 matrix ke liye rank + nullity = 7, na ki 4. Aur nullity(Aᵀ) = m − r alag cheez hai." },
      ],
      ex: {
        problem: "A ek 5×8 matrix hai jiska rank 3 hai. dim N(A), dim N(Aᵀ), dim C(A), dim C(Aᵀ) nikalo. Kya Ax = b har b ke liye solvable hai?",
        steps: [
          { title: "Values note karo", detail: "m = 5 (rows), n = 8 (columns), r = 3" },
          { title: "N(A) — Rⁿ mein rehta hai", detail: "dim N(A) = n − r = 8 − 3 = **5**" },
          { title: "N(Aᵀ) — Rᵐ mein rehta hai", detail: "dim N(Aᵀ) = m − r = 5 − 3 = **2**" },
          { title: "Column aur row space", detail: "dim C(A) = r = 3, R⁵ ka subspace\ndim C(Aᵀ) = r = 3, R⁸ ka subspace" },
          { title: "Solvability check", detail: "Har b ∈ R⁵ ke liye solvable tabhi jab C(A) = R⁵ ⟹ r = m = 5\nYahan r = 3 < 5 ⟹ **NAHI**, sirf un b ke liye jo 3-dimensional C(A) mein hain." },
          { title: "Verification", detail: "3 + 5 = 8 = n ✓\n3 + 2 = 5 = m ✓" },
        ],
        answer: "dim N(A) = 5, dim N(Aᵀ) = 2, dim C(A) = dim C(Aᵀ) = 3. Har b ke liye solvable nahi.",
      },
      quiz: { q: "Ek 6×4 matrix ka nullity 0 hai. Rank kya hai?", opts: ["6", "4", "2", "0"], a: 1, why: "rank + nullity = n = 4 ⟹ rank = 4. Yeh full COLUMN rank hai (tall matrix)." },
    },
    {
      id: "basis", title: "Basis & Dimension", body: [
        { t: "f", label: "Definition", x: "Ek set basis hai agar:\n1. vectors **linearly independent** hain, AUR\n2. wo poora space **span** karte hain", note: "Basis = minimal set jo poore space ko cover karta hai, bina kisi redundancy ke." },
        { t: "list", x: [
          "**Unique representation:** basis ke saath har vector v ka **exactly ek** combination hota hai — double counting possible hi nahi.",
          "**Bases unique nahi hote** — Rⁿ ke infinitely many bases hain. Kisi bhi invertible n×n matrix ke columns ek valid basis hain.",
          "Basis mein vectors ki ginti hamesha same rehti hai — usi ko **dimension** kehte hain.",
          "Basis ek **choice** hai, fixed coordinate system nahi. Same vector, alag basis, alag coordinates.",
        ]},
        { t: "tip", x: "Agar aapko pehle se pata hai ki dim V = k, to sirf **k vectors** ke liye ek hi cheez check karni hai — independence YA spanning. Ek aa gayi to doosri automatically aa jaati hai." },
      ],
      ex: {
        problem: "Kya {[1,1,0]ᵀ, [1,0,1]ᵀ, [0,1,1]ᵀ} R³ ka basis hai? Agar haan, to v = [2, 3, 5]ᵀ ke coordinates is basis mein nikalo.",
        steps: [
          { title: "Matrix banao aur det nikalo", detail: "A = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]\ndet = 1(0·1 − 1·1) − 1(1·1 − 1·0) + 0\n    = 1(−1) − 1(1) + 0 = −2 ≠ 0" },
          { title: "Conclusion", detail: "det ≠ 0 ⟹ columns independent aur 3 vectors R³ mein ⟹ **basis hai** ✓\n(3 = dim R³, isliye independence hi kaafi thi)" },
          { title: "Coordinates ke liye Ac = v solve karo", detail: "c₁ + c₂     = 2\nc₁      + c₃ = 3\n     c₂ + c₃ = 5" },
          { title: "Elimination", detail: "R2 − R1: −c₂ + c₃ = 1\nR3: c₂ + c₃ = 5\nJodo: 2c₃ = 6 ⟹ c₃ = 3" },
          { title: "Back-substitute", detail: "c₂ + 3 = 5 ⟹ c₂ = 2\nc₁ + 2 = 2 ⟹ c₁ = 0" },
          { title: "Verify", detail: "0[1,1,0] + 2[1,0,1] + 3[0,1,1] = [2, 3, 5] ✓" },
        ],
        answer: "Haan, basis hai (det = −2 ≠ 0). Coordinates: v = 0·v₁ + 2·v₂ + 3·v₃, yaani [0, 2, 3]ᵀ.",
      },
      quiz: { q: "R⁴ mein 5 vectors diye hain jo poora R⁴ span karte hain. Kya wo basis hain?", opts: ["Haan", "Nahi — 5 > 4 isliye dependent hain", "Sirf agar orthogonal hon", "Pata nahi chal sakta"], a: 1, why: "Span karte hain par independent nahi (5 > 4). Basis ke liye dono chahiye — inme se ek vector hataana padega." },
    },
    {
      id: "four", title: "Four Fundamental Subspaces — The Big Picture", vis: "rref", body: [
        { t: "table", head: ["Subspace", "Notation", "Dimension", "Kahan rehta hai", "Basis"], rows: [
          ["Row space", "C(Aᵀ)", "r", "Rⁿ", "R ke non-zero rows"],
          ["Column space", "C(A)", "r", "Rᵐ", "A ke ORIGINAL pivot columns"],
          ["Null space", "N(A)", "n − r", "Rⁿ", "Special solutions (Rx = 0)"],
          ["Left null space", "N(Aᵀ)", "m − r", "Rᵐ", "Aᵀy = 0 solve karke"],
        ]},
        { t: "f", label: "Fundamental Theorem — Part 1 & 2", x: "dim C(A) = dim C(Aᵀ) = r\nr + (n − r) = n     r + (m − r) = m\nN(A) ⊥ C(Aᵀ)  in Rⁿ\nN(Aᵀ) ⊥ C(A)  in Rᵐ", note: "Rⁿ do orthogonal complement mein bat jaata hai: row space aur null space. Rᵐ bhi: column space aur left null space." },
        { t: "f", label: "Effect of row operations", x: "UNCHANGED:  Row space aur Null space\nCHANGED:    Column space aur Left null space", note: "Row reduction rows ka linear combination hai — isliye row space wahi rehta hai. Par columns puri tarah badal jaate hain." },
        { t: "p", x: "**Mapping picture:** A row space ke har vector ko column space mein bhejta hai, aur yeh mapping **one-to-one** hai (r → r). Null space ka sab kuch 0 par gir jaata hai. Har x ko x = x_row + x_null likha ja sakta hai, aur Ax = A·x_row." },
        { t: "tip", x: "Left null space nikaalne ke liye Aᵀ par dobara mehnat mat karo. **[A | I]** par elimination karo — jab A ke rows zero ban jaayein, unke saamne I-side ke rows hi N(Aᵀ) ka basis hain (yaani yᵀA = 0)." },
      ],
      ex: {
        problem: "A = [[1, 2, 3], [2, 4, 6], [1, 1, 1]] ke chaaron subspaces ka basis aur dimension nikalo.",
        steps: [
          { title: "[A | I] par elimination", detail: "[1 2 3 | 1 0 0]\n[2 4 6 | 0 1 0]\n[1 1 1 | 0 0 1]\nR2 − 2R1: [0 0 0 | −2 1 0]\nR3 − R1:  [0 −1 −2 | −1 0 1]" },
          { title: "Rows arrange karo", detail: "[1  2  3 |  1 0 0]\n[0 −1 −2 | −1 0 1]\n[0  0  0 | −2 1 0]  ← A-side zero row\nrank r = 2, m = 3, n = 3" },
          { title: "Row space C(Aᵀ) — dim r = 2", detail: "REF ke non-zero rows: {[1, 2, 3]ᵀ, [0, −1, −2]ᵀ}\nR³ mein ek plane" },
          { title: "Column space C(A) — dim r = 2", detail: "Pivot columns = 1, 2 ⟹ **ORIGINAL A** ke columns:\n{[1, 2, 1]ᵀ, [2, 4, 1]ᵀ}" },
          { title: "Null space N(A) — dim n − r = 1", detail: "RREF: [1 0 −1; 0 1 2; 0 0 0]\nx₃ free = 1 ⟹ x₁ = 1, x₂ = −2\nBasis: {[1, −2, 1]ᵀ}\nCheck: A[1,−2,1]ᵀ = [1−4+3, 2−8+6, 1−2+1]ᵀ = [0,0,0]ᵀ ✓" },
          { title: "Left null space N(Aᵀ) — dim m − r = 1", detail: "Zero row ke saamne I-side: [−2, 1, 0]\nBasis: {[−2, 1, 0]ᵀ}\nCheck: −2·row1 + 1·row2 + 0·row3 = −2[1,2,3] + [2,4,6] = [0,0,0] ✓" },
          { title: "Orthogonality verify karo", detail: "Row space ka [1,2,3] · null space ka [1,−2,1] = 1 − 4 + 3 = 0 ✓\nColumn space ka [1,2,1] · left null [−2,1,0] = −2 + 2 + 0 = 0 ✓" },
        ],
        answer: "r = 2. Row space & column space dim 2, N(A) = span{[1,−2,1]ᵀ}, N(Aᵀ) = span{[−2,1,0]ᵀ}. Dono orthogonality relations verified.",
      },
      quiz: { q: "Row operations ke baad kaunse do subspaces badal jaate hain?", opts: ["Row space aur null space", "Column space aur left null space", "Saare chaar", "Koi nahi"], a: 1, why: "Rows ka combination lene se row space aur null space same rehte hain, lekin columns ki geometry badal jaati hai." },
    },
    {
      id: "solvability", title: "Solvability of Ax = b — decision flow", body: [
        { t: "num", x: [
          "**b ∈ C(A)?** Nahi ⟹ **NO SOLUTION** (R mein zero row par b side non-zero).",
          "Haan ⟹ solution exist karta hai. Ab **N(A) = {0}?**",
          "Haan ⟹ **UNIQUE solution** x = x_p.",
          "Nahi ⟹ **INFINITE solutions** x = x_p + x_n.",
        ]},
        { t: "f", label: "Rank ki bhasha mein", x: "rank(A) < rank[A|b]      → no solution\nrank(A) = rank[A|b] = n  → unique\nrank(A) = rank[A|b] < n  → infinite" },
        { t: "list", x: [
          "**Linearity:** x₁ aur x₂ dono Ax = b solve karte hain ⟹ x₁ − x₂ ∈ N(A).",
          "Ax = b **har** b ∈ Rᵐ ke liye solvable ⟺ A ka har row mein pivot ⟺ rank = m ⟺ N(Aᵀ) = {0}.",
        ]},
      ],
      ex: {
        problem: "Kis b = [b₁, b₂, b₃]ᵀ ke liye Ax = b solvable hai, jahan A = [[1, 2], [2, 4], [3, 6]]?",
        steps: [
          { title: "A ki structure dekho", detail: "col2 = 2 × col1 ⟹ rank = 1\nC(A) = span{[1, 2, 3]ᵀ} — R³ mein ek LINE" },
          { title: "Augmented matrix reduce karo", detail: "[1 2 | b₁]\n[2 4 | b₂]\n[3 6 | b₃]\nR2 − 2R1: [0, 0 | b₂ − 2b₁]\nR3 − 3R1: [0, 0 | b₃ − 3b₁]" },
          { title: "Consistency conditions", detail: "Do zero rows ban gayi. Solvable tabhi jab dono b-side bhi 0 hon:\n**b₂ − 2b₁ = 0** aur **b₃ − 3b₁ = 0**" },
          { title: "Conditions ko C(A) se milao", detail: "b₂ = 2b₁, b₃ = 3b₁ ⟹ b = b₁·[1, 2, 3]ᵀ\nExactly wahi line jo C(A) hai ✓" },
          { title: "Left null space se cross-check", detail: "N(Aᵀ) ke vectors: y ⊥ [1,2,3]ᵀ\ny₁ = [−2, 1, 0]ᵀ, y₂ = [−3, 0, 1]ᵀ\nCondition yᵀb = 0: −2b₁ + b₂ = 0 aur −3b₁ + b₃ = 0 ✓ same" },
          { title: "Ek example", detail: "b = [1, 2, 3]ᵀ ⟹ x₁ + 2x₂ = 1, infinite solutions:\nx = [1, 0]ᵀ + t[−2, 1]ᵀ" },
        ],
        answer: "Solvable ⟺ b₂ = 2b₁ aur b₃ = 3b₁, yaani b ∈ C(A) = span{[1,2,3]ᵀ}. Tab hamesha infinite solutions (nullity = 1).",
      },
      quiz: { q: "Ax = b solvable hai aur nullity(A) = 2. Kitne solutions?", opts: ["Exactly 1", "Exactly 2", "Infinite — ek 2D plane (shifted)", "Koi nahi"], a: 2, why: "Solution set = x_p + N(A), aur N(A) 2-dimensional hai. Isliye solution set ek shifted plane hai." },
    },
  ],
};

const CH4 = {
  id: "ch4", n: "04", title: "Determinants",
  blurb: "Geometric meaning, pivot method, saari properties, cofactor & adjugate, Cramer's rule aur exam traps.",
  topics: [
    {
      id: "detbasics", title: "Determinant — meaning & pivot method", vis: "transform", body: [
        { t: "p", x: "Determinant ek scalar hai jo sirf **square** matrices ke liye defined hai. Geometrically wo row (ya column) vectors se bane parallelepiped ka **signed volume** hai — 2×2 mein area, 3×3 mein volume. Sign orientation batata hai." },
        { t: "f", label: "Pivot method — kisi bhi n×n ke liye", x: "det(A) = (−1)^r × (u₁₁ × u₂₂ × ... × uₙₙ)", note: "r = elimination mein kitne **row swaps** hue. uᵢᵢ = U (upper triangular) ke diagonal entries = pivots." },
        { t: "list", x: [
          "Row addition/subtraction (Rᵢ − m·Rⱼ) se det **bilkul nahi badalta**.",
          "Sirf row **swap** sign flip karta hai.",
          "Invertible A (saare pivots non-zero): |A| = (−1)^(swaps) × pivots ka product.",
          "Non-invertible (koi zero pivot): |A| = 0.",
        ]},
        { t: "tip", x: "4×4 ya bade determinant ke liye cofactor **mat** lagao — 24 terms ban jaate hain. Elimination karke pivots ka product lo, sirf 6 steps. Har row swap par ek minus." },
      ],
      ex: {
        problem: "det nikalo: A = [[0, 1, 2], [1, 2, 3], [2, 5, 8]] — elimination method se.",
        steps: [
          { title: "Pivot position (1,1) par 0 hai", detail: "Row swap chahiye: R1 ↔ R2\n[1 2 3]\n[0 1 2]\n[2 5 8]\nSwap count r = 1" },
          { title: "Column 1 clear karo", detail: "R3 − 2R1 = [0, 1, 2]\n[1 2 3]\n[0 1 2]\n[0 1 2]" },
          { title: "Column 2 clear karo", detail: "R3 − R2 = [0, 0, 0]\nU = [[1, 2, 3], [0, 1, 2], [0, 0, 0]]" },
          { title: "Pivots ka product", detail: "Pivots = 1, 1, 0\ndet = (−1)¹ × (1 × 1 × 0) = 0" },
          { title: "Verify — dependency dhoondo", detail: "Original mein: row3 = [2,5,8]\nrow1 + 2·row2 = [0,1,2] + [2,4,6] = [2, 5, 8] = row3 ✓\nRows dependent ⟹ det = 0, singular matrix." },
        ],
        answer: "det A = 0. Matrix singular hai kyunki row3 = row1 + 2·row2. Rank = 2, nullity = 1.",
      },
      quiz: { q: "Elimination mein 3 row swaps hue aur pivots 2, −1, 4 mile. det = ?", opts: ["8", "−8", "5", "−5"], a: 0, why: "Pivots ka product = (2)(−1)(4) = −8. 3 swaps ⟹ factor (−1)³ = −1. det = (−1)(−8) = **8**." },
    },
    {
      id: "detprops", title: "Core Properties & Exam Traps", body: [
        { t: "table", head: ["#", "Property", "Formula"], rows: [
          ["1", "Transpose", "|Aᵀ| = |A|"],
          ["2", "Product", "|AB| = |A|·|B|"],
          ["3", "Commutativity of det", "|AB| = |BA| (scalars hain, isliye)"],
          ["4", "Power", "det(Aⁿ) = (det A)ⁿ"],
          ["5", "Inverse", "det(A⁻¹) = 1/det(A)"],
          ["6", "Similarity invariance", "det(PAP⁻¹) = det(A)"],
          ["7", "Orthogonal matrix", "QᵀQ = I ⟹ det(Q) = ±1"],
        ]},
        { t: "f", label: "Scalar multiplication rule", x: "|k · A(n×n)| = kⁿ · |A|", note: "Ek row ko k se multiply karne se det k guna hota hai. n×n matrix mein n rows hain, isliye kⁿ." },
        { t: "trap", label: "Sabse common mistake", x: "**|kA| = k|A| likhna GALAT hai.** Yeh sirf 1×1 matrices ke liye sahi hai. 3×3 ke liye |2A| = 2³|A| = 8|A|." },
        { t: "trap", x: "**det(A + B) ≠ det(A) + det(B)** — determinant additive **nahi** hai. Sirf multiplicative hai." },
        { t: "trap", x: "**GALAT:** det(A) = diagonal entries ka product. **SAHI:** det = REF ke **pivots** ka product. General matrix mein diagonal entries aur pivots alag cheezein hain (triangular matrix ek exception hai)." },
        { t: "list", x: [
          "**det = 0 forcing conditions:** do same rows/columns · ek zero row/column · rows (ya columns) linearly dependent.",
          "**LU se:** det(L) = 1 (unit lower triangular), isliye det(A) = det(U) = U ke diagonal ka product.",
          "det(U⁻¹L⁻¹A) = det(I) = 1 hamesha, kyunki A⁻¹ = U⁻¹L⁻¹.",
          "**True:** det(Aⁿ) = 0 ⟹ (det A)ⁿ = 0 ⟹ det A = 0 ⟹ A not invertible.",
        ]},
      ],
      ex: {
        problem: "A ek 4×4 matrix hai jiska det(A) = 3. Nikalo: (a) det(2A), (b) det(A³), (c) det(A⁻¹), (d) det(2A⁻¹Aᵀ).",
        steps: [
          { title: "(a) Scalar rule — n = 4", detail: "det(2A) = 2⁴ · det(A) = 16 × 3 = **48**\n(Galat hota: 2 × 3 = 6)" },
          { title: "(b) Power rule", detail: "det(A³) = (det A)³ = 3³ = **27**" },
          { title: "(c) Inverse rule", detail: "det(A⁻¹) = 1/det(A) = **1/3**" },
          { title: "(d) Todo aur multiply karo", detail: "det(2A⁻¹Aᵀ) = det(2 · (A⁻¹Aᵀ))\n= 2⁴ · det(A⁻¹) · det(Aᵀ)" },
          { title: "(d) Values daalo", detail: "= 16 × (1/3) × 3\n= 16 × 1 = **16**\n(kyunki det(Aᵀ) = det(A) = 3)" },
        ],
        answer: "(a) 48 (b) 27 (c) 1/3 (d) 16",
      },
      quiz: { q: "A 3×3 hai aur det(A) = 5. det(−A) = ?", opts: ["−5", "5", "−125", "−15"], a: 0, why: "det(−A) = (−1)³ · det(A) = −5. Odd order matrix mein sign flip hota hai; even order mein nahi." },
    },
    {
      id: "cofactor", title: "Cofactor, Adjugate & Matrix Identities", body: [
        { t: "f", label: "Cofactor expansion", x: "det A = Σⱼ a₁ⱼ C₁ⱼ,     Cᵢⱼ = (−1)^(i+j) Mᵢⱼ", note: "Mᵢⱼ = minor = row i aur column j hataake bachi matrix ka determinant." },
        { t: "f", label: "Sign pattern", x: "[ +  −  + ]\n[ −  +  − ]\n[ +  −  + ]" },
        { t: "f", label: "Inverse via adjugate", x: "A⁻¹ = Cᵀ / |A|", note: "Cᵀ = adjugate (classical adjoint). Har column Cramer's rule se aata hai." },
        { t: "table", head: ["Identity", "Statement"], rows: [
          ["Fundamental", "**A · Cᵀ = |A| · I**"],
          ["Cofactor det", "**|C| = |A|^(n−1)** for n×n"],
          ["Singular case", "|A| = 0 ⟹ A·Cᵀ = 0 ⟹ Cᵀ ke columns N(A) mein hain"],
        ]},
        { t: "p", x: "**|C| = |A|^(n−1) ka derivation:** |A|·|C| = |A·Cᵀ| = ||A|·I| = |A|ⁿ ⟹ |C| = |A|^(n−1)." },
        { t: "tip", x: "Us row ya column se expand karo jisme sabse zyada **zeros** hain — har zero ek poora minor bacha deta hai." },
      ],
      ex: {
        problem: "A = [[1, 0, 2], [3, 1, 0], [0, 4, 1]] ke liye cofactor matrix C, det A, aur A⁻¹ nikalo. Phir |C| verify karo.",
        steps: [
          { title: "Row 1 se det nikalo (ek zero hai)", detail: "det = 1·C₁₁ + 0·C₁₂ + 2·C₁₃\nC₁₁ = +det[[1,0],[4,1]] = 1(1) − 0(4) = 1\nC₁₃ = +det[[3,1],[0,4]] = 3(4) − 1(0) = 12\ndet = 1(1) + 2(12) = 25" },
          { title: "Baaki cofactors — row 1", detail: "C₁₂ = −det[[3,0],[0,1]] = −(3 − 0) = −3" },
          { title: "Row 2 ke cofactors", detail: "C₂₁ = −det[[0,2],[4,1]] = −(0 − 8) = 8\nC₂₂ = +det[[1,2],[0,1]] = 1\nC₂₃ = −det[[1,0],[0,4]] = −4" },
          { title: "Row 3 ke cofactors", detail: "C₃₁ = +det[[0,2],[1,0]] = 0 − 2 = −2\nC₃₂ = −det[[1,2],[3,0]] = −(0 − 6) = 6\nC₃₃ = +det[[1,0],[3,1]] = 1" },
          { title: "C aur adjugate likho", detail: "C = [[1, −3, 12], [8, 1, −4], [−2, 6, 1]]\nCᵀ = [[1, 8, −2], [−3, 1, 6], [12, −4, 1]]" },
          { title: "A⁻¹ = Cᵀ/|A|", detail: "A⁻¹ = (1/25)·[[1, 8, −2], [−3, 1, 6], [12, −4, 1]]" },
          { title: "|C| verify karo", detail: "Formula: |C| = |A|^(n−1) = 25² = 625\nDirect check bhi yahi dega ✓" },
        ],
        answer: "det A = 25, A⁻¹ = (1/25)[[1,8,−2],[−3,1,6],[12,−4,1]], |C| = 625.",
      },
      quiz: { q: "A 4×4 singular matrix hai. A·Cᵀ = ?", opts: ["I", "Zero matrix", "4I", "Cᵀ"], a: 1, why: "A·Cᵀ = |A|·I = 0·I = zero matrix. Cᵀ ke saare columns N(A) mein baith jaate hain." },
    },
    {
      id: "cramer", title: "Cramer's Rule", body: [
        { t: "f", label: "Statement", x: "Ax = b, A invertible (det A ≠ 0):\n\nxᵢ = det(Bᵢ) / det(A)", note: "Bᵢ = matrix A jisme **column i ki jagah b** rakh diya gaya ho." },
        { t: "f", label: "3×3 ke liye", x: "B₁ = [b  a₂  a₃],  x₁ = |B₁|/|A|\nB₂ = [a₁  b  a₃],  x₂ = |B₂|/|A|\nB₃ = [a₁  a₂  b],  x₃ = |B₃|/|A|" },
        { t: "p", x: "**Derivation:** A · [e₁ ... x ... eₙ] = [a₁ ... b ... aₙ] = Bᵢ. Dono side det lo: det(A)·xᵢ = det(Bᵢ)." },
        { t: "trap", x: "Cramer's rule ke liye **det(A) ≠ 0 zaroori hai**. Singular A par rule undefined hai — 0/0 form aata hai, us se koi conclusion nahi nikaal sakte." },
        { t: "tip", x: "Cramer chhote systems (2×2, 3×3) aur theory ke liye sundar hai. Computation ke liye **elimination bahut tez** hai — Cramer ko n+1 determinants chahiye, matlab O(n⁴) ya bura." },
      ],
      ex: {
        problem: "Cramer's rule se solve karo: 2x + y − z = 3, x − y + z = 1, 3x + 2y + z = 10.",
        steps: [
          { title: "A aur b likho", detail: "A = [[2, 1, −1], [1, −1, 1], [3, 2, 1]],  b = [3, 1, 10]ᵀ" },
          { title: "det A nikalo (row 1 se)", detail: "= 2·det[[−1,1],[2,1]] − 1·det[[1,1],[3,1]] + (−1)·det[[1,−1],[3,2]]\n= 2(−1 − 2) − 1(1 − 3) − 1(2 + 3)\n= 2(−3) − 1(−2) − 1(5) = −6 + 2 − 5 = −9" },
          { title: "det B₁ (col 1 → b)", detail: "B₁ = [[3, 1, −1], [1, −1, 1], [10, 2, 1]]\n= 3(−1 − 2) − 1(1 − 10) + (−1)(2 + 10)\n= −9 + 9 − 12 = −12\nx = −12/−9 = 4/3" },
          { title: "det B₂ (col 2 → b)", detail: "B₂ = [[2, 3, −1], [1, 1, 1], [3, 10, 1]]\n= 2(1 − 10) − 3(1 − 3) + (−1)(10 − 3)\n= −18 + 6 − 7 = −19\ny = −19/−9 = 19/9" },
          { title: "det B₃ (col 3 → b)", detail: "B₃ = [[2, 1, 3], [1, −1, 1], [3, 2, 10]]\n= 2(−10 − 2) − 1(10 − 3) + 3(2 + 3)\n= −24 − 7 + 15 = −16\nz = −16/−9 = 16/9" },
          { title: "Verify (equation 2)", detail: "x − y + z = 12/9 − 19/9 + 16/9 = 9/9 = 1 ✓" },
        ],
        answer: "x = 4/3, y = 19/9, z = 16/9 (det A = −9).",
      },
      quiz: { q: "Cramer's rule lagane ke liye zaroori shart?", opts: ["A symmetric ho", "det(A) ≠ 0", "A ke saare entries positive hon", "b ≠ 0"], a: 1, why: "Formula mein det(A) denominator mein hai. Singular matrix par rule undefined ho jaata hai." },
    },
  ],
};

const CH5 = {
  id: "ch5", n: "05", title: "Orthogonality & Projection Matrices",
  blurb: "Line aur subspace par projection, normal equations, AᵀA ki properties, orthogonal matrices, Gram-Schmidt aur least squares.",
  topics: [
    {
      id: "projline", title: "Projection onto a Line", vis: "project", body: [
        { t: "p", x: "b ko line (span of a) par giraana hai. Sabse nazdeek point p wahi hai jahan error **e = b − p** line ke **perpendicular** ho." },
        { t: "f", label: "Derivation", x: "aᵀ(b − a x̂) = 0\naᵀb − aᵀa x̂ = 0\n⟹  x̂ = aᵀb / aᵀa\n⟹  p = a x̂ = a (aᵀb / aᵀa)\n⟹  P = a aᵀ / aᵀa", note: "**P rank-1 matrix hai.** Dhyaan do: aᵀa ek NUMBER hai, aaᵀ ek poora MATRIX." },
        { t: "trap", x: "aᵀa (scalar) aur aaᵀ (matrix) ko ulat dena sabse common galti hai. a ∈ Rⁿ ke liye aᵀa 1×1 hai, aaᵀ n×n hai." },
        { t: "list", x: [
          "Agar b pehle se line par hai → p = b, e = 0.",
          "Agar b line ke perpendicular hai → p = 0, e = b.",
          "P² = P, Pᵀ = P, aur rank(P) = 1, det(P) = 0 (n ≥ 2 ke liye).",
          "P ke eigenvalues sirf **1 (a ki direction mein) aur 0** hain.",
        ]},
      ],
      ex: {
        problem: "b = [1, 2, 3]ᵀ ko a = [1, 1, 1]ᵀ ki line par project karo. x̂, p, e, P nikalo aur aᵀe = 0 verify karo.",
        steps: [
          { title: "aᵀa aur aᵀb nikalo", detail: "aᵀa = 1 + 1 + 1 = 3\naᵀb = 1(1) + 1(2) + 1(3) = 6" },
          { title: "x̂ nikalo", detail: "x̂ = aᵀb / aᵀa = 6/3 = 2" },
          { title: "p = x̂a", detail: "p = 2 · [1, 1, 1]ᵀ = [2, 2, 2]ᵀ\n(Yeh b ka average hai — makes sense, a saare 1s hai)" },
          { title: "e = b − p", detail: "e = [1, 2, 3]ᵀ − [2, 2, 2]ᵀ = [−1, 0, 1]ᵀ" },
          { title: "Orthogonality verify", detail: "aᵀe = 1(−1) + 1(0) + 1(1) = 0 ✓" },
          { title: "Projection matrix P = aaᵀ/aᵀa", detail: "aaᵀ = [[1,1,1],[1,1,1],[1,1,1]]\nP = (1/3)·[[1,1,1],[1,1,1],[1,1,1]]\nCheck: Pb = (1/3)[6, 6, 6]ᵀ = [2,2,2]ᵀ = p ✓" },
          { title: "P² = P verify", detail: "P² = (1/9)·[[3,3,3],[3,3,3],[3,3,3]] = (1/3)[[1,1,1],[1,1,1],[1,1,1]] = P ✓" },
        ],
        answer: "x̂ = 2, p = [2,2,2]ᵀ, e = [−1,0,1]ᵀ, P = (1/3)·ones(3). rank(P) = 1, eigenvalues {1, 0, 0}.",
      },
      quiz: { q: "Projection matrix P ke eigenvalues kya ho sakte hain?", opts: ["Sirf 1", "Sirf 0", "Sirf 0 aur 1", "Koi bhi real number"], a: 2, why: "P² = P ⟹ λ² = λ ⟹ λ(λ−1) = 0 ⟹ λ = 0 ya 1." },
    },
    {
      id: "projsub", title: "Projection onto a Subspace — Normal Equations", body: [
        { t: "num", x: [
          "Projection **p = Ax̂** hamesha C(A) mein hoti hai.",
          "Error **e = b − Ax̂** ko C(A) ke perpendicular hona chahiye.",
          "Orthogonality ka matlab: e, Aᵀ ke null space mein hai ⟹ **Aᵀ(b − Ax̂) = 0**.",
          "Expand: Aᵀb − AᵀAx̂ = 0.",
          "**Normal equations: AᵀA x̂ = Aᵀb**.",
          "x̂ = (AᵀA)⁻¹Aᵀb  aur  p = A(AᵀA)⁻¹Aᵀ b.",
        ]},
        { t: "f", label: "Projection matrix", x: "P = A(AᵀA)⁻¹Aᵀ", note: "Special case: agar A ke columns **orthonormal** hain (A = Q), to QᵀQ = I aur **P = QQᵀ** — inverse ki zarurat hi nahi." },
        { t: "table", head: ["Property of P", "Statement"], rows: [
          ["Symmetric", "Pᵀ = P"],
          ["Idempotent", "P² = P — dobara project karne se kuch naya nahi"],
          ["Rank", "rank(P) = rank(A) = r; trace(P) = r"],
          ["Complement", "**I − P** bhi projection hai — N(Aᵀ) par"],
        ]},
        { t: "trap", x: "P = A(AᵀA)⁻¹Aᵀ ko simplify karke AA⁻¹(Aᵀ)⁻¹Aᵀ = I **mat** likh dena. Yeh sirf tab valid hai jab A square aur invertible ho — us case mein sach mein P = I hota hai (kyunki C(A) = poora space)." },
      ],
      ex: {
        problem: "b = [6, 0, 0]ᵀ ko A = [[1, 0], [1, 1], [1, 2]] ke column space par project karo.",
        steps: [
          { title: "AᵀA nikalo", detail: "Aᵀ = [[1,1,1],[0,1,2]]\nAᵀA = [[3, 3], [3, 5]]\n(top-left = 1+1+1 = 3, off-diag = 0+1+2 = 3, bottom-right = 0+1+4 = 5)" },
          { title: "Aᵀb nikalo", detail: "Aᵀb = [1(6)+1(0)+1(0), 0(6)+1(0)+2(0)]ᵀ = [6, 0]ᵀ" },
          { title: "Normal equations solve karo", detail: "[3  3][x̂₁]   [6]\n[3  5][x̂₂] = [0]\nR2 − R1: 2x̂₂ = −6 ⟹ x̂₂ = −3\n3x̂₁ + 3(−3) = 6 ⟹ x̂₁ = 5" },
          { title: "p = Ax̂", detail: "p = 5·[1,1,1]ᵀ + (−3)·[0,1,2]ᵀ\n  = [5, 5, 5]ᵀ − [0, 3, 6]ᵀ = [5, 2, −1]ᵀ" },
          { title: "e = b − p aur check", detail: "e = [6,0,0]ᵀ − [5,2,−1]ᵀ = [1, −2, 1]ᵀ\nAᵀe = [1−2+1, 0−2+2]ᵀ = [0, 0]ᵀ ✓ perpendicular" },
          { title: "P bhi nikaal lo", detail: "(AᵀA)⁻¹ = (1/6)[[5, −3], [−3, 3]]\nP = A(AᵀA)⁻¹Aᵀ = (1/6)·[[5, 2, −1], [2, 2, 2], [−1, 2, 5]]\ntrace(P) = (5+2+5)/6 = 2 = rank ✓" },
        ],
        answer: "x̂ = [5, −3]ᵀ, p = [5, 2, −1]ᵀ, e = [1, −2, 1]ᵀ. trace(P) = 2 = rank(A).",
      },
      quiz: { q: "Agar b pehle se C(A) mein ho to projection p kya hoga?", opts: ["p = 0", "p = b", "p = Ab", "p = b/2"], a: 1, why: "Jo vector already subspace mein hai use project karne ki zarurat nahi — Pb = b, aur error e = 0." },
    },
    {
      id: "ata", title: "Properties of AᵀA & Orthogonal Complements", body: [
        { t: "f", label: "Property 1", x: "N(AᵀA) = N(A)   ⟹   rank(AᵀA) = rank(A)", note: "**Proof:** AᵀAx = 0 ⟹ xᵀAᵀAx = 0 ⟹ (Ax)ᵀ(Ax) = 0 ⟹ ‖Ax‖² = 0 ⟹ Ax = 0." },
        { t: "f", label: "Property 2 — invertibility", x: "AᵀA invertible  ⟺  A ke columns **linearly independent** (full column rank)", note: "Independent columns ⟹ N(A) = {0} ⟹ N(AᵀA) = {0} ⟹ AᵀA invertible. Agar B ka rank m hai to BBᵀ invertible hoga." },
        { t: "list", x: [
          "**P projects onto C(A)**, aur **I − P projects onto N(Aᵀ)**.",
          "**Residual e = b − Ax̂ hamesha left null space N(Aᵀ) mein hota hai** — least squares ka core fact.",
          "AᵀA hamesha **symmetric** aur **positive semi-definite** hai (xᵀAᵀAx = ‖Ax‖² ≥ 0).",
        ]},
        { t: "table", head: ["Concept", "Meaning"], rows: [
          ["**Orthogonal subspaces** (V ⊥ W)", "Sirf itna: P_V P_W = 0. P_V + P_W ≠ I kyunki wo poora space cover nahi karte (jaise R³ mein do lines)."],
          ["**Orthogonal complements** (V ⊕ V⊥ = Rⁿ)", "Poora space cover: **P_V + P_V⊥ = I** aur **P_V P_V⊥ = 0**. Har b ka perfect split hota hai."],
        ]},
      ],
      ex: {
        problem: "A = [[1, 1], [1, 2], [1, 3]] ke liye dikhao ki AᵀA invertible hai, aur rank(AᵀA) = rank(A) verify karo.",
        steps: [
          { title: "A ka rank nikalo", detail: "col2 = c·col1? [1,2,3] ≠ c[1,1,1] kisi c ke liye\n⟹ columns independent ⟹ rank(A) = 2 (full column rank)" },
          { title: "AᵀA compute karo", detail: "AᵀA = [[1+1+1, 1+2+3], [1+2+3, 1+4+9]]\n    = [[3, 6], [6, 14]]" },
          { title: "det check", detail: "det(AᵀA) = 3(14) − 6(6) = 42 − 36 = 6 ≠ 0\n⟹ invertible ✓ ⟹ rank(AᵀA) = 2 = rank(A) ✓" },
          { title: "Ab dependent columns ka case dekho", detail: "B = [[1, 2], [1, 2], [1, 2]] lo (col2 = 2·col1, rank 1)\nBᵀB = [[3, 6], [6, 12]]\ndet = 36 − 36 = 0 ⟹ NOT invertible" },
          { title: "Confirm", detail: "rank(BᵀB) = 1 = rank(B) ✓\nProperty dono cases mein sahi utri." },
        ],
        answer: "AᵀA = [[3,6],[6,14]], det = 6 ≠ 0 ⟹ invertible. rank(AᵀA) = rank(A) = 2 dono cases mein verify hua.",
      },
      quiz: { q: "A ek 5×7 matrix hai. AᵀA (7×7) invertible ho sakta hai?", opts: ["Haan agar rank 5 ho", "Nahi kabhi", "Haan agar A symmetric ho", "Sirf agar A square ho"], a: 1, why: "rank(AᵀA) = rank(A) ≤ min(5,7) = 5 < 7. 7×7 matrix ka rank 7 chahiye tha, isliye kabhi invertible nahi." },
    },
    {
      id: "orthomat", title: "Orthonormal Vectors, Q, Rotation & Reflection", body: [
        { t: "f", label: "Orthonormal", x: "qᵢᵀqⱼ = 0  (i ≠ j)   aur   qᵢᵀqᵢ = 1", note: "Aise columns wali matrix Q ke liye **QᵀQ = I**." },
        { t: "trap", x: "**QQᵀ = I sirf tab jab Q square ho.** Agar Q tall hai (m > n), to QᵀQ = I par QQᵀ ek **projection matrix** hai, identity nahi." },
        { t: "list", x: [
          "**Length preservation:** ‖Qx‖ = ‖x‖ — sirf rotation ya reflection, koi stretching nahi.",
          "**Projection with Q:** x̂ = Qᵀb aur P = QQᵀ — inverse lagane ki zarurat hi nahi.",
          "Har term qᵢ(qᵢᵀb) line qᵢ par ki independent projection hai.",
          "Q ke eigenvalues ka |λ| = 1 (complex ho sakte hain).",
        ]},
        { t: "f", label: "Rotation matrix", x: "R = [[cos θ, −sin θ], [sin θ, cos θ]]", note: "θ se anti-clockwise ghumata hai. RᵀR = I, **det(R) = +1**." },
        { t: "f", label: "Reflection (Householder)", x: "H = I − 2uuᵀ   (u unit vector)", note: "u ke perpendicular line/plane ke across reflect karta hai. **Hᵀ = H** (symmetric), **HᵀH = I**, **det(H) = −1**, aur **Hu = −u**." },
      ],
      ex: {
        problem: "u = [1, 1]ᵀ/√2 ke liye Householder H = I − 2uuᵀ nikalo. Uska det, Hu, aur H² check karo.",
        steps: [
          { title: "uuᵀ nikalo", detail: "u = [1/√2, 1/√2]ᵀ\nuuᵀ = [[1/2, 1/2], [1/2, 1/2]]" },
          { title: "H = I − 2uuᵀ", detail: "2uuᵀ = [[1, 1], [1, 1]]\nH = [[1,0],[0,1]] − [[1,1],[1,1]] = [[0, −1], [−1, 0]]" },
          { title: "det aur symmetry check", detail: "det(H) = 0(0) − (−1)(−1) = −1 ✓ (reflection ka signature)\nHᵀ = [[0,−1],[−1,0]] = H ✓ symmetric" },
          { title: "Hu = −u verify", detail: "Hu = [[0,−1],[−1,0]]·[1/√2, 1/√2]ᵀ = [−1/√2, −1/√2]ᵀ = −u ✓" },
          { title: "H² = I check", detail: "H² = [[0,−1],[−1,0]]² = [[1, 0], [0, 1]] = I ✓\nDo baar reflect = wapas original" },
          { title: "Geometric meaning", detail: "H, x aur y ko swap karke minus laga deta hai — yeh line y = −x ke across reflection hai (jo u = [1,1] ke perpendicular hai).\nEigenvalues: +1 (u ke perpendicular direction), −1 (u ki direction)." },
        ],
        answer: "H = [[0,−1],[−1,0]]. det = −1, Hu = −u, H² = I, eigenvalues {1, −1}.",
      },
      quiz: { q: "Q tall hai (5×3) aur QᵀQ = I₃. QQᵀ kya hai?", opts: ["I₅", "5×5 projection matrix, rank 3", "Zero matrix", "I₃"], a: 1, why: "Tall Q ke liye QQᵀ ek projection matrix hai C(Q) par, rank 3. Identity sirf square Q ke liye milti hai." },
    },
    {
      id: "gramschmidt", title: "Gram-Schmidt Process & A = QR", body: [
        { t: "p", x: "**Goal:** independent vectors {x₁,...,x_k} se orthonormal basis {q₁,...,q_k} banao — **wahi subspace** span karte hue. Idea: har naye vector se uska wo hissa **ghata do** jo pehle wale q's ke span mein hai." },
        { t: "f", label: "Algorithm", x: "Step 1:  v₁ = x₁,          q₁ = v₁/‖v₁‖\nStep 2:  v₂ = x₂ − (v₁ᵀx₂/v₁ᵀv₁)v₁,   q₂ = v₂/‖v₂‖\nStep k:  v_k = x_k − Σᵢ₌₁^(k−1) (vᵢᵀx_k/vᵢᵀvᵢ)vᵢ,   q_k = v_k/‖v_k‖" },
        { t: "f", label: "A = QR", x: "A = QR,  R = QᵀA  upper triangular", note: "R upper triangular isliye hai kyunki har naya vector sirf **apne se pehle wale** q's use karta hai." },
        { t: "trap", x: "Agar kisi step par **v_k = 0** aa jaaye, to x_k pehle wale vectors ke span mein tha — set **dependent** hai aur Gram-Schmidt naya orthonormal vector nahi bana sakta. Aur agar v_k bahut chhota hai, to ‖v_k‖ se divide karna numerical error badha deta hai (isliye practice mein Modified Gram-Schmidt use hota hai)." },
      ],
      ex: {
        problem: "a = [1, 1, 0]ᵀ, b = [1, 0, 1]ᵀ par Gram-Schmidt lagao aur A = QR likho.",
        steps: [
          { title: "q₁ nikalo", detail: "‖a‖ = √(1+1+0) = √2\nq₁ = [1/√2, 1/√2, 0]ᵀ ≈ [0.707, 0.707, 0]ᵀ" },
          { title: "b ka q₁ par component", detail: "q₁ᵀb = (1/√2)(1) + (1/√2)(0) + 0(1) = 1/√2 ≈ 0.707" },
          { title: "B = b − (q₁ᵀb)q₁", detail: "(q₁ᵀb)q₁ = (1/√2)·[1/√2, 1/√2, 0]ᵀ = [1/2, 1/2, 0]ᵀ\nB = [1, 0, 1]ᵀ − [0.5, 0.5, 0]ᵀ = [0.5, −0.5, 1]ᵀ" },
          { title: "q₂ = B/‖B‖", detail: "‖B‖ = √(0.25 + 0.25 + 1) = √1.5 ≈ 1.2247\nq₂ = [0.408, −0.408, 0.816]ᵀ" },
          { title: "Orthogonality verify", detail: "q₁ᵀq₂ = 0.707(0.408) + 0.707(−0.408) + 0 = 0 ✓\n‖q₂‖ = √(0.1667+0.1667+0.6667) = 1 ✓" },
          { title: "R = QᵀA banao", detail: "R₁₁ = ‖a‖ = √2 ≈ 1.414\nR₁₂ = q₁ᵀb = 0.707\nR₂₂ = ‖B‖ = 1.2247\nR₂₁ = 0 (hamesha, upper triangular)\nR = [[1.414, 0.707], [0, 1.225]]" },
        ],
        answer: "q₁ = [0.707, 0.707, 0]ᵀ, q₂ = [0.408, −0.408, 0.816]ᵀ, R = [[1.414, 0.707], [0, 1.225]], aur A = QR.",
      },
      quiz: { q: "Gram-Schmidt mein v₃ = 0 aa gaya. Kya matlab?", opts: ["Calculation galat hai", "x₃ pehle wale vectors ke span mein hai", "x₃ orthogonal hai", "Set orthonormal hai"], a: 1, why: "v₃ = 0 ka matlab x₃ ka poora hissa pehle wale q's ne cover kar liya — set linearly dependent hai." },
    },
    {
      id: "leastsq", title: "Least Squares — best fit line", body: [
        { t: "p", x: "Jab Ax = b solve nahi hota (b, C(A) se bahar hai), tab hum **‖b − Ax‖² minimize** karte hain. Wahi normal equations dete hain: **AᵀA x̂ = Aᵀb**." },
        { t: "f", label: "Line fitting y = C + Dt", x: "A = [[1, t₁], [1, t₂], ..., [1, tₘ]],   x = [C, D]ᵀ,   b = [y₁, ..., yₘ]ᵀ", note: "Pehla column saare 1s (constant term), doosra column t values. Parabola chahiye? Teesra column t² jod do — method wahi rehta hai." },
        { t: "list", x: [
          "Best fit line **vertical errors ka square sum** minimize karti hai (eᵢ = yᵢ − (C + Dtᵢ)).",
          "Minimum error ka value = ‖e‖² = ‖b‖² − ‖p‖².",
          "Error vector e hamesha N(Aᵀ) mein hota hai, isliye Σeᵢ = 0 aur Σtᵢeᵢ = 0.",
        ]},
      ],
      ex: {
        problem: "Points (t, y) = (0, 1), (1, 3), (2, 4), (3, 4) par best fit line y = C + Dt nikalo.",
        steps: [
          { title: "A aur b banao", detail: "A = [[1,0],[1,1],[1,2],[1,3]],  b = [1, 3, 4, 4]ᵀ" },
          { title: "AᵀA nikalo", detail: "(AᵀA)₁₁ = 4 (number of points)\n(AᵀA)₁₂ = Σtᵢ = 0+1+2+3 = 6\n(AᵀA)₂₂ = Σtᵢ² = 0+1+4+9 = 14\nAᵀA = [[4, 6], [6, 14]]" },
          { title: "Aᵀb nikalo", detail: "(Aᵀb)₁ = Σyᵢ = 1+3+4+4 = 12\n(Aᵀb)₂ = Σtᵢyᵢ = 0(1)+1(3)+2(4)+3(4) = 0+3+8+12 = 23\nAᵀb = [12, 23]ᵀ" },
          { title: "Normal equations solve karo", detail: "4C + 6D = 12\n6C + 14D = 23\nPehli × 1.5: 6C + 9D = 18\nGhatao: 5D = 5 ⟹ D = 1" },
          { title: "C nikalo", detail: "4C + 6(1) = 12 ⟹ 4C = 6 ⟹ C = 1.5" },
          { title: "Errors aur checks", detail: "Line: y = 1.5 + t\nPredicted: 1.5, 2.5, 3.5, 4.5\ne = [1−1.5, 3−2.5, 4−3.5, 4−4.5] = [−0.5, 0.5, 0.5, −0.5]\nΣeᵢ = 0 ✓   Σtᵢeᵢ = 0 + 0.5 + 1 − 1.5 = 0 ✓\n‖e‖² = 4(0.25) = 1 (minimum possible)" },
        ],
        answer: "Best fit: **y = 1.5 + t**. Minimum squared error ‖e‖² = 1.",
      },
      quiz: { q: "Least squares mein error vector e kis subspace mein hota hai?", opts: ["C(A)", "N(A)", "N(Aᵀ) — left null space", "Row space"], a: 2, why: "Aᵀe = 0 by construction, isliye e left null space N(Aᵀ) mein hota hai — C(A) ke bilkul perpendicular." },
    },
  ],
};

const CH6 = {
  id: "ch6", n: "06", title: "Eigenvalues & Eigenvectors",
  blurb: "Characteristic equation, AM vs GM, diagonalization, powers, complex λ, idempotent, spectral theorem, similarity aur definiteness.",
  topics: [
    {
      id: "eigdef", title: "Definition & Geometric Intuition", vis: "eigen", body: [
        { t: "f", label: "Definition", x: "Ax = λx,   x ≠ 0", note: "λ = eigenvalue (scalar), x = eigenvector. A sirf x ko **stretch/shrink** karta hai — direction nahi badalta." },
        { t: "list", x: [
          "**General vector u:** Au ghum jaata hai aur alag direction mein chala jaata hai.",
          "**Eigenvector v:** Av usi line par rehta hai, bas lambaai badalti hai.",
          "λ eigenvalue hai ⟺ (A − λI)x = 0 ka **non-trivial** solution hai ⟺ det(A − λI) = 0.",
          "**Eigenspace** for λ = N(A − λI) — isme zero vector aur us λ ke saare eigenvectors hain. Yeh ek subspace hai.",
        ]},
        { t: "trap", x: "Eigenvector **kabhi unique nahi** hota — koi bhi non-zero multiple bhi eigenvector hai. Isliye jawab mein direction likho ya normalize karo. Par eigen**value** unique hoti hai." },
      ],
      ex: {
        problem: "Verify karo ki x = [1, 1]ᵀ, A = [[3, 1], [1, 3]] ka eigenvector hai. λ nikalo. Phir [1, −1]ᵀ bhi check karo.",
        steps: [
          { title: "Ax compute karo", detail: "Ax = [[3,1],[1,3]]·[1,1]ᵀ = [3+1, 1+3]ᵀ = [4, 4]ᵀ" },
          { title: "λx ke form mein likho", detail: "[4, 4]ᵀ = 4·[1, 1]ᵀ = 4x ✓\n⟹ x eigenvector hai, **λ₁ = 4**" },
          { title: "Doosra vector check karo", detail: "A[1,−1]ᵀ = [3−1, 1−3]ᵀ = [2, −2]ᵀ = 2·[1,−1]ᵀ\n⟹ **λ₂ = 2**" },
          { title: "trace/det se cross-check", detail: "λ₁ + λ₂ = 4 + 2 = 6 = trace = 3 + 3 ✓\nλ₁ · λ₂ = 8 = det = 9 − 1 ✓" },
          { title: "Orthogonality note karo", detail: "A symmetric hai, aur eigenvectors [1,1] · [1,−1] = 1 − 1 = 0 — perpendicular ✓\nYahi spectral theorem hai." },
          { title: "Multiple bhi eigenvector hai", detail: "A[5,5]ᵀ = [20,20]ᵀ = 4[5,5]ᵀ ✓ — poori line eigenvectors hai" },
        ],
        answer: "λ₁ = 4 with x₁ = [1,1]ᵀ; λ₂ = 2 with x₂ = [1,−1]ᵀ. Symmetric matrix, eigenvectors orthogonal.",
      },
      quiz: { q: "Ax = 0 ka non-trivial solution x hai. Iska matlab?", opts: ["λ = 1 eigenvalue hai", "λ = 0 eigenvalue hai", "A invertible hai", "x = 0"], a: 1, why: "Ax = 0 = 0·x, isliye λ = 0 eigenvalue hai (aur A singular hai)." },
    },
    {
      id: "findeig", title: "Finding Eigenvalues — characteristic equation & tricks", body: [
        { t: "f", label: "Characteristic equation", x: "det(A − λI) = 0", note: "Degree-n polynomial in λ; uske n roots (multiplicity ke saath) hi eigenvalues hain." },
        { t: "f", label: "2×2 shortcut — exam mein yahi lagao", x: "λ² − (trace)λ + (det) = 0", note: "trace = a + d = λ₁ + λ₂;  det = ad − bc = λ₁λ₂" },
        { t: "list", x: [
          "**Σλᵢ = trace(A)** = a₁₁ + a₂₂ + ... + aₙₙ",
          "**Πλᵢ = det(A)**",
          "Koi λ = 0 ⟹ det = 0 ⟹ A singular.",
          "**Triangular / diagonal matrices:** eigenvalues seedhe diagonal entries hain.",
          "**Row-sum trick:** har row ka sum s ho, to **s ek eigenvalue** hai with x = [1,1,...,1]ᵀ.",
          "**Column-sum trick:** har column ka sum s ho, to s eigenvalue hai (Aᵀ aur A dono ka).",
        ]},
        { t: "table", head: ["Matrix", "Eigenvalue", "Eigenvector"], rows: [
          ["A⁻¹", "1/λ", "**same x**"],
          ["A²", "λ²", "same x"],
          ["A^k", "λ^k", "same x"],
          ["A + kI", "λ + k", "same x"],
          ["A² + A + 2I", "λ² + λ + 2", "same x"],
          ["Aᵀ", "same λ", "**different x**"],
        ]},
        { t: "tip", x: "A ka koi bhi polynomial (A², A^k, A + kI, ...) ke eigenvectors **bilkul same** rehte hain, sirf eigenvalues transform hoti hain. Commuting matrices (AB = BA) bhi same n independent eigenvectors share karte hain; unke liye AB ke eigenvalues λ·β aur A+B ke λ+β." },
      ],
      ex: {
        problem: "A = [[2, 1, 1], [1, 2, 1], [1, 1, 2]] ke saare eigenvalues aur eigenvectors nikalo.",
        steps: [
          { title: "Row-sum trick lagao", detail: "Har row ka sum = 2 + 1 + 1 = 4\n⟹ **λ = 4** eigenvalue hai, x = [1, 1, 1]ᵀ ke saath\nVerify: A[1,1,1]ᵀ = [4,4,4]ᵀ ✓" },
          { title: "A − I ki structure dekho", detail: "A − I = [[1,1,1],[1,1,1],[1,1,1]] — rank 1!\n⟹ N(A − I) ki dimension = 3 − 1 = 2\n⟹ **λ = 1** eigenvalue hai with GM = 2" },
          { title: "trace se confirm", detail: "trace = 2+2+2 = 6\nEigenvalues: 4, 1, 1 ⟹ sum = 6 ✓\ndet = 4(1)(1) = 4" },
          { title: "λ = 1 ke eigenvectors", detail: "(A − I)x = 0 ⟹ x₁ + x₂ + x₃ = 0 (ek hi equation)\nFree: x₂, x₃\ns₁ = [−1, 1, 0]ᵀ,  s₂ = [−1, 0, 1]ᵀ" },
          { title: "AM vs GM check", detail: "λ = 4: AM = 1, GM = 1 ✓\nλ = 1: AM = 2, GM = 2 ✓\nTotal independent eigenvectors = 3 ⟹ **diagonalizable**" },
          { title: "Orthogonality note", detail: "A symmetric hai. [1,1,1] · [−1,1,0] = 0 ✓ aur [1,1,1] · [−1,0,1] = 0 ✓\n(s₁ aur s₂ aapas mein orthogonal nahi, par Gram-Schmidt se bana sakte hain)" },
        ],
        answer: "λ = 4 (x = [1,1,1]ᵀ) aur λ = 1 double (eigenspace = plane x₁+x₂+x₃ = 0). Diagonalizable, det = 4, trace = 6.",
      },
      quiz: { q: "A ke eigenvalues 2, 3, 5 hain. A⁻¹ + 2I ke eigenvalues?", opts: ["2.5, 2.33, 2.2", "4, 5, 7", "0.5, 0.33, 0.2", "1, 1.5, 2.5"], a: 0, why: "A⁻¹ ke λ = 1/2, 1/3, 1/5. +2I se: 2.5, 2.333, 2.2. Eigenvectors nahi badalte." },
    },
    {
      id: "amgm", title: "Distinct vs Repeated — AM & GM", vis: "eigen", body: [
        { t: "f", label: "Do multiplicities", x: "AM (Algebraic Multiplicity) = char. polynomial mein (λ − λᵢ) ki power\nGM (Geometric Multiplicity) = dim N(A − λI) = us λ ke independent eigenvectors" },
        { t: "f", label: "The critical inequality", x: "1 ≤ GM ≤ AM   (hamesha)" },
        { t: "table", head: ["Case", "Example", "Result"], rows: [
          ["n distinct eigenvalues", "[[3,0],[0,1]]", "Independent eigenvectors guaranteed → **hamesha diagonalizable**"],
          ["Repeated λ, GM = AM", "[[2,0],[0,2]]", "Poora eigenspace mila → **diagonalizable ✓**"],
          ["Repeated λ, GM < AM", "[[2,1],[0,2]]", "**Defective** — diagonalize nahi hoga ✗"],
          ["Complex λ", "[[0,−1],[1,0]]", "Rotation — koi real eigenvector nahi"],
        ]},
        { t: "trap", x: "Repeated eigenvalue ka matlab **automatically** non-diagonalizable nahi hai! Har repeated λ ke liye GM alag se compute karna padta hai. I matrix ke saare λ repeated hain par wo perfectly diagonal hai." },
        { t: "f", label: "Conjugate pair rule (real A)", x: "λ = a + bi eigenvalue ⟹ λ̄ = a − bi bhi eigenvalue\nx eigenvector for λ ⟹ x̄ eigenvector for λ̄", note: "Real matrices ke liye complex eigenvalues hamesha jodon mein aate hain." },
      ],
      ex: {
        problem: "A = [[3, 1, 0], [0, 3, 0], [0, 0, 2]] ke liye AM aur GM nikalo. Diagonalizable hai?",
        steps: [
          { title: "Eigenvalues — triangular matrix", detail: "Upper triangular hai ⟹ eigenvalues = diagonal entries\nλ = 3, 3, 2\nChar. poly: (λ − 3)²(λ − 2) = 0" },
          { title: "AM likho", detail: "AM(λ = 3) = 2\nAM(λ = 2) = 1" },
          { title: "GM(λ = 3) nikalo", detail: "A − 3I = [[0, 1, 0], [0, 0, 0], [0, 0, −1]]\nRank = 2 (rows 1 aur 3 non-zero, independent)\nGM = dim N(A − 3I) = 3 − 2 = **1**" },
          { title: "Eigenvector for λ = 3", detail: "x₂ = 0 (row 1 se), −x₃ = 0 ⟹ x₃ = 0\nx₁ free ⟹ x = [1, 0, 0]ᵀ — sirf ek direction" },
          { title: "GM(λ = 2) nikalo", detail: "A − 2I = [[1, 1, 0], [0, 1, 0], [0, 0, 0]]\nRank = 2 ⟹ GM = 3 − 2 = 1 = AM ✓\nEigenvector: x₁ + x₂ = 0, x₂ = 0 ⟹ x = [0, 0, 1]ᵀ" },
          { title: "Final verdict", detail: "λ = 3: GM = 1 < AM = 2  ← **DEFECTIVE**\nTotal independent eigenvectors = 1 + 1 = 2 < 3\n⟹ **NOT diagonalizable**" },
        ],
        answer: "λ=3: AM=2, GM=1 (defective). λ=2: AM=1, GM=1. Sirf 2 independent eigenvectors ⟹ diagonalizable NAHI.",
      },
      quiz: { q: "5×5 matrix ka ek λ hai jiska AM = 3, GM = 3, aur baaki 2 distinct λ. Diagonalizable?", opts: ["Nahi", "Haan — total 5 independent eigenvectors", "Pata nahi chal sakta", "Sirf agar symmetric ho"], a: 1, why: "GM ka total = 3 + 1 + 1 = 5 = n. Enough independent eigenvectors ⟹ diagonalizable." },
    },
    {
      id: "diag", title: "Diagonalization & Powers of A", body: [
        { t: "f", label: "The theorem", x: "A diagonalizable ⟺ uske n linearly independent eigenvectors hain\n\nA = X Λ X⁻¹      ⟺      X⁻¹AX = Λ", note: "X ke columns = eigenvectors, Λ = eigenvalues ka diagonal matrix. **Order match karna zaroori hai.**" },
        { t: "num", x: [
          "det(A − λI) = 0 se eigenvalues nikalo.",
          "Har λ ke liye (A − λI)x = 0 solve karo — nullspace ka basis = eigenvectors.",
          "Total independent eigenvectors = n? Haan ⟹ diagonalizable. X aur Λ banao.",
          "A = XΛX⁻¹ likho (Λ mein λ ka order X ke columns se match kare).",
        ]},
        { t: "f", label: "Powers — the killer application", x: "A^k = XΛ^k X⁻¹,   Λ^k = diag(λ₁^k, ..., λₙ^k)", note: "Sirf diagonal entries ko power dena hai. u_k = A^k u₀ = c₁λ₁^k x₁ + c₂λ₂^k x₂ + ..." },
        { t: "table", head: ["Invertibility", "Diagonalizability"], rows: [
          ["**Eigenvalues** ke baare mein: λ = 0 hai ya nahi?", "**Eigenvectors** ke baare mein: kitne independent hain?"],
        ]},
        { t: "trap", x: "**Invertible ≠ Diagonalizable, Singular ≠ non-diagonalizable.** [[2,1],[0,2]] invertible hai par diagonalizable nahi. [[0,0],[0,0]] singular hai par perfectly diagonal hai." },
        { t: "list", x: [
          "**Long-run behaviour:** sabse bada |λ| dominate karta hai. A^k → 0 agar saare |λ| < 1.",
          "**Markov matrix** (har column sum 1): λ₁ = 1 hamesha, baaki |λ| < 1; steady state = λ=1 ka eigenvector.",
          "**Fibonacci:** A = [[1,1],[1,0]], λ = (1 ± √5)/2 — golden ratio 1.618 hi growth rate hai.",
          "**Differential equation** du/dt = Au: u(t) = c₁e^(λ₁t)x₁ + ..., stability ki shart **Re(λ) < 0**.",
        ]},
      ],
      ex: {
        problem: "A = [[1, 1], [1, 0]] (Fibonacci matrix) ko diagonalize karo aur A^k ka formula nikalo.",
        steps: [
          { title: "Characteristic equation", detail: "trace = 1, det = (1)(0) − (1)(1) = −1\nλ² − λ − 1 = 0" },
          { title: "Eigenvalues", detail: "λ = (1 ± √5)/2\nλ₁ = 1.618 (golden ratio φ), λ₂ = −0.618" },
          { title: "Eigenvector for λ₁", detail: "(A − λ₁I)x = 0 ⟹ (1 − λ₁)x₁ + x₂ = 0\nx₂ = (λ₁ − 1)x₁. λ₁ − 1 = 0.618 = 1/λ₁\nx₁ = [λ₁, 1]ᵀ = [1.618, 1]ᵀ" },
          { title: "Eigenvector for λ₂", detail: "x₂ = [λ₂, 1]ᵀ = [−0.618, 1]ᵀ" },
          { title: "X aur Λ banao", detail: "X = [[1.618, −0.618], [1, 1]]\nΛ = [[1.618, 0], [0, −0.618]]\ndet X = 1.618 + 0.618 = 2.236 = √5" },
          { title: "A^k formula", detail: "A^k = XΛ^kX⁻¹\nFibonacci: F_k = (λ₁^k − λ₂^k)/√5  (Binet formula)\nCheck k=5: (11.09 − (−0.09))/2.236 = 5 ✓ (F₅ = 5)" },
          { title: "Long-run growth", detail: "|λ₂| = 0.618 < 1 ⟹ wo term dab jaata hai\nF_{k+1}/F_k → λ₁ = 1.618 — golden ratio ✓" },
        ],
        answer: "λ = (1±√5)/2. A^k = XΛ^kX⁻¹ deta hai Binet formula F_k = (φ^k − ψ^k)/√5. Growth rate = φ ≈ 1.618.",
      },
      quiz: { q: "A ke eigenvalues 0.5 aur 0.9 hain. k → ∞ par A^k?", opts: ["Blow up karega", "→ 0 (zero matrix)", "Steady state par rukega", "Oscillate karega"], a: 1, why: "Dono |λ| < 1, isliye λ^k → 0 aur A^k → 0. System stable hai." },
    },
    {
      id: "rankeig", title: "Rank vs Eigenvalues, Idempotent & AB/BA", body: [
        { t: "f", label: "Rank aur eigenvalues ka rishta", x: "rank(A) = non-zero eigenvalues ki ginti  — **SIRF jab A diagonalizable ho**\n\nGeneral n×n ke liye:  n − rank = # of ZERO eigenvalues ki GM = dim N(A)", note: "Symmetric matrices hamesha diagonalizable hain, isliye unke liye rank = # non-zero λ hamesha sahi." },
        { t: "trap", x: "A = [[0,1],[0,0]] ke dono eigenvalues 0 hain (non-zero λ ki ginti = 0), par rank = 1! Kyunki A defective hai. Isliye 'rank = non-zero eigenvalues' rule **sirf diagonalizable matrices** par lagta hai." },
        { t: "f", label: "Idempotent matrix", x: "A² = A  ⟹  λ² = λ  ⟹  λ(λ − 1) = 0  ⟹  **λ = 0 ya 1 only**" },
        { t: "table", head: ["Case", "Result"], rows: [
          ["Sirf λ = 1", "A = Iₙ (agar diagonalizable)"],
          ["Sirf λ = 0", "A = zero matrix"],
          ["Dono 0 aur 1 (most common)", "rank r ⟹ r eigenvectors for λ=1, (n−r) for λ=0 ⟹ total n ⟹ **ALWAYS diagonalizable**"],
        ]},
        { t: "f", label: "Rank = Trace (idempotent ke liye)", x: "rank(A) = trace(A)  jab A² = A", note: "Proof: trace = Σλ = (#of 1s) + (#of 0s)·0 = # of λ=1 = rank." },
        { t: "f", label: "AB vs BA", x: "A(m×n) · B(n×m):  AB aur BA ke **non-zero eigenvalues IDENTICAL** hote hain", note: "Proof: ABx = λx ⟹ B(ABx) = B(λx) ⟹ (BA)(Bx) = λ(Bx). Isliye y = Bx, BA ka eigenvector hai same λ ke saath. Zero eigenvalues alag ho sakte hain kyunki AB aur BA ki dimensions alag hain." },
      ],
      ex: {
        problem: "P = (1/3)·[[1,1,1],[1,1,1],[1,1,1]] ke liye dikhao ki idempotent hai, aur rank = trace verify karo.",
        steps: [
          { title: "P² compute karo", detail: "ones(3)·ones(3) = 3·ones(3)  (har entry = 1+1+1 = 3)\nP² = (1/9)·3·ones(3) = (1/3)·ones(3) = P ✓ **idempotent**" },
          { title: "trace nikalo", detail: "trace(P) = 1/3 + 1/3 + 1/3 = 1" },
          { title: "rank nikalo", detail: "Saare rows identical hain ⟹ rank = 1\n**rank = trace = 1** ✓" },
          { title: "Eigenvalues confirm karo", detail: "Row sums = 1 ⟹ λ = 1 with x = [1,1,1]ᵀ\nrank 1 ⟹ nullity = 2 ⟹ λ = 0 with GM = 2\nEigenvalues: {1, 0, 0}" },
          { title: "Checks", detail: "Σλ = 1 + 0 + 0 = 1 = trace ✓\nΠλ = 0 = det ✓ (singular)\n# non-zero λ = 1 = rank ✓ (P diagonalizable hai)" },
          { title: "Meaning", detail: "P ek projection matrix hai [1,1,1]ᵀ ki line par.\nPᵀ = P bhi hai ⟹ orthogonal projection.\nI − P bhi idempotent, rank 2." },
        ],
        answer: "P² = P ✓, trace = rank = 1, eigenvalues {1, 0, 0}. Yeh [1,1,1]ᵀ line par orthogonal projection hai.",
      },
      quiz: { q: "A ek 5×5 idempotent matrix hai jiska trace 3 hai. rank aur nullity?", opts: ["rank 3, nullity 2", "rank 2, nullity 3", "rank 5, nullity 0", "Pata nahi", ], a: 0, why: "Idempotent ke liye rank = trace = 3. Rank-nullity se nullity = 5 − 3 = 2." },
    },
    {
      id: "spectral", title: "Symmetric Matrices & Spectral Theorem", body: [
        { t: "f", label: "Spectral theorem (Aᵀ = A)", x: "S = Q Λ Qᵀ    (Q orthogonal, QᵀQ = I)", note: "n **real** eigenvalues · GM = AM hamesha · alag λ ke eigenspaces mutually **orthogonal** · hamesha diagonalizable." },
        { t: "p", x: "**Proof (distinct λ ke eigenvectors orthogonal hain):** Sx₁ = λ₁x₁, Sx₂ = λ₂x₂. Tab (λ₁x₁)ᵀx₂ = (Sx₁)ᵀx₂ = x₁ᵀSᵀx₂ = x₁ᵀSx₂ = x₁ᵀ(λ₂x₂). Yaani λ₁(x₁ᵀx₂) = λ₂(x₁ᵀx₂) ⟹ (λ₁ − λ₂)(x₁ᵀx₂) = 0. λ₁ ≠ λ₂ hai, isliye **x₁ᵀx₂ = 0**." },
        { t: "f", label: "Spectral decomposition", x: "S = λ₁q₁q₁ᵀ + λ₂q₂q₂ᵀ + ... + λₙqₙqₙᵀ", note: "Har qᵢqᵢᵀ ek **rank-1 projection matrix** hai direction qᵢ par. Poori matrix in projections ka weighted sum hai — yahi soch aage SVD banti hai." },
        { t: "f", label: "Pivot vs eigenvalue connection", x: "Product of pivots = det(A) = product of eigenvalues", note: "**Symmetric matrices ke liye: pivots aur eigenvalues ke SIGNS hamesha same hote hain.** 2 positive λ ⟹ 2 positive pivots. Yeh positive-definiteness test ki jaan hai." },
        { t: "tip", x: "Symmetric matrix mein repeated eigenvalue ho, to us eigenspace ke andar **Gram-Schmidt** lagao — orthonormal set mil jaayega. Isliye symmetric matrix hamesha orthogonally diagonalizable rehti hai." },
      ],
      ex: {
        problem: "S = [[2, 1], [1, 2]] ka spectral decomposition S = QΛQᵀ nikalo aur outer-product form likho.",
        steps: [
          { title: "Eigenvalues", detail: "trace = 4, det = 4 − 1 = 3\nλ² − 4λ + 3 = 0 ⟹ (λ−3)(λ−1) = 0\nλ₁ = 3, λ₂ = 1" },
          { title: "Eigenvector for λ = 3", detail: "(S − 3I) = [[−1, 1], [1, −1]]\n−x₁ + x₂ = 0 ⟹ x₁ = x₂\nx₁ = [1, 1]ᵀ → normalize: q₁ = [1/√2, 1/√2]ᵀ" },
          { title: "Eigenvector for λ = 1", detail: "(S − I) = [[1, 1], [1, 1]]\nx₁ + x₂ = 0 ⟹ x₂ = −x₁\nq₂ = [1/√2, −1/√2]ᵀ" },
          { title: "Orthogonality verify", detail: "q₁ᵀq₂ = 1/2 − 1/2 = 0 ✓ (symmetric ki guarantee)" },
          { title: "Q aur Λ likho", detail: "Q = (1/√2)·[[1, 1], [1, −1]],  Λ = [[3, 0], [0, 1]]\nQᵀQ = I ✓ (orthogonal matrix)" },
          { title: "Outer product form", detail: "q₁q₁ᵀ = (1/2)[[1,1],[1,1]]\nq₂q₂ᵀ = (1/2)[[1,−1],[−1,1]]\nS = 3·(1/2)[[1,1],[1,1]] + 1·(1/2)[[1,−1],[−1,1]]\n = [[1.5,1.5],[1.5,1.5]] + [[0.5,−0.5],[−0.5,0.5]]\n = [[2, 1], [1, 2]] ✓" },
        ],
        answer: "S = QΛQᵀ with Q = (1/√2)[[1,1],[1,−1]], Λ = diag(3,1). Outer form: S = 3q₁q₁ᵀ + 1·q₂q₂ᵀ.",
      },
      quiz: { q: "Symmetric matrix ke 3 pivots hain: 2, −1, 5. Eigenvalues ke signs?", opts: ["Saare positive", "2 positive, 1 negative", "Saare negative", "Pata nahi chal sakta"], a: 1, why: "Symmetric matrices mein pivots aur eigenvalues ke signs match karte hain — 2 positive pivots ⟹ 2 positive λ, 1 negative pivot ⟹ 1 negative λ." },
    },
    {
      id: "similar", title: "Similar Matrices & Invariants", body: [
        { t: "f", label: "Definition", x: "A ~ B  agar koi invertible Z ho aisi ki  B = Z⁻¹AZ   ⟺   A = ZBZ⁻¹" },
        { t: "list", x: [
          "**Same characteristic polynomial** ⟹ same eigenvalues (same multiplicities ke saath).",
          "**Same determinant, trace, aur rank.**",
          "**Same diagonalizability** — ek diagonalizable hai to doosra bhi.",
          "**Transitivity:** B ~ A aur C ~ A ⟹ B ~ C.",
          "**Powers aur inverses:** A^k ~ B^k, A⁻¹ ~ B⁻¹.",
          "**Eigenvectors:** x, A ka eigenvector ⟹ Z⁻¹x, B ka eigenvector (same λ).",
        ]},
        { t: "p", x: "**Proof (char. poly same):** |B − λI| = |Z⁻¹AZ − λZ⁻¹IZ| = |Z⁻¹(A − λI)Z| = |Z⁻¹|·|A − λI|·|Z| = |A − λI|, kyunki |Z⁻¹Z| = 1." },
        { t: "trap", label: "Same eigenvalues ≠ Similar", x: "A = [[2,1],[0,2]] aur B = [[2,0],[0,2]] — dono ke eigenvalues {2,2} hain. Par **similar NAHI hain**: B diagonal hai (GM = 2), A defective hai (GM = 1). Similarity GM ko preserve karti hai." },
        { t: "tip", x: "**Efficient check:** agar A aur B **dono diagonalizable** hain, to wo similar hain ⟺ unke eigenvalues same hain (dono usi Λ ke similar hain)." },
      ],
      ex: {
        problem: "Kya A = [[3, 1], [0, 3]] aur B = [[3, 0], [0, 3]] similar hain? Reason do.",
        steps: [
          { title: "Eigenvalues compare karo", detail: "A upper triangular: λ = 3, 3\nB diagonal: λ = 3, 3\nSame eigenvalues ✓" },
          { title: "trace aur det compare karo", detail: "trace(A) = 6 = trace(B) ✓\ndet(A) = 9 = det(B) ✓\nSab invariants match — par yeh kaafi nahi!" },
          { title: "GM compute karo — A ke liye", detail: "A − 3I = [[0, 1], [0, 0]], rank = 1\nGM = 2 − 1 = 1" },
          { title: "GM — B ke liye", detail: "B − 3I = [[0, 0], [0, 0]], rank = 0\nGM = 2 − 0 = 2" },
          { title: "Decisive argument", detail: "Agar A ~ B, to B = Z⁻¹AZ.\nPar B = 3I, aur Z⁻¹(3I)Z = 3I hamesha.\nYaani agar B ~ A hota, to A khud 3I hota — jo galat hai (A ≠ 3I)." },
          { title: "Conclusion", detail: "GM alag hai (1 vs 2) ⟹ **similar NAHI hain**.\nB diagonalizable hai, A defective. Similarity yeh property preserve karti hai." },
        ],
        answer: "NAHI similar. Same λ, trace, det, rank sab hone ke bawajood GM alag hai (1 vs 2). 3I sirf apne aap ke similar hota hai.",
      },
      quiz: { q: "A ~ B aur A ke eigenvalues 1, 2, 3 hain. det(B) = ?", opts: ["6", "1", "0", "Pata nahi"], a: 0, why: "Similar matrices ke eigenvalues same hote hain, isliye det(B) = 1×2×3 = 6." },
    },
    {
      id: "definite", title: "Definiteness — PD, PSD, ND, NSD", vis: "quad", body: [
        { t: "table", head: ["Type", "Eigenvalues", "Pivots", "xᵀAx", "Shape"], rows: [
          ["**Positive Definite**", "saare λ > 0", "saare > 0", "> 0 (x ≠ 0)", "Upward bowl"],
          ["**Positive Semi-Def.**", "saare λ ≥ 0", "saare ≥ 0", "≥ 0", "Valley (ek direction flat)"],
          ["**Negative Definite**", "saare λ < 0", "saare < 0", "< 0 (x ≠ 0)", "Inverted bowl"],
          ["**Negative Semi-Def.**", "saare λ ≤ 0", "saare ≤ 0", "≤ 0", "Inverted valley"],
          ["**Indefinite**", "mixed signs", "mixed", "dono signs", "Saddle point"],
        ]},
        { t: "f", label: "PD ke 4 equivalent tests", x: "1. Saare eigenvalues λᵢ > 0\n2. Saare pivots > 0 (elimination bina row swaps)\n3. Saare n **leading principal minors** > 0\n4. xᵀAx > 0 for all x ≠ 0", note: "2×2 ke liye seedha: **a > 0 aur ac − b² > 0**." },
        { t: "trap", label: "Negative definite ka minor test", x: "ND ke liye leading principal minors **alternate** karte hain aur **D₁ < 0 se shuru** hote hain: D₁ < 0, D₂ > 0, D₃ < 0, ... Rule: Dᵢ > 0 agar i even, Dᵢ < 0 agar i odd. **Agar D₁ positive hai to matrix ND ho hi nahi sakta.**" },
        { t: "f", label: "AᵀA ka status", x: "AᵀA hamesha **positive semi-definite** hai:  xᵀ(AᵀA)x = ‖Ax‖² ≥ 0\nA ke columns independent  ⟹  AᵀA **positive definite**" },
        { t: "tip", label: "Sabse tez elimination test", x: "PD matrix ke **saare diagonal entries positive** hone chahiye (x = eᵢ rakho to xᵀAx = aᵢᵢ). Agar koi aᵢᵢ ≤ 0 dikhe, turant PD reject kar do — aur kuch check karne ki zarurat nahi." },
      ],
      ex: {
        problem: "A = [[2, −1, 0], [−1, 2, −1], [0, −1, 2]] positive definite hai? Teeno tests lagao.",
        steps: [
          { title: "Test 0 — diagonal quick check", detail: "Diagonal = 2, 2, 2 — saare positive ✓ (PD possible hai, aage badho)" },
          { title: "Test 3 — leading principal minors", detail: "D₁ = 2 > 0 ✓\nD₂ = det[[2,−1],[−1,2]] = 4 − 1 = 3 > 0 ✓\nD₃ = det(A) = 2(4−1) − (−1)(−2−0) + 0 = 6 − 2 = 4 > 0 ✓" },
          { title: "Test 2 — pivots (elimination)", detail: "R2 + (1/2)R1: [0, 1.5, −1]\nR3 + (2/3)R2: [0, 0, 2 − 2/3] = [0, 0, 4/3]\nPivots = 2, 1.5, 4/3 — **saare positive** ✓" },
          { title: "Pivots aur minors ka rishta", detail: "pivot₁ = D₁ = 2\npivot₂ = D₂/D₁ = 3/2 = 1.5 ✓\npivot₃ = D₃/D₂ = 4/3 ✓\n(Yeh formula hamesha kaam karta hai)" },
          { title: "Test 1 — eigenvalues", detail: "Yeh classic tridiagonal matrix hai:\nλ = 2 − 2cos(kπ/4) for k = 1,2,3\nλ = 2 − √2 ≈ 0.586, 2, 2 + √2 ≈ 3.414\n**Saare positive** ✓" },
          { title: "Test 4 — energy", detail: "xᵀAx = 2x₁² + 2x₂² + 2x₃² − 2x₁x₂ − 2x₂x₃\n= x₁² + (x₁−x₂)² + (x₂−x₃)² + x₃²\nSaare squares ⟹ ≥ 0, aur = 0 sirf x = 0 par ⟹ **PD** ✓" },
        ],
        answer: "Haan, positive definite. Pivots {2, 1.5, 1.333}, minors {2, 3, 4}, λ {0.586, 2, 3.414} — chaaron test pass.",
      },
      quiz: { q: "A symmetric hai, D₁ = 3 aur D₂ = −5. Kya keh sakte ho?", opts: ["Positive definite", "Negative definite", "Indefinite", "Positive semi-definite"], a: 2, why: "PD ke liye saare minors > 0 chahiye (D₂ fail). ND ke liye D₁ < 0 chahiye (fail). Isliye indefinite — mixed sign eigenvalues." },
    },
  ],
};

const CH7 = {
  id: "ch7", n: "07", title: "Quadratic Forms",
  blurb: "Polynomial se matrix banana, classification, principal axes theorem aur unit sphere par max/min.",
  topics: [
    {
      id: "qfdef", title: "Definition & Construction", vis: "quad", body: [
        { t: "f", label: "Definition", x: "Q(x) = xᵀAx,   A symmetric n×n", note: "Har term degree **exactly 2** hai — pure squares xᵢ² ya cross-terms xᵢxⱼ. Koi linear term nahi, koi constant nahi." },
        { t: "f", label: "Polynomial → Matrix algorithm", x: "Aᵢᵢ = xᵢ² ka coefficient\nAᵢⱼ = Aⱼᵢ = ½ × (xᵢxⱼ ka coefficient)", note: "Cross-term coefficient ko **hamesha aadha-aadha** dono positions mein baanto — taaki matrix symmetric rahe." },
        { t: "trap", x: "Poora cross-term coefficient ek hi position mein daal dena galat hai. 6xy ke liye A₁₂ = A₂₁ = 3 (na ki 6 aur 0). Symmetric matrix ke bina eigenvalue tests kaam nahi karenge." },
        { t: "f", label: "Pro-tip formula", x: "f(x₁,...,xₙ) = (x₁ + 2x₂ + ... + nxₙ)² = xᵀAx ke liye:\ntrace(A) = 1² + 2² + ... + n² = n(n+1)(2n+1)/6\nsum of ALL elements = (1 + 2 + ... + n)² = [n(n+1)/2]²", note: "Aisa A = vvᵀ hota hai jahan v = [1,2,...,n]ᵀ — rank 1, isliye ek hi non-zero eigenvalue = ‖v‖²." },
      ],
      ex: {
        problem: "Q(x) = 3x₁² + 2x₂² + 5x₃² + 4x₁x₂ − 6x₁x₃ ke liye symmetric matrix A banao aur classify karo.",
        steps: [
          { title: "Diagonal entries", detail: "A₁₁ = 3 (x₁² ka coeff)\nA₂₂ = 2\nA₃₃ = 5" },
          { title: "Off-diagonal — aadha-aadha", detail: "x₁x₂ ka coeff = 4 ⟹ A₁₂ = A₂₁ = 2\nx₁x₃ ka coeff = −6 ⟹ A₁₃ = A₃₁ = −3\nx₂x₃ ka coeff = 0 ⟹ A₂₃ = A₃₂ = 0" },
          { title: "A likho", detail: "A = [[3, 2, −3], [2, 2, 0], [−3, 0, 5]]\nSymmetric ✓" },
          { title: "Verify karo (expand karke)", detail: "xᵀAx = 3x₁² + 2x₂² + 5x₃² + 2(2)x₁x₂ + 2(−3)x₁x₃ + 2(0)x₂x₃\n     = 3x₁² + 2x₂² + 5x₃² + 4x₁x₂ − 6x₁x₃ ✓" },
          { title: "Classification — minors", detail: "D₁ = 3 > 0 ✓\nD₂ = det[[3,2],[2,2]] = 6 − 4 = 2 > 0 ✓\nD₃ = 3(10 − 0) − 2(10 − 0) + (−3)(0 + 6)\n   = 30 − 20 − 18 = −8 < 0 ✗" },
          { title: "Verdict", detail: "D₃ < 0 ⟹ PD nahi. D₁ > 0 ⟹ ND bhi nahi.\n⟹ **INDEFINITE** (saddle point) — kuch λ positive, kuch negative.\nSince det = −8 < 0 aur 3×3 hai, ek ya teen negative λ hain." },
        ],
        answer: "A = [[3,2,−3],[2,2,0],[−3,0,5]]. Minors: 3, 2, −8 ⟹ **Indefinite** (saddle).",
      },
      quiz: { q: "Q = x² + 6xy + y² ka matrix?", opts: ["[[1,6],[6,1]]", "[[1,3],[3,1]]", "[[1,0],[6,1]]", "[[1,6],[0,1]]"], a: 1, why: "Cross-term 6 ko aadha-aadha baanto: A₁₂ = A₂₁ = 3. Matrix symmetric hona chahiye." },
    },
    {
      id: "principal", title: "Change of Variables & Principal Axes Theorem", body: [
        { t: "p", x: "**Motivation:** cross-terms (x₁x₂) ki wajah se quadratic form ko graph karna, optimize karna ya classify karna mushkil ho jaata hai. Coordinate transformation se unhe **poori tarah khatam** kar sakte hain." },
        { t: "f", label: "Change of variable", x: "x = Py  ⟹  xᵀAx = (Py)ᵀA(Py) = yᵀ(PᵀAP)y" },
        { t: "f", label: "Principal Axes Theorem", x: "A symmetric ⟹ ek **orthogonal** change of variable x = Py exist karta hai jo xᵀAx ko **bina cross-terms** wale yᵀΛy mein badal deta hai.\n\nQ(y) = λ₁y₁² + λ₂y₂² + ... + λₙyₙ²", note: "P orthogonal (P⁻¹ = Pᵀ), Λ = eigenvalues ka diagonal. **P ke columns = A ke unit eigenvectors = Principal Axes**." },
        { t: "p", x: "**Derivation:** A = PΛPᵀ (spectral theorem). Substituting x = Py: Q = yᵀ(PᵀAP)y, aur PᵀAP = Pᵀ(PΛPᵀ)P = (PᵀP)Λ(PᵀP) = IΛI = Λ. Cross-terms gayab kyunki Λ diagonal hai." },
        { t: "trap", x: "Distinct eigenvalues ke eigenvectors automatically orthogonal hote hain (symmetric matrix mein). Lekin agar kisi λ ki **multiplicity > 1** hai, to us eigenspace ki dimension > 1 hai — tab aapko us eigenspace ke andar **Gram-Schmidt** lagana padega, warna P orthogonal nahi banega." },
        { t: "list", x: [
          "Ellipse xᵀAx = 1 ke **axes ki direction** = eigenvectors.",
          "Axes ki **lambaai** = 1/√λ — bada λ, chhota axis.",
          "**Condition number** λmax/λmin jitna bada, ellipse utni patli aur numerically utni bekaar.",
        ]},
      ],
      ex: {
        problem: "Q = 5x₁² + 4x₁x₂ + 5x₂² ko principal axes mein likho aur uske level curve ki shape batao.",
        steps: [
          { title: "Matrix banao", detail: "A₁₁ = 5, A₂₂ = 5, A₁₂ = A₂₁ = 4/2 = 2\nA = [[5, 2], [2, 5]]" },
          { title: "Eigenvalues nikalo", detail: "trace = 10, det = 25 − 4 = 21\nλ² − 10λ + 21 = 0 ⟹ (λ−7)(λ−3) = 0\nλ₁ = 7, λ₂ = 3" },
          { title: "Eigenvectors", detail: "λ = 7: (A−7I) = [[−2,2],[2,−2]] ⟹ x₁ = x₂ ⟹ u₁ = [1,1]ᵀ/√2\nλ = 3: (A−3I) = [[2,2],[2,2]] ⟹ x₂ = −x₁ ⟹ u₂ = [1,−1]ᵀ/√2" },
          { title: "P banao aur verify orthogonal", detail: "P = (1/√2)·[[1, 1], [1, −1]]\nPᵀP = I ✓, det P = −1 (reflection + rotation)" },
          { title: "Naya form likho", detail: "x = Py substitute karo:\nQ = 7y₁² + 3y₂²\n**Cross-term gayab** ✓" },
          { title: "Shape identify karo", detail: "Dono λ > 0 ⟹ **positive definite** ⟹ upward bowl\nLevel curve Q = 1: 7y₁² + 3y₂² = 1 → **ELLIPSE**\nSemi-axes: 1/√7 ≈ 0.378 (y₁ direction, [1,1] wali) aur 1/√3 ≈ 0.577 (y₂ direction)" },
        ],
        answer: "Q = 7y₁² + 3y₂² with principal axes [1,1]ᵀ/√2 aur [1,−1]ᵀ/√2. Level curve = ellipse, semi-axes 1/√7 aur 1/√3.",
      },
      quiz: { q: "xᵀAx = 1 ka level curve ellipse hai. A ke baare mein kya pata chala?", opts: ["A singular hai", "A positive definite hai", "A indefinite hai", "A skew-symmetric hai"], a: 1, why: "Bounded closed ellipse tabhi banti hai jab dono λ > 0. Mixed signs se hyperbola banti hai." },
    },
    {
      id: "maxmin", title: "Max & Min on the Unit Sphere", body: [
        { t: "f", label: "Rayleigh quotient result", x: "‖x‖ = 1 par:   max Q(x) = λ₁ (largest),  at x = u₁\n                 min Q(x) = λₙ (smallest), at x = uₙ", note: "λ₁ ≥ λ₂ ≥ ... ≥ λₙ sorted, aur uᵢ unke unit eigenvectors." },
        { t: "p", x: "**Proof one line mein:** x = Py rakho. Q = yᵀΛy = Σλᵢyᵢ². ‖x‖ = 1 aur P orthogonal ⟹ ‖y‖ = 1 ⟹ Σyᵢ² = 1. Isliye Σλᵢyᵢ² ≤ λ₁Σyᵢ² = λ₁(1) = λ₁. Equality tabhi jab y = e₁, yaani x = Pe₁ = u₁." },
        { t: "f", label: "Constrained maximum (iterative)", x: "Agar ‖x‖ = 1 AUR x ⊥ u₁:   max Q(x) = λ₂  at x = u₂\nGeneral: x ⊥ u₁,...,u_{k−1} aur ‖x‖ = 1  ⟹  max Q(x) = λ_k", note: "Har baar ek eigen-direction hatate jao, agla eigenvalue maximum ban jaata hai. Yahi PCA ka principle hai." },
      ],
      ex: {
        problem: "A = [[3, 0, 0], [0, 7, 0], [0, 0, 2]] ke liye unit sphere par Q(x) = xᵀAx ka max, min, aur x ⊥ u_max ke saath constrained max nikalo.",
        steps: [
          { title: "Eigenvalues padho", detail: "Diagonal matrix hai ⟹ λ = 3, 7, 2\nSort: λ₁ = 7, λ₂ = 3, λ₃ = 2" },
          { title: "Eigenvectors", detail: "λ₁ = 7 → u₁ = e₂ = [0,1,0]ᵀ\nλ₂ = 3 → u₂ = e₁ = [1,0,0]ᵀ\nλ₃ = 2 → u₃ = e₃ = [0,0,1]ᵀ" },
          { title: "Maximum", detail: "max Q = λ₁ = **7**, attained at x = [0, 1, 0]ᵀ\nCheck: Q = 3(0) + 7(1) + 2(0) = 7 ✓" },
          { title: "Minimum", detail: "min Q = λ₃ = **2**, at x = [0, 0, 1]ᵀ\nCheck: Q = 3(0) + 7(0) + 2(1) = 2 ✓" },
          { title: "Constrained max (x ⊥ u₁)", detail: "x₂ = 0 ki condition\nQ = 3x₁² + 2x₃² with x₁² + x₃² = 1\nMax tab jab poora weight bade coefficient par ⟹ x₁ = 1\nmax = λ₂ = **3** at [1, 0, 0]ᵀ ✓" },
          { title: "General bound", detail: "Har unit x ke liye: 2 ≤ Q(x) ≤ 7\nYaani λmin ≤ xᵀAx ≤ λmax — yeh bound hamesha sahi hai." },
        ],
        answer: "max = 7 (at e₂), min = 2 (at e₃), constrained max (⊥ e₂) = 3 (at e₁).",
      },
      quiz: { q: "A symmetric ke λ = −2, 1, 5 hain. ‖x‖ = 1 par xᵀAx ki range?", opts: ["[0, 5]", "[−2, 5]", "[1, 5]", "[−2, 1]"], a: 1, why: "Range hamesha [λmin, λmax] = [−2, 5] hoti hai." },
    },
  ],
};

const CH8 = {
  id: "ch8", n: "08", title: "Singular Value Decomposition (SVD)",
  blurb: "A = UΣVᵀ, computation, four subspaces, Av = σu, geometry, aur λ vs σ ke saare traps.",
  topics: [
    {
      id: "svdmotiv", title: "Motivation — matrix as three simple steps", vis: "svd", body: [
        { t: "p", x: "Har m×n matrix A ek linear map hai: Rⁿ se leta hai, Rᵐ mein deta hai. SVD is transformation ko **teen samajhne laayak steps** mein tod deta hai." },
        { t: "f", label: "A = UΣVᵀ ⟹ Ax = U(Σ(Vᵀx))", x: "Step 1: Vᵀx      → Rⁿ mein **rotate** karo (length same)\nStep 2: Σ(Vᵀx)   → axes ke saath **scale** karo, dimension badlo (Rⁿ → Rᵐ)\nStep 3: U(ΣVᵀx)  → Rᵐ mein **rotate** karo (length same)" },
        { t: "tip", label: "Key insight", x: "Orthogonal matrix se multiply karna sirf **rotate/reflect** karta hai — length kabhi nahi badalta. Isliye poori stretching/shrinking **sirf Σ** karta hai." },
        { t: "f", label: "Core theorem", x: "A(m×n) = U(m×m) · Σ(m×n) · Vᵀ(n×n)", note: "**U**: orthogonal, columns = left singular vectors uᵢ ∈ Rᵐ, UᵀU = UUᵀ = I.\n**Σ**: A jaisi shape, diagonal par σ₁ ≥ σ₂ ≥ ... ≥ σ_r > 0, baaki 0.\n**V**: orthogonal, columns = right singular vectors vᵢ ∈ Rⁿ, VᵀV = VVᵀ = I." },
        { t: "p", x: "**Square matrices ke liye:** U, Σ, V teeno n×n hote hain aur |det(A)| = σ₁σ₂...σₙ." },
      ],
      ex: {
        problem: "A = [[2, 0], [0, 1]] ke liye batao ki unit circle ka image kya hai aur σ values kya hain.",
        steps: [
          { title: "A pehle se diagonal hai", detail: "A = Σ hi hai (U = V = I)\nσ₁ = 2, σ₂ = 1" },
          { title: "Unit circle par action", detail: "x = [cos t, sin t]ᵀ, ‖x‖ = 1\nAx = [2cos t, sin t]ᵀ = [y₁, y₂]ᵀ" },
          { title: "Ellipse equation nikalo", detail: "cos t = y₁/2, sin t = y₂\ncos²t + sin²t = 1 ⟹ (y₁/2)² + y₂² = 1\ny₁²/4 + y₂²/1 = 1 — **ELLIPSE**" },
          { title: "Semi-axes identify karo", detail: "Semi-major = 2 = σ₁ (x-direction)\nSemi-minor = 1 = σ₂ (y-direction)" },
          { title: "Length bounds", detail: "σ₂ ≤ ‖Ax‖ ≤ σ₁ for all unit x\n1 ≤ ‖Ax‖ ≤ 2\nMax stretch v₁ = [1,0]ᵀ direction mein, min v₂ = [0,1]ᵀ mein" },
          { title: "General rule", detail: "Har matrix ke liye: circle → ellipse, aur ellipse ke semi-axes hamesha σᵢ hote hain.\nAgar A rank-deficient hai to ellipse flat ho jaati hai (koi σ = 0)." },
        ],
        answer: "Unit circle → ellipse y₁²/4 + y₂² = 1. σ₁ = 2, σ₂ = 1. Length bounds: 1 ≤ ‖Ax‖ ≤ 2.",
      },
      quiz: { q: "Ax ki length ka maximum kya hai jab ‖x‖ = 1?", opts: ["λmax", "σ₁ (largest singular value)", "det A", "trace A"], a: 1, why: "‖A‖₂ = σ₁ = max ‖Ax‖/‖x‖. Yeh operator 2-norm ki definition hai." },
    },
    {
      id: "svdcompute", title: "How to Compute SVD — full procedure", body: [
        { t: "num", x: [
          "**AᵀA (n×n symmetric) compute karo.** Uske eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₙ ≥ 0 aur unit eigenvectors v₁,...,vₙ nikalo. Yeh columns **V** banate hain.",
          "**Singular values:** σᵢ = √λᵢ. Decreasing order mein sort karo.",
          "**Non-zero σᵢ ke liye:** uᵢ = Avᵢ / σᵢ. Yeh unit vectors column space wala hissa banate hain.",
          "**Zero σᵢ ke liye:** N(Aᵀ) nikalo (Aᵀy = 0 solve karo), aur **Gram-Schmidt** se orthonormal u_{r+1},...,u_m banao.",
          "**Alternative:** seedha AAᵀ compute karo — uske eigenvectors = U ke columns. Non-zero eigenvalues wahi σᵢ² hote hain.",
        ]},
        { t: "f", label: "Key relations", x: "AᵀA = VΣᵀΣVᵀ = VΣ²Vᵀ\nAAᵀ = UΣΣᵀUᵀ = UΣ²Uᵀ", note: "Yaani AᵀA aur AAᵀ ke non-zero eigenvalues **same** hain, aur wo σᵢ² hain." },
        { t: "tip", x: "Exam mein hamesha **chhoti wali** matrix choose karo. A 100×3 hai? AᵀA sirf 3×3 hoga — usko lo. A 3×100 hai? AAᵀ 3×3 hoga." },
      ],
      ex: {
        problem: "A = [[3, 0], [4, 5]] ka poora SVD nikalo.",
        steps: [
          { title: "AᵀA compute karo", detail: "Aᵀ = [[3, 4], [0, 5]]\nAᵀA = [[9+16, 0+20], [0+20, 0+25]] = [[25, 20], [20, 25]]" },
          { title: "AᵀA ke eigenvalues", detail: "trace = 50, det = 625 − 400 = 225\nλ² − 50λ + 225 = 0 ⟹ λ = (50 ± √(2500−900))/2 = (50 ± 40)/2\nλ₁ = 45, λ₂ = 5" },
          { title: "Singular values", detail: "σ₁ = √45 = 3√5 ≈ 6.708\nσ₂ = √5 ≈ 2.236\nCheck: σ₁σ₂ = √225 = 15 = |det A| = |15 − 0| ✓" },
          { title: "V nikalo (AᵀA ke eigenvectors)", detail: "λ=45: (AᵀA − 45I) = [[−20, 20], [20, −20]] ⟹ v₁ = [1,1]ᵀ/√2\nλ=5: [[20,20],[20,20]] ⟹ v₂ = [1,−1]ᵀ/√2\nV = (1/√2)[[1, 1], [1, −1]]" },
          { title: "U nikalo: uᵢ = Avᵢ/σᵢ", detail: "Av₁ = (1/√2)[3, 9]ᵀ\nu₁ = Av₁/σ₁ = (1/√2)[3,9]ᵀ/(3√5) = [1, 3]ᵀ/√10\n\nAv₂ = (1/√2)[3, −1]ᵀ\nu₂ = Av₂/σ₂ = (1/√2)[3,−1]ᵀ/√5 = [3, −1]ᵀ/√10" },
          { title: "Verify aur likho", detail: "u₁ᵀu₂ = (3 − 3)/10 = 0 ✓ orthonormal\nU = (1/√10)[[1, 3], [3, −1]]\nΣ = [[6.708, 0], [0, 2.236]]\nA = UΣVᵀ ✓" },
        ],
        answer: "σ₁ = 3√5, σ₂ = √5. V = (1/√2)[[1,1],[1,−1]], U = (1/√10)[[1,3],[3,−1]]. σ₁σ₂ = 15 = |det A| ✓",
      },
      quiz: { q: "AᵀA ke eigenvalues 16, 9, 0 hain. A ke singular values?", opts: ["16, 9, 0", "4, 3, 0", "256, 81, 0", "8, 4.5, 0"], a: 1, why: "σᵢ = √λᵢ(AᵀA) = √16, √9, √0 = 4, 3, 0. rank = 2 (do non-zero σ)." },
    },
    {
      id: "svdsubspaces", title: "Four Subspaces from SVD & Av = σu", body: [
        { t: "f", label: "The fundamental relation", x: "Av₁ = σ₁u₁,  Av₂ = σ₂u₂,  ...,  Av_r = σ_r u_r\nAv_{r+1} = 0,  ...,  Av_n = 0", note: "Stack karo: AV = UΣ. V orthogonal hai (V⁻¹ = Vᵀ), isliye **A = UΣVᵀ**." },
        { t: "table", head: ["Subspace", "SVD basis", "Dimension"], rows: [
          ["**Row space C(Aᵀ)**", "v₁, ..., v_r (V ke pehle r columns)", "r"],
          ["**Null space N(A)**", "v_{r+1}, ..., v_n (V ke last n−r)", "n − r"],
          ["**Column space C(A)**", "u₁, ..., u_r (U ke pehle r columns)", "r"],
          ["**Left null space N(Aᵀ)**", "u_{r+1}, ..., u_m (U ke last m−r)", "m − r"],
        ]},
        { t: "f", label: "Rank", x: "rank(A) = r = non-zero singular values ki ginti = dim(row space) = dim(col space)" },
        { t: "p", x: "**Why uᵢ ⊥ uⱼ?** Kyunki vᵢ ⊥ vⱼ: u₂ᵀu₁ = (Av₂/σ₂)ᵀ(Av₁/σ₁) = v₂ᵀAᵀAv₁/(σ₁σ₂) = v₂ᵀ(λ₁v₁)/(σ₁σ₂) = 0. **Orthogonal input → orthogonal output, automatically.**" },
        { t: "tip", x: "SVD hi wo cheez hai jo chaaron subspaces ke liye ek saath **perfect orthonormal basis** deti hai, aur A ko un basis mein bilkul diagonal bana deti hai. Gram-Schmidt/RREF se aisa nahi milta." },
      ],
      ex: {
        problem: "A = [[1, 1], [2, 2]] ka SVD nikalo aur chaaron subspaces ke basis identify karo.",
        steps: [
          { title: "Rank pehchano", detail: "row2 = 2 × row1, col2 = col1\n⟹ rank r = 1\nm = n = 2, isliye: dim N(A) = 1, dim N(Aᵀ) = 1" },
          { title: "AᵀA compute karo", detail: "AᵀA = [[1+4, 1+4], [1+4, 1+4]] = [[5, 5], [5, 5]]\ntrace = 10, det = 0 ⟹ λ = 10, 0" },
          { title: "σ aur V", detail: "σ₁ = √10 ≈ 3.162, σ₂ = 0\nλ=10: [[−5,5],[5,−5]] ⟹ v₁ = [1,1]ᵀ/√2\nλ=0: v₂ = [1,−1]ᵀ/√2" },
          { title: "u₁ nikalo", detail: "Av₁ = (1/√2)[1+1, 2+2]ᵀ = (1/√2)[2, 4]ᵀ\nu₁ = Av₁/σ₁ = (1/√2)[2,4]ᵀ/√10 = [1, 2]ᵀ/√5" },
          { title: "u₂ — left null space se", detail: "u₂ ⊥ u₁ ⟹ u₂ = [2, −1]ᵀ/√5\nVerify: Aᵀu₂ = [[1,2],[1,2]]·[2,−1]ᵀ/√5 = [0, 0]ᵀ ✓" },
          { title: "Chaaron subspaces likho", detail: "Row space = span{v₁} = span{[1,1]ᵀ}\nNull space = span{v₂} = span{[1,−1]ᵀ}  (check: A[1,−1]ᵀ = [0,0]ᵀ ✓)\nColumn space = span{u₁} = span{[1,2]ᵀ}\nLeft null space = span{u₂} = span{[2,−1]ᵀ}" },
          { title: "Outer product form", detail: "A = σ₁u₁v₁ᵀ = √10 · [1,2]ᵀ[1,1]/(√5·√2)\n= (√10/√10)·[[1,1],[2,2]] = A ✓ (rank-1, ek hi term)" },
        ],
        answer: "σ₁ = √10, σ₂ = 0, rank 1. Row space span{[1,1]}, N(A) span{[1,−1]}, C(A) span{[1,2]}, N(Aᵀ) span{[2,−1]}.",
      },
      quiz: { q: "A ek 4×6 matrix hai jiske 3 non-zero singular values hain. dim N(A) = ?", opts: ["1", "3", "2", "6"], a: 1, why: "rank = 3 (non-zero σ ki ginti). n = 6, isliye dim N(A) = 6 − 3 = 3." },
    },
    {
      id: "svdprops", title: "Five Key Properties & Traps", body: [
        { t: "f", label: "1. Subspace summary", x: "U → [u₁...u_r | u_{r+1}...u_m]   (Col space | Left null space)\nV → [v₁...v_r | v_{r+1}...v_n]   (Row space | Null space)\nrank(A) = r = # non-zero σ" },
        { t: "f", label: "2. AᵀA eigenvalues vs σ", x: "AᵀA ke eigenvalues: σ₁², σ₂², ..., σ_r², phir zeros\nA ke singular values: σ₁, σ₂, ..., σ_r, phir zeros", note: "**Trap:** agar Σ = diag(2, 4, 0, 0) diya ho to rank 2 hai aur σ = 2, 4. Par **A ke eigenvalues aap batah hi nahi sakte** — extra info chahiye." },
        { t: "trap", label: "3. Powers ka trap (bahut important)", x: "A ke σ hain, par **A² ke σ, σ² NAHI hote**. Reason: A² = (UΣVᵀ)(UΣVᵀ) = UΣ(VᵀU)ΣVᵀ. VᵀU general mein I nahi hota, isliye A² ka SVD UΣ²Vᵀ nahi hai. **Squaring A does NOT square its singular values.**" },
        { t: "f", label: "4. Inverse — yeh WORK karta hai", x: "A = UΣVᵀ  ⟹  A⁻¹ = VΣ⁻¹Uᵀ,  Σ⁻¹ = diag(1/σ₁, 1/σ₂, ...)", note: "A⁻¹ ke singular values **1/σ** hote hain — powers ke ulat, yeh rule sach hai." },
        { t: "f", label: "5. Symmetric matrix: SVD = Eigendecomposition", x: "A = A ᵀ ⟹ A = QΛQᵀ, aur U = V = Q, Σ = |Λ|", note: "σᵢ = **|λᵢ|**, na ki λᵢ. Singular values hamesha ≥ 0 hote hain; eigenvalues negative ho sakte hain. Positive (semi)definite ke liye σᵢ = λᵢ exactly." },
        { t: "table", head: ["Property", "Eigenvalues (λ)", "Singular values (σ)"], rows: [
          ["Sign", "negative, complex ho sakte", "**hamesha ≥ 0**"],
          ["Exists for", "sirf square matrices", "**koi bhi m × n**"],
          ["Equation", "Ax = λx (same space)", "Av = σu (**different spaces**)"],
          ["For Aᵀ", "same λ", "same σ"],
          ["For A⁻¹", "1/λ", "1/σ"],
          ["For A²", "λ²", "**NOT σ² in general**"],
          ["Symmetric A", "real λ", "σᵢ = |λᵢ|"],
          ["PSD symmetric", "λᵢ ≥ 0", "σᵢ = λᵢ"],
        ]},
      ],
      ex: {
        problem: "A = [[0, 2], [−1, 0]] ke eigenvalues aur singular values dono nikalo aur compare karo.",
        steps: [
          { title: "Eigenvalues", detail: "trace = 0, det = 0 − (2)(−1) = 2\nλ² − 0λ + 2 = 0 ⟹ λ² = −2\nλ = ±i√2 — **complex!**" },
          { title: "AᵀA compute karo", detail: "Aᵀ = [[0, −1], [2, 0]]\nAᵀA = [[0+1, 0+0], [0+0, 4+0]] = [[1, 0], [0, 4]]" },
          { title: "Singular values", detail: "AᵀA ke eigenvalues = 1, 4\nσ₁ = 2, σ₂ = 1 — **real aur positive**, hamesha" },
          { title: "Cross-check", detail: "σ₁σ₂ = 2 = |det A| ✓\n|λ₁||λ₂| = √2·√2 = 2 ✓ (magnitudes match karte hain, values nahi)" },
          { title: "Powers ka trap check karo", detail: "A² = [[0,2],[−1,0]]² = [[−2, 0], [0, −2]] = −2I\nA² ke singular values = 2, 2\nAgar σ² rule sach hota to 4, 1 aate — **par 2, 2 aaye** ✗\nTrap confirm ✓" },
          { title: "Conclusion", detail: "Eigenvalues complex, singular values real. Non-symmetric matrix mein dono ka koi seedha rishta nahi hota (bas |det| = Π|λ| = Πσ)." },
        ],
        answer: "λ = ±i√2 (complex), σ = 2, 1 (real positive). A² ke σ = 2, 2 — na ki 4, 1. Powers ka trap verify hua.",
      },
      quiz: { q: "A symmetric hai with λ = −5, 3. Singular values?", opts: ["−5, 3", "5, 3", "25, 9", "3, −5"], a: 1, why: "Symmetric ke liye σᵢ = |λᵢ| = 5, 3. Singular values kabhi negative nahi hote." },
    },
    {
      id: "svdreduced", title: "Reduced SVD & Outer Product Form", body: [
        { t: "f", label: "Reduced (thin) SVD", x: "A(m×n) = U(m×r) · Σ(r×r) · Vᵀ(r×n)", note: "σ = 0 wale vectors drop kar do — wo kuch contribute hi nahi karte. Sirf **column space (u₁...u_r)** aur **row space (v₁...v_r)** bachte hain." },
        { t: "f", label: "Outer product form — the 4th way", x: "A = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + ... + σ_r u_r v_rᵀ", note: "Har term ek **rank-1 matrix** hai. σ_{r+1} = ... = 0 hain, isliye wo terms gayab. Yeh SVD ke fundamental building blocks hain." },
        { t: "p", x: "**Analogy:** symmetric S ka eigendecomposition S = λ₁q₁q₁ᵀ + ... + λₙqₙqₙᵀ hota hai. **SVD isko HAR matrix par generalize karta hai** — square hone ki bhi zarurat nahi, symmetric ki to bilkul nahi." },
        { t: "trap", x: "**SVD unique nahi hai:** −uᵢ, −vᵢ bhi valid hain kyunki σᵢ(−uᵢ)(−vᵢ)ᵀ = σᵢuᵢvᵢᵀ. Signs hamesha **jodon mein** flip ho sakte hain. Repeated σ ho to aur bhi freedom hai." },
      ],
      ex: {
        problem: "A = [[4, 0], [3, −5]] ko rank-1 pieces (outer product form) mein todo.",
        steps: [
          { title: "AᵀA nikalo", detail: "AᵀA = [[16+9, 0−15], [0−15, 0+25]] = [[25, −15], [−15, 25]]" },
          { title: "Eigenvalues", detail: "trace = 50, det = 625 − 225 = 400\nλ² − 50λ + 400 = 0 ⟹ λ = (50 ± √(2500−1600))/2 = (50 ± 30)/2\nλ₁ = 40, λ₂ = 10" },
          { title: "σ values", detail: "σ₁ = √40 = 2√10 ≈ 6.325\nσ₂ = √10 ≈ 3.162\nCheck: σ₁σ₂ = √400 = 20 = |det A| = |−20| ✓" },
          { title: "V nikalo", detail: "λ=40: [[−15,−15],[−15,−15]] ⟹ v₁ = [1,−1]ᵀ/√2\nλ=10: [[15,−15],[−15,15]] ⟹ v₂ = [1,1]ᵀ/√2" },
          { title: "U nikalo", detail: "Av₁ = (1/√2)[4, 3+5]ᵀ = (1/√2)[4, 8]ᵀ\nu₁ = (1/√2)[4,8]ᵀ/(2√10) = [1, 2]ᵀ/√5\n\nAv₂ = (1/√2)[4, 3−5]ᵀ = (1/√2)[4, −2]ᵀ\nu₂ = (1/√2)[4,−2]ᵀ/√10 = [2, −1]ᵀ/√5" },
          { title: "Outer product form likho", detail: "A = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ\nu₁v₁ᵀ = (1/√10)[[1,−1],[2,−2]]\nu₂v₂ᵀ = (1/√10)[[2,2],[−1,−1]]\n\nA = 2√10·(1/√10)[[1,−1],[2,−2]] + √10·(1/√10)[[2,2],[−1,−1]]\n  = [[2,−2],[4,−4]] + [[2,2],[−1,−1]] = [[4, 0], [3, −5]] ✓" },
        ],
        answer: "A = 2√10·u₁v₁ᵀ + √10·u₂v₂ᵀ, jahan u₁=[1,2]ᵀ/√5, v₁=[1,−1]ᵀ/√2, u₂=[2,−1]ᵀ/√5, v₂=[1,1]ᵀ/√2.",
      },
      quiz: { q: "A ka rank 3 hai. Outer product form mein kitne terms honge?", opts: ["1", "3", "Matrix ke size jitne", "Infinite"], a: 1, why: "Non-zero σ ki ginti = rank = 3, isliye exactly 3 rank-1 terms." },
    },
  ],
};

const CH9 = {
  id: "ch9", n: "09", title: "Norms & Rank-k Approximation",
  blurb: "Vector norms, Frobenius/spectral/nuclear norms, truncated SVD aur Eckart–Young–Mirsky theorem.",
  topics: [
    {
      id: "vecnorm", title: "Vector Norms", body: [
        { t: "f", label: "Norm axioms (teenon zaroori)", x: "1. Non-negativity: ‖v‖ ≥ 0, aur ‖v‖ = 0 ⟺ v = 0\n2. Homogeneity: ‖kv‖ = |k|·‖v‖\n3. Triangle inequality: ‖v + w‖ ≤ ‖v‖ + ‖w‖" },
        { t: "table", head: ["Norm", "Name", "Formula", "Note"], rows: [
          ["‖x‖₂", "Euclidean / 2-norm", "√(Σxᵢ²) = √(xᵀx)", "**Default** — unspecified ho to yahi"],
          ["‖x‖₁", "Manhattan / Taxicab", "Σ|xᵢ|", "1-norm"],
          ["‖x‖∞", "Maximum / Chebyshev", "max|xᵢ|", "∞-norm"],
        ]},
        { t: "f", label: "General ℓp norm", x: "‖x‖_p = (Σ|xᵢ|^p)^(1/p),   p > 0", note: "p = 1, 2, ∞ teenon isi se nikalte hain." },
      ],
      ex: {
        problem: "x = [3, −4, 12]ᵀ ke liye ‖x‖₁, ‖x‖₂, ‖x‖∞ nikalo aur inequality ‖x‖∞ ≤ ‖x‖₂ ≤ ‖x‖₁ verify karo.",
        steps: [
          { title: "1-norm", detail: "‖x‖₁ = |3| + |−4| + |12| = 3 + 4 + 12 = 19" },
          { title: "2-norm", detail: "‖x‖₂ = √(9 + 16 + 144) = √169 = 13" },
          { title: "∞-norm", detail: "‖x‖∞ = max(3, 4, 12) = 12" },
          { title: "Inequality verify", detail: "12 ≤ 13 ≤ 19 ✓\nYeh ordering hamesha sach hoti hai." },
          { title: "Unit vector", detail: "x̂ = x/‖x‖₂ = [3/13, −4/13, 12/13]ᵀ\nCheck: (9 + 16 + 144)/169 = 1 ✓" },
        ],
        answer: "‖x‖₁ = 19, ‖x‖₂ = 13, ‖x‖∞ = 12. Ordering ‖x‖∞ ≤ ‖x‖₂ ≤ ‖x‖₁ verified.",
      },
      quiz: { q: "Rⁿ mein ‖x‖₂ ≤ ‖x‖₁ hamesha sach hai?", opts: ["Haan", "Nahi", "Sirf positive entries ke liye", "Sirf n = 1 ke liye"], a: 0, why: "Squares ka sum ≤ (sum of absolute values)² kyunki cross-terms non-negative hote hain. Equality tabhi jab at most ek entry non-zero ho." },
    },
    {
      id: "matnorm", title: "Matrix Norms — Frobenius, Spectral, Nuclear", vis: "svd", body: [
        { t: "f", label: "Frobenius norm", x: "‖A‖_F = √(Σᵢⱼ aᵢⱼ²)  = √(trace(AᵀA))  = √(σ₁² + σ₂² + ... + σ_r²)", note: "Saare entries ke squares ka sum ka root. Matrix ko ek lambe vector ki tarah treat karta hai." },
        { t: "f", label: "Spectral norm (operator 2-norm)", x: "‖A‖₂ = max(‖Ax‖₂ / ‖x‖₂) = max_{‖u‖=1} ‖Au‖₂ = **σ₁**", note: "**Proof:** ‖A‖₂² = max (xᵀAᵀAx)/(xᵀx) = λ₁(AᵀA) = σ₁². Maximizer x = v₁." },
        { t: "f", label: "Nuclear norm (trace norm)", x: "‖A‖* = ‖σ‖₁ = Σᵢ σᵢ", note: "Saare singular values ka sum — σ vector ka ℓ₁ norm." },
        { t: "table", head: ["Norm", "Symbol", "Entries se", "σ ke terms mein"], rows: [
          ["Frobenius", "‖A‖_F", "√(Σaᵢⱼ²)", "√(Σσᵢ²)"],
          ["Spectral", "‖A‖₂", "max‖Ax‖/‖x‖", "σ₁"],
          ["Nuclear", "‖A‖*", "—", "Σσᵢ"],
        ]},
        { t: "tip", x: "Teenon **unitarily invariant** hain: ‖UAV‖ = ‖A‖ kisi bhi orthogonal U, V ke liye. Isi wajah se Eckart–Young–Mirsky theorem teenon norms ke liye ek saath sach hota hai." },
      ],
      ex: {
        problem: "A = [[3, 0], [4, 5]] ke teenon norms nikalo (σ₁ = 3√5, σ₂ = √5 pehle nikal chuke hain).",
        steps: [
          { title: "Frobenius — entries se", detail: "‖A‖_F = √(9 + 0 + 16 + 25) = √50 ≈ 7.071" },
          { title: "Frobenius — σ se cross-check", detail: "σ₁² + σ₂² = 45 + 5 = 50\n√50 ≈ 7.071 ✓ dono match" },
          { title: "Spectral norm", detail: "‖A‖₂ = σ₁ = 3√5 ≈ 6.708\nYeh maximum stretch factor hai" },
          { title: "Nuclear norm", detail: "‖A‖* = σ₁ + σ₂ = 3√5 + √5 = 4√5 ≈ 8.944" },
          { title: "Ordering note karo", detail: "‖A‖₂ ≤ ‖A‖_F ≤ ‖A‖*\n6.708 ≤ 7.071 ≤ 8.944 ✓\nYeh ordering hamesha sach hai." },
          { title: "trace identity verify", detail: "trace(AᵀA) = trace([[25,20],[20,25]]) = 50 = ‖A‖_F² ✓\nΣσᵢ² = Σaᵢⱼ² — ek elegant identity" },
        ],
        answer: "‖A‖_F = √50 ≈ 7.071, ‖A‖₂ = 3√5 ≈ 6.708, ‖A‖* = 4√5 ≈ 8.944.",
      },
      quiz: { q: "trace(AᵀA) kis ke barabar hai?", opts: ["Σσᵢ", "Σσᵢ² = ‖A‖_F²", "σ₁", "det(A)"], a: 1, why: "trace(AᵀA) = AᵀA ke eigenvalues ka sum = Σσᵢ² = saare entries ke squares ka sum = ‖A‖_F²." },
    },
    {
      id: "rankk", title: "Rank-k Approximation & Eckart–Young–Mirsky", vis: "rankk", body: [
        { t: "f", label: "Truncated SVD", x: "A_K = Σᵢ₌₁^K σᵢuᵢvᵢᵀ    (1 ≤ K ≤ r)", note: "**rank(A_K) = K.** Sirf top-K terms rakho — yeh low-rank (compact) approximation hai." },
        { t: "f", label: "Error E = A − A_k", x: "E = Σᵢ₌ₖ₊₁^r σᵢuᵢvᵢᵀ", note: "Discard kiya hua tail — jo pehle se hi SVD form mein hai, singular values σ_{k+1},...,σ_r ke saath." },
        { t: "table", head: ["Error norm", "Value", "Meaning"], rows: [
          ["‖A − A_k‖_F", "√(Σᵢ₌ₖ₊₁^r σᵢ²)", "Chhode gaye σ ke squares ka root"],
          ["‖A − A_k‖₂", "**σ_{k+1}**", "Bas agla singular value"],
          ["‖A − A_k‖*", "Σᵢ₌ₖ₊₁^r σᵢ", "Chhode gaye σ ka sum"],
        ]},
        { t: "f", label: "Eckart–Young–Mirsky Theorem ★", x: "Har B jiska rank(B) ≤ k ke liye:\n\n‖A − A_k‖  ≤  ‖A − B‖", note: "**A_k hi A ki BEST rank-k approximation hai** — koi doosri rank-k matrix isse nazdeek nahi ho sakti. Aur yeh Frobenius, spectral aur nuclear teenon norms ke liye ek saath sach hai." },
        { t: "f", label: "Geometric view — P = V_k V_kᵀ", x: "V_k = V ke pehle k columns.  P = V_kV_kᵀ", note: "P, span(v₁,...,v_k) par orthogonal projection hai (P² = P, Pᵀ = P). **AP = A_k**, aur A_k ke rows = A ke rows ki projections V_k subspace par." },
      ],
      ex: {
        problem: "A ke singular values 10, 6, 3, 1 hain. Rank-2 approximation ke teeno error norms nikalo, aur batao ki 90% Frobenius energy ke liye kitna k chahiye.",
        steps: [
          { title: "Values note karo", detail: "σ = 10, 6, 3, 1 (rank r = 4)\nk = 2 ⟹ σ₁, σ₂ rakhe; σ₃, σ₄ chhode" },
          { title: "Spectral error", detail: "‖A − A₂‖₂ = σ₃ = **3**\n(Bas agla singular value, aur kuch nahi)" },
          { title: "Frobenius error", detail: "‖A − A₂‖_F = √(σ₃² + σ₄²) = √(9 + 1) = √10 ≈ **3.162**" },
          { title: "Nuclear error", detail: "‖A − A₂‖* = σ₃ + σ₄ = 3 + 1 = **4**" },
          { title: "Total energy nikalo", detail: "‖A‖_F² = 100 + 36 + 9 + 1 = 146" },
          { title: "Energy fractions check karo", detail: "k=1: 100/146 = 68.5%\nk=2: 136/146 = 93.2% ✓ **90% cross ho gaya**\nk=3: 145/146 = 99.3%" },
          { title: "Eckart-Young ka matlab", detail: "Koi bhi rank-2 matrix B ke liye ‖A − B‖_F ≥ 3.162.\nA₂ hi optimal hai — isse behtar rank-2 approximation exist hi nahi karti." },
        ],
        answer: "‖A−A₂‖₂ = 3, ‖A−A₂‖_F = √10 ≈ 3.162, ‖A−A₂‖* = 4. 90% energy ke liye **k = 2** kaafi hai (93.2%).",
      },
      quiz: { q: "Rank-3 approximation ka spectral error kya hoga?", opts: ["σ₃", "σ₄", "σ₁+σ₂+σ₃", "√(σ₁²+σ₂²+σ₃²)"], a: 1, why: "‖A − A_k‖₂ = σ_{k+1}. k = 3 ke liye error = σ₄ — pehla chhoda hua singular value." },
    },
  ],
};

const CH10 = {
  id: "ch10", n: "10", title: "Partition (Block) Matrices",
  blurb: "Edge-to-edge cuts, conformability, block multiplication, transpose, determinant ke teen cases, eigenvalues aur inverse.",
  topics: [
    {
      id: "blockdef", title: "Definition, Cuts & Conformability", vis: "block", body: [
        { t: "p", x: "Block matrix wo matrix hai jise continuous horizontal aur vertical lines se **sub-matrices (blocks)** mein tod diya gaya ho. Poori matrix ko ek chhote grid ki tarah treat karte hain jiske 'elements' khud matrices hain." },
        { t: "list", x: [
          "**Divide & conquer:** badi computationally mehngi matrices ko chhote solvable tukdon mein todna.",
          "**Structure reveal:** block-diagonal ya block-triangular patterns dikh jaate hain, jinse inversion aur determinant bahut aasaan ho jaate hain.",
        ]},
        { t: "trap", label: "Edge-to-edge rule", x: "Partition lines **poori width/height** cover karni chahiye. Aadhe raste ruk jaane wale cuts allowed nahi — warna blocks well-defined rectangular matrices nahi rahenge aur block-level operations possible hi nahi hongi." },
        { t: "f", label: "Block addition", x: "[A B; C D] + [E F; G H] = [A+E  B+F; C+G  D+H]", note: "**Conformability:** dono matrices ka overall size same ho, AUR har corresponding sub-block pair identically sized ho — yaani cut lines **bilkul same indices** par hon." },
      ],
      ex: {
        problem: "M ek 5×5 matrix hai. Row cut 3 ke baad, column cut 2 ke baad. Chaaron blocks ke sizes batao aur check karo ki det formula lag sakta hai ya nahi.",
        steps: [
          { title: "Cuts note karo", detail: "Rows: 3 | 2  (upar 3 rows, neeche 2)\nColumns: 2 | 3  (baayen 2 cols, daayen 3)" },
          { title: "Block sizes", detail: "A = 3 × 2  (top-left)\nB = 3 × 3  (top-right)\nC = 2 × 2  (bottom-left)\nD = 2 × 3  (bottom-right)" },
          { title: "Det formula ki shart check karo", detail: "Formula ke liye **A aur D dono square** hone chahiye.\nA = 3×2 — square NAHI ✗\nD = 2×3 — square NAHI ✗" },
          { title: "Conclusion", detail: "Is partition par block determinant formulas **lag hi nahi sakte**." },
          { title: "Sahi partition kya hoti?", detail: "Row cut aur column cut **same jagah** par (dono 3 ke baad ya dono 2 ke baad):\nRow 2|3, Col 2|3 ⟹ A = 2×2 ✓, D = 3×3 ✓\nAb formula valid hai." },
        ],
        answer: "A(3×2), B(3×3), C(2×2), D(2×3). A aur D square nahi ⟹ det formula invalid. Row aur column cut same index par chahiye.",
      },
      quiz: { q: "Block determinant formulas ke liye kya zaroori hai?", opts: ["Saare blocks square hon", "A aur D (diagonal blocks) square hon", "B aur C zero hon", "Matrix symmetric ho"], a: 1, why: "Sirf diagonal blocks A aur D square hone chahiye. B aur C koi bhi conformable size ke ho sakte hain." },
    },
    {
      id: "blockmul", title: "Block Multiplication & Transpose", body: [
        { t: "f", label: "Block multiplication", x: "[A B; C D] · [E F; G H] = [AE+BG  AF+BH;  CE+DG  CF+DH]", note: "Bilkul normal matrix multiplication jaisa — blocks ko scalars ki tarah treat karo, bas dimensions conformable honi chahiye." },
        { t: "f", label: "The one rule", x: "Left matrix ka **column-cut** = Right matrix ka **row-cut**", note: "**Free:** left matrix ka row cut (output ki row structure decide karta hai) aur right matrix ka column cut. **MUST MATCH:** sirf inner partition." },
        { t: "trap", label: "Critical exam trap", x: "Overall sizes compatible hone ke bawajood (dono 4×4 ho), agar left ka internal col-cut aur right ka row-cut match nahi karte to block product **poori tarah fail** ho jaayega. **Hamesha inner partition verify karo.**" },
        { t: "f", label: "Block transpose — DO steps", x: "M = [A B; C D]  ⟹  Mᵀ = [Aᵀ Cᵀ; Bᵀ Dᵀ]", note: "1. Macro-grid mein blocks ki position swap karo: B (top-right) → bottom-left, C → top-right.\n2. **Har individual block ko bhi transpose karo** — yeh step sabse zyada bhoola jaata hai." },
      ],
      ex: {
        problem: "M = [[I₂, B], [0, I₂]] jahan B = [[1, 2], [3, 4]]. M², M⁻¹ aur Mᵀ nikalo.",
        steps: [
          { title: "M² — block multiplication", detail: "[I B; 0 I]·[I B; 0 I]\nTop-left: I·I + B·0 = I\nTop-right: I·B + B·I = 2B\nBottom-left: 0·I + I·0 = 0\nBottom-right: 0·B + I·I = I" },
          { title: "M² likho", detail: "M² = [I  2B; 0  I], jahan 2B = [[2,4],[6,8]]\nPattern: **M^k = [I  kB; 0  I]**" },
          { title: "M⁻¹ — upper triangular formula", detail: "[A B; 0 D]⁻¹ = [A⁻¹  −A⁻¹BD⁻¹; 0  D⁻¹]\nYahan A = D = I:\nM⁻¹ = [I  −B; 0  I]" },
          { title: "Verify M·M⁻¹ = I", detail: "[I B; 0 I]·[I −B; 0 I]\nTop-right: I(−B) + B(I) = −B + B = 0 ✓\nTop-left = I, bottom-right = I ✓" },
          { title: "Mᵀ — dono steps lagao", detail: "Step 1 — positions swap: [Iᵀ 0ᵀ; Bᵀ Iᵀ]\nStep 2 — har block transpose: Bᵀ = [[1,3],[2,4]]\nMᵀ = [[I, 0], [Bᵀ, I]] — lower triangular ban gaya" },
          { title: "Mᵀ explicit likho", detail: "Mᵀ = [[1,0,0,0], [0,1,0,0], [1,3,1,0], [2,4,0,1]]\nNote: B upper-right se lower-left aaya AUR transpose bhi hua." },
        ],
        answer: "M² = [I 2B; 0 I], M^k = [I kB; 0 I], M⁻¹ = [I −B; 0 I], Mᵀ = [I 0; Bᵀ I].",
      },
      quiz: { q: "Block transpose mein kya bhoola jaata hai?", opts: ["Blocks ki positions swap karna", "Har individual block ko bhi transpose karna", "Sizes check karna", "Sign badalna"], a: 1, why: "Do steps hain: grid mein positions swap karo AUR har block ko andar se transpose karo. Doosra step aksar chhoot jaata hai." },
    },
    {
      id: "blockdet", title: "Determinant of a Block Matrix — teen cases", body: [
        { t: "f", label: "General formulas", x: "D invertible ho:  det(M) = det(A − BD⁻¹C) · det(D)\nA invertible ho:  det(M) = det(D − CA⁻¹B) · det(A)", note: "(D − CA⁻¹B) ko **Schur complement** kehte hain. **Golden constraint:** A aur D dono square hone chahiye." },
        { t: "p", x: "**Proof (block Gaussian elimination):** [I 0; −CA⁻¹ I]·[A B; C D] = [A B; 0 D − CA⁻¹B]. Elimination matrix triangular hai with I on diagonal, isliye uska det = 1. Right side block-upper-triangular hai, to uska det = diagonal blocks ka product: det(M) = det(A)·det(D − CA⁻¹B)." },
        { t: "table", head: ["Case", "Condition", "Formula"], rows: [
          ["**I — Triangular block**", "C = 0 ya B = 0 (ya dono)", "det(M) = det(A) · det(D)"],
          ["**II — Commuting blocks**", "AC = CA", "det(M) = det(AD − CB)"],
          ["**II (variant)**", "BD = DB", "det(M) = det(DA − BC)"],
          ["**III — Symmetric pair**", "M = [A B; B A]", "det(M) = det(A + B)·det(A − B)"],
        ]},
        { t: "trap", label: "Critical exam trap", x: "**det(M) ≠ det(A)·det(D)** jab B aur C dono non-zero hon. Simple product formula sirf tab lagti hai jab **kam se kam ek off-diagonal block zero matrix ho**." },
        { t: "f", label: "Block eigenvalues", x: "M block-triangular (C = 0 ya B = 0)  ⟹  eigenvalues of M = (eigenvalues of A) ∪ (eigenvalues of D)", note: "**Proof:** det(M − λI) = det(A − λI)·det(D − λI) = 0 ⟹ ek ya doosra factor zero. **Yeh union formula sirf triangular case mein sach hai** — chaaron blocks non-zero ho to polynomials mix ho jaate hain." },
      ],
      ex: {
        problem: "M = [[A, B], [B, A]] jahan A = [[2, 0], [0, 2]] aur B = [[1, 0], [0, 1]]. det(M) aur eigenvalues nikalo.",
        steps: [
          { title: "Case III recognize karo", detail: "M = [A B; B A] — symmetric pair form\nFormula: det(M) = det(A + B) · det(A − B)" },
          { title: "A + B aur A − B", detail: "A + B = [[3, 0], [0, 3]] = 3I\nA − B = [[1, 0], [0, 1]] = I" },
          { title: "Determinants", detail: "det(A + B) = 9\ndet(A − B) = 1\ndet(M) = 9 × 1 = **9**" },
          { title: "Galat tareeke se check karo", detail: "Agar galti se det(A)·det(D) = 4 × 4 = 16 likh dete...\nWo GALAT hota, kyunki B ≠ 0 hai ✗" },
          { title: "Eigenvalues nikalo", detail: "M = [2I I; I 2I]. Vectors [x, x] aur [x, −x] try karo:\nM[x,x]ᵀ = [2x + x, x + 2x]ᵀ = 3[x,x]ᵀ ⟹ λ = 3 (twice)\nM[x,−x]ᵀ = [2x − x, x − 2x]ᵀ = 1·[x,−x]ᵀ ⟹ λ = 1 (twice)" },
          { title: "Verify", detail: "Eigenvalues: 3, 3, 1, 1\nΠλ = 9 = det(M) ✓\nΣλ = 8 = trace(M) = 2+2+2+2 ✓\nNote: eigenvalues = eig(A+B) ∪ eig(A−B) — symmetric pair ki khoobi." },
        ],
        answer: "det(M) = det(A+B)·det(A−B) = 9 × 1 = 9. Eigenvalues: 3, 3, 1, 1.",
      },
      quiz: { q: "M = [[A, B], [0, D]] jahan A ke eigenvalues 1, 2 aur D ke 3, 4 hain. M ke eigenvalues?", opts: ["1, 2, 3, 4", "4, 6", "B par depend karta hai", "Pata nahi"], a: 0, why: "Block-triangular hai (C = 0), isliye eigenvalues ka union: {1,2} ∪ {3,4}. B ka koi effect nahi." },
    },
    {
      id: "blockinv", title: "Inverse of Block Matrices", body: [
        { t: "f", label: "Case I — block diagonal", x: "M = [A 0; 0 D]  ⟹  M⁻¹ = [A⁻¹ 0; 0 D⁻¹]", note: "Verify: [A 0; 0 D][A⁻¹ 0; 0 D⁻¹] = [AA⁻¹ 0; 0 DD⁻¹] = [I 0; 0 I] ✓" },
        { t: "f", label: "Case II — block upper triangular", x: "M = [A B; 0 D]  ⟹  M⁻¹ = [A⁻¹  −A⁻¹BD⁻¹;  0  D⁻¹]" },
        { t: "num", x: [
          "Unknown X = [P Q; R S] rakho, MX = I solve karo.",
          "**Bottom-left:** 0·P + D·R = 0 ⟹ DR = 0 ⟹ **R = 0**.",
          "**Bottom-right:** 0·Q + D·S = I ⟹ **S = D⁻¹**.",
          "**Top-left:** AP + BR = I, aur R = 0 ⟹ AP = I ⟹ **P = A⁻¹**.",
          "**Top-right:** AQ + BS = 0 ⟹ AQ = −BD⁻¹ ⟹ **Q = −A⁻¹BD⁻¹**.",
        ]},
        { t: "trap", label: "Invertibility condition", x: "In formulas ke liye **A aur D dono square aur invertible** hone chahiye. Inverse exist karta hai ⟺ det(A) ≠ 0 **AUR** det(D) ≠ 0. Agar koi bhi diagonal block singular hai, to poori M singular hai — chahe B kuch bhi ho." },
      ],
      ex: {
        problem: "M = [[2, 0, 1, 3], [0, 2, 4, 1], [0, 0, 1, 0], [0, 0, 0, 1]] ka inverse nikalo (block form use karke).",
        steps: [
          { title: "Blocks identify karo", detail: "A = [[2, 0], [0, 2]] = 2I₂\nB = [[1, 3], [4, 1]]\nC = 0 (bottom-left)\nD = [[1, 0], [0, 1]] = I₂\n⟹ block **upper triangular**" },
          { title: "Invertibility check", detail: "det(A) = 4 ≠ 0 ✓\ndet(D) = 1 ≠ 0 ✓\n⟹ M invertible hai, aur det(M) = 4 × 1 = 4" },
          { title: "A⁻¹ aur D⁻¹", detail: "A⁻¹ = (1/2)I = [[0.5, 0], [0, 0.5]]\nD⁻¹ = I = [[1, 0], [0, 1]]" },
          { title: "Q = −A⁻¹BD⁻¹ nikalo", detail: "A⁻¹B = 0.5 · [[1,3],[4,1]] = [[0.5, 1.5], [2, 0.5]]\nBD⁻¹ part: I se multiply, koi change nahi\nQ = −[[0.5, 1.5], [2, 0.5]]" },
          { title: "M⁻¹ assemble karo", detail: "M⁻¹ = [[0.5, 0, −0.5, −1.5],\n       [0, 0.5, −2, −0.5],\n       [0, 0, 1, 0],\n       [0, 0, 0, 1]]" },
          { title: "Verify (row 1)", detail: "Row1 of M · Col1 of M⁻¹ = 2(0.5) + 0 + 1(0) + 3(0) = 1 ✓\nRow1 · Col3 = 2(−0.5) + 0(−2) + 1(1) + 3(0) = −1 + 1 = 0 ✓" },
        ],
        answer: "M⁻¹ = [[0.5, 0, −0.5, −1.5], [0, 0.5, −2, −0.5], [0, 0, 1, 0], [0, 0, 0, 1]], det(M) = 4.",
      },
      quiz: { q: "M = [[A, B], [0, D]] mein A singular hai. M ke baare mein?", opts: ["M phir bhi invertible ho sakti hai agar B accha ho", "M singular hai", "Depends on D", "M symmetric hai"], a: 1, why: "det(M) = det(A)·det(D) = 0·det(D) = 0. Koi bhi diagonal block singular ho to poori M singular." },
    },
  ],
};

const CH11 = {
  id: "ch11", n: "11", title: "Algorithm Complexity",
  blurb: "Forward elimination aur back substitution ki exact operation counts, aur O(n³) kyun.",
  topics: [
    {
      id: "complexity", title: "Cost of solving Ax = b", vis: "complexity", body: [
        { t: "p", x: "**Two-phase pipeline:** Ax = b → (forward elimination) → Ux = c → (back substitution) → solution x." },
        { t: "f", label: "Cost per pivot step", x: "Row op Rᵢ ← Rᵢ − k·Rⱼ (row length m+1, b column included):\n1 multiplication (k = aᵢⱼ/aⱼⱼ) + m multiplications (scale) + m additions (subtract)", note: "Pivot column k par bacha submatrix (n−k) × (n−k+1) hai. Cost ≈ 2(n−k)²." },
        { t: "f", label: "Forward elimination total", x: "Cost = 2 Σₖ₌₁ⁿ (k² − 1) = 2[n(n+1)(2n+1)/6 − n] = **(2n³ + 3n² − 5n)/3**", note: "Leading term ≈ (2/3)n³ — yahi dominate karta hai." },
        { t: "f", label: "Back substitution total", x: "Multiplications: 1 + 2 + ... + n = n(n+1)/2\nAdditions: 0 + 1 + ... + (n−1) = n(n−1)/2\nTotal = **n² operations** ⟹ O(n²)" },
        { t: "f", label: "Grand total", x: "Total = (2n³ + 3n² − 5n)/3 + n² = **(2n³ + 6n² − 5n)/3**  ⟹  **O(n³)**", note: "Back substitution ≈ n² hai jo n³ ke saamne bilkul negligible hai. Isliye complexity puri tarah forward pass se aati hai." },
        { t: "tip", x: "**LU ka asli fayda yahin dikhta hai:** ek baar A = LU nikal lo (⅔n³ cost), phir har naye b ke liye sirf do triangular solves karo (2n² cost). 100 alag b hon to elimination 100 baar karne ki jagah ek baar." },
      ],
      ex: {
        problem: "n = 100 ke liye forward elimination aur back substitution ki exact operation counts nikalo, aur ratio batao.",
        steps: [
          { title: "Forward elimination", detail: "(2n³ + 3n² − 5n)/3 with n = 100\n= (2,000,000 + 30,000 − 500)/3\n= 2,029,500/3 = **676,500 operations**" },
          { title: "Back substitution", detail: "n² = 100² = **10,000 operations**" },
          { title: "Ratio", detail: "10,000 / 676,500 ≈ 0.0148 = **1.48%**\nBack substitution total ka sirf 1.5% hai" },
          { title: "Total", detail: "(2n³ + 6n² − 5n)/3 = (2,000,000 + 60,000 − 500)/3\n= 2,059,500/3 = **686,500 operations**" },
          { title: "Approximation check", detail: "(2/3)n³ = (2/3)(10⁶) = 666,667\nActual 686,500 — sirf 3% farak\n⟹ (2/3)n³ ek accha estimate hai" },
          { title: "n = 1000 ke liye scaling", detail: "n 10× badha ⟹ cost 10³ = 1000× badhega\n≈ 6.7 × 10⁸ operations\nJabki back substitution sirf 10⁶ — ratio 0.15% reh gaya" },
        ],
        answer: "Forward = 676,500 ops, back = 10,000 ops (1.48%), total = 686,500 ≈ (2/3)n³. Complexity O(n³).",
      },
      quiz: { q: "Matrix size double karne par elimination cost kitni badhegi?", opts: ["2×", "4×", "8×", "16×"], a: 2, why: "Cost ∝ n³, isliye n → 2n par cost 2³ = 8 guna ho jaati hai." },
    },
  ],
};

const CONTENT = [CH1, CH2, CH3, CH4, CH5, CH6, CH7, CH8, CH9, CH10, CH11];

/* ============================== APP SHELL ============================== */
const FLAT = CONTENT.flatMap((c) => c.topics.map((t) => ({ ...t, ch: c })));

export default function App() {
  const [sel, setSel] = useState(null);          // topic id, null = home
  const [q, setQ] = useState("");
  const [openCh, setOpenCh] = useState(() => new Set(["ch1"]));
  const [done, setDone] = useState(() => new Set());
  const [navOpen, setNavOpen] = useState(false);
  const mainRef = useRef(null);

  const results = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return null;
    return FLAT.filter((t) =>
      t.title.toLowerCase().includes(s) ||
      t.ch.title.toLowerCase().includes(s) ||
      JSON.stringify(t.body).toLowerCase().includes(s)
    ).slice(0, 25);
  }, [q]);

  const idx = sel ? FLAT.findIndex((t) => t.id === sel) : -1;
  const topic = idx >= 0 ? FLAT[idx] : null;

  useEffect(() => { if (mainRef.current) mainRef.current.scrollTop = 0; }, [sel]);

  const go = (id) => { setSel(id); setNavOpen(false); setQ(""); const t = FLAT.find((x) => x.id === id); if (t) setOpenCh((s) => new Set(s).add(t.ch.id)); };
  const toggleCh = (id) => setOpenCh((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const markDone = (id) => setDone((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });

  const Vis = topic && topic.vis ? VISUALS[topic.vis] : null;
  const pct = Math.round((done.size / FLAT.length) * 100);

  return (
    <div className="flex h-screen w-full bg-stone-100 text-stone-900 font-sans overflow-hidden">
      {/* ---------- SIDEBAR ---------- */}
      <aside className={"fixed md:static z-30 h-full w-72 shrink-0 bg-white border-r border-stone-300 flex flex-col transition-transform " + (navOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0")}>
        <div className="px-4 py-3.5 border-b border-stone-200">
          <button onClick={() => { setSel(null); setNavOpen(false); }} className="text-left w-full">
            <div className="font-serif text-lg font-semibold leading-tight text-stone-900">Linear Algebra</div>
            <div className="text-xs text-stone-500 mt-0.5">GATE DA · complete visual handbook</div>
          </button>
          <div className="mt-3 relative">
            <Search size={14} className="absolute left-2.5 top-2.5 text-stone-400" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Topic search…"
              className="w-full pl-8 pr-2 py-1.5 text-sm bg-stone-100 border border-stone-200 rounded-lg focus:outline-none focus:border-indigo-400" />
          </div>
          <div className="mt-3">
            <div className="flex justify-between text-xs text-stone-500 mb-1">
              <span>{done.size} / {FLAT.length} topics done</span><span>{pct}%</span>
            </div>
            <div className="h-1.5 bg-stone-200 rounded-full overflow-hidden">
              <div className="h-full bg-emerald-500 transition-all" style={{ width: pct + "%" }} />
            </div>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto px-2 py-2">
          {results ? (
            <div>
              <div className="px-2 py-1 text-xs font-semibold text-stone-500">{results.length} results</div>
              {results.map((t) => (
                <button key={t.id} onClick={() => go(t.id)} className="w-full text-left px-2 py-1.5 rounded-lg hover:bg-stone-100">
                  <div className="text-sm text-stone-800">{t.title}</div>
                  <div className="text-xs text-stone-400">{t.ch.n} · {t.ch.title}</div>
                </button>
              ))}
              {results.length === 0 && <div className="px-2 py-2 text-sm text-stone-500">Kuch nahi mila.</div>}
            </div>
          ) : (
            CONTENT.map((c) => {
              const open = openCh.has(c.id);
              const cDone = c.topics.filter((t) => done.has(t.id)).length;
              return (
                <div key={c.id} className="mb-0.5">
                  <button onClick={() => toggleCh(c.id)} className="w-full flex items-center gap-1.5 px-2 py-2 rounded-lg hover:bg-stone-100 text-left">
                    {open ? <ChevronDown size={14} className="text-stone-400 shrink-0" /> : <ChevronRight size={14} className="text-stone-400 shrink-0" />}
                    <span className="text-xs font-mono text-stone-400">{c.n}</span>
                    <span className="text-sm font-medium text-stone-800 flex-1 leading-snug">{c.title}</span>
                    {cDone > 0 && <span className="text-xs text-emerald-600 font-medium">{cDone}/{c.topics.length}</span>}
                  </button>
                  {open && (
                    <div className="ml-5 border-l border-stone-200 pl-1">
                      {c.topics.map((t) => (
                        <button key={t.id} onClick={() => go(t.id)}
                          className={"w-full text-left px-2.5 py-1.5 rounded-md text-sm flex items-center gap-1.5 " +
                            (sel === t.id ? "bg-indigo-50 text-indigo-800 font-medium" : "text-stone-600 hover:bg-stone-100")}>
                          {done.has(t.id) && <Check size={12} className="text-emerald-600 shrink-0" />}
                          <span className="leading-snug">{t.title}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </nav>
      </aside>
      {navOpen && <div onClick={() => setNavOpen(false)} className="fixed inset-0 bg-black/30 z-20 md:hidden" />}

      {/* ---------- MAIN ---------- */}
      <main ref={mainRef} className="flex-1 overflow-y-auto">
        <div className="md:hidden sticky top-0 z-10 bg-white border-b border-stone-200 px-3 py-2 flex items-center gap-2">
          <button onClick={() => setNavOpen(true)} className="p-1.5 rounded-lg hover:bg-stone-100"><Menu size={18} /></button>
          <span className="text-sm font-medium">{topic ? topic.title : "Linear Algebra"}</span>
        </div>

        {!topic ? (
          <div className="max-w-3xl mx-auto px-5 md:px-10 py-10">
            <h1 className="font-serif text-4xl md:text-5xl font-semibold leading-tight text-stone-900">
              Linear Algebra, poori tarah<br />visual — GATE DA ke liye
            </h1>
            <p className="mt-4 text-stone-600 leading-relaxed max-w-2xl">
              {CONTENT.length} chapters, {FLAT.length} topics. Har topic mein concept, formulas, GATE traps,
              ek <b className="text-stone-900">step-by-step solved example</b>, aur jahan geometry madad karti hai wahan ek live interactive tool.
              Left sidebar se koi bhi topic khol lo — ya search karo.
            </p>
            <div className="mt-8 grid sm:grid-cols-2 gap-3">
              {CONTENT.map((c) => (
                <button key={c.id} onClick={() => { toggleCh(c.id); go(c.topics[0].id); }}
                  className="text-left p-4 rounded-xl border border-stone-300 bg-white hover:border-indigo-400 hover:shadow-sm transition">
                  <div className="flex items-baseline gap-2">
                    <span className="font-mono text-xs text-indigo-500">{c.n}</span>
                    <span className="font-semibold text-stone-900 leading-snug">{c.title}</span>
                  </div>
                  <div className="text-sm text-stone-500 mt-1.5 leading-relaxed">{c.blurb}</div>
                  <div className="text-xs text-stone-400 mt-2">{c.topics.length} topics</div>
                </button>
              ))}
            </div>
            <div className="mt-8 rounded-xl border border-amber-300 bg-amber-50 p-4">
              <div className="font-semibold text-amber-900 text-sm mb-1">Padhne ka tareeka</div>
              <p className="text-sm text-stone-700 leading-relaxed">
                Pehle concept + formulas padho, phir <b>solved example ke steps ek-ek karke kholo</b> (pehle khud try karo,
                phir step reveal karo). Visual tool mein apni values daal kar dekho ki numbers kaise badalte hain.
                Ant mein quick-check attempt karo aur topic ko done mark kar do.
              </p>
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-5 md:px-10 py-8 pb-24">
            <div className="flex items-center gap-2 text-xs text-stone-500 mb-2">
              <span className="font-mono">{topic.ch.n}</span>
              <ChevronRight size={12} />
              <span>{topic.ch.title}</span>
            </div>
            <h1 className="font-serif text-3xl font-semibold text-stone-900 leading-tight mb-1">{topic.title}</h1>
            <div className="flex items-center gap-2 mb-6">
              <button onClick={() => markDone(topic.id)}
                className={"text-xs px-2.5 py-1 rounded-full border flex items-center gap-1 " +
                  (done.has(topic.id) ? "bg-emerald-50 border-emerald-400 text-emerald-700" : "bg-white border-stone-300 text-stone-600 hover:border-stone-500")}>
                <Check size={12} /> {done.has(topic.id) ? "Done" : "Mark as done"}
              </button>
              <span className="text-xs text-stone-400">Topic {idx + 1} of {FLAT.length}</span>
            </div>

            <Blocks list={topic.body} />
            {Vis && <Vis />}
            <Example ex={topic.ex} />
            <Quiz q={topic.quiz} />

            <div className="mt-10 flex justify-between gap-3 border-t border-stone-200 pt-5">
              {idx > 0 ? (
                <button onClick={() => go(FLAT[idx - 1].id)} className="text-left flex-1 p-3 rounded-lg border border-stone-300 bg-white hover:border-stone-500">
                  <div className="text-xs text-stone-400">← Pichla</div>
                  <div className="text-sm text-stone-800 font-medium leading-snug">{FLAT[idx - 1].title}</div>
                </button>
              ) : <div className="flex-1" />}
              {idx < FLAT.length - 1 ? (
                <button onClick={() => go(FLAT[idx + 1].id)} className="text-right flex-1 p-3 rounded-lg border border-stone-300 bg-white hover:border-stone-500">
                  <div className="text-xs text-stone-400">Agla →</div>
                  <div className="text-sm text-stone-800 font-medium leading-snug">{FLAT[idx + 1].title}</div>
                </button>
              ) : <div className="flex-1" />}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
