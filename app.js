// NPP Digital Twin — NPPAD accident diagnosis (Task E) — BETTER-RESULTS build.
// Standalone, offline. Diagnosis: the TaskE_NPPAD (4) package's four
// classifiers — logistic regression, logistic regression +, 1D CNN and GRU
// (panel switcher; CNN/GRU probabilities averaged over seeds 0/1/2) — one
// prediction per evaluated window (W = 30/60/120/300/600 s), shown as
// window snapshots, not per-step scores.
// Default setting: onset-aligned windows (t0 = 100 s), realistic
// instrumentation. Scene data: the same 12 held-out test runs, t = 0..700 s.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const $ = id => document.getElementById(id);
const clamp01 = v => Math.max(0, Math.min(1, v));
const lerp = (a, b, f) => a + (b - a) * f;

/* ---------------- state ---------------- */
const S = {
  manifest: null, run: null, runMeta: null,
  frame: 0, playing: true, speed: 1, scrubbing: false,
  reveal: false, chartSel: [], lastShownSec: -1, toastShown: false,
  setting: 't0100_realistic', // active package setting (key in manifest.settings)
  model: 'CNN', // active diagnosis model (key in manifest.models)
  bases: {}, // per-channel pre-onset baseline of the loaded run
};

/* Diagnosis: the package's four classifiers, reported once per evaluated
   window W. A snapshot for window W becomes available at t = t0 + W and
   stays on the panel until the next window completes. CNN probabilities
   live in run.diag (the original bake); the other models in run.diagModels. */
const WINDOWS = [30, 60, 120, 300, 600];
function settingMeta() {
  return S.manifest ? S.manifest.settings[S.setting] : null;
}
function modelMeta() {
  return S.manifest ? S.manifest.models[S.model] : null;
}
function diagTable(setting) {
  if (!S.run) return null;
  return S.model === 'CNN' ? S.run.diag[setting] : S.run.diagModels[S.model][setting];
}
function currentSnapshot() {
  if (!S.run || !S.manifest) return null;
  const t = Math.round(S.frame) * 10;
  const t0 = settingMeta().t0;
  let best = null;
  for (const W of WINDOWS) if (t0 + W <= t) best = W;
  return best ? { W: best, row: diagTable(S.setting)[String(best)] } : null;
}
/* ---------------- data ---------------- */
async function loadManifest() {
  const r = await fetch('data/manifest.json');
  S.manifest = await r.json();
}
async function loadRun(id) {
  const meta = S.manifest.runs.find(r => r.id === id);
  const r = await fetch('data/' + meta.data);
  S.run = await r.json();
  S.runMeta = meta;
  S.bases = {};
  for (const code of Object.keys(S.run.channels)) {
    const s = S.run.channels[code];
    S.bases[code] = (s[0] + s[1] + s[2]) / 3; // t = 0,10,20 s: steady state
  }
  S.frame = 0; S.toastShown = false;
  document.querySelectorAll('.run-chip').forEach(ch =>
    ch.classList.toggle('active', ch.dataset.id === id));
  $('hudRun').textContent = `${S.run.class} · ${S.run.file} · sev ${S.run.severity}`;
  buildChips();
  S.lastShownSec = -1;
  updateAll(true);
}
function val(code, frameFloat = S.frame) {
  const s = S.run.channels[code];
  const i = Math.max(0, Math.min(s.length - 1, Math.floor(frameFloat)));
  const j = Math.min(s.length - 1, i + 1);
  return lerp(s[i], s[j], frameFloat - Math.floor(frameFloat));
}
function frac(code) { // fractional change vs the run's own steady state
  const b = S.bases[code];
  if (!b || Math.abs(b) < 1e-9) return 0;
  return (val(code) - b) / Math.abs(b);
}

/* ---------------- three.js scene ---------------- */
const container = $('sceneContainer');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.xr.enabled = true;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1220);
scene.fog = new THREE.Fog(0x0b1220, 55, 120);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 300);
camera.position.set(22, 15, 25);
const dolly = new THREE.Group();
dolly.add(camera);
scene.add(dolly);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 5.5, 0);
controls.enableDamping = true;
controls.maxDistance = 80;

scene.add(new THREE.HemisphereLight(0xbdd5ff, 0x1a2436, 0.85));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(25, 35, 18);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x88aaff, 0.5);
fill.position.set(-20, 12, -15);
scene.add(fill);

const MAT = {
  steel:  new THREE.MeshStandardMaterial({ color: 0x93a9c4, metalness: 0.65, roughness: 0.35 }),
  dark:   new THREE.MeshStandardMaterial({ color: 0x54657e, metalness: 0.6, roughness: 0.5 }),
  hot:    new THREE.MeshStandardMaterial({ color: 0xc7a17a, metalness: 0.55, roughness: 0.4 }),
  cold:   new THREE.MeshStandardMaterial({ color: 0x7f9fc4, metalness: 0.55, roughness: 0.4 }),
  feed:   new THREE.MeshStandardMaterial({ color: 0x5f8f6e, metalness: 0.5, roughness: 0.45 }),
  water:  new THREE.MeshStandardMaterial({ color: 0x2f7fd0, transparent: true, opacity: 0.78,
                                           emissive: 0x0a2f55, roughness: 0.2 }),
  shell:  new THREE.MeshPhysicalMaterial({ color: 0x9fc4e8, transparent: true, opacity: 0.09,
                                           roughness: 0.15, metalness: 0, side: THREE.DoubleSide,
                                           depthWrite: false }),
  floor:  new THREE.MeshStandardMaterial({ color: 0x131e30, roughness: 0.95 }),
};
function mesh(geo, mat, x = 0, y = 0, z = 0, parent = scene) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  parent.add(m);
  return m;
}
function tube(p1, p2, r, mat, parent = scene) {
  const a = new THREE.Vector3(...p1), b = new THREE.Vector3(...p2);
  const d = b.clone().sub(a), len = d.length();
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 14), mat);
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  parent.add(m);
  return m;
}
function makeLabel(text, pos, scale = 1) {
  const cv = document.createElement('canvas');
  cv.width = 512; cv.height = 112;
  const g = cv.getContext('2d');
  g.font = '600 46px "Segoe UI", sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.shadowColor = 'rgba(0,0,0,0.9)'; g.shadowBlur = 10;
  g.fillStyle = '#dbe7ff';
  g.fillText(text, 256, 56);
  const tex = new THREE.CanvasTexture(cv);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false,
                                                         transparent: true }));
  sp.scale.set(7.2 * scale, 1.58 * scale, 1);
  sp.position.set(...pos);
  scene.add(sp);
  return sp;
}

/* --- static plant --- */
mesh(new THREE.CylinderGeometry(17.5, 17.5, 1, 56), MAT.floor, 0, 0.5, 0);       // base slab
mesh(new THREE.CylinderGeometry(15, 15, 13, 56, 1, true), MAT.shell, 0, 7.5, 0); // containment wall
const dome = mesh(new THREE.SphereGeometry(15, 56, 24, 0, Math.PI * 2, 0, Math.PI / 2),
                  MAT.shell, 0, 14, 0);
dome.scale.y = 0.42;

/* reactor vessel + core + rods */
mesh(new THREE.CylinderGeometry(2.3, 2.3, 6, 28), MAT.steel, 0, 5, 0);
const vBottom = mesh(new THREE.SphereGeometry(2.3, 28, 14, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), MAT.steel, 0, 2, 0);
mesh(new THREE.SphereGeometry(2.3, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2), MAT.steel, 0, 8, 0);
const coreMat = new THREE.MeshStandardMaterial({ color: 0x2a6db0, emissive: 0x1a4a80,
                                                   emissiveIntensity: 0.4, roughness: 0.4 });
const coreMesh = mesh(new THREE.CylinderGeometry(1.55, 1.55, 3.4, 24), coreMat, 0, 4.1, 0);
const rods = new THREE.Group();
for (const [rx, rz] of [[0, 0], [0.8, 0], [-0.8, 0], [0, 0.8], [0, -0.8], [0.57, 0.57], [-0.57, -0.57]]) {
  const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 5.2, 8), MAT.dark);
  rod.position.set(rx, 10.6, rz);
  rods.add(rod);
}
const rodDrive = mesh(new THREE.CylinderGeometry(1.15, 1.15, 0.5, 20), MAT.dark, 0, 13.2, 0);
rods.add(rodDrive);
scene.add(rods);

/* pressurizer */
mesh(new THREE.CylinderGeometry(1.3, 1.5, 1.2, 20), MAT.dark, 6.8, 1.6, -4.5);
mesh(new THREE.CylinderGeometry(1.15, 1.15, 6.4, 22), MAT.steel, 6.8, 5.4, -4.5);
mesh(new THREE.SphereGeometry(1.15, 22, 12, 0, Math.PI * 2, 0, Math.PI / 2), MAT.steel, 6.8, 8.6, -4.5);
const pzWater = mesh(new THREE.CylinderGeometry(0.94, 0.94, 1, 18), MAT.water.clone(), 6.8, 2.4, -4.5);

/* steam generators + pumps, loops A (z<0) and B (z>0) */
const sgWater = {}, sgPos = { A: [-8.5, -5.5], B: [-8.5, 5.5] };
for (const L of ['A', 'B']) {
  const z = sgPos[L][1];
  mesh(new THREE.CylinderGeometry(1.9, 1.9, 8.6, 26), MAT.steel, -8.5, 5.3, z);
  mesh(new THREE.SphereGeometry(1.9, 26, 12, 0, Math.PI * 2, 0, Math.PI / 2), MAT.steel, -8.5, 9.6, z);
  mesh(new THREE.CylinderGeometry(2.0, 2.1, 0.9, 26), MAT.dark, -8.5, 1.45, z);
  sgWater[L] = mesh(new THREE.CylinderGeometry(1.62, 1.62, 1, 20), MAT.water.clone(), -8.5, 2, z);
  // reactor coolant pump
  mesh(new THREE.CylinderGeometry(1.05, 1.05, 1.3, 20), MAT.dark, -4.2, 2.1, z);
  mesh(new THREE.CylinderGeometry(0.55, 0.55, 1.7, 14), MAT.steel, -4.2, 3.6, z);
  // hot leg (vessel -> SG), cold leg (SG -> pump -> vessel)
  tube([0, 6.4, z * 0.3], [-8.5, 6.9, z], 0.5, MAT.hot);
  tube([-8.5, 2.7, z], [-4.2, 2.5, z], 0.5, MAT.cold);
  tube([-4.2, 2.5, z], [-0.4, 3.1, z * 0.28], 0.5, MAT.cold);
  // steam line (SG top -> header -> turbine)
  tube([-8.5, 10.4, z], [-8.5, 12.4, z], 0.42, MAT.steel);
  tube([-8.5, 12.4, z], [9, 12.4, z * 0.45], 0.42, MAT.steel);
  tube([9, 12.4, z * 0.45], [18.5, 7.5, 0], 0.42, MAT.steel);
  // feedwater line (turbine hall -> SG)
  tube([16.5, 3.4, z * 0.55], [-8.5, 4.7, z], 0.3, MAT.feed);
}
/* surge line: pressurizer -> hot leg A */
tube([6.8, 2.3, -4.5], [1.5, 5.2, -3.6], 0.22, MAT.cold);
tube([1.5, 5.2, -3.6], [-3.2, 6.55, -4.3], 0.22, MAT.cold);

/* turbine hall */
mesh(new THREE.BoxGeometry(6.5, 4.6, 7.5), MAT.dark, 20.5, 3.3, 0);
mesh(new THREE.CylinderGeometry(1.1, 1.1, 5.4, 18), MAT.steel, 20.5, 6.4, 0)
  .rotation.z = Math.PI / 2;

/* labels */
makeLabel('Reactor vessel', [0, 15.6, 0]);
makeLabel('Pressurizer', [6.8, 11.4, -4.5], 0.85);
makeLabel('Steam generator A', [-8.5, 13.6, -5.5], 0.85);
makeLabel('Steam generator B', [-8.5, 13.6, 5.5], 0.85);
makeLabel('RCP A', [-4.2, 5.9, -5.5], 0.6);
makeLabel('RCP B', [-4.2, 5.9, 5.5], 0.6);
makeLabel('Turbine hall', [20.5, 9.6, 0], 0.85);
makeLabel('Containment', [12.8, 17.6, 7.5], 0.9);

/* break marker + light + steam particles */
const breakMarker = new THREE.Mesh(new THREE.SphereGeometry(0.62, 18, 14),
  new THREE.MeshBasicMaterial({ color: 0xff4433, transparent: true, opacity: 0 }));
breakMarker.visible = false;
scene.add(breakMarker);
const breakLight = new THREE.PointLight(0xff5533, 0, 26);
scene.add(breakLight);
const BREAK_POS = {
  rcs:   [-1.7, 3.2, -3.1],
  sgA:   [-8.5, 5.6, -5.5],  sgB: [-8.5, 5.6, 5.5],
  steamA: [0, 12.4, -3.6],   steamB: [0, 12.4, 3.6],
  steamOut: [16.8, 8.8, 0],
  feedA: [2.5, 4.1, -5.1],   feedB: [2.5, 4.1, 5.1],
  core:  [0, 11.2, 0],
  none:  [20.5, 6.2, 0],
};
const P_N = 260;
const pGeo = new THREE.BufferGeometry();
const pPos = new Float32Array(P_N * 3).fill(-50);
pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
  color: 0xdfeaff, size: 0.55, transparent: true, opacity: 0.75, depthWrite: false }));
particles.frustumCulled = false;
scene.add(particles);
const pState = Array.from({ length: P_N }, () => ({ life: 0, vx: 0, vy: 0, vz: 0 }));
let pCursor = 0;

function breakInfo() {
  const b = S.run.break;
  if (b.kind === 'rcs') return { pos: BREAK_POS.rcs,
    I: clamp01(Math.max(Math.abs(frac('P')) * 5, Math.abs(frac('LVPZ')) * 2.5)) };
  if (b.kind === 'sg') { const L = b.loop; return { pos: BREAK_POS['sg' + L],
    I: clamp01(Math.abs(frac('LSG' + L)) * 3.5 + Math.abs(frac('WST' + L)) * 1.5) }; }
  if (b.kind === 'steam') { const L = b.loop; return { pos: BREAK_POS['steam' + L],
    I: clamp01(Math.abs(frac('WST' + L)) * 3) }; }
  if (b.kind === 'steamout') return { pos: BREAK_POS.steamOut,
    I: clamp01(Math.max(Math.abs(frac('WSTA')), Math.abs(frac('WSTB'))) * 3) };
  if (b.kind === 'feed') { const L = b.loop; return { pos: BREAK_POS['feed' + L],
    I: clamp01(Math.abs(frac('WFW' + L)) * 3) }; }
  if (b.kind === 'core') return { pos: BREAK_POS.core,
    I: clamp01(Math.abs(frac('PWR')) * 2.2) };
  return { pos: BREAK_POS.none,
    I: clamp01(Math.abs(frac('WSTA')) * 1.5 + Math.abs(frac('PWR')) * 0.8) };
}

/* drive the 3D plant from the current frame's channel values */
function updatePlant(now) {
  // water levels
  const setFill = (m, fracFill, baseY, maxH) => {
    const h = Math.max(0.06, fracFill * maxH);
    m.scale.y = h;
    m.position.y = baseY + h / 2;
  };
  setFill(sgWater.A, clamp01(val('LSGA') / 23), 1.95, 6.4);
  setFill(sgWater.B, clamp01(val('LSGB') / 23), 1.95, 6.4);
  setFill(pzWater, clamp01(val('LVPZ') / 100), 2.35, 5.6);
  const pDev = clamp01(Math.abs(frac('P')) * 4);
  pzWater.material.emissive.setRGB(0.04 + pDev * 0.5, 0.18 - pDev * 0.1, 0.33 - pDev * 0.2);
  // core heat + rods
  const heat = clamp01(frac('TPCT') * 4 + Math.max(0, val('VOID')) * 5 +
                       Math.max(0, frac('PWR')) * 1.2);
  coreMat.emissive.setRGB(lerp(0.10, 1.0, heat), lerp(0.29, 0.30, heat), lerp(0.50, 0.08, heat));
  coreMat.emissiveIntensity = 0.35 + heat * 1.9;
  const rodIn = clamp01(0.5 - frac('PWR') * 1.5);
  rods.position.y = lerp(1.7, -1.9, rodIn);
  // break marker + particles
  const { pos, I } = breakInfo();
  breakMarker.position.set(...pos);
  breakLight.position.set(...pos);
  const active = I > 0.04 && S.frame >= 2;
  breakMarker.visible = active;
  if (active) {
    const pulse = 0.8 + 0.35 * Math.sin(now * 0.011);
    breakMarker.scale.setScalar(pulse * (0.6 + I));
    breakMarker.material.opacity = 0.3 + 0.7 * I;
    breakLight.intensity = 60 * I;
    if (I > 0.12 && Math.random() < I * 0.85) {
      const p = pState[pCursor];
      p.life = 1;
      p.vx = (Math.random() - 0.5) * 0.05; p.vy = 0.06 + Math.random() * 0.08;
      p.vz = (Math.random() - 0.5) * 0.05;
      pPos[pCursor * 3] = pos[0]; pPos[pCursor * 3 + 1] = pos[1]; pPos[pCursor * 3 + 2] = pos[2];
      pCursor = (pCursor + 1) % P_N;
    }
  } else breakLight.intensity = 0;
  for (let i = 0; i < P_N; i++) {
    const p = pState[i];
    if (p.life <= 0) continue;
    p.life -= 0.012;
    pPos[i * 3] += p.vx; pPos[i * 3 + 1] += p.vy; pPos[i * 3 + 2] += p.vz;
    if (p.life <= 0) pPos[i * 3 + 1] = -50;
  }
  pGeo.attributes.position.needsUpdate = true;
}

/* ---------------- UI: run bar, diagnosis, tiles ---------------- */
function buildRunBar() {
  const bar = $('runBar');
  bar.innerHTML = '';
  for (const r of S.manifest.runs) {
    const b = document.createElement('button');
    b.className = 'run-chip';
    b.dataset.id = r.id;
    b.innerHTML = `${r.class}<small>sev ${r.severity}</small>`;
    b.title = `Test run ${r.file}`;
    b.onclick = () => { loadRun(r.id); S.playing = true; syncPlayBtn(); };
    bar.appendChild(b);
  }
}
function updateDiagnosis() {
  const t = Math.round(S.frame) * 10;
  const meta = settingMeta();
  const snap = S.run ? currentSnapshot() : null;
  if (!snap) {
    $('diagIdle').classList.remove('hidden');
    $('diagBars').classList.add('hidden');
    if (S.run && meta) {
      const e0 = meta.t0 + WINDOWS[0];
      $('diagIdle').innerHTML =
        'Collecting the first diagnosis window.<br>' +
        `The ${modelMeta().label} reports one diagnosis per window — the W = 30 s snapshot ` +
        `completes at t = ${e0} s (window ${meta.t0}–${e0} s).`;
    }
    $('hudW').textContent = 'W = —';
  } else {
    const { W, row } = snap;
    const order = row.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    $('diagIdle').classList.add('hidden');
    $('diagBars').classList.remove('hidden');
    order.slice(0, 3).forEach(([p, ci], r) => {
      const rowEl = $('barRow' + r);
      rowEl.classList.toggle('top', r === 0);
      rowEl.querySelector('.bar-label').textContent = S.manifest.classes[ci];
      rowEl.querySelector('.bar-fill').style.width = (p * 100).toFixed(1) + '%';
      rowEl.querySelector('.bar-val').textContent = (p * 100).toFixed(0) + '%';
    });
    $('hudW').textContent = `W = ${W} s`;
    const mm = modelMeta();
    $('snapNote').textContent =
      `Snapshot W = ${W} s · window ${meta.t0}–${meta.t0 + W} s · ${mm.label} — ${mm.note} · ` +
      'top-3 of 12 accident classes. ' +
      'The panel updates at the evaluated windows only (W = 30/60/120/300/600 s).';
    const pred = S.manifest.classes[order[0][1]];
    if (S.reveal) {
      $('truthLabel').textContent = `${S.run.class} (sev ${S.run.severity})`;
      const ok = pred === S.run.class;
      $('verdictLabel').textContent = ok ? `correct — ${pred}` : `model says ${pred}`;
      $('verdictLabel').className = ok ? 'ok' : 'no';
    }
  }
  $('hudPhase').textContent =
    t < 20 ? 'Steady state'
    : t < 110 ? 'Accident inserted (20 s) — plant response not yet visible'
    : t <= 400 ? 'Transient — response developing' : 'Transient — developed';
  $('hudTime').textContent = `t = ${t} s`;
  $('clockLabel').textContent = `t = ${t} s`;
}
function setSetting(key) {
  if (!S.manifest || !S.manifest.settings[key]) return;
  S.setting = key;
  const sel = $('settingSel');
  if (sel) sel.value = key;
  const meta = S.manifest.settings[key];
  $('settingLabel').textContent = meta.label;
  const mm = modelMeta();
  $('f1Line').textContent = `Test macro-F1, ${mm.label} (package report, ` +
    'results/macroF1_vs_W.csv): ' +
    WINDOWS.map(W => `W=${W} ${mm.f1[key][String(W)].toFixed(3)}`).join(' · ');
  updateDiagnosis();
  drawChart();
}
function setModel(key) {
  if (!S.manifest || !S.manifest.models[key]) return;
  S.model = key;
  const sel = $('modelSel');
  if (sel) sel.value = key;
  $('modelTag').textContent = modelMeta().tag;
  setSetting(S.setting);
}
function buildModels() {
  const sel = $('modelSel');
  sel.innerHTML = '';
  for (const key of S.manifest.model_order) {
    const o = document.createElement('option');
    o.value = key; o.textContent = S.manifest.models[key].label;
    sel.appendChild(o);
  }
  setModel(S.manifest.default_model);
}
function buildSettings() {
  const sel = $('settingSel');
  sel.innerHTML = '';
  for (const [key, meta] of Object.entries(S.manifest.settings)) {
    const o = document.createElement('option');
    o.value = key; o.textContent = meta.label;
    sel.appendChild(o);
  }
  setSetting(S.manifest.default_setting);
}
function updateTiles() {
  const f = (code, d = 1) => val(code).toFixed(d);
  $('tileP').textContent = f('P');
  $('tileLVPZ').textContent = f('LVPZ');
  $('tilePWR').textContent = f('PWR');
  $('tileDNBR').textContent = f('DNBR', 2);
  $('tileLSGA').textContent = f('LSGA', 2);
  $('tileLSGB').textContent = f('LSGB', 2);
}

/* ---------------- channel chart (canvas 2D) ---------------- */
const CHART_COLORS = ['#4da3ff', '#ffb454', '#53d769', '#ff5d5d', '#c792ff', '#4dd0e1',
                      '#ff8a5c', '#9ccc65', '#f06292', '#ffd54f', '#64b5f6', '#a1887f',
                      '#7ee0c2', '#e6ee9c', '#ffab91', '#b39ddb', '#90caf9'];
function chColor(code) {
  return CHART_COLORS[Object.keys(S.manifest.channels).indexOf(code) % CHART_COLORS.length];
}
function buildChips() {
  const wrap = $('chartChips');
  wrap.innerHTML = '';
  if (!S.chartSel.length) S.chartSel = [...S.manifest.default_chart_channels];
  for (const code of Object.keys(S.manifest.channels)) {
    const meta = S.manifest.channels[code];
    const chip = document.createElement('button');
    chip.className = 'chip' + (S.chartSel.includes(code) ? ' on' : '');
    chip.dataset.code = code;
    chip.style.color = S.chartSel.includes(code) ? chColor(code) : '';
    chip.innerHTML = `<span class="swatch"></span>${code}`;
    chip.title = meta.name + (meta.unit ? ` (${meta.unit})` : '');
    chip.onclick = () => {
      const i = S.chartSel.indexOf(code);
      if (i >= 0) S.chartSel.splice(i, 1);
      else { S.chartSel.push(code); if (S.chartSel.length > 6) S.chartSel.shift(); }
      buildChips(); drawChart();
    };
    wrap.appendChild(chip);
  }
}
function drawChart() {
  const cv = $('chart');
  const dpr = Math.min(devicePixelRatio, 2);
  const W = cv.clientWidth, H = cv.clientHeight;
  if (!W) return;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const padL = 10, padR = 10, padT = 10, padB = 18;
  const iw = W - padL - padR, ih = H - padT - padB;
  const x = t => padL + (t / 700) * iw;
  g.strokeStyle = '#22344f'; g.fillStyle = '#8fa8c8'; g.font = '10px sans-serif';
  for (const t of [0, 175, 350, 525, 700]) {
    g.beginPath(); g.moveTo(x(t), padT); g.lineTo(x(t), padT + ih); g.stroke();
    g.fillText(t + ' s', x(t) - 10, H - 5);
  }
  if (!S.run) return;
  // handout onset line (20 s) + the active setting's window origin (t0)
  g.strokeStyle = 'rgba(255,93,93,0.75)'; g.setLineDash([4, 4]);
  g.beginPath(); g.moveTo(x(20), padT); g.lineTo(x(20), padT + ih); g.stroke();
  g.setLineDash([]);
  g.fillStyle = '#ff9c9c'; g.fillText('onset 20 s', x(20) + 4, padT + 10);
  const t0 = settingMeta() ? settingMeta().t0 : 100;
  if (t0 !== 20) {
    g.strokeStyle = 'rgba(120,200,255,0.8)'; g.setLineDash([2, 3]);
    g.beginPath(); g.moveTo(x(t0), padT); g.lineTo(x(t0), padT + ih); g.stroke();
    g.setLineDash([]);
    g.fillStyle = '#9fd8ff'; g.fillText(`window origin t0 = ${t0} s`, x(t0) + 4, padT + 22);
  }
  const times = S.run.times;
  for (const code of S.chartSel) {
    const s = S.run.channels[code];
    let mn = Math.min(...s), mx = Math.max(...s);
    if (mx - mn < 1e-9) { mx += 0.5; mn -= 0.5; }
    g.strokeStyle = chColor(code); g.lineWidth = 1.6;
    g.beginPath();
    s.forEach((v, i) => {
      const px = x(times[i]), py = padT + ih - ((v - mn) / (mx - mn)) * ih;
      i ? g.lineTo(px, py) : g.moveTo(px, py);
    });
    g.stroke(); g.lineWidth = 1;
    const cur = val(code);
    g.fillStyle = chColor(code);
    g.fillText(`${code} ${cur.toFixed(2)}`, padL + 4, padT + 12 + S.chartSel.indexOf(code) * 13);
  }
  // playback cursor
  const cx = x(S.frame * 10);
  g.strokeStyle = 'rgba(255,255,255,0.85)';
  g.beginPath(); g.moveTo(cx, padT); g.lineTo(cx, padT + ih); g.stroke();
}

/* ---------------- transport + main loop ---------------- */
function syncPlayBtn() { $('playBtn').innerHTML = S.playing ? '&#10074;&#10074;' : '&#9654;'; }
$('playBtn').onclick = () => { S.playing = !S.playing; syncPlayBtn(); };
$('restartBtn').onclick = () => { S.frame = 0; S.toastShown = false; updateAll(true); };
$('speedSel').onchange = e => { S.speed = parseFloat(e.target.value); };
const scrub = $('scrub');
scrub.addEventListener('input', () => {
  S.frame = parseFloat(scrub.value);
  S.scrubbing = true;
  updateAll(true);
});
scrub.addEventListener('change', () => { S.scrubbing = false; });
$('revealToggle').onchange = e => {
  S.reveal = e.target.checked;
  $('truthBox').classList.toggle('hidden', !S.reveal);
  updateDiagnosis();
};
$('settingSel').onchange = e => setSetting(e.target.value);
$('modelSel').onchange = e => setModel(e.target.value);
function updateAll(force = false) {
  if (!S.run) return;
  updatePlant(performance.now());
  const sec = Math.floor(S.frame * 10);
  if (force || sec !== S.lastShownSec) {
    S.lastShownSec = sec;
    if (!S.scrubbing) scrub.value = S.frame;
    updateDiagnosis();
    updateTiles();
    drawChart();
  }
}

const clock = new THREE.Clock();
function tick() {
  const dt = Math.min(clock.getDelta(), 0.5); // faithful sim clock even on slow frames
  if (S.playing && !S.scrubbing && S.run) {
    const prev = S.frame;
    S.frame = Math.min(70, S.frame + dt * 2 * S.speed);
    if (S.frame >= 70) { S.playing = false; syncPlayBtn(); }
    if (prev < 11 && S.frame >= 11 && !S.toastShown) {
      S.toastShown = true;
      const toast = $('onsetToast');
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 1700);
    }
  }
  controls.update();
  updateAll();
  renderer.render(scene, camera);
}
renderer.setAnimationLoop(tick);

function resize() {
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  drawChart();
}
new ResizeObserver(resize).observe(container);
new ResizeObserver(drawChart).observe($('chart'));
window.addEventListener('resize', resize);

/* ---------------- WebXR (HydroTwin pattern: session on the same renderer) --- */
const vrBtn = $('vrButton');
if (navigator.xr && navigator.xr.isSessionSupported) {
  navigator.xr.isSessionSupported('immersive-vr').then(ok => {
    vrBtn.disabled = !ok;
    vrBtn.textContent = ok ? 'Enter VR' : 'VR n/a here';
  }).catch(() => { vrBtn.textContent = 'VR n/a here'; });
} else vrBtn.textContent = 'VR n/a here';
vrBtn.onclick = async () => {
  try {
    const session = await navigator.xr.requestSession('immersive-vr',
      { optionalFeatures: ['local-floor', 'bounded-floor'] });
    session.addEventListener('end', () => dolly.position.set(0, 0, 0));
    dolly.position.set(13, 0.2, 15); // stand off to the side of the plant in VR
    await renderer.xr.setSession(session);
  } catch (e) { console.warn('VR session failed', e); }
};

/* ---------------- boot + debug handle for verification ---------------- */
window.__NPP_DT__ = {
  state: () => {
    const snap = currentSnapshot();
    const row = snap ? snap.row : null;
    return {
      run: S.run ? S.run.class : null,
      frame: S.frame, t: Math.round(S.frame) * 10,
      setting: S.setting,
      model: S.model,
      W: snap ? snap.W : null,
      probaSum: row ? row.reduce((a, b) => a + b, 0) : null,
      top1: row ? S.manifest.classes[row.indexOf(Math.max(...row))] : null,
      top1Conf: row ? Math.max(...row) : null,
      sceneObjects: scene.children.length,
      playing: S.playing,
    };
  },
  setFrame: f => { S.frame = f; updateAll(true); },
  setSetting,
  setModel,
  loadRun,
  ready: () => !!S.run,
  renderInfo: () => ({ calls: renderer.info.render.calls,
                       triangles: renderer.info.render.triangles }),
  readPixels: () => { // render synchronously, then sample the drawing buffer
    renderer.render(scene, camera);
    const gl = renderer.getContext();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const px = new Uint8Array(4 * 32 * 32);
    gl.readPixels(w / 2 - 16, h / 2 - 16, 32, 32, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let distinct = new Set();
    for (let i = 0; i < px.length; i += 4) distinct.add(px[i] + ',' + px[i + 1] + ',' + px[i + 2]);
    return { distinctColors: distinct.size, center: [px[0], px[1], px[2]] };
  },
  chartSample: () => { // count non-background pixels in the chart canvas
    const cv = $('chart'), g = cv.getContext('2d');
    const img = g.getImageData(0, 0, cv.width, cv.height).data;
    let lit = 0;
    for (let i = 0; i < img.length; i += 16) if (img[i] + img[i + 1] + img[i + 2] > 130) lit++;
    return lit;
  },
};

(async () => {
  await loadManifest();
  buildRunBar();
  buildModels();
  buildSettings();
  await loadRun(S.manifest.runs[0].id);
  resize();
  syncPlayBtn();
})();
