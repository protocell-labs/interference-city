// main.js (ES module)

// MODULE IMPORTS

// Imports from your importmap
import * as THREE from 'three';
import { OrbitControls } from 'OrbitControls';
import { GLTFLoader } from 'GLTFLoader';
import { EffectComposer } from 'EffectComposer';
import { RenderPass } from 'RenderPass';
import { UnrealBloomPass } from 'UnrealBloomPass';
import { ShaderPass } from 'ShaderPass';

// -------------------------------
// Constants & helpers
// -------------------------------
const F_MIN = 3000;          // 3 kHz
const F_MAX = 3000000000;    // 3 GHz
const SOURCE_RANGE = 6;      // world range for source sliders
const GRID_SIZE = 8;         // point cloud extent
const AMPLITUDE_SCALE = 0.4; // vertical scaling of wave

function mapSliderToFrequency(v) {
  const t = v / 100; // slider 0..100
  const logMin = Math.log10(F_MIN);
  const logMax = Math.log10(F_MAX);
  const logF = logMin + (logMax - logMin) * t;
  return Math.pow(10, logF);
}

function getVisualLambdaFromFrequency(f) {
  if (!isFinite(f) || f <= 0) return 2.0;
  const norm = (f - F_MIN) / (F_MAX - F_MIN); // 0..1
  const clamped = Math.max(0, Math.min(1, norm));
  // Low freq → long λ, high freq → shorter λ
  return 4.0 - 3.0 * clamped; // 4 → 1 units
}

function mapSliderToResolution(densityValue) {
  const base = 15;
  const step = 10;
  return base + (densityValue - 1) * step; // 1→15, 2→25, ...
}

function mapSliderToSource(v) {
  const t = v / 100; // v ∈ [-100,100] → t ∈ [-1,1]
  return t * SOURCE_RANGE;
}

// Size vs wavelength rule
function classifyInteraction(objectSize, lambdaVis) {
  const ratio = objectSize / lambdaVis;

  if (ratio < 0.1) {
    // object size < 1/10 λ → wave mostly passes
    return { transmit: 0.9, absorb: 0.1, reflect: 0.0 };
  } else if (ratio < 2.0) {
    // same order as λ → scattering & reflection
    return { transmit: 0.3, absorb: 0.2, reflect: 0.5 };
  } else {
    // much larger than λ → strong reflection/absorption
    return { transmit: 0.1, absorb: 0.6, reflect: 0.3 };
  }
}

// Spherical wave from a point source
function radioAmplitudePointSource(t, f, px, py, pz, sx, sy, sz) {
  const dx = px - sx;
  const dy = py - sy;
  const dz = pz - sz;

  const r = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6;

  const lambdaVis = getVisualLambdaFromFrequency(f);
  const k = (2 * Math.PI) / lambdaVis;

  const norm = (f - F_MIN) / (F_MAX - F_MIN); // 0..1
  const omega = 2 * Math.PI * (0.5 + norm * 2.0); // 0.5..2.5 Hz

  const phase = k * r - omega * t;
  return Math.sin(phase) / (1.0 + 0.3 * r);
}

// -------------------------------
// Scene setup
// -------------------------------
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(
  50,
  window.innerWidth / window.innerHeight,
  0.1,
  200
);
camera.position.set(10, 10, 18);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Orbit controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Lights
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(10, 15, 8);
scene.add(dirLight);

const ambient = new THREE.AmbientLight(0xffffff, 0.3);
scene.add(ambient);

// -------------------------------
// Obstacles (GLB meshes)
// -------------------------------
const obstacles = []; // { root, maxSize }

const raycaster = new THREE.Raycaster();
const tmpDir = new THREE.Vector3();
const tmpPoint = new THREE.Vector3();

if (GLTFLoader) {
  const loader = new GLTFLoader();

  // Example obstacle – change the path to your own GLB
  loader.load(
    "models/obstacle1.glb",
    (gltf) => {
      const root = gltf.scene;
      root.position.set(0, 0, 0);
      scene.add(root);

      root.updateWorldMatrix(true, true);
      const bbox = new THREE.Box3().setFromObject(root);
      const size = new THREE.Vector3();
      bbox.getSize(size);
      const maxSize = Math.max(size.x, size.y, size.z);

      obstacles.push({ root, maxSize });
      console.log("Loaded obstacle1.glb, size:", maxSize.toFixed(2));
    },
    undefined,
    (err) => console.warn("Failed to load obstacle1.glb", err)
  );

  // Add more GLBs if you like by calling loader.load(...) again
}

function findObstacleFromHit(hitObject) {
  for (const o of obstacles) {
    let current = hitObject;
    while (current) {
      if (current === o.root) return o;
      current = current.parent;
    }
  }
  return null;
}

function occludingObstacle(sourcePos, pointPos) {
  if (obstacles.length === 0) return null;

  tmpDir.subVectors(pointPos, sourcePos).normalize();
  const maxDist = sourcePos.distanceTo(pointPos);

  raycaster.set(sourcePos, tmpDir);
  const roots = obstacles.map((o) => o.root);
  const hits = raycaster.intersectObjects(roots, true);

  if (hits.length === 0) return null;

  const hit = hits[0];
  if (hit.distance < maxDist - 1e-3) {
    return findObstacleFromHit(hit.object);
  }

  return null;
}

// -------------------------------
// Point cloud grid
// -------------------------------
let points = null;
let geometry = null;
let basePositions = null;
let currentResolution = 0;

function buildPointCloud(resolution) {
  if (points) {
    scene.remove(points);
    points.geometry.dispose();
    points.material.dispose();
    points = null;
  }

  currentResolution = resolution;

  const numPoints = resolution * resolution * resolution;
  basePositions = new Float32Array(numPoints * 3);
  const positions = new Float32Array(numPoints * 3);

  let i = 0;
  for (let ix = 0; ix < resolution; ix++) {
    const x = (ix / (resolution - 1) - 0.5) * GRID_SIZE;
    for (let iy = 0; iy < resolution; iy++) {
      const y = (iy / (resolution - 1) - 0.5) * GRID_SIZE;
      for (let iz = 0; iz < resolution; iz++) {
        const z = (iz / (resolution - 1) - 0.5) * GRID_SIZE;

        const idx = i * 3;
        basePositions[idx] = x;
        basePositions[idx + 1] = y;
        basePositions[idx + 2] = z;

        positions[idx] = x;
        positions[idx + 1] = y;
        positions[idx + 2] = z;

        i++;
      }
    }
  }

  geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color: 0x00ffff,
    size: 0.06,
    sizeAttenuation: true,
  });

  points = new THREE.Points(geometry, material);
  scene.add(points);
}

// -------------------------------
// Two sources (spheres) + sliders
// -------------------------------
const source1Mesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.3, 24, 16),
  new THREE.MeshBasicMaterial({ color: 0xff4040 })
);
scene.add(source1Mesh);

const source2Mesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.3, 24, 16),
  new THREE.MeshBasicMaterial({ color: 0x40ff40 })
);
scene.add(source2Mesh);

const source1Pos = new THREE.Vector3();
const source2Pos = new THREE.Vector3();

function updateSource1FromSliders() {
  const sx = Number(source1XSlider.value);
  const sy = Number(source1YSlider.value);
  const sz = Number(source1ZSlider.value);

  source1Pos.set(
    mapSliderToSource(sx),
    mapSliderToSource(sy),
    mapSliderToSource(sz)
  );

  source1Mesh.position.copy(source1Pos);
}

function updateSource2FromSliders() {
  const sx = Number(source2XSlider.value);
  const sy = Number(source2YSlider.value);
  const sz = Number(source2ZSlider.value);

  source2Pos.set(
    mapSliderToSource(sx),
    mapSliderToSource(sy),
    mapSliderToSource(sz)
  );

  source2Mesh.position.copy(source2Pos);
}

// -------------------------------
// UI wiring
// -------------------------------
const densitySlider = document.getElementById("densitySlider");

const freq1Slider = document.getElementById("freq1Slider");
const source1XSlider = document.getElementById("source1XSlider");
const source1YSlider = document.getElementById("source1YSlider");
const source1ZSlider = document.getElementById("source1ZSlider");

const freq2Slider = document.getElementById("freq2Slider");
const source2XSlider = document.getElementById("source2XSlider");
const source2YSlider = document.getElementById("source2YSlider");
const source2ZSlider = document.getElementById("source2ZSlider");

const labelsDiv = document.getElementById("labels");

let densitySliderVal = Number(densitySlider.value);
let freq1SliderVal = Number(freq1Slider.value);
let freq2SliderVal = Number(freq2Slider.value);

let freq1 = mapSliderToFrequency(freq1SliderVal);
let freq2 = mapSliderToFrequency(freq2SliderVal);
let resolution = mapSliderToResolution(densitySliderVal);

function updateLabels() {
  const MHz1 = freq1 / 1e6;
  const MHz2 = freq2 / 1e6;
  const numPoints = resolution * resolution * resolution;

  labelsDiv.innerHTML =
    "Grid: " +
    resolution +
    " × " +
    resolution +
    " × " +
    resolution +
    " ≈ " +
    numPoints.toLocaleString() +
    " points" +
    "<br>Source 1 freq: " +
    freq1.toExponential(3) +
    " Hz (" +
    MHz1.toFixed(3) +
    " MHz)" +
    "<br>Source 1 pos: (" +
    source1Pos.x.toFixed(2) +
    ", " +
    source1Pos.y.toFixed(2) +
    ", " +
    source1Pos.z.toFixed(2) +
    ")" +
    "<br>Source 2 freq: " +
    freq2.toExponential(3) +
    " Hz (" +
    MHz2.toFixed(3) +
    " MHz)" +
    "<br>Source 2 pos: (" +
    source2Pos.x.toFixed(2) +
    ", " +
    source2Pos.y.toFixed(2) +
    ", " +
    source2Pos.z.toFixed(2) +
    ")";
}

densitySlider.addEventListener("input", () => {
  densitySliderVal = Number(densitySlider.value);
  resolution = mapSliderToResolution(densitySliderVal);
  buildPointCloud(resolution);
  updateLabels();
});

freq1Slider.addEventListener("input", () => {
  freq1SliderVal = Number(freq1Slider.value);
  freq1 = mapSliderToFrequency(freq1SliderVal);
  updateLabels();
});

freq2Slider.addEventListener("input", () => {
  freq2SliderVal = Number(freq2Slider.value);
  freq2 = mapSliderToFrequency(freq2SliderVal);
  updateLabels();
});

source1XSlider.addEventListener("input", () => {
  updateSource1FromSliders();
  updateLabels();
});
source1YSlider.addEventListener("input", () => {
  updateSource1FromSliders();
  updateLabels();
});
source1ZSlider.addEventListener("input", () => {
  updateSource1FromSliders();
  updateLabels();
});

source2XSlider.addEventListener("input", () => {
  updateSource2FromSliders();
  updateLabels();
});
source2YSlider.addEventListener("input", () => {
  updateSource2FromSliders();
  updateLabels();
});
source2ZSlider.addEventListener("input", () => {
  updateSource2FromSliders();
  updateLabels();
});

// Initial setup
updateSource1FromSliders();
updateSource2FromSliders();
buildPointCloud(resolution);
updateLabels();

// -------------------------------
// Animation loop
// -------------------------------
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);

  const t = clock.getElapsedTime();
  const lambdaVis1 = getVisualLambdaFromFrequency(freq1);
  const lambdaVis2 = getVisualLambdaFromFrequency(freq2);

  if (points && geometry && basePositions) {
    const pos = geometry.attributes.position.array;
    const numPoints = basePositions.length / 3;

    for (let i = 0; i < numPoints; i++) {
      const idx = i * 3;

      const x0 = basePositions[idx];
      const y0 = basePositions[idx + 1];
      const z0 = basePositions[idx + 2];

      tmpPoint.set(x0, y0, z0);

      let A1 = radioAmplitudePointSource(
        t,
        freq1,
        x0,
        y0,
        z0,
        source1Pos.x,
        source1Pos.y,
        source1Pos.z
      );

      let A2 = radioAmplitudePointSource(
        t,
        freq2,
        x0,
        y0,
        z0,
        source2Pos.x,
        source2Pos.y,
        source2Pos.z
      );

      // Occlusion & attenuation for source 1
      const obs1 = occludingObstacle(source1Pos, tmpPoint);
      if (obs1) {
        const intr = classifyInteraction(obs1.maxSize, lambdaVis1);
        A1 *= intr.transmit;
      }

      // Occlusion & attenuation for source 2
      const obs2 = occludingObstacle(source2Pos, tmpPoint);
      if (obs2) {
        const intr = classifyInteraction(obs2.maxSize, lambdaVis2);
        A2 *= intr.transmit;
      }

      const A_total = A1 + A2;

      pos[idx] = x0;
      pos[idx + 1] = y0 + A_total * AMPLITUDE_SCALE;
      pos[idx + 2] = z0;
    }

    geometry.attributes.position.needsUpdate = true;
  }

  scene.rotation.y += 0.0015;
  controls.update();
  renderer.render(scene, camera);
}

animate();

// Resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
