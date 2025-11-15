// MODULE IMPORTS

import * as THREE from 'three';
import { OrbitControls } from 'OrbitControls';
import { GLTFLoader } from 'GLTFLoader';
import { EffectComposer } from 'EffectComposer';
import { RenderPass } from 'RenderPass';
import { UnrealBloomPass } from 'UnrealBloomPass';
import { ShaderPass } from 'ShaderPass';



// ---------- Basic three.js setup ----------
const container = document.getElementById('app');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);

const BG_COLOR = 0x05070a; // dark background for main scene
renderer.setClearColor(BG_COLOR, 1.0);

container.appendChild(renderer.domElement);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / window.innerHeight,
    0.1,
    100
);
camera.position.set(0, 6, 10);
camera.lookAt(0, 0, 0);

// Simple light (mainly for the grid; the points use a custom shader)
const light = new THREE.DirectionalLight(0xffffff, 1.0);
light.position.set(5, 10, 5);
scene.add(light);
scene.add(new THREE.AmbientLight(0x404040));

// ---------- Simulation settings ----------
const SIM_SIZE = 128;
const PLANE_SIZE = 10; // world-space size of the simulated area

// Render target options
const rtOptions = {
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    type: THREE.FloatType,
    depthBuffer: false,
    stencilBuffer: false,
};

// Three RTs to avoid feedback loops:
// rtCurr: u^t, rtPrev: u^{t-1}, rtTemp: u^{t+1}
let rtCurr = new THREE.WebGLRenderTarget(SIM_SIZE, SIM_SIZE, rtOptions);
let rtPrev = new THREE.WebGLRenderTarget(SIM_SIZE, SIM_SIZE, rtOptions);
let rtTemp = new THREE.WebGLRenderTarget(SIM_SIZE, SIM_SIZE, rtOptions);

// RT for impulses / sources
let sourceRT = new THREE.WebGLRenderTarget(SIM_SIZE, SIM_SIZE, rtOptions);

// Helper: clear a render target to black (no source, no height)
function clearRT(rt) {
    renderer.setRenderTarget(rt);
    renderer.setClearColor(0x000000, 1.0); // IMPORTANT: pure black for simulation data
    renderer.clear();
    renderer.setRenderTarget(null);
    renderer.setClearColor(BG_COLOR, 1.0); // restore scene background
}

// Initialize all simulation RTs to zero
clearRT(rtCurr);
clearRT(rtPrev);
clearRT(rtTemp);
clearRT(sourceRT);

// ---------- Simulation quad (compute pass) ----------
const simScene = new THREE.Scene();
const simCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const simMaterial = new THREE.ShaderMaterial({
    uniforms: {
        u_prev: { value: rtCurr.texture }, // u^t
        u_prevPrev: { value: rtPrev.texture }, // u^{t-1}
        u_sources: { value: sourceRT.texture },
        u_resolution: { value: new THREE.Vector2(SIM_SIZE, SIM_SIZE) },
        u_c: { value: 0.3 },   // wave speed (lower = more stable)
        u_damping: { value: 0.01 },  // damping factor
    },
    vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position, 1.0);
    }
  `,
    fragmentShader: /* glsl */`
    precision highp float;

    varying vec2 vUv;
    uniform sampler2D u_prev;
    uniform sampler2D u_prevPrev;
    uniform sampler2D u_sources;
    uniform vec2 u_resolution;
    uniform float u_c;
    uniform float u_damping;

    // Simple cross-shaped obstacle mask
    float obstacleMask(vec2 p) {
      float m = 0.0;

      // Vertical bar in the middle
      if (p.x > 0.47 && p.x < 0.53 && p.y > 0.2 && p.y < 0.8) {
        m = 1.0;
      }

      // Horizontal bar in the middle
      if (p.y > 0.47 && p.y < 0.53 && p.x > 0.2 && p.x < 0.8) {
        m = 1.0;
      }

      return m;
    }

    void main() {
      // Obstacles: pin displacement to 0.
      float obs = obstacleMask(vUv);
      if (obs > 0.5) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
      }

      vec2 texel = 1.0 / u_resolution;

      float center = texture2D(u_prev, vUv).r;
      float up     = texture2D(u_prev, vUv + vec2(0.0, texel.y)).r;
      float down   = texture2D(u_prev, vUv - vec2(0.0, texel.y)).r;
      float left   = texture2D(u_prev, vUv - vec2(texel.x, 0.0)).r;
      float right  = texture2D(u_prev, vUv + vec2(texel.x, 0.0)).r;

      float lap = (up + down + left + right - 4.0 * center);

      float prevPrev = texture2D(u_prevPrev, vUv).r;
      float src      = texture2D(u_sources, vUv).r;

      // Discrete wave equation with damping
      float next = (2.0 - u_damping) * center
                 - (1.0 - u_damping) * prevPrev
                 + u_c * u_c * lap
                 + src;

      // Clamp to avoid runaway blowups from numerical noise
      next = clamp(next, -5.0, 5.0);

      gl_FragColor = vec4(next, 0.0, 0.0, 1.0);
    }
  `
});

const simQuad = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    simMaterial
);
simScene.add(simQuad);

// ---------- Source drawing quad ----------
const sourceScene = new THREE.Scene();
const sourceMaterial = new THREE.ShaderMaterial({
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthTest: false,
    depthWrite: false,
    uniforms: {
        u_center: { value: new THREE.Vector2(0.5, 0.5) },
        u_radius: { value: 0.02 },
        u_strength: { value: 2.0 },
    },
    vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position, 1.0);
    }
  `,
    fragmentShader: /* glsl */`
    precision highp float;
    varying vec2 vUv;
    uniform vec2 u_center;
    uniform float u_radius;
    uniform float u_strength;

    void main() {
      float d = distance(vUv, u_center);
      float impulse = exp(- (d * d) / (u_radius * u_radius)) * u_strength;
      gl_FragColor = vec4(impulse, 0.0, 0.0, 1.0);
    }
  `
});

const sourceQuad = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    sourceMaterial
);
sourceScene.add(sourceQuad);

// Pending impulses for this frame
const pendingSources = [];

// ---------- Point cloud visualization ----------

// Build a grid of points matching the simulation resolution
const pointGeometry = new THREE.BufferGeometry();
const num = SIM_SIZE;
const positions = new Float32Array(num * num * 3);
const uvs = new Float32Array(num * num * 2);

let i3 = 0, i2 = 0;
for (let y = 0; y < num; y++) {
    for (let x = 0; x < num; x++) {
        const u = x / (num - 1);
        const v = y / (num - 1);

        const posX = (u - 0.5) * PLANE_SIZE;
        const posZ = (v - 0.5) * PLANE_SIZE;

        positions[i3++] = posX;
        positions[i3++] = 0.0;  // y will be displaced by height
        positions[i3++] = posZ;

        uvs[i2++] = u;
        uvs[i2++] = v;
    }
}

pointGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
pointGeometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

const pointsMaterial = new THREE.ShaderMaterial({
    uniforms: {
        u_heightMap: { value: rtCurr.texture },
        u_amplitude: { value: 1.0 },
        u_pointSize: { value: 0.25 },
    },
    vertexShader: /* glsl */`
    uniform sampler2D u_heightMap;
    uniform float u_amplitude;
    uniform float u_pointSize;

    varying float vHeight;

    void main() {
      vec2 vUv = uv;
      float h = texture2D(u_heightMap, vUv).r;
      vHeight = h;

      // Displace vertically by height
      vec3 displaced = vec3(position.x, position.y + h * u_amplitude, position.z);
      vec4 worldPos = modelMatrix * vec4(displaced, 1.0);
      gl_Position = projectionMatrix * viewMatrix * worldPos;

      float dist = length(worldPos.xyz - cameraPosition);
      gl_PointSize = u_pointSize * (1.0 / dist) * 50.0;
    }
  `,
    fragmentShader: /* glsl */`
    precision highp float;
    varying float vHeight;

    void main() {
      // Circular points
      vec2 c = gl_PointCoord - 0.5;
      if (dot(c, c) > 0.25) discard;

      float h = vHeight;

      // Symmetric blue–white–red map
      // h < 0 -> blue, h > 0 -> red, around 0 -> white-ish
      float t = clamp(h * 1.5 + 0.5, 0.0, 1.0);

      vec3 blue  = vec3(0.1, 0.2, 0.9);
      vec3 white = vec3(1.0, 1.0, 1.0);
      vec3 red   = vec3(1.0, 0.2, 0.1);

      // Blend via white in the middle
      vec3 midColor = mix(blue, white, smoothstep(0.0, 0.5, t));
      vec3 color    = mix(midColor, red, smoothstep(0.5, 1.0, t));

      gl_FragColor = vec4(color, 1.0);
    }
  `,
    depthTest: true,
    depthWrite: true,
});


const points = new THREE.Points(pointGeometry, pointsMaterial);
scene.add(points);

// Optional: a simple grid to see scale
const grid = new THREE.GridHelper(PLANE_SIZE, 20, 0x444444, 0x222222);
scene.add(grid);

// ---------- Invisible click plane for raycasting ----------

const clickPlaneGeom = new THREE.PlaneGeometry(PLANE_SIZE, PLANE_SIZE);
const clickPlaneMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.0,   // invisible, but still raycast-able
});
const clickPlane = new THREE.Mesh(clickPlaneGeom, clickPlaneMat);
clickPlane.rotation.x = -Math.PI / 2;
scene.add(clickPlane);

// ---------- Raycaster for click-to-emit ----------
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function onPointerDown(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = - ((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(clickPlane);

    if (intersects.length > 0) {
        const uv = intersects[0].uv; // [0,1]x[0,1] on the plane

        // Flip both axes to match how the simulation grid is laid out
        const uvx = uv.x;
        const uvy = 1.0 - uv.y;

        pendingSources.push({
            uvx,
            uvy,
            strength: 3.0,
            radius: 0.03,
        });
    }

}

renderer.domElement.addEventListener('pointerdown', onPointerDown);

// ---------- Simulation step ----------
function stepSimulation() {
    // 1) Clear sourceRT to pure black (no background bias!)
    renderer.setRenderTarget(sourceRT);
    renderer.setClearColor(0x000000, 1.0);
    renderer.clear();

    // 2) Draw all pending sources into sourceRT (additive)
    for (const src of pendingSources) {
        sourceMaterial.uniforms.u_center.value.set(src.uvx, src.uvy);
        sourceMaterial.uniforms.u_strength.value = src.strength;
        sourceMaterial.uniforms.u_radius.value = src.radius;
        renderer.render(sourceScene, simCamera);
    }
    pendingSources.length = 0; // impulses are one-frame

    renderer.setRenderTarget(null);
    renderer.setClearColor(BG_COLOR, 1.0); // restore main background

    // 3) Run compute shader:
    //    read from rtCurr (u^t) and rtPrev (u^{t-1}), write into rtTemp (u^{t+1})
    simMaterial.uniforms.u_prev.value = rtCurr.texture;
    simMaterial.uniforms.u_prevPrev.value = rtPrev.texture;

    renderer.setRenderTarget(rtTemp);
    renderer.render(simScene, simCamera);
    renderer.setRenderTarget(null);

    // 4) Rotate targets: newPrev = oldCurr, newCurr = rtTemp
    const oldPrev = rtPrev;
    rtPrev = rtCurr;
    rtCurr = rtTemp;
    rtTemp = oldPrev;

    // 5) Update visualization height map for the points
    pointsMaterial.uniforms.u_heightMap.value = rtCurr.texture;
}

// ---------- Resize handling ----------
function onWindowResize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}
window.addEventListener('resize', onWindowResize);

// ---------- Animation loop ----------
function animate() {
    requestAnimationFrame(animate);
    stepSimulation();
    renderer.render(scene, camera);
}

animate();
