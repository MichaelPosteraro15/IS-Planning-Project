import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let planData = { problem: { grid_size: 5, snow: {}, balls: {}, ball_size: {}, character: "2,1" }, frames: [], isNumeric: false };
let currentFrame = 0;
let isPlaying = false;
let speed = 1;
let snowSpeed = 0.2;
let scene, camera, renderer, controls, grid = {}, balls = {}, character, snowman, mixer, particles, particleVelocities = [], snowParticles = [], snowParticleVelocities = [], pathLine, spotlight;
const radius = { 0: 0.15, 1: 0.25, 2: 0.35 };
const clock = new THREE.Clock();
let textDisplayTime = 0;
let textFadeTime = 0;
let currentTime = 0;

const controlPanel = document.getElementById('controlPanel');
let isDragging = false, currentX, currentY, isResizing = false, startX, startY, startWidth, startHeight;
controlPanel.addEventListener('mousedown', e => {
    if (e.target.id === 'minimizeBtn') return;
    if (e.target.id === 'resizeHandle') {
        isResizing = true;
        startX = e.clientX;
        startY = e.clientY;
        startWidth = parseInt(document.defaultView.getComputedStyle(controlPanel).width, 10);
        startHeight = parseInt(document.defaultView.getComputedStyle(controlPanel).height, 10);
        controlPanel.style.cursor = 'se-resize';
    } else {
        isDragging = true;
        currentX = e.clientX - parseFloat(controlPanel.style.left || 0);
        currentY = e.clientY - parseFloat(controlPanel.style.top || 0);
        controlPanel.style.cursor = 'grabbing';
    }
});
document.addEventListener('mousemove', e => {
    if (isDragging) {
        let newLeft = e.clientX - currentX;
        let newTop = e.clientY - currentY;
        newLeft = Math.max(0, Math.min(newLeft, window.innerWidth * 0.35 - controlPanel.offsetWidth));
        newTop = Math.max(40, Math.min(newTop, window.innerHeight * 0.5 - controlPanel.offsetHeight));
        controlPanel.style.left = `${newLeft}px`;
        controlPanel.style.top = `${newTop}px`;
        controlPanel.style.transform = 'none';
    } else if (isResizing) {
        const width = startWidth + (e.clientX - startX);
        const height = startHeight + (e.clientY - startY);
        controlPanel.style.width = `${Math.max(200, Math.min(400, width))}px`;
        controlPanel.style.height = `${Math.max(150, Math.min(600, height))}px`;
    }
});
document.addEventListener('mouseup', () => {
    isDragging = false;
    isResizing = false;
    controlPanel.style.cursor = 'move';
});

document.getElementById('minimizeBtn').addEventListener('click', () => {
    controlPanel.classList.toggle('minimized');
    if (controlPanel.classList.contains('minimized')) {
        document.getElementById('minimizeBtn').textContent = '+';
        document.getElementById('planFile').style.display = 'none';
        document.querySelectorAll('#controlPanel button, #controlPanel label, #controlPanel input[type="range"]').forEach(el => el.style.display = 'none');
        document.getElementById('resizeHandle').style.display = 'none';
    } else {
        document.getElementById('minimizeBtn').textContent = '-';
        document.getElementById('planFile').style.display = 'block';
        document.querySelectorAll('#controlPanel button, #controlPanel label, #controlPanel input[type="range"]').forEach(el => el.style.display = 'block');
        document.getElementById('resizeHandle').style.display = 'block';
    }
});

function parsePlanFile(fileContent) {
    try {
        const lines = fileContent.split('\n').map(line => line.trim().toLowerCase()).filter(line => line && !line.startsWith(';'));
        const frames = [];
        let current_state = {
            snow: Object.fromEntries(Array(5).fill().flatMap((_, x) => Array(5).fill().map((_, y) => [[x, y].join(','), x === 0 || x === 4]))),
            balls: { ball_0: "1,2", ball_1: "2,2", ball_2: "3,2" },
            ball_size: { ball_0: 2, ball_1: 1, ball_2: 0 },
            character: "2,1",
            step_text: null,
            type: null,
            alpha: 0,
            direction: null,
            time: 0
        };
        let step_count = 0;
        let isNumeric = lines.some(line => /^\d+\.\d+\s*:\s*/.test(line));
        planData.isNumeric = isNumeric;

        for (let line of lines) {
            let action, time;
            if (isNumeric) {
                const match = line.match(/(\d+\.\d+)\s*:\s*\((.*?)\)\s*(?:;.*)?$/);
                if (!match) continue;
                time = parseFloat(match[1]);
                action = match[2].replace(/-/g, '_').split(/\s+/).filter(s => s);
            } else {
                const match = line.match(/\((.*?)\)\s*(?:;.*)?$/);
                if (!match) continue;
                time = step_count + 1;
                action = match[1].replace(/-/g, '_').split(/\s+/).filter(s => s);
            }

            if (!action.length) continue;

            const action_name = action[0];
            let step_text;
            if (["move_character", "move", "move_to", "move_char"].includes(action_name) && action.length >= 3) {
                const from_loc = current_state.character.replace('loc_', '').replace('loc-', '');
                const to_loc = action[2].replace('loc_', '').replace('loc-', '');
                const direction = action.length > 3 ? action[3] : '';
                step_text = `Step ${time}: move_character loc_${from_loc.split(',')[0]}_${from_loc.split(',')[1]} loc_${to_loc.split(/[_,]/)[0]}_${to_loc.split(/[_,]/)[1]} ${direction}`;
            } else if (["move_ball", "push", "roll", "roll_ball"].includes(action_name) && action.length >= 4) {
                const ball = action[1].replace('ball-', 'ball_');
                const to_cell = action[3].replace('loc_', '').replace('loc-', '');
                step_text = `Step ${time}: move_ball ${ball} loc_${to_cell.split(/[_,]/)[0]}_${to_cell.split(/[_,]/)[1]} ${action.length >= 5 ? action[4] : ''}`;
            } else if (action_name === "goal" && action.length >= 4) {
                step_text = `Step ${time}: Goal: Snowman built at loc_1_3`;
            } else {
                continue;
            }

            if (["move_character", "move", "move_to", "move_char"].includes(action_name) && action.length >= 3) {
                const loc = action[2].replace('loc_', '').replace('loc-', '');
                const [x, y] = loc.split(/[_\s]/).map(Number);
                for (let i = 0; i < 10; i++) {
                    const frame = JSON.parse(JSON.stringify(current_state));
                    frame.alpha = i / 10;
                    frame.step_text = i === 0 ? step_text : null;
                    frame.type = action_name;
                    frame.direction = action.length > 3 ? action[3] : null;
                    frame.time = time;
                    frame.start = current_state.character;
                    frame.end = `${x-1},${y-1}`;
                    frames.push(frame);
                }
                current_state.character = `${x-1},${y-1}`;
            } else if (["move_ball", "push", "roll", "roll_ball"].includes(action_name) && action.length >= 4) {
                const ball = action[1].replace('ball-', 'ball_');
                const from_cell = action[2].replace('loc_', '').replace('loc-', '');
                const to_cell = action[3].replace('loc_', '').replace('loc-', '');
                const [sx, sy] = from_cell.split(/[_\s]/).map(Number);
                const [ex, ey] = to_cell.split(/[_\s]/).map(Number);
                for (let i = 0; i < 10; i++) {
                    const frame = JSON.parse(JSON.stringify(current_state));
                    frame.alpha = i / 10;
                    frame.step_text = i === 0 ? step_text : null;
                    frame.type = action_name;
                    frame.ball = ball;
                    frame.start = current_state.balls[ball];
                    frame.end = `${ex-1},${ey-1}`;
                    frame.direction = action.length >= 5 ? action[4] : null;
                    frame.time = time;
                    if (current_state.snow[frame.end] && current_state.ball_size[ball] < 2) {
                        frame.ball_size[ball] = Math.min(frame.ball_size[ball] + 1, 2);
                        frame.snow[frame.end] = false;
                    }
                    frames.push(frame);
                }
                current_state.balls[ball] = `${ex-1},${ey-1}`;
                if (current_state.snow[`${ex-1},${ey-1}`] && current_state.ball_size[ball] < 2) {
                    current_state.ball_size[ball] = Math.min(current_state.ball_size[ball] + 1, 2);
                    current_state.snow[`${ex-1},${ey-1}`] = false;
                }
            } else if (action_name === "goal" && action.length >= 4) {
                for (let i = 0; i < 10; i++) {
                    const frame = JSON.parse(JSON.stringify(current_state));
                    frame.alpha = i / 10;
                    frame.step_text = i === 0 ? step_text : null;
                    frame.type = "goal";
                    frame.time = time;
                    frame.balls[action[1]] = "0,2";
                    frame.balls[action[2]] = "0,2";
                    frame.balls[action[3]] = "0,2";
                    frames.push(frame);
                }
                current_state.balls[action[1]] = "0,2";
                current_state.balls[action[2]] = "0,2";
                current_state.balls[action[3]] = "0,2";
            }

            step_count += 1;
        }

        if (!frames.length) throw new Error("No valid frames parsed from TXT");
        return frames;
    } catch (e) {
        console.error(`Error parsing file: ${e}`);
        return [];
    }
}

async function loadPlanData(file) {
    try {
        if (!file) throw new Error("No file selected");
        if (!file.name.endsWith('.txt')) throw new Error("Only .txt files are supported");
        const frames = parsePlanFile(await file.text());
        if (!frames.length) throw new Error("No valid frames parsed from TXT");
        planData = {
            problem: {
                grid_size: 5,
                snow: Object.fromEntries(Array(5).fill().flatMap((_, x) => Array(5).fill().map((_, y) => [[x, y].join(','), x === 0 || x === 4]))),
                balls: { ball_0: "1,2", ball_1: "2,2", ball_2: "3,2" },
                ball_size: { ball_0: 2, ball_1: 1, ball_2: 0 },
                character: "2,1"
            },
            frames,
            isNumeric: planData.isNumeric
        };
        currentFrame = 0;
        document.getElementById('step').max = Math.max(0, Math.floor(planData.frames.length / 10) - 1);
        document.getElementById('step').value = 0;
        document.getElementById('step').disabled = false;
        ['playPause', 'stepForward', 'stepBackward', 'reset'].forEach(id => document.getElementById(id).disabled = false);
        document.getElementById('stepLogText').value = planData.frames[0]?.step_text || '';
        document.getElementById('timeDisplay').textContent = `Time: 0s`;
        await initSceneObjects();
        if (planData.frames.length > 0) updateFrame(planData.frames[0]);
    } catch (err) {
        console.error('Error loading plan:', err);
        document.getElementById('stepLogText').value = `Error: ${err.message}`;
    }
}

async function initSceneObjects() {
    try {
        const { problem } = planData;
        grid = {};
        balls = {};

        const groundGeometry = new THREE.PlaneGeometry(7, 7);
        const groundMaterial = new THREE.MeshStandardMaterial({ color: 0xE0FFFF, roughness: 0.8, metalness: 0.1 });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.set(2.5, -0.01, 2.5);
        ground.receiveShadow = true;
        scene.add(ground);

        for (let x = 0; x < problem.grid_size; x++) {
            for (let y = 0; y < problem.grid_size; y++) {
                const coord = `${x},${y}`;
                const geometry = new THREE.PlaneGeometry(1, 1);
                const material = new THREE.MeshStandardMaterial({ color: problem.snow[coord] ? 0xE0FFFF : 0x90EE90, side: THREE.DoubleSide });
                const plane = new THREE.Mesh(geometry, material);
                plane.rotation.x = -Math.PI / 2;
                plane.position.set(x + 0.5, 0, y + 0.5);
                plane.receiveShadow = true;
                scene.add(plane);
                grid[coord] = plane;
            }
        }

        const wallMaterial = new THREE.MeshStandardMaterial({ color: 0xADD8E6, transparent: true, opacity: 0.7, roughness: 0.3, metalness: 0.5 });
        const wallHeight = 0.5;
        const wallThickness = 0.2;
        [
            { geometry: new THREE.BoxGeometry(7, wallHeight, wallThickness), position: [2.5, wallHeight / 2, -0.1] },
            { geometry: new THREE.BoxGeometry(7, wallHeight, wallThickness), position: [2.5, wallHeight / 2, 5.1] },
            { geometry: new THREE.BoxGeometry(wallThickness, wallHeight, 5), position: [-0.1, wallHeight / 2, 2.5] },
            { geometry: new THREE.BoxGeometry(wallThickness, wallHeight, 5), position: [5.1, wallHeight / 2, 2.5] }
        ].forEach(({ geometry, position }) => {
            const wall = new THREE.Mesh(geometry, wallMaterial);
            wall.position.set(...position);
            wall.castShadow = true;
            wall.receiveShadow = true;
            scene.add(wall);
        });

        const treeGeometry = new THREE.ConeGeometry(0.3, 0.8, 8);
        const treeMaterial = new THREE.MeshStandardMaterial({ color: 0xE0FFFF, roughness: 0.9 });
        const treePositions = [[-1, 0, -1], [6, 0, -1], [-1, 0, 6], [6, 0, 6], [2.5, 0, -1], [2.5, 0, 6], [-1, 0, 2.5], [6, 0, 2.5]];
        treePositions.forEach(pos => {
            const tree = new THREE.Mesh(treeGeometry, treeMaterial);
            tree.position.set(pos[0], pos[1] + 0.4, pos[2]);
            tree.castShadow = true;
            tree.receiveShadow = true;
            scene.add(tree);
        });

        if (planData.frames.length > 0) {
            for (let b in problem.balls) {
                if (!balls[b]) {
                    const geometry = new THREE.SphereGeometry(radius[problem.ball_size[b]], 32, 32);
                    const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0.2 });
                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.castShadow = true;
                    sphere.receiveShadow = true;
                    const [x, y] = problem.balls[b].split(',').map(Number);
                    sphere.position.set(x + 0.5, radius[problem.ball_size[b]], y + 0.5);
                    scene.add(sphere);
                    balls[b] = sphere;
                }
            }

            if (character) scene.remove(character);
            const loader = new GLTFLoader();
            try {
                const gltf = await new Promise((resolve, reject) => loader.load('https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb', resolve, undefined, reject));
                character = gltf.scene;
                character.scale.set(0.4, 0.4, 0.4);
                character.castShadow = true;
                character.receiveShadow = true;
                if (gltf.animations.length > 0) {
                    mixer = new THREE.AnimationMixer(character);
                    gltf.animations.forEach(clip => mixer.clipAction(clip).play());
                }
            } catch (e) {
                character = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.5, 32), new THREE.MeshStandardMaterial({ color: 0x00ff00 }));
                character.castShadow = true;
            }
            scene.add(character);
            const [cx, cy] = problem.character.split(',').map(Number);
            character.position.set(cx + 0.5, 0, cy + 0.5);

            if (snowman) scene.remove(snowman);
            try {
                const gltf = await new Promise((resolve, reject) => loader.load('https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb', resolve, undefined, reject));
                snowman = gltf.scene.clone();
                snowman.scale.set(0.3, 0.3, 0.3);
                snowman.castShadow = true;
                snowman.receiveShadow = true;
            } catch (e) {
                snowman = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.6, 0.2), new THREE.MeshStandardMaterial({ color: 0xFFD700 }));
                snowman.castShadow = true;
            }
        }

        spotlight = new THREE.SpotLight(0xffffff, 0, 5, Math.PI / 4, 0.5, 2);
        spotlight.position.set(2.5, 3, 2.5);
        spotlight.castShadow = true;
        scene.add(spotlight);
        scene.add(spotlight.target);
    } catch (err) {
        console.error('Error initializing scene objects:', err);
    }
}

async function init() {
    try {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, 0.65 * window.innerWidth / window.innerHeight, 0.1, 1000);
        renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('canvas'), antialias: true });
        renderer.setSize(window.innerWidth * 0.65, window.innerHeight - 70);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        scene.fog = new THREE.Fog(0x87CEEB, 3, 10);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
        directionalLight.position.set(5, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        scene.add(directionalLight);

        const skyboxLoader = new THREE.CubeTextureLoader();
        skyboxLoader.load([
            'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_px.jpg',
            'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_nx.jpg',
            'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_py.jpg',
            'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_ny.jpg',
            'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_pz.jpg',
            'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_nz.jpg'
        ], texture => { scene.background = texture; }, undefined, () => { scene.background = new THREE.Color(0x87CEEB); });

        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 2;
        controls.maxDistance = 10;

        const particleGeometry = new THREE.BufferGeometry();
        const particleCount = 2000;
        const particlePositions = new Float32Array(particleCount * 3);
        const particleMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.015, transparent: true, opacity: 0.9 });
        for (let i = 0; i < particleCount; i++) {
            particlePositions[i * 3] = 0;
            particlePositions[i * 3 + 1] = 0;
            particlePositions[i * 3 + 2] = 0;
            particleVelocities.push(new THREE.Vector3((Math.random() - 0.5) * 0.25 + 0.05, Math.random() * 0.35, (Math.random() - 0.5) * 0.25));
        }
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
        particles = new THREE.Points(particleGeometry, particleMaterial);
        scene.add(particles);
        particles.visible = false;

        const snowParticleMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
        function dropSnowflake() {
            const size = 0.009 + Math.random() * 0.012;
            const geometry = new THREE.CircleGeometry(size, 32);
            const mesh = new THREE.Mesh(geometry, snowParticleMaterial);
            mesh.position.set(Math.random() * 7 - 1, Math.random() * 5 + 3, Math.random() * 7 - 1);
            const velocity = new THREE.Vector3((Math.random() - 0.5) * 0.05 + 0.02 * Math.sin(clock.getElapsedTime()), -snowSpeed, (Math.random() - 0.5) * 0.05 + 0.02 * Math.cos(clock.getElapsedTime()));
            snowParticles.push(mesh);
            snowParticleVelocities.push(velocity);
            scene.add(mesh);
        }
        setInterval(dropSnowflake, 300);

        const pathGeometry = new THREE.BufferGeometry();
        const pathMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.5 });
        pathLine = new THREE.Line(pathGeometry, pathMaterial);
        scene.add(pathLine);

        camera.position.set(planData.problem.grid_size / 2, 3, planData.problem.grid_size);
        controls.target.set(planData.problem.grid_size / 2, 0, planData.problem.grid_size / 2);

        await initSceneObjects();
        animate();
    } catch (err) {
        console.error('Initialization error:', err);
    }
}

function updateFrame(f) {
    try {
        if (!f) return;

        if (f.step_text) {
            textDisplayTime = 2;
            textFadeTime = 1;
            document.getElementById('stepLogText').value = f.step_text + '\n' + document.getElementById('stepLogText').value;
            currentTime = f.time;
            document.getElementById('timeDisplay').textContent = `Time: ${currentTime.toFixed(1)}s`;
        }

        for (let coord in f.snow) {
            if (grid[coord]) {
                grid[coord].material.color.set(f.snow[coord] ? 0xE0FFFF : 0x90EE90);
                grid[coord].material.needsUpdate = true;
            }
        }

        for (let b in f.balls) {
            if (!balls[b]) {
                const geometry = new THREE.SphereGeometry(radius[f.ball_size[b]], 32, 32);
                const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0.2 });
                balls[b] = new THREE.Mesh(geometry, material);
                balls[b].castShadow = true;
                balls[b].receiveShadow = true;
                scene.add(balls[b]);
            }
            const [x, y] = f.balls[b].split(',').map(Number);
            let posX = x + 0.5, posZ = y + 0.5, posY = radius[f.ball_size[b]];
            if (f.type === "goal" && f.balls.ball_0 === "0,2" && f.balls.ball_1 === "0,2" && f.balls.ball_2 === "0,2") {
                posX = 0.5;
                posZ = 2.5;
                posY = b === "ball_2" ? 0.15 : b === "ball_1" ? 0.55 : 1.05;
            } else if (['move_ball', 'push', 'roll', 'roll_ball'].includes(f.type) && f.ball === b) {
                const [sx, sy] = f.start.split(',').map(Number);
                const [ex, ey] = f.end.split(',').map(Number);
                posX = sx + 0.5 + f.alpha * (ex - sx);
                posZ = sy + 0.5 + f.alpha * (ey - sy);
            }
            balls[b].geometry = new THREE.SphereGeometry(radius[f.ball_size[b]], 32, 32);
            balls[b].position.set(posX, posY, posZ);
        }

        if (character && f.character) {
            let cx, cz, rotationY = 0;
            if (['move_ball', 'push', 'roll', 'roll_ball'].includes(f.type)) {
                const [bx, by] = f.start.split(',').map(Number);
                const [ex, ey] = f.end.split(',').map(Number);
                cx = bx + 0.5;
                cz = by + 0.5;
                rotationY = f.direction === 'left' ? Math.PI : f.direction === 'right' ? 0 : f.direction === 'up' ? Math.PI / 2 : f.direction === 'down' ? -Math.PI / 2 : 0;
                if (f.alpha > 0) {
                    cx += f.alpha * 0.2 * (ex - bx);
                    cz += f.alpha * 0.2 * (ey - by);
                }
                spotlight.position.set(cx, 2, cz);
                spotlight.target.position.set(balls[f.ball].position.x, balls[f.ball].position.y, balls[f.ball].position.z);
            } else {
                const [x, y] = f.character.split(',').map(Number);
                cx = x + 0.5;
                cz = y + 0.5;
            }
            character.position.set(cx, 0, cz);
            character.rotation.y = rotationY;
        }

        if (snowman) scene.remove(snowman);
        if (f.step_text && f.step_text.includes('Goal') && f.balls.ball_0 === "0,2" && f.balls.ball_1 === "0,2" && f.balls.ball_2 === "0,2") {
            snowman = snowman.clone();
            snowman.position.set(0.5, 0, 2.5);
            snowman.rotation.y = Math.PI / 4;
            scene.add(snowman);
        }
    } catch (err) {
        console.error('Error updating frame:', err);
    }
}

function resetScene() {
    try {
        planData = {
            problem: {
                grid_size: 5,
                snow: Object.fromEntries(Array(5).fill().flatMap((_, x) => Array(5).fill().map((_, y) => [[x, y].join(','), x === 0 || x === 4]))),
                balls: { ball_0: "1,2", ball_1: "2,2", ball_2: "3,2" },
                ball_size: { ball_0: 2, ball_1: 1, ball_2: 0 },
                character: "2,1"
            },
            frames: [],
            isNumeric: false
        };
        currentFrame = 0;
        isPlaying = false;
        currentTime = 0;
        document.getElementById('planFile').value = '';
        document.getElementById('playPause').textContent = 'Play';
        document.getElementById('step').value = 0;
        document.getElementById('step').max = 0;
        document.getElementById('step').disabled = true;
        document.getElementById('stepLogText').value = '';
        document.getElementById('timeDisplay').textContent = 'Time: 0s';
        ['playPause', 'stepForward', 'stepBackward', 'reset'].forEach(id => document.getElementById(id).disabled = true);
        scene.clear();
        grid = {};
        balls = {};
        character = null;
        snowman = null;
        particles = null;
        snowParticles = [];
        snowParticleVelocities = [];
        pathLine = null;
        spotlight = null;
        mixer = null;
        renderer.dispose();
        renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('canvas'), antialias: true });
        renderer.setSize(window.innerWidth * 0.65, window.innerHeight - 70);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        camera = new THREE.PerspectiveCamera(75, 0.65 * window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(2.5, 3, 5);
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 2;
        controls.maxDistance = 10;
        controls.target.set(2.5, 0, 2.5);
        init();
    } catch (err) {
        console.error('Error resetting scene:', err);
    }
}

function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    controls.update();

    if (isPlaying && planData.frames.length > 0 && currentFrame < planData.frames.length - 1) {
        updateFrame(planData.frames[Math.floor(currentFrame)]);
        currentFrame += speed;
        document.getElementById('step').value = Math.floor(currentFrame / 10);
        if (currentFrame >= planData.frames.length - 1) {
            isPlaying = false;
            document.getElementById('playPause').textContent = 'Play';
        }
    }

    snowParticles.forEach((mesh, i) => {
        const velocity = snowParticleVelocities[i];
        mesh.position.x += velocity.x * delta;
        mesh.position.y += velocity.y * delta * snowSpeed;
        mesh.position.z += velocity.z * delta;
        if (mesh.position.y < 0) {
            scene.remove(mesh);
            snowParticles.splice(i, 1);
            snowParticleVelocities.splice(i, 1);
        }
    });

    if (mixer && isPlaying && ['move_character', 'move', 'move_to', 'move_char', 'move_ball', 'push', 'roll', 'roll_ball'].includes(planData.frames[Math.floor(currentFrame)]?.type)) {
        mixer.update(delta);
    }

    renderer.render(scene, camera);
}

document.getElementById('planFile').addEventListener('change', async e => await loadPlanData(e.target.files[0]));
document.getElementById('playPause').addEventListener('click', () => {
    if (planData.frames.length === 0) return;
    isPlaying = !isPlaying;
    document.getElementById('playPause').textContent = isPlaying ? 'Pause' : 'Play';
});
document.getElementById('stepForward').addEventListener('click', () => {
    if (planData.frames.length === 0) return;
    if (!isPlaying && currentFrame < planData.frames.length - 10) {
        currentFrame = Math.min(currentFrame + 10, planData.frames.length - 1);
        document.getElementById('step').value = Math.floor(currentFrame / 10);
        updateFrame(planData.frames[Math.floor(currentFrame)]);
    }
});
document.getElementById('stepBackward').addEventListener('click', () => {
    if (planData.frames.length === 0) return;
    if (!isPlaying && currentFrame >= 10) {
        currentFrame -= 10;
        document.getElementById('step').value = Math.floor(currentFrame / 10);
        updateFrame(planData.frames[Math.floor(currentFrame)]);
    }
});
document.getElementById('reset').addEventListener('click', resetScene);
document.getElementById('speed').addEventListener('input', e => speed = Number(e.target.value));
document.getElementById('snowSpeed').addEventListener('input', e => snowSpeed = Number(e.target.value));
document.getElementById('step').addEventListener('input', e => {
    if (planData.frames.length === 0) return;
    currentFrame = Number(e.target.value) * 10;
    isPlaying = false;
    document.getElementById('playPause').textContent = 'Play';
    updateFrame(planData.frames[Math.floor(currentFrame)]);
});

window.addEventListener('resize', () => {
    camera.aspect = 0.65 * window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth * 0.65, window.innerHeight - 70);
    controlPanel.style.left = '10px';
    controlPanel.style.top = '50px';
    stepLogPanel.style.left = '10px';
    stepLogPanel.style.top = '70%';
    stepLogPanel.style.width = `calc(35% - 20px)`;
    stepLogPanel.style.height = '30%';
});

init();