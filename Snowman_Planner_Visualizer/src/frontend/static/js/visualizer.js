import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/loaders/GLTFLoader.js';

// Configuration constants
const CONFIG = {
    RADIUS: { 0: 0.15, 1: 0.25, 2: 0.35 }, // Radii for small, medium, large balls
    SUBSTEPS: 10, // Number of animation substeps per plan action
    GOAL_POS: '2,0', // Target grid position for snowman (0-indexed)
    COLORS: {
        SNOW: 0xE0FFFF, // Light Cyan for snow cells
        GRID_DEFAULT: 0x90EE90, // Light Green for non-snow goal cells
        GRID_UNTOUCHED: 0x556B2F, // Dark Olive Green for other cells
        CHARACTER_DEFAULT: 0x8B4513, // SaddleBrown for soldier model
        CHARACTER_FALLBACK: 0xFFFF00, // Yellow for fallback box
        BARRIER: 0xFF0000 // Red for forbidden icons
    },
    SPECIAL_POSITIONS: ['1,1', '1,3', '3,1', '3,3'], // Forbidden grid positions
    ANIMATION: {
        FRAME_RATE: 60, // Target frame rate
        SNOW_SPEED_DEFAULT: 0.2, // Default snow particle speed
        SPEED_DEFAULT: 1 // Default animation speed
    }
};

// Global state
const state = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    grid: {}, // Grid plane meshes
    balls: {}, // Ball meshes
    character: null, // Character mesh/model
    mixer: null, // Animation mixer for character
    spotlight: null, // Spotlight following character
    snowParticles: [], // Snow particle system
    snowParticleVelocities: [], // Velocities for snow particles
    pathLine: null, // Path line (currently unused)
    forbiddenIcons: {}, // Forbidden icon meshes
    clock: new THREE.Clock(),
    planData: {
        problem: {
            grid_size: 5,
            snow: Object.fromEntries(
                Array(5).fill().flatMap((_, x) => 
                    Array(5).fill().map((_, y) => [[x, y].join(','), x === 0 || x === 4])
                )
            ),
            balls: {},
            ball_size: {},
            character: '2,1'
        },
        frames: [],
        isNumeric: false
    },
    currentFrame: 0,
    isPlaying: false,
    speed: CONFIG.ANIMATION.SPEED_DEFAULT,
    snowSpeed: CONFIG.ANIMATION.SNOW_SPEED_DEFAULT,
    currentTime: 0,
    startTime: null,
    problemFile: null,
    planFile: null,
    frameRate: 0,
    lastFrameTime: performance.now()
};

// UI elements
const UI = {
    playIcon: document.getElementById('playIcon'),
    pauseIcon: document.getElementById('pauseIcon'),
    playPauseText: document.getElementById('playPauseText'),
    controlPanel: document.getElementById('controlPanel')
};

/**
 * Utility Functions
 */

/**
 * Shows a pop-up message.
 * @param {string} popupId - ID of the pop-up element.
 * @param {string} messageId - ID of the message element within the pop-up.
 * @param {string} message - Message content (HTML supported).
 */
function showPopup(popupId, messageId, message) {
    document.querySelectorAll('#errorPopup, #helpPopup, #metricsPopup, #aboutPopup')
        .forEach(p => p.style.display = 'none');
    const popup = document.getElementById(popupId);
    document.getElementById(messageId).innerHTML = message;
    popup.style.display = 'block';
    popup.classList.add('fadeIn');
    setTimeout(() => popup.classList.remove('fadeIn'), 500);
}

/**
 * Hides a pop-up.
 * @param {string} popupId - ID of the pop-up element.
 */
function hidePopup(popupId) {
    const popup = document.getElementById(popupId);
    popup.classList.add('fadeOut');
    setTimeout(() => {
        popup.style.display = 'none';
        popup.classList.remove('fadeOut');
    }, 500);
}

/**
 * Parses a location string (e.g., "loc_3_1") into [row, col].
 * @param {string} loc - Location string.
 * @returns {number[]} - [row, col] in 0-based indexing.
 */
function parseLoc(loc) {
    try {
        const [, row, col] = loc.match(/loc_(\d+)_(\d+)/) || [];
        if (!row || !col) throw new Error(`Invalid location format: ${loc}`);
        return [parseInt(row) - 1, parseInt(col) - 1];
    } catch (e) {
        showPopup('errorPopup', 'errorMessage', `Error parsing location '${loc}': ${e.message}`);
        throw e;
    }
}

/**
 * PDDL Parsing and Frame Building
 */

/**
 * Parses PDDL problem file content.
 * @param {string} content - PDDL file content.
 * @returns {object} - Parsed problem data.
 */
function parseProblem(content) {
    if (!content.trim()) throw new Error("Empty problem file");
    
    const result = {
        snow: {},
        balls: {},
        ball_size: {},
        character: null,
        grid_size: 5,
        domain: 'unknown'
    };
    const gridPositions = new Set();

    // Extract domain
    const domainMatch = content.match(/:domain (\S+)/);
    if (domainMatch) result.domain = domainMatch[1];

    // Parse snow locations
    content.replace(/\(:location_type (\S+) (\d+)\)/g, (_, loc, type) => {
        const coord = parseLoc(loc).join(',');
        result.snow[coord] = type === '1';
        gridPositions.add(coord);
    });
    content.replace(/\(snow (\S+)\)/g, (_, loc) => {
        const coord = parseLoc(loc).join(',');
        result.snow[coord] = true;
        gridPositions.add(coord);
    });

    // Parse ball positions
    content.replace(/\(ball_at (\S+) (\S+)\)/g, (_, ball, loc) => {
        const coord = parseLoc(loc);
        result.balls[ball] = coord;
        gridPositions.add(coord.join(','));
    });

    // Parse ball sizes
    content.replace(/\(:ball_size (\S+) (\d+)\)/g, (_, ball, size) => {
        const sizeInt = parseInt(size);
        if (![0, 1, 2].includes(sizeInt)) throw new Error(`Invalid ball size ${size} for ${ball}`);
        result.ball_size[ball] = sizeInt;
    });
    content.replace(/\(ball_size_(small|medium|large) (\S+)\)/g, (_, sizeStr, ball) => {
        result.ball_size[ball] = { small: 0, medium: 1, large: 2 }[sizeStr.toLowerCase()];
    });

    // Parse character position
    const charMatch = content.match(/\(character_at (\S+)\)/);
    if (charMatch) {
        result.character = parseLoc(charMatch[1]).join(',');
        gridPositions.add(result.character);
    } else {
        throw new Error("No character position found");
    }

    // Set default ball sizes
    for (const ball in result.balls) {
        if (!(ball in result.ball_size)) result.ball_size[ball] = 0;
    }

    // Calculate grid size
    if (gridPositions.size) {
        const coords = Array.from(gridPositions).map(c => c.split(',').map(Number));
        result.grid_size = Math.max(...coords.flat()) + 1;
    }

    // Ensure all grid cells have snow state
    for (let r = 0; r < result.grid_size; r++) {
        for (let c = 0; c < result.grid_size; c++) {
            if (!(r + ',' + c in result.snow)) result.snow[r + ',' + c] = false;
        }
    }

    console.log("[parseProblem] Parsed:", result);
    return result;
}

/**
 * Parses plan file content into action steps.
 * @param {string} content - Plan file content.
 * @returns {string[]} - Array of action strings.
 */
function parsePlan(content) {
    if (!content.trim()) throw new Error("Empty plan file");
    
    const steps = content.trim().split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.startsWith(';'))
        .map(line => line.replace(/^\d+[.:]?\s*/, '').replace(/^\(|\)$/g, '').trim())
        .filter(line => ['move', 'move_to', 'move_ball', 'push', 'roll', 'roll_ball', 'goal', 'move_character']
            .some(k => line.toLowerCase().includes(k)));

    if (!steps.length) throw new Error("No valid actions found");
    console.log("[parsePlan] Parsed steps:", steps);
    return steps;
}

/**
 * Builds animation frames from problem and plan data.
 * @param {object} prob - Parsed problem data.
 * @param {string[]} plan - Parsed plan actions.
 * @returns {object[]} - Array of frame objects.
 */
function buildFrames(prob, plan) {
    const frames = [];
    const state = {
        snow: { ...prob.snow },
        balls: Object.fromEntries(Object.entries(prob.balls).map(([k, v]) => [k, v.join(',')])),
        ball_size: { ...prob.ball_size },
        character: prob.character,
        grid_size: prob.grid_size,
        isNumeric: prob.domain.includes('snowman_numeric')
    };

    frames.push({
        type: 'initial',
        balls: { ...state.balls },
        ball_size: { ...state.ball_size },
        snow: { ...state.snow },
        character: state.character,
        grid_size: state.grid_size,
        time: 0,
        alpha: 0
    });

    let step_count = 0;
    for (const action of plan) {
        const parts = action.split(/\s+/);
        try {
            if (['move_character', 'move', 'move_to'].includes(parts[0])) {
                if (parts.length < 3) throw new Error(`Invalid move action: ${action}`);
                const start = parseLoc(parts[1]);
                const end = parseLoc(parts[2]);
                const direction = parts[3] || null;
                
                for (let t = 0; t < CONFIG.SUBSTEPS; t++) {
                    frames.push({
                        type: 'move',
                        start: start.join(','),
                        end: end.join(','),
                        alpha: t / (CONFIG.SUBSTEPS - 1),
                        balls: { ...state.balls },
                        ball_size: { ...state.ball_size },
                        snow: { ...state.snow },
                        character: state.character,
                        grid_size: state.grid_size,
                        time: step_count + t / CONFIG.SUBSTEPS,
                        direction
                    });
                }
                state.character = end.join(',');
            } else if (['move_ball', 'push', 'roll', 'roll_ball'].includes(parts[0])) {
                if (parts.length < 5) throw new Error(`Invalid move_ball action: ${action}`);
                const [, ball, fromCell, , toCell] = parts;
                const start = parseLoc(fromCell);
                const end = parseLoc(toCell);
                const direction = parts[5] || null;

                // Character moves to ball's start
                for (let t = 0; t < CONFIG.SUBSTEPS; t++) {
                    frames.push({
                        type: 'move_to_ball',
                        start: state.character,
                        end: start.join(','),
                        alpha: t / (CONFIG.SUBSTEPS - 1),
                        balls: { ...state.balls },
                        ball_size: { ...state.ball_size },
                        snow: { ...state.snow },
                        character: state.character,
                        grid_size: state.grid_size,
                        time: step_count + t / CONFIG.SUBSTEPS,
                        direction
                    });
                }
                state.character = start.join(',');

                // Ball movement
                for (let t = 0; t < CONFIG.SUBSTEPS; t++) {
                    frames.push({
                        type: 'move_ball',
                        ball,
                        start: start.join(','),
                        end: end.join(','),
                        alpha: t / (CONFIG.SUBSTEPS - 1),
                        balls: { ...state.balls },
                        ball_size: { ...state.ball_size },
                        snow: { ...state.snow },
                        character: state.character,
                        grid_size: state.grid_size,
                        time: step_count + t / CONFIG.SUBSTEPS,
                        direction
                    });
                }
                state.balls[ball] = end.join(',');

                if (state.snow[end.join(',')]) {
                    state.ball_size[ball] = Math.min(state.ball_size[ball] + 1, 2);
                    state.snow[end.join(',')] = false;
                }
            } else if (parts[0] === 'goal') {
                const ballsAtGoal = Object.entries(state.balls).filter(([_, pos]) => pos === CONFIG.GOAL_POS);
                for (let t = 0; t < CONFIG.SUBSTEPS; t++) {
                    frames.push({
                        type: 'goal',
                        balls: { ...state.balls },
                        ball_size: { ...state.ball_size },
                        snow: { ...state.snow },
                        character: state.character,
                        grid_size: state.grid_size,
                        time: step_count + t / CONFIG.SUBSTEPS,
                        alpha: t / (CONFIG.SUBSTEPS - 1)
                    });
                }
            } else {
                for (let t = 0; t < CONFIG.SUBSTEPS; t++) {
                    frames.push({
                        type: 'static',
                        balls: { ...state.balls },
                        ball_size: { ...state.ball_size },
                        snow: { ...state.snow },
                        character: state.character,
                        grid_size: state.grid_size,
                        time: step_count + t / CONFIG.SUBSTEPS,
                        alpha: t / (CONFIG.SUBSTEPS - 1)
                    });
                }
            }
            step_count++;
        } catch (e) {
            console.error(`Error processing action '${action}' at step ${step_count + 1}:`, e);
            showPopup('errorPopup', 'errorMessage', `Error in action '${action}': ${e.message}`);
            for (let t = 0; t < CONFIG.SUBSTEPS; t++) {
                frames.push({
                    type: 'error',
                    balls: { ...state.balls },
                    ball_size: { ...state.ball_size },
                    snow: { ...state.snow },
                    character: state.character,
                    grid_size: state.grid_size,
                    time: step_count + t / CONFIG.SUBSTEPS,
                    alpha: t / (CONFIG.SUBSTEPS - 1)
                });
            }
            step_count++;
        }
    }
    console.log(`[buildFrames] Generated ${frames.length} frames, isNumeric: ${state.isNumeric}`);
    return frames;
}

/**
 * Scene Management
 */

/**
 * Creates a 3D forbidden icon for special grid cells.
 * @returns {THREE.Group} - Group containing barrier parts.
 */
function createForbiddenIcon() {
    const group = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({ 
        color: CONFIG.COLORS.BARRIER, 
        roughness: 0.5, 
        metalness: 0.1 
    });

    // Base barrier
    const base = new THREE.Mesh(
        new THREE.BoxGeometry(0.8, 0.05, 0.2),
        material
    );
    base.position.y = 0.025;
    base.castShadow = true;
    base.receiveShadow = true;
    group.add(base);

    // Vertical posts
    const postGeometry = new THREE.BoxGeometry(0.05, 0.3, 0.05);
    const post1 = new THREE.Mesh(postGeometry, material);
    post1.position.set(-0.3, 0.175, 0);
    post1.castShadow = true;
    post1.receiveShadow = true;
    group.add(post1);

    const post2 = new THREE.Mesh(postGeometry, material);
    post2.position.set(0.3, 0.175, 0);
    post2.castShadow = true;
    post2.receiveShadow = true;
    group.add(post2);

    return group;
}

/**
 * Clears dynamic scene objects (balls, character, spotlight).
 */
function clearDynamicSceneObjects() {
    const objectsToRemove = [];
    state.scene.traverse(obj => {
        if (obj.name && (
            obj.name.startsWith('ball_') ||
            obj.name === 'character' ||
            obj.name === 'spotlight'
        )) {
            objectsToRemove.push(obj);
        }
    });

    objectsToRemove.forEach(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            Array.isArray(obj.material) 
                ? obj.material.forEach(mat => mat.dispose())
                : obj.material.dispose();
        }
        state.scene.remove(obj);
    });

    state.balls = {};
    state.character = null;
    if (state.mixer) {
        state.mixer.stopAllAction();
        state.mixer = null;
    }
    state.spotlight = null;
    console.log('[clearDynamicSceneObjects] Dynamic objects cleared');
}

/**
 * Initializes static environment (ground, grid, walls, trees, forbidden icons).
 * @param {object} problemData - Problem data for static setup.
 */
async function initStaticEnvironment(problemData) {
    console.log('[initStaticEnvironment] Initializing static environment');

    // Clear existing static objects
    const staticObjects = [];
    state.scene.traverse(obj => {
        if (obj.name && (
            obj.name.startsWith('grid_') ||
            obj.name === 'ground_plane' ||
            obj.name.startsWith('wall_') ||
            obj.name.startsWith('tree_') ||
            obj.name.startsWith('forbidden_icon_')
        )) {
            staticObjects.push(obj);
        }
    });
    staticObjects.forEach(obj => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            Array.isArray(obj.material) 
                ? obj.material.forEach(mat => mat.dispose())
                : obj.material.dispose();
        }
        state.scene.remove(obj);
    });
    state.grid = {};
    state.forbiddenIcons = {};

    // Ground plane
    const ground = new THREE.Mesh(
        new THREE.PlaneGeometry(problemData.grid_size + 2, problemData.grid_size + 2),
        new THREE.MeshStandardMaterial({ 
            color: CONFIG.COLORS.SNOW, 
            roughness: 0.8, 
            metalness: 0.1 
        })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(problemData.grid_size / 2, -0.01, problemData.grid_size / 2);
    ground.receiveShadow = true;
    ground.name = 'ground_plane';
    state.scene.add(ground);

    // Grid planes
    const [goalX, goalY] = CONFIG.GOAL_POS.split(',').map(Number);
    for (let x = 0; x < problemData.grid_size; x++) {
        for (let y = 0; y < problemData.grid_size; y++) {
            const coord = `${x},${y}`;
            const plane = new THREE.Mesh(
                new THREE.PlaneGeometry(1, 1),
                new THREE.MeshStandardMaterial({
                    color: CONFIG.SPECIAL_POSITIONS.includes(coord) ? 0xFFFFFF :
                           problemData.snow[coord] ? CONFIG.COLORS.SNOW :
                           (x === goalX && y === goalY) ? CONFIG.COLORS.GRID_DEFAULT :
                           CONFIG.COLORS.GRID_UNTOUCHED,
                    side: THREE.DoubleSide
                })
            );
            plane.rotation.x = -Math.PI / 2;
            plane.position.set(x + 0.5, 0, y + 0.5);
            plane.receiveShadow = true;
            plane.name = `grid_${coord}`;
            state.scene.add(plane);
            state.grid[coord] = plane;

            if (CONFIG.SPECIAL_POSITIONS.includes(coord)) {
                const icon = createForbiddenIcon();
                icon.position.set(x + 0.5, 0, y + 0.5);
                icon.name = `forbidden_icon_${coord}`;
                state.scene.add(icon);
                state.forbiddenIcons[coord] = icon;
            }
        }
    }

    // Walls
    const wallMaterial = new THREE.MeshStandardMaterial({ 
        color: 0xADD8E6, 
        transparent: true, 
        opacity: 0.7, 
        roughness: 0.3, 
        metalness: 0.5 
    });
    const wallHeight = 0.5;
    const wallThickness = 0.2;
    const halfGridSize = problemData.grid_size / 2;
    [
        { geometry: new THREE.BoxGeometry(problemData.grid_size + wallThickness * 2, wallHeight, wallThickness), position: [halfGridSize, wallHeight / 2, -wallThickness / 2], name: 'wall_top' },
        { geometry: new THREE.BoxGeometry(problemData.grid_size + wallThickness * 2, wallHeight, wallThickness), position: [halfGridSize, wallHeight / 2, problemData.grid_size + wallThickness / 2], name: 'wall_bottom' },
        { geometry: new THREE.BoxGeometry(wallThickness, wallHeight, problemData.grid_size + wallThickness * 2), position: [-wallThickness / 2, wallHeight / 2, halfGridSize], name: 'wall_left' },
        { geometry: new THREE.BoxGeometry(wallThickness, wallHeight, problemData.grid_size + wallThickness * 2), position: [problemData.grid_size + wallThickness / 2, wallHeight / 2, halfGridSize], name: 'wall_right' }
    ].forEach(({ geometry, position, name }) => {
        const wall = new THREE.Mesh(geometry, wallMaterial);
        wall.position.set(...position);
        wall.castShadow = true;
        wall.receiveShadow = true;
        wall.name = name;
        state.scene.add(wall);
    });

    // Trees
    const treeGeometry = new THREE.ConeGeometry(0.3, 0.8, 8);
    const treeMaterial = new THREE.MeshStandardMaterial({ color: 0x228B22, roughness: 0.9 });
    const treePositions = [
        [-1, 0, -1], [problemData.grid_size + 1, 0, -1],
        [-1, 0, problemData.grid_size + 1], [problemData.grid_size + 1, 0, problemData.grid_size + 1],
        [halfGridSize, 0, -1], [halfGridSize, 0, problemData.grid_size + 1],
        [-1, 0, halfGridSize], [problemData.grid_size + 1, 0, halfGridSize]
    ];
    treePositions.forEach((pos, index) => {
        const tree = new THREE.Mesh(treeGeometry, treeMaterial);
        tree.position.set(pos[0], pos[1] + 0.4, pos[2]);
        tree.castShadow = true;
        tree.receiveShadow = true;
        tree.name = `tree_${index}`;
        state.scene.add(tree);
    });

    console.log('[initStaticEnvironment] Static environment initialized');
}

/**
 * Initializes dynamic scene objects (balls, character, spotlight).
 * @param {object} problemData - Current problem data.
 */
async function initDynamicSceneObjects(problemData) {
    console.log('[initDynamicSceneObjects] Initializing dynamic objects:', problemData);

    const loader = new GLTFLoader();

    // Load character model
    if (!state.character) {
        try {
            const gltf = await loader.loadAsync('https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb');
            state.character = gltf.scene;
            state.character.scale.set(0.4, 0.4, 0.4);
            state.character.castShadow = true;
            state.character.receiveShadow = true;
            state.character.name = 'character';

            // Handle animations
            if (gltf.animations.length) {
                state.mixer = new THREE.AnimationMixer(state.character);
                const walkAction = state.mixer.clipAction(gltf.animations.find(clip => clip.name.toLowerCase().includes('walk')) || gltf.animations[0]);
                walkAction.play();
            }

            state.character.traverse(obj => {
                if (obj.isMesh && obj.material) {
                    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
                    materials.forEach(mat => {
                        if (['MeshStandardMaterial', 'MeshBasicMaterial'].includes(mat.type)) {
                            mat.color.set(CONFIG.COLORS.CHARACTER_DEFAULT);
                        }
                    });
                }
            });
            state.scene.add(state.character);
            console.log('[initDynamicSceneObjects] Character loaded');
        } catch (e) {
            console.warn('[initDynamicSceneObjects] Fallback to box character:', e);
            state.character = new THREE.Mesh(
                new THREE.BoxGeometry(0.3, 0.8, 0.3),
                new THREE.MeshStandardMaterial({ color: CONFIG.COLORS.CHARACTER_FALLBACK })
            );
            state.character.position.y = 0.4;
            state.character.castShadow = true;
            state.character.name = 'character';
            state.scene.add(state.character);
        }
    }

    // Set character position
    const [cx, cy] = problemData.character.split(',').map(Number);
    state.character.position.set(cx + 0.5, 0, cy + 0.5);
    state.character.visible = true;

    // Create balls
    for (const ballName in state.balls) {
        state.scene.remove(state.balls[ballName]);
        if (state.balls[ballName].geometry) state.balls[ballName].geometry.dispose();
        if (state.balls[ballName].material) {
            Array.isArray(state.balls[ballName].material)
                ? state.balls[ballName].material.forEach(m => m.dispose())
                : state.balls[ballName].material.dispose();
        }
    }
    state.balls = {};

    for (const b in problemData.balls) {
        const ball = new THREE.Mesh(
            new THREE.SphereGeometry(CONFIG.RADIUS[problemData.ball_size[b]], 32, 32),
            new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0.2 })
        );
        ball.castShadow = true;
        ball.receiveShadow = true;
        const [x, y] = problemData.balls[b];
        ball.position.set(x + 0.5, CONFIG.RADIUS[problemData.ball_size[b]], y + 0.5);
        ball.name = `ball_${b}`;
        state.scene.add(ball);
        state.balls[b] = ball;
        console.log(`[initDynamicSceneObjects] Added ball ${b} (size ${problemData.ball_size[b]}) at (${x + 0.5}, ${CONFIG.RADIUS[problemData.ball_size[b]]}, ${y + 0.5})`);
    }

    // Spotlight
    if (state.spotlight) {
        state.scene.remove(state.spotlight);
        state.scene.remove(state.spotlight.target);
    }
    state.spotlight = new THREE.SpotLight(0xffffff, 0.8, 5, Math.PI / 4, 0.5, 2);
    state.spotlight.position.set(cx + 0.5, 3, cy + 0.5);
    state.spotlight.castShadow = true;
    state.scene.add(state.spotlight);
    state.scene.add(state.spotlight.target);
    console.log('[initDynamicSceneObjects] Spotlight initialized');
}

/**
 * Initializes core Three.js components.
 */
async function initCoreThreeJs() {
    state.scene = new THREE.Scene();
    state.camera = new THREE.PerspectiveCamera(75, 0.65 * window.innerWidth / window.innerHeight, 0.1, 1000);
    state.renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('canvas'), antialias: true });
    state.renderer.setSize(window.innerWidth * 0.65, window.innerHeight - 70);
    state.renderer.shadowMap.enabled = true;
    state.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    state.scene.fog = new THREE.Fog(0x87CEEB, 3, 10);

    // Lighting
    state.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(5, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.set(2048, 2048);
    state.scene.add(directionalLight);

    // Skybox
    new THREE.CubeTextureLoader().load([
        'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_px.jpg',
        'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_nx.jpg',
        'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_py.jpg',
        'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_ny.jpg',
        'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_pz.jpg',
        'https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/textures/cube/MilkyWay/dark-s_nz.jpg'
    ], texture => {
        state.scene.background = texture;
    }, undefined, () => {
        state.scene.background = new THREE.Color(0x87CEEB);
    });

    // Orbit controls
    state.controls = new OrbitControls(state.camera, state.renderer.domElement);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.05;
    state.controls.minDistance = 2;
    state.controls.maxDistance = 10;
    state.camera.position.set(state.planData.problem.grid_size / 2, 3, state.planData.problem.grid_size);
    state.controls.target.set(state.planData.problem.grid_size / 2, 0, state.planData.problem.grid_size / 2);

    // Snow particles
    const snowMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
    function dropSnowflake() {
        const size = 0.009 + Math.random() * 0.012;
        const mesh = new THREE.Mesh(new THREE.CircleGeometry(size, 32), snowMaterial);
        mesh.position.set(
            Math.random() * (state.planData.problem.grid_size + 2) - 1,
            Math.random() * 5 + 3,
            Math.random() * (state.planData.problem.grid_size + 2) - 1
        );
        const velocity = new THREE.Vector3(
            (Math.random() - 0.5) * 0.05 + 0.02 * Math.sin(state.clock.getElapsedTime()),
            -state.snowSpeed,
            (Math.random() - 0.5) * 0.05 + 0.02 * Math.cos(state.clock.getElapsedTime())
        );
        state.snowParticles.push(mesh);
        state.snowParticleVelocities.push(velocity);
        state.scene.add(mesh);
    }
    setInterval(dropSnowflake, 200); // Increased frequency for more snow

    // Path line (unused)
    state.pathLine = new THREE.Line(
        new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.5 })
    );
    state.scene.add(state.pathLine);

    console.log('[initCoreThreeJs] Core Three.js initialized');
    animate();
}

/**
 * Updates the scene based on the current frame.
 * @param {object} frame - Current frame data.
 */
function updateFrame(frame) {
    if (!frame) return;

    console.log(`[updateFrame] Frame ${Math.floor(state.currentFrame)}: ${frame.type}`);

    // Update grid colors
    const [goalX, goalY] = CONFIG.GOAL_POS.split(',').map(Number);
    for (let x = 0; x < frame.grid_size; x++) {
        for (let y = 0; y < frame.grid_size; y++) {
            const coord = `${x},${y}`;
            if (state.grid[coord]) {
                state.grid[coord].material.color.set(
                    CONFIG.SPECIAL_POSITIONS.includes(coord) ? 0xFFFFFF :
                    frame.snow[coord] ? CONFIG.COLORS.SNOW :
                    (x === goalX && y === goalY) ? CONFIG.COLORS.GRID_DEFAULT :
                    CONFIG.COLORS.GRID_UNTOUCHED
                );
                state.grid[coord].material.needsUpdate = true;
            }
        }
    }

    // Handle ball stacking at goal
    const ballsAtGoal = Object.entries(frame.balls)
        .filter(([_, pos]) => pos === CONFIG.GOAL_POS)
        .map(([b]) => ({ name: b, size: frame.ball_size[b] }))
        .sort((a, b) => b.size - a.size);

    // Update balls
    for (const b in frame.balls) {
        if (!state.balls[b]) {
            console.warn(`[updateFrame] Creating ball ${b} dynamically`);
            const ball = new THREE.Mesh(
                new THREE.SphereGeometry(CONFIG.RADIUS[frame.ball_size[b]], 32, 32),
                new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0.2 })
            );
            ball.castShadow = true;
            ball.receiveShadow = true;
            ball.name = `ball_${b}`;
            state.scene.add(ball);
            state.balls[b] = ball;
        }

        if (state.balls[b].geometry.parameters.radius !== CONFIG.RADIUS[frame.ball_size[b]]) {
            state.balls[b].geometry.dispose();
            state.balls[b].geometry = new THREE.SphereGeometry(CONFIG.RADIUS[frame.ball_size[b]], 32, 32);
        }

        let posX, posY, posZ;
        if (frame.balls[b] === CONFIG.GOAL_POS) {
            let currentHeight = 0;
            for (const { name, size } of ballsAtGoal) {
                if (name === b) {
                    posY = currentHeight + CONFIG.RADIUS[size];
                    break;
                }
                currentHeight += CONFIG.RADIUS[size] * 2;
            }
            posX = goalX + 0.5;
            posZ = goalY + 0.5;
        } else if (['move_ball', 'push', 'roll', 'roll_ball'].includes(frame.type) && frame.ball === b) {
            const [sx, sy] = frame.start.split(',').map(Number);
            const [ex, ey] = frame.end.split(',').map(Number);
            posX = sx + 0.5 + frame.alpha * (ex - sx);
            posZ = sy + 0.5 + frame.alpha * (ey - sy);
            posY = CONFIG.RADIUS[frame.ball_size[b]];
        } else {
            const [x, y] = frame.balls[b].split(',').map(Number);
            posX = x + 0.5;
            posZ = y + 0.5;
            posY = CONFIG.RADIUS[frame.ball_size[b]];
        }
        state.balls[b].position.set(posX, posY, posZ);
        state.balls[b].visible = true;
    }

    // Snowman visibility
    const ballsAtGoalCount = ballsAtGoal.length;
    const sizesAtGoal = new Set(ballsAtGoal.map(b => b.size));
    const isSnowmanFormed = ballsAtGoalCount === 3 && sizesAtGoal.has(0) && sizesAtGoal.has(1) && sizesAtGoal.has(2);

    if (state.character) {
        state.character.visible = !(isSnowmanFormed && frame.character === CONFIG.GOAL_POS);
    }

    // Update character
    if (state.character && state.character.visible && frame.character) {
        let cx, cz, rotationY = 0;
        if (frame.type === 'move_to_ball') {
            const [sx, sy] = frame.start.split(',').map(Number);
            const [ex, ey] = frame.end.split(',').map(Number);
            cx = sx + 0.5 + frame.alpha * (ex - sx);
            cz = sy + 0.5 + frame.alpha * (ey - sy);
            rotationY = {
                left: Math.PI,
                right: 0,
                up: Math.PI / 2,
                down: -Math.PI / 2
            }[frame.direction] || 0;
            state.spotlight.position.set(cx, 2, cz);
            state.spotlight.target.position.set(cx, 0, cz);
        } else if (['move_ball', 'push', 'roll', 'roll_ball'].includes(frame.type)) {
            const [bx, by] = frame.start.split(',').map(Number);
            const [ex, ey] = frame.end.split(',').map(Number);
            cx = bx + 0.5 + frame.alpha * (ex - bx);
            cz = by + 0.5 + frame.alpha * (ey - by);
            rotationY = {
                left: Math.PI,
                right: 0,
                up: Math.PI / 2,
                down: -Math.PI / 2
            }[frame.direction] || 0;
            state.spotlight.position.set(cx, 2, cz);
            state.spotlight.target.position.set(state.balls[frame.ball].position.x, state.balls[frame.ball].position.y, state.balls[frame.ball].position.z);
        } else if (frame.type === 'move') {
            const [sx, sy] = frame.start.split(',').map(Number);
            const [ex, ey] = frame.end.split(',').map(Number);
            cx = sx + 0.5 + frame.alpha * (ex - sx);
            cz = sy + 0.5 + frame.alpha * (ey - sy);
            rotationY = {
                left: Math.PI,
                right: 0,
                up: Math.PI / 2,
                down: -Math.PI / 2
            }[frame.direction] || 0;
            state.spotlight.position.set(cx, 2, cz);
            state.spotlight.target.position.set(cx, 0, cz);
        } else {
            const [x, y] = frame.character.split(',').map(Number);
            cx = x + 0.5;
            cz = y + 0.5;
            state.spotlight.position.set(cx, 2, cz);
            state.spotlight.target.position.set(cx, 0, cz);
        }
        state.character.position.set(cx, 0, cz);
        state.character.rotation.y = rotationY;
    }
}

/**
 * File Handling
 */

/**
 * Reads file content as text.
 * @param {File} file - File to read.
 * @returns {Promise<string>} - File content.
 */
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
        reader.readAsText(file);
    });
}

/**
 * Handles file selection and loading.
 * @param {Event} e - File input change event.
 */
async function selectFiles(e) {
    const problemInput = document.getElementById('problemFile');
    const planInput = document.getElementById('planFile');
    const changedInput = e.target.id;

    console.log(`[selectFiles] Input changed: ${changedInput}`);

    if (changedInput === 'problemFile' && problemInput.files[0]) {
        if (!problemInput.files[0].name.endsWith('.pddl')) {
            showPopup('errorPopup', 'errorMessage', 'Please select a .pddl problem file');
            problemInput.value = '';
            state.problemFile = null;
            return;
        }
        state.problemFile = problemInput.files[0];
    } else if (changedInput === 'planFile' && planInput.files[0]) {
        if (!planInput.files[0].name.endsWith('.txt') && !planInput.files[0].name.endsWith('.plan')) {
            showPopup('errorPopup', 'errorMessage', 'Please select a .txt or .plan file');
            planInput.value = '';
            state.planFile = null;
            return;
        }
        state.planFile = planInput.files[0];
    }

    if (state.problemFile && state.planFile) {
        try {
            resetScene(false);
            const [problemContent, planContent] = await Promise.all([
                readFile(state.problemFile),
                readFile(state.planFile)
            ]);
            await loadFiles(problemContent, planContent);
            if (state.planData.frames.length) {
                updateFrame(state.planData.frames[0]);
                state.renderer.render(state.scene, state.camera);
            }
        } catch (err) {
            showPopup('errorPopup', 'errorMessage', `Error reading files: ${err.message}`);
            resetScene(false);
        }
    } else {
        ['playPause', 'stepForward', 'stepBackward', 'reset', 'step'].forEach(id => {
            document.getElementById(id).disabled = true;
        });
    }
}

/**
 * Loads and processes problem and plan files.
 * @param {string} problemContent - Problem file content.
 * @param {string} planContent - Plan file content.
 */
async function loadFiles(problemContent, planContent) {
    try {
        state.startTime = performance.now();
        const problem = parseProblem(problemContent);
        const plan = parsePlan(planContent);
        state.planData = {
            problem,
            frames: buildFrames(problem, plan),
            isNumeric: problem.domain.includes('snowman_numeric')
        };
        state.currentFrame = 0;
        state.isPlaying = false;
        state.currentTime = 0;

        // Update UI
        const stepSlider = document.getElementById('step');
        stepSlider.max = Math.max(0, Math.floor(state.planData.frames.length / CONFIG.SUBSTEPS) - 1);
        stepSlider.value = 0;
        stepSlider.disabled = false;
        ['playPause', 'stepForward', 'stepBackward', 'reset'].forEach(id => {
            document.getElementById(id).disabled = false;
        });
        UI.playIcon.style.display = 'block';
        UI.pauseIcon.style.display = 'none';
        UI.playPauseText.textContent = 'Play';

        clearDynamicSceneObjects();
        await initDynamicSceneObjects(state.planData.problem);
        if (state.planData.frames.length) {
            updateFrame(state.planData.frames[0]);
            state.renderer.render(state.scene, state.camera);
        }
    } catch (err) {
        showPopup('errorPopup', 'errorMessage', `Error loading plan: ${err.message}`);
        resetScene(false);
    }
}

/**
 * Resets the scene to initial state.
 * @param {boolean} clearFiles - Whether to clear file inputs.
 */
function resetScene(clearFiles = true) {
    state.planData = {
        problem: { ...state.planData.problem },
        frames: [],
        isNumeric: false
    };
    state.currentFrame = 0;
    state.isPlaying = false;
    state.currentTime = 0;
    state.startTime = null;

    if (clearFiles) {
        state.problemFile = null;
        state.planFile = null;
        document.getElementById('problemFile').value = '';
        document.getElementById('planFile').value = '';
    }

    UI.playIcon.style.display = 'block';
    UI.pauseIcon.style.display = 'none';
    UI.playPauseText.textContent = 'Play';
    document.getElementById('step').value = 0;
    document.getElementById('step').max = 0;
    document.getElementById('step').disabled = true;
    ['playPause', 'stepForward', 'stepBackward', 'reset'].forEach(id => {
        document.getElementById(id).disabled = true;
    });

    clearDynamicSceneObjects();
    initStaticEnvironment(state.planData.problem);
    state.renderer.render(state.scene, state.camera);
}

/**
 * Generates plan metrics.
 * @returns {string} - HTML formatted metrics.
 */
function getMetrics() {
    const totalSteps = Math.floor(state.planData.frames.length / CONFIG.SUBSTEPS);
    const planDuration = state.currentTime.toFixed(2);
    const ballsAtGoal = state.planData.frames.length
        ? Object.values(state.planData.frames[state.planData.frames.length - 1].balls)
            .filter(pos => pos === CONFIG.GOAL_POS).length
        : 0;
    return `
        <strong>Plan Metrics:</strong><br>
        Total Steps: ${totalSteps}<br>
        Plan Duration: ${planDuration} seconds<br>
        Balls at Goal: ${ballsAtGoal}<br>
        Frame Rate: ${state.frameRate.toFixed(1)} FPS
    `;
}

/**
 * Animation loop.
 */
function animate() {
    requestAnimationFrame(animate);
    const delta = state.clock.getDelta();
    state.controls.update();

    // Update frame rate
    const now = performance.now();
    state.frameRate = 1000 / (now - state.lastFrameTime);
    state.lastFrameTime = now;

    if (state.isPlaying && state.planData.frames.length && state.currentFrame < state.planData.frames.length - 1) {
        updateFrame(state.planData.frames[Math.floor(state.currentFrame)]);
        state.currentFrame += state.speed;
        document.getElementById('step').value = Math.floor(state.currentFrame / CONFIG.SUBSTEPS);
        state.currentTime = state.startTime ? (performance.now() - state.startTime) / 1000 : 0;
        if (state.currentFrame >= state.planData.frames.length - 1) {
            state.isPlaying = false;
            UI.playIcon.style.display = 'block';
            UI.pauseIcon.style.display = 'none';
            UI.playPauseText.textContent = 'Play';
        }
    }

    // Update snow particles
    state.snowParticles.forEach((mesh, i) => {
        const velocity = state.snowParticleVelocities[i];
        mesh.position.addScaledVector(velocity, delta * state.snowSpeed);
        if (mesh.position.y < 0) {
            state.scene.remove(mesh);
            state.snowParticles.splice(i, 1);
            state.snowParticleVelocities.splice(i, 1);
        }
    });

    // Update character animations
    if (state.mixer && state.isPlaying && state.planData.frames[Math.floor(state.currentFrame)]?.type.includes('move')) {
        state.mixer.update(delta);
    }

    state.renderer.render(state.scene, state.camera);
}

/**
 * Event Listeners
 */
function setupEventListeners() {
    // Control panel drag and resize
    let isDragging = false, isResizing = false, currentX, currentY, startX, startY, startWidth, startHeight;
    UI.controlPanel.addEventListener('mousedown', e => {
        if (e.target.id === 'minimizeBtn') return;
        if (e.target.id === 'resizeHandle') {
            isResizing = true;
            startX = e.clientX;
            startY = e.clientY;
            startWidth = parseInt(getComputedStyle(UI.controlPanel).width);
            startHeight = parseInt(getComputedStyle(UI.controlPanel).height);
            UI.controlPanel.style.cursor = 'se-resize';
        } else {
            isDragging = true;
            currentX = e.clientX - parseFloat(UI.controlPanel.style.left || 0);
            currentY = e.clientY - parseFloat(UI.controlPanel.style.top || 0);
            UI.controlPanel.style.cursor = 'grabbing';
        }
    });

    document.addEventListener('mousemove', e => {
        if (isDragging) {
            const newLeft = Math.max(0, Math.min(e.clientX - currentX, window.innerWidth * 0.35 - UI.controlPanel.offsetWidth));
            const newTop = Math.max(40, Math.min(e.clientY - currentY, window.innerHeight * 0.5 - UI.controlPanel.offsetHeight));
            UI.controlPanel.style.left = `${newLeft}px`;
            UI.controlPanel.style.top = `${newTop}px`;
        } else if (isResizing) {
            const width = Math.max(200, Math.min(400, startWidth + (e.clientX - startX)));
            const height = Math.max(150, Math.min(600, startHeight + (e.clientY - startY)));
            UI.controlPanel.style.width = `${width}px`;
            UI.controlPanel.style.height = `${height}px`;
        }
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        isResizing = false;
        UI.controlPanel.style.cursor = 'move';
    });

    // Minimize/maximize
    document.getElementById('minimizeBtn').addEventListener('click', () => {
        UI.controlPanel.classList.toggle('minimized');
        const isMinimized = UI.controlPanel.classList.contains('minimized');
        document.getElementById('minimizeBtn').textContent = isMinimized ? '+' : '-';
        ['problemFile', 'planFile'].forEach(id => {
            document.getElementById(id).style.display = isMinimized ? 'none' : 'block';
        });
        document.querySelectorAll('#controlPanel button, #controlPanel label, #controlPanel input[type="range"]').forEach(el => {
            el.style.display = isMinimized ? 'none' : 'block';
        });
        document.getElementById('resizeHandle').style.display = isMinimized ? 'none' : 'block';
    });

    // Menu interactions
    document.getElementById('menuBtn').addEventListener('click', () => {
        const menu = document.getElementById('menuDropdown');
        menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
    });

    ['help', 'metrics', 'about'].forEach(type => {
        document.getElementById(`${type}Btn`).addEventListener('click', () => {
            showPopup(`${type}Popup`, `${type}Message`, 
                type === 'metrics' ? getMetrics() : document.getElementById(`${type}Message`).innerHTML);
            document.getElementById('menuDropdown').style.display = 'none';
        });
        document.getElementById(`${type}Close`).addEventListener('click', () => hidePopup(`${type}Popup`));
    });

    // File inputs
    ['problemFile', 'planFile'].forEach(id => {
        document.getElementById(id).addEventListener('change', selectFiles);
    });

    // Playback controls
    document.getElementById('playPause').addEventListener('click', () => {
        if (!state.planData.frames.length) {
            showPopup('errorPopup', 'errorMessage', 'Please select both problem (.pddl) and plan (.txt or .plan) files');
            return;
        }
        state.isPlaying = !state.isPlaying;
        UI.playIcon.style.display = state.isPlaying ? 'none' : 'block';
        UI.pauseIcon.style.display = state.isPlaying ? 'block' : 'none';
        UI.playPauseText.textContent = state.isPlaying ? 'Pause' : 'Play';
        if (state.isPlaying && !state.startTime) state.startTime = performance.now();
    });

    document.getElementById('stepForward').addEventListener('click', () => {
        if (!state.planData.frames.length) {
            showPopup('errorPopup', 'errorMessage', 'Please select both problem (.pddl) and plan (.txt or .plan) files');
            return;
        }
        if (!state.isPlaying && state.currentFrame < state.planData.frames.length - CONFIG.SUBSTEPS) {
            state.currentFrame = Math.min(state.currentFrame + CONFIG.SUBSTEPS, state.planData.frames.length - 1);
            document.getElementById('step').value = Math.floor(state.currentFrame / CONFIG.SUBSTEPS);
            updateFrame(state.planData.frames[Math.floor(state.currentFrame)]);
        }
    });

    document.getElementById('stepBackward').addEventListener('click', () => {
        if (!state.planData.frames.length) {
            showPopup('errorPopup', 'errorMessage', 'Please select both problem (.pddl) and plan (.txt or .plan) files');
            return;
        }
        if (!state.isPlaying && state.currentFrame >= CONFIG.SUBSTEPS) {
            state.currentFrame -= CONFIG.SUBSTEPS;
            document.getElementById('step').value = Math.floor(state.currentFrame / CONFIG.SUBSTEPS);
            updateFrame(state.planData.frames[Math.floor(state.currentFrame)]);
        }
    });

    document.getElementById('reset').addEventListener('click', () => resetScene(true));

    document.getElementById('speed').addEventListener('input', e => {
        state.speed = Number(e.target.value);
    });

    document.getElementById('snowSpeed').addEventListener('input', e => {
        state.snowSpeed = Number(e.target.value);
    });

    document.getElementById('step').addEventListener('input', e => {
        if (!state.planData.frames.length) {
            showPopup('errorPopup', 'errorMessage', 'Please select both problem (.pddl) and plan (.txt or .plan) files');
            return;
        }
        state.currentFrame = Number(e.target.value) * CONFIG.SUBSTEPS;
        state.isPlaying = false;
        UI.playIcon.style.display = 'block';
        UI.pauseIcon.style.display = 'none';
        UI.playPauseText.textContent = 'Play';
        updateFrame(state.planData.frames[Math.floor(state.currentFrame)]);
    });

    window.addEventListener('resize', () => {
        state.camera.aspect = 0.65 * window.innerWidth / window.innerHeight;
        state.camera.updateProjectionMatrix();
        state.renderer.setSize(window.innerWidth * 0.65, window.innerHeight - 70);
        UI.controlPanel.style.left = '10px';
        UI.controlPanel.style.top = '50px';
    });
}

/**
 * Initialize application
 */
window.onload = async () => {
    await initCoreThreeJs();
    await initStaticEnvironment(state.planData.problem);
    setupEventListeners();
    state.renderer.render(state.scene, state.camera);
};