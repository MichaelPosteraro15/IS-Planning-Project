// Import necessary Three.js modules
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Configuration constants
const RADIUS = { 0: 0.15, 1: 0.25, 2: 0.35 }; // Radii for small, medium, large balls
const SUBSTEPS = 10; // Number of animation substeps per plan action
const GOAL_POS = '2,0'; // The target grid position for the snowman (X,Y string in 0-indexed) - Reverted to original
const SNOW_COLOR = 0xE0FFFF; // Color for cells with snow (Light Cyan)
const REGULAR_GRID_COLOR = 0x90EE90; // Default color for non-snow grid cells (Light Green)
const UNTOUCHED_GRID_COLOR = 0x556B2F; // Dark Olive Green for non-snow, non-goal, non-special grid cells
// Special positions from the Python script (0-indexed)
const SPECIAL_GRID_POSITIONS = ['1,1', '1,3', '3,1', '3,3'];
const CHARACTER_DEFAULT_COLOR = 0x8B4513; // SaddleBrown for soldier model
const FALLBACK_CHARACTER_COLOR = 0xFFFF00; // Yellow for fallback box character

// Global state variables for the visualization
// Default problem state for initial display of static environment
let defaultProblemData = {
    grid_size: 5,
    snow: Object.fromEntries(Array(5).fill().flatMap((_, x) => Array(5).fill().map((_, y) => [[x, y].join(','), x === 0 || x === 4]))),
    balls: {}, // No balls initially
    ball_size: {}, // No ball sizes initially
    character: '2,1' // Default character position for static display
};
let planData = { problem: defaultProblemData, frames: [], isNumeric: false }; // planData will be updated on file load

let currentFrame = 0; // Current frame in the animation sequence
let isPlaying = false; // Animation play/pause state
let speed = 1; // Animation playback speed multiplier
let snowSpeed = 0.2; // Speed of falling snowflakes
// Three.js scene elements
let scene, camera, renderer, controls,
    grid = {}, // Stores references to grid plane meshes
    balls = {}, // Stores references to ball meshes
    character, // Reference to character mesh/model
    mixer, // Animation mixer for character
    particles, particleVelocities = [], // General particle system (not used for snow)
    snowParticles = [], snowParticleVelocities = [], // Snow particle system
    pathLine, // Path line (currently unused)
    spotlight; // Spotlight following character
let forbiddenIcons = {}; // To store references to the 3D forbidden icons

const clock = new THREE.Clock(); // Clock for animation timing
let currentTime = 0; // Current elapsed time of the plan
let startTime = null; // Timestamp when playback started
// Removed snowman and snowmanInstance as they are no longer used for the visual snowman

// Declare problemFile and planFile globally
let problemFile = null;
let planFile = null;

// UI elements for play/pause icon and text toggle
const playIcon = document.getElementById('playIcon');
const pauseIcon = document.getElementById('pauseIcon');
const playPauseText = document.getElementById('playPauseText');

/**
 * Shows a specified pop-up message.
 * @param {string} popupId - The ID of the pop-up element to show.
 * @param {string} messageId - The ID of the element within the pop-up to display the message.
 * @param {string} message - The message content (can include HTML).
 */
function showPopup(popupId, messageId, message) {
    // Hide all pop-ups first
    document.querySelectorAll('#errorPopup, #helpPopup, #metricsPopup, #aboutPopup').forEach(p => p.style.display = 'none');
    const popup = document.getElementById(popupId);
    const messageEl = document.getElementById(messageId);
    messageEl.innerHTML = message; // Use innerHTML to preserve formatting
    popup.style.display = 'block';
}

/**
 * Hides a specified pop-up message.
 * @param {string} popupId - The ID of the pop-up element to hide.
 */
function hidePopup(popupId) {
    document.getElementById(popupId).style.display = 'none';
}

/**
 * Parses a location string (e.g., "loc_3_1") into a [row, col] array.
 * Adjusts for 0-based indexing if input is 1-based.
 * @param {string} loc - The location string.
 * @returns {number[]} - An array [row, col].
 * @throws {Error} If the location format is invalid.
 */
function parseLoc(loc) {
    try {
        const parts = loc.split('_');
        if (parts.length < 3) throw new Error(`Invalid location format: ${loc}`);
        return [parseInt(parts[1]) - 1, parseInt(parts[2]) - 1]; // Convert to 0-based index
    } catch (e) {
        console.error(`Error parsing location '${loc}': ${e}`);
        showPopup('errorPopup', 'errorMessage', `Error parsing location '${loc}': ${e.message}`);
        throw e;
    }
}

/**
 * Parses the content of a PDDL problem file to extract initial state information.
 * @param {string} content - The PDDL problem file content.
 * @returns {object} - An object containing parsed problem data (snow, balls, ball_size, character, grid_size, domain).
 * @throws {Error} If the problem file is empty or contains invalid data.
 */
function parseProblem(content) {
    try {
        if (!content.trim()) throw new Error("Problem file is empty");

        let snow = {}, balls = {}, ballSize = {}, character = null;
        let gridPositions = new Set(); // To determine grid size dynamically
        let domain = 'unknown';

        // Extract domain name
        const domainMatch = content.match(/:domain (\S+)/);
        if (domainMatch) domain = domainMatch[1];

        // Parse snow locations based on location_type (numeric domain)
        const locTypeRegex = /\(= \(location_type (\S+)\) (\d+)\)/g;
        let match;
        while ((match = locTypeRegex.exec(content)) !== null) {
            const loc = match[1];
            const t = match[2];
            const coord = parseLoc(loc);
            snow[coord.join(',')] = (t === '1'); // '1' indicates snow
            gridPositions.add(coord.join(','));
        }

        // Parse snow locations (classic domain)
        const snowRegex = /\(snow (\S+)\)/g;
        while ((match = snowRegex.exec(content)) !== null) {
            const loc = match[1];
            const coord = parseLoc(loc);
            snow[coord.join(',')] = true;
            gridPositions.add(coord.join(','));
        }

        // Parse ball positions
        const ballAtRegex = /\(ball_at (\S+) (\S+)\)/g;
        while ((match = ballAtRegex.exec(content)) !== null) {
            const [_, ball, loc] = match;
            gridPositions.add(parseLoc(loc).join(','));
            balls[ball] = parseLoc(loc);
        }

        // Parse ball sizes (numeric domain)
        const ballSizeNumRegex = /\(= \(ball_size (\S+)\) (\d+)\)/g;
        while ((match = ballSizeNumRegex.exec(content)) !== null) {
            const [_, ball, size] = match;
            const sizeInt = parseInt(size);
            if (![0, 1, 2].includes(sizeInt)) throw new Error(`Invalid ball size ${size} for ball ${ball}. Must be 0, 1, or 2.`);
            ballSize[ball] = sizeInt;
        }

        // Parse ball sizes (classic domain)
        const ballSizeClassicRegex = /\(ball_size_(small|medium|large) (\S+)\)/g;
        while ((match = ballSizeClassicRegex.exec(content)) !== null) {
            const [_, sizeStr, ball] = match;
            const sizeMap = { 'small': 0, 'medium': 1, 'large': 2 };
            ballSize[ball] = sizeMap[sizeStr.toLowerCase()];
        }

        // Parse character position
        const charMatch = content.match(/\(character_at (\S+)\)/);
        if (charMatch) {
            character = parseLoc(charMatch[1]);
            gridPositions.add(character.join(','));
        }

        // Basic validation
        if (Object.keys(balls).length === 0) console.warn("No balls found in problem file. This might be expected for some initial states.");
        if (!character) throw new Error("No character position found in problem file.");

        // Assign default size 0 if not specified for a ball
        for (let ball in balls) {
            if (!(ball in ballSize)) ballSize[ball] = 0;
        }

        // Determine grid size based on max coordinates found
        let gridSize = 5; // Default grid size
        if (gridPositions.size > 0) {
            const coords = Array.from(gridPositions).map(c => c.split(',').map(Number));
            const maxR = Math.max(...coords.map(c => c[0]));
            const maxC = Math.max(...coords.map(c => c[1]));
            gridSize = Math.max(maxR, maxC) + 1;
        }

        // Ensure all grid cells have a snow state (default to false if not specified)
        for (let r = 0; r < gridSize; r++) {
            for (let c = 0; c < gridSize; c++) {
                if (!(r + ',' + c in snow)) snow[r + ',' + c] = false;
            }
        }
        console.log("[parseProblem] Parsed problem data:", { snow, balls, ball_size: ballSize, character: character.join(','), grid_size: gridSize, domain });
        return { snow, balls, ball_size: ballSize, character: character.join(','), grid_size: gridSize, domain };
    } catch (e) {
        console.error(`Error parsing problem file: ${e}`);
        showPopup('errorPopup', 'errorMessage', `Error parsing problem file: ${e.message}`);
        throw e;
    }
}

/**
 * Parses the content of a plan file into an array of action strings.
 * @param {string} content - The plan file content.
 * @returns {string[]} - An array of parsed plan steps.
 * @throws {Error} If the plan file is empty or contains no valid actions.
 */
function parsePlan(content) {
    try {
        if (!content.trim()) throw new Error("Plan file is empty");

        const steps = [];
        const lines = content.trim().split('\n');
        for (let line of lines) {
            line = line.trim();
            // Skip empty lines or comments
            if (!line || line.startsWith(';')) continue;

            // Remove timestamp (e.g., "0.001: ") and parentheses
            let cleanedLine = line.replace(/^\d+\.\d+:\s*/, '').replace(/^\d+[.:]?\s*/, '');
            if (cleanedLine.startsWith('(') && cleanedLine.endsWith(')')) {
                cleanedLine = cleanedLine.slice(1, -1).trim();
            }
            // Only add lines that look like valid actions
            if (cleanedLine && ['move', 'move_to', 'move_ball', 'push', 'roll', 'roll_ball', 'goal', 'move_character'].some(k => cleanedLine.toLowerCase().includes(k))) {
                steps.push(cleanedLine);
            }
        }

        if (steps.length === 0) throw new Error("No valid actions found in plan file.");
        console.log("[parsePlan] Parsed plan steps:", steps);
        return steps;
    } catch (e) {
        console.error(`Error parsing plan file: ${e}`);
        showPopup('errorPopup', 'errorMessage', `Error parsing plan file: ${e.message}`);
        throw e;
    }
}

/**
 * Builds a sequence of animation frames based on the problem and plan.
 * Each frame represents a snapshot of the scene at a given substep.
 * This function applies the "good snowman rule" for ball sizing at the goal.
 * It now *allows* movement into forbidden cells if the plan dictates it,
 * aligning with the Python visualizer's behavior.
 * @param {object} prob - Parsed problem data.
 * @param {string[]} plan - Parsed plan actions.
 * @returns {object[]} - An array of frame objects.
 * @throws {Error} If there's an error during frame building.
 */
function buildFrames(prob, plan) {
    try {
        const frames = [];
        // Deep copy initial state
        const state = {
            snow: { ...prob.snow },
            balls: Object.fromEntries(Object.entries(prob.balls).map(([k, v]) => [k, v.join(',')])),
            ball_size: { ...prob.ball_size },
            character: prob.character,
            grid_size: prob.grid_size,
            isNumeric: prob.domain.includes('snowman_numeric')
        };

        // Add initial frame
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
        for (let action of plan) {
            const parts = action.split(/\s+/);
            let currentCharacterPos = state.character;
            let currentBallPos = { ...state.balls };

            try {
                if (['move_character', 'move', 'move_to'].includes(parts[0])) {
                    if (parts.length < 3) throw new Error(`Invalid move action: ${action}`);
                    const start = parseLoc(parts[1]);
                    const end = parseLoc(parts[2]);
                    const endCoordString = end.join(',');
                    const direction = parts.length > 3 ? parts[3] : null;

                    for (let t = 0; t < SUBSTEPS; t++) {
                        const alpha = t / (SUBSTEPS - 1);
                        frames.push({
                            type: 'move',
                            start: start.join(','),
                            end: end.join(','),
                            alpha,
                            balls: { ...state.balls },
                            ball_size: { ...state.ball_size },
                            snow: { ...state.snow },
                            character: currentCharacterPos,
                            grid_size: state.grid_size,
                            time: step_count + alpha,
                            direction
                        });
                    }
                    state.character = end.join(',');
                } else if (['move_ball', 'push', 'roll', 'roll_ball'].includes(parts[0])) {
                    if (parts.length < 5) throw new Error(`Invalid move_ball action: ${action}`);
                    const [_, ball, fromCell, midCell, toCell] = parts;
                    const start = parseLoc(fromCell);
                    const end = parseLoc(toCell);
                    const endCoordString = end.join(',');
                    const direction = parts.length > 5 ? parts[5] : null;

                    const charStart = state.character;
                    for (let t = 0; t < SUBSTEPS; t++) {
                        const alpha = t / (SUBSTEPS - 1);
                        frames.push({
                            type: 'move_to_ball',
                            start: charStart,
                            end: start.join(','),
                            alpha,
                            balls: { ...state.balls },
                            ball_size: { ...state.ball_size },
                            snow: { ...state.snow },
                            character: state.character,
                            grid_size: state.grid_size,
                            time: step_count + alpha,
                            direction
                        });
                    }
                    state.character = start.join(',');

                    for (let t = 0; t < SUBSTEPS; t++) {
                        const alpha = t / (SUBSTEPS - 1);
                        frames.push({
                            type: 'move_ball',
                            ball,
                            start: start.join(','),
                            end: end.join(','),
                            alpha,
                            balls: { ...state.balls },
                            ball_size: { ...state.ball_size },
                            snow: { ...state.snow },
                            character: state.character,
                            grid_size: state.grid_size,
                            time: step_count + alpha,
                            direction
                        });
                    }
                    state.balls[ball] = end.join(',');

                    if (state.snow[end.join(',')] && (state.isNumeric || !state.isNumeric)) {
                        state.ball_size[ball] = Math.min(state.ball_size[ball] + 1, 2);
                        state.snow[end.join(',')] = false;
                    }
                } else if (parts[0] === 'goal') {
                    const ballsAtGoal = Object.entries(state.balls).filter(([_, pos]) => pos === GOAL_POS).map(([b]) => b);
                    console.log(`Processing goal action at step ${step_count + 1}, balls at ${GOAL_POS}: ${ballsAtGoal.length}`);

                    if (ballsAtGoal.length >= 3) {
                        ballsAtGoal.forEach(ballName => {
                            state.balls[ballName] = GOAL_POS;
                        });
                    }

                    for (let t = 0; t < SUBSTEPS; t++) {
                        const alpha = t / (SUBSTEPS - 1);
                        frames.push({
                            type: 'goal',
                            balls: { ...state.balls },
                            ball_size: { ...state.ball_size },
                            snow: { ...state.snow },
                            character: state.character,
                            grid_size: state.grid_size,
                            time: step_count + alpha,
                            alpha
                        });
                    }
                } else {
                    console.warn(`Unknown action '${action}' on step ${step_count + 1}`);
                    for (let t = 0; t < SUBSTEPS; t++) {
                        const alpha = t / (SUBSTEPS - 1);
                        frames.push({
                            type: 'static',
                            balls: { ...state.balls },
                            ball_size: { ...state.ball_size },
                            snow: { ...state.snow },
                            character: state.character,
                            grid_size: state.grid_size,
                            time: step_count + alpha,
                            alpha
                        });
                    }
                }
                step_count++;
            } catch (e) {
                console.error(`Error processing action '${action}' on step ${step_count + 1}: ${e}`);
                showPopup('errorPopup', 'errorMessage', `Error processing action '${action}' on step ${step_count + 1}: ${e.message}`);
                for (let t = 0; t < SUBSTEPS; t++) {
                    const alpha = t / (SUBSTEPS - 1);
                    frames.push({
                        type: 'error',
                        balls: { ...state.balls },
                        ball_size: { ...state.ball_size },
                        snow: { ...state.snow },
                        character: state.character,
                        grid_size: state.grid_size,
                        time: step_count + alpha,
                        alpha
                    });
                }
                step_count++;
            }
        }
        console.log(`Total frames generated: ${frames.length}, isNumeric: ${state.isNumeric}`);
        return frames;
    } catch (e) {
        console.error(`Error building frames: ${e}`);
        showPopup('errorPopup', 'errorMessage', `Error building frames: ${e.message}`);
        throw e;
    }
}

/**
 * Reads a file as text.
 * @param {File} file - The file to read.
 * @returns {Promise<string>} - A promise that resolves with the file content.
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
 * Handles file selection and initiates loading of problem and plan files.
 * @param {Event} e - The change event from the file input.
 */
async function selectFiles(e) {
    const problemInput = document.getElementById('problemFile');
    const planInput = document.getElementById('planFile');
    const changedInput = e.target.id;

    console.log(`File input changed: ${changedInput}`);

    if (changedInput === 'problemFile' && problemInput.files[0]) {
        if (!problemInput.files[0].name.endsWith('.pddl')) {
            console.error('Invalid problem file selected');
            showPopup('errorPopup', 'errorMessage', 'Please select a valid problem file (.pddl).');
            problemInput.value = '';
            problemFile = null;
            return;
        }
        problemFile = problemInput.files[0];
        console.log(`Problem file selected: ${problemFile.name}`);
    } else if (changedInput === 'planFile' && planInput.files[0]) {
        if (!planInput.files[0].name.endsWith('.txt') && !planInput.files[0].name.endsWith('.plan')) {
            console.error('Invalid plan file selected');
            showPopup('errorPopup', 'errorMessage', 'Please select a valid plan file (.txt or .plan).');
            planInput.value = '';
            planFile = null;
            return;
        }
        planFile = planInput.files[0];
        console.log(`Plan file selected: ${planFile.name}`);
    }

    if (problemFile && planFile) {
        console.log('Both files selected, attempting to load');
        try {
            resetScene(false);
            const [problemContent, planContent] = await Promise.all([
                readFile(problemFile),
                readFile(planFile)
            ]);
            await loadFiles(problemContent, planContent, problemFile.name, planFile.name);
            console.log('Files loaded successfully, rendering initial frame');
            if (planData.frames.length > 0) {
                updateFrame(planData.frames[0]);
                renderer.render(scene, camera);
            }
        } catch (err) {
            console.error(`Error reading files: ${err.message}`);
            showPopup('errorPopup', 'errorMessage', `Error reading files: ${err.message}`);
            resetScene(false);
        }
    } else {
        console.log('Waiting for both files to be selected');
        ['playPause', 'stepForward', 'stepBackward', 'reset', 'step'].forEach(id => {
            document.getElementById(id).disabled = true;
        });
    }
}

/**
 * Loads and processes the problem and plan files.
 * @param {string} problemContent - Content of the PDDL problem file.
 * @param {string} planContent - Content of the plan file.
 * @param {string} problemFileName - Name of the problem file.
 * @param {string} planFileName - Name of the plan file.
 * @returns {Promise<void>}
 */
async function loadFiles(problemContent, planContent, problemFileName, planFileName) {
    try {
        console.log(`Loading files: ${problemFileName}, ${planFileName}`);
        startTime = performance.now();
        const problem = parseProblem(problemContent);
        const plan = parsePlan(planContent);
        planData = {
            problem,
            frames: buildFrames(problem, plan),
            isNumeric: problem.domain.includes('snowman_numeric')
        };
        currentFrame = 0;
        isPlaying = false;
        currentTime = 0;

        console.log("Parsed problem data (initial state):", planData.problem);
        console.log("First frame data:", planData.frames[0]);

        document.getElementById('step').max = Math.max(0, Math.floor(planData.frames.length / SUBSTEPS) - 1);
        document.getElementById('step').value = 0;
        document.getElementById('step').disabled = false;
        ['playPause', 'stepForward', 'stepBackward', 'reset'].forEach(id => document.getElementById(id).disabled = false);
        playIcon.style.display = 'block';
        pauseIcon.style.display = 'none';
        playPauseText.textContent = 'Play';

        console.log('Clearing existing dynamic scene objects before re-initialization.');
        clearDynamicSceneObjects();
        console.log('Initializing dynamic scene objects with loaded data.');
        await initDynamicSceneObjects(planData.problem);
        console.log(`Dynamic scene initialized, frames: ${planData.frames.length}`);
        console.log("[loadFiles] Current state of 'balls' (Three.js meshes) after initDynamicSceneObjects:", Object.keys(balls));

        if (planData.frames.length > 0) {
            console.log('Rendering first frame');
            updateFrame(planData.frames[0]);
            renderer.render(scene, camera);
        }
    }
    catch (err) {
        console.error('Error loading plan:', err);
        showPopup('errorPopup', 'errorMessage', `Error loading plan: ${err.message}`);
        resetScene(false);
    }
}

/**
 * Creates a 3D barrier icon for forbidden grid cells.
 * @returns {THREE.Group} A Three.js group containing the barrier parts.
 */
function createForbiddenIcon() {
    const barrierGroup = new THREE.Group();
    const barrierMaterial = new THREE.MeshStandardMaterial({ color: 0xFF0000, roughness: 0.5, metalness: 0.1 });

    const baseGeometry = new THREE.BoxGeometry(0.8, 0.05, 0.2);
    const baseMesh = new THREE.Mesh(baseGeometry, barrierMaterial);
    baseMesh.position.y = 0.025;
    baseMesh.castShadow = true;
    baseMesh.receiveShadow = true;
    barrierGroup.add(baseMesh);

    const postGeometry = new THREE.BoxGeometry(0.05, 0.3, 0.05);
    const post1 = new THREE.Mesh(postGeometry, barrierMaterial);
    post1.position.set(-0.3, 0.15 + 0.025, 0);
    post1.castShadow = true;
    post1.receiveShadow = true;
    barrierGroup.add(post1);

    const post2 = new THREE.Mesh(postGeometry, barrierMaterial);
    post2.position.set(0.3, 0.15 + 0.025, 0);
    post2.castShadow = true;
    post2.receiveShadow = true;
    barrierGroup.add(post2);

    return barrierGroup;
}

/**
 * Clears only the dynamic scene objects (balls, character).
 * The static environment (ground, grid, walls, trees, forbidden icons) remains.
 */
function clearDynamicSceneObjects() {
    console.log('Clearing dynamic scene objects...');
    const objectsToRemove = [];
    scene.traverse(object => {
        if (object.name && (
            object.name.startsWith('ball_') ||
            object.name === 'character'
        )) {
            objectsToRemove.push(object);
        }
    });

    objectsToRemove.forEach(object => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
            if (Array.isArray(object.material)) {
                object.material.forEach(mat => mat.dispose());
            } else {
                object.material.dispose();
            }
        }
        scene.remove(object);
    });

    balls = {};
    character = null;
    
    if (mixer) {
        mixer.stopAllAction();
        mixer = null;
    }
    if (spotlight) {
        spotlight.target.position.set(defaultProblemData.grid_size / 2, 0, defaultProblemData.grid_size / 2);
        spotlight.position.set(defaultProblemData.grid_size / 2, 3, defaultProblemData.grid_size / 2);
    }
    console.log('Dynamic scene objects cleared.');
}

/**
 * Initializes the static environment objects (ground, grid, walls, trees, forbidden icons).
 * This runs once on initial page load and when explicitly resetting the static environment.
 * @param {object} problemDataForStatic - Use defaultProblemData for initial static display.
 */
async function initStaticEnvironment(problemDataForStatic) {
    console.log('Initializing static environment objects...');

    const staticObjectsToRemove = [];
    scene.traverse(object => {
        if (object.name && (
            object.name.startsWith('grid_') ||
            object.name === 'ground_plane' ||
            object.name.startsWith('wall_') ||
            object.name.startsWith('tree_') ||
            object.name.startsWith('forbidden_icon_')
        )) {
            staticObjectsToRemove.push(object);
        }
    });
    staticObjectsToRemove.forEach(object => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
            if (Array.isArray(object.material)) object.material.forEach(m => m.dispose());
            else object.material.dispose();
        }
        scene.remove(object);
    });
    grid = {};
    forbiddenIcons = {};

    const groundGeometry = new THREE.PlaneGeometry(problemDataForStatic.grid_size + 2, problemDataForStatic.grid_size + 2);
    const groundMaterial = new THREE.MeshStandardMaterial({ color: SNOW_COLOR, roughness: 0.8, metalness: 0.1 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(problemDataForStatic.grid_size / 2, -0.01, problemDataForStatic.grid_size / 2);
    ground.receiveShadow = true;
    ground.name = 'ground_plane';
    scene.add(ground);
    console.log('Ground added.');

    const [goalX, goalY] = GOAL_POS.split(',').map(Number);
    for (let x = 0; x < problemDataForStatic.grid_size; x++) {
        for (let y = 0; y < problemDataForStatic.grid_size; y++) {
            const coord = `${x},${y}`;
            const geometry = new THREE.PlaneGeometry(1, 1);
            let materialColor;

            if (SPECIAL_GRID_POSITIONS.includes(coord)) {
                materialColor = 0xFFFFFF;
                const icon = createForbiddenIcon();
                icon.position.set(x + 0.5, 0, y + 0.5);
                icon.name = `forbidden_icon_${coord}`;
                scene.add(icon);
                forbiddenIcons[coord] = icon;
            } else if (problemDataForStatic.snow[coord]) {
                materialColor = SNOW_COLOR;
            } else if (x === goalX && y === goalY) {
                materialColor = REGULAR_GRID_COLOR;
            } else {
                materialColor = UNTOUCHED_GRID_COLOR;
            }

            const material = new THREE.MeshStandardMaterial({ color: materialColor, side: THREE.DoubleSide });
            const plane = new THREE.Mesh(geometry, material);
            plane.rotation.x = -Math.PI / 2;
            plane.position.set(x + 0.5, 0, y + 0.5);
            plane.receiveShadow = true;
            plane.name = `grid_${coord}`;
            scene.add(plane);
            grid[coord] = plane;
        }
    }
    console.log('Grid planes added.');

    const wallMaterial = new THREE.MeshStandardMaterial({ color: 0xADD8E6, transparent: true, opacity: 0.7, roughness: 0.3, metalness: 0.5 });
    const wallHeight = 0.5;
    const wallThickness = 0.2;
    const halfGridSize = problemDataForStatic.grid_size / 2;
    [
        { geometry: new THREE.BoxGeometry(problemDataForStatic.grid_size + wallThickness * 2, wallHeight, wallThickness), position: [halfGridSize, wallHeight / 2, -wallThickness / 2], name: 'wall_top' },
        { geometry: new THREE.BoxGeometry(problemDataForStatic.grid_size + wallThickness * 2, wallHeight, wallThickness), position: [halfGridSize, wallHeight / 2, problemDataForStatic.grid_size + wallThickness / 2], name: 'wall_bottom' },
        { geometry: new THREE.BoxGeometry(wallThickness, wallHeight, problemDataForStatic.grid_size + wallThickness * 2), position: [-wallThickness / 2, wallHeight / 2, halfGridSize], name: 'wall_left' },
        { geometry: new THREE.BoxGeometry(wallThickness, wallHeight, problemDataForStatic.grid_size + wallThickness * 2), position: [problemDataForStatic.grid_size + wallThickness / 2, wallHeight / 2, halfGridSize], name: 'wall_right' }
    ].forEach(({ geometry, position, name }) => {
        const wall = new THREE.Mesh(geometry, wallMaterial);
        wall.position.set(...position);
        wall.castShadow = true;
        wall.receiveShadow = true;
        wall.name = name;
        scene.add(wall);
    });
    console.log('Walls added.');

    const treeGeometry = new THREE.ConeGeometry(0.3, 0.8, 8);
    const treeMaterial = new THREE.MeshStandardMaterial({ color: 0x228B22, roughness: 0.9 });
    const treePositions = [
        [-1, 0, -1], [problemDataForStatic.grid_size + 1, 0, -1],
        [-1, 0, problemDataForStatic.grid_size + 1], [problemDataForStatic.grid_size + 1, 0, problemDataForStatic.grid_size + 1],
        [halfGridSize, 0, -1], [halfGridSize, 0, problemDataForStatic.grid_size + 1],
        [-1, 0, halfGridSize], [problemDataForStatic.grid_size + 1, 0, halfGridSize]
    ];
    treePositions.forEach((pos, index) => {
        const tree = new THREE.Mesh(treeGeometry, treeMaterial);
        tree.position.set(pos[0], pos[1] + 0.4, pos[2]);
        tree.castShadow = true;
        tree.receiveShadow = true;
        tree.name = `tree_${index}`;
        scene.add(tree);
    });
    console.log('Trees added.');
}

/**
 * Initializes or re-initializes only the dynamic 3D objects (balls, character).
 * This is called after files are loaded.
 * @param {object} problemData - The current problem data (initial state).
 * @returns {Promise<void>}
 */
async function initDynamicSceneObjects(problemData) {
    console.log('[initDynamicSceneObjects] Setting up dynamic scene objects with problem data:', problemData);

    const loader = new GLTFLoader();

    if (!character) {
        try {
            const gltf = await new Promise((resolve, reject) => loader.load('https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb', resolve, undefined, reject));
            character = gltf.scene;
            character.scale.set(0.4, 0.4, 0.4);
            character.castShadow = true;
            character.receiveShadow = true;
            character.name = 'character';
            if (gltf.animations.length > 0) {
                mixer = new THREE.AnimationMixer(character);
                gltf.animations.forEach(clip => mixer.clipAction(clip).play());
            }
            character.traverse(obj => {
                if (obj.isMesh && obj.material) {
                    if (Array.isArray(obj.material)) {
                        obj.material.forEach(mat => {
                            if (mat.type === 'MeshStandardMaterial' || mat.type === 'MeshBasicMaterial') {
                                mat.color.set(CHARACTER_DEFAULT_COLOR);
                            }
                        });
                    } else {
                        if (obj.material.type === 'MeshStandardMaterial' || obj.material.type === 'MeshBasicMaterial') {
                            obj.material.color.set(CHARACTER_DEFAULT_COLOR);
                        }
                    }
                }
            });
            console.log('[initDynamicSceneObjects] Character loaded.');
        } catch (e) {
            console.warn('[initDynamicSceneObjects] Failed to load character model, using fallback (yellow box).', e);
            character = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.8, 0.3), new THREE.MeshStandardMaterial({ color: FALLBACK_CHARACTER_COLOR }));
            character.position.y = 0.4;
            character.castShadow = true;
            character.name = 'character';
        }
        scene.add(character);
    }
    if (character) {
        character.visible = true;
    }
    const [cx, cy] = problemData.character.split(',').map(Number);
    character.position.set(cx + 0.5, 0, cy + 0.5);
    console.log(`[initDynamicSceneObjects] Character positioned at (${cx + 0.5}, 0, ${cy + 0.5}).`);

    for (const ballName in balls) {
        scene.remove(balls[ballName]);
        if (balls[ballName].geometry) balls[ballName].geometry.dispose();
        if (balls[ballName].material) {
            if (Array.isArray(balls[ballName].material)) balls[ballName].material.forEach(m => m.dispose());
            else balls[ballName].material.dispose();
        }
        delete balls[ballName];
    }
    balls = {};

    for (let b in problemData.balls) {
        const geometry = new THREE.SphereGeometry(RADIUS[problemData.ball_size[b]], 32, 32);
        const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0.2 });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.castShadow = true;
        sphere.receiveShadow = true;
        const [x, y] = problemData.balls[b];
        sphere.position.set(x + 0.5, RADIUS[problemData.ball_size[b]], y + 0.5);
        sphere.name = `ball_${b}`;
        scene.add(sphere);
        balls[b] = sphere;
        console.log(`[initDynamicSceneObjects] Added ball ${b} (size ${problemData.ball_size[b]}) at position (${x + 0.5}, ${RADIUS[problemData.ball_size[b]]}, ${y + 0.5}).`);
    }
    console.log("[initDynamicSceneObjects] All initial balls created:", Object.keys(balls));

    if (spotlight) { scene.remove(spotlight); scene.remove(spotlight.target); }
    spotlight = new THREE.SpotLight(0xffffff, 0, 5, Math.PI / 4, 0.5, 2);
    spotlight.position.set(cx + 0.5, 3, cy + 0.5);
    spotlight.castShadow = true;
    scene.add(spotlight);
    scene.add(spotlight.target);
    console.log('[initDynamicSceneObjects] Spotlight added/re-added.');
}

/**
 * Initializes the core Three.js scene, camera, renderer, and controls.
 * This is called once when the page loads.
 * @returns {Promise<void>}
 */
async function initCoreThreeJs() {
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

        camera.position.set(defaultProblemData.grid_size / 2, 3, defaultProblemData.grid_size);
        controls.target.set(defaultProblemData.grid_size / 2, 0, defaultProblemData.grid_size / 2);

        const snowParticleMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 });
        function dropSnowflake() {
            const size = 0.009 + Math.random() * 0.012;
            const geometry = new THREE.CircleGeometry(size, 32);
            const mesh = new THREE.Mesh(geometry, snowParticleMaterial);
            mesh.position.set(Math.random() * (defaultProblemData.grid_size + 2) - 1, Math.random() * 5 + 3, Math.random() * (defaultProblemData.grid_size + 2) - 1);
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

        console.log('Core Three.js setup complete.');
        animate();
    } catch (err) {
        console.error('Core Three.js Initialization error:', err);
        showPopup('errorPopup', 'errorMessage', `Core Three.js Initialization error: ${err.message}`);
    }
}

/**
 * Updates the 3D scene based on the current animation frame data.
 * This is where the visual representation of the plan is updated.
 * @param {object} f - The current frame object containing state information.
 */
function updateFrame(f) {
    try {
        if (!f) return;

        console.log(`[updateFrame] Processing frame: ${Math.floor(currentFrame)}, type: ${f.type}`);
        console.log("[updateFrame] Frame's f.balls state:", f.balls);
        console.log("[updateFrame] Current scene's 'balls' (meshes) before update:", Object.keys(balls));

        const [goalX, goalY] = GOAL_POS.split(',').map(Number);
        for (let x = 0; x < f.grid_size; x++) {
            for (let y = 0; y < f.grid_size; y++) {
                const coord = `${x},${y}`;
                if (grid[coord]) {
                    let newColor;

                    if (SPECIAL_GRID_POSITIONS.includes(coord)) {
                        newColor = 0xFFFFFF;
                    } else if (f.snow[coord]) {
                        newColor = SNOW_COLOR;
                    } else if (x === goalX && y === goalY) {
                        newColor = REGULAR_GRID_COLOR;
                    } else {
                        newColor = UNTOUCHED_GRID_COLOR;
                    }

                    grid[coord].material.color.set(newColor);
                    grid[coord].material.needsUpdate = true;
                }
            }
        }

        const ballsCurrentlyAtGoal = [];
        for (let b in f.balls) {
            if (f.balls[b] === GOAL_POS) {
                ballsCurrentlyAtGoal.push({ name: b, size: f.ball_size[b] });
            }
        }

        ballsCurrentlyAtGoal.sort((a, b) => b.size - a.size);

        const [goalCellX, goalCellY] = GOAL_POS.split(',').map(Number);

        for (let b in f.balls) {
            if (!balls[b]) {
                console.warn(`[updateFrame] Creating ball ${b} dynamically. Should have been initialized.`);
                const geometry = new THREE.SphereGeometry(RADIUS[f.ball_size[b]], 32, 32);
                const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.6, metalness: 0.2 });
                balls[b] = new THREE.Mesh(geometry, material);
                balls[b].castShadow = true;
                balls[b].receiveShadow = true;
                balls[b].name = `ball_${b}`;
                scene.add(balls[b]);
            } else {
                if (balls[b].geometry.parameters.radius !== RADIUS[f.ball_size[b]]) {
                    balls[b].geometry.dispose();
                    balls[b].geometry = new THREE.SphereGeometry(RADIUS[f.ball_size[b]], 32, 32);
                    console.log(`[updateFrame] Updated geometry for ball: ${b} to size ${f.ball_size[b]} at frame ${Math.floor(currentFrame)}`);
                }
            }

            let posX, posY, posZ;
            if (f.balls[b] === GOAL_POS) {
                let currentHeight = 0;
                for (let i = 0; i < ballsCurrentlyAtGoal.length; i++) {
                    const stackedBall = ballsCurrentlyAtGoal[i];
                    if (stackedBall.name === b) {
                        posY = currentHeight + RADIUS[stackedBall.size];
                        break;
                    }
                    currentHeight += (RADIUS[stackedBall.size] * 2);
                }
                posX = goalCellX + 0.5;
                posZ = goalCellY + 0.5;
                balls[b].visible = true;

            } else if (['move_ball', 'push', 'roll', 'roll_ball'].includes(f.type) && f.ball === b) {
                const [sx, sy] = f.start.split(',').map(Number);
                const [ex, ey] = f.end.split(',').map(Number);
                posX = sx + 0.5 + f.alpha * (ex - sx);
                posZ = sy + 0.5 + f.alpha * (ey - sy);
                posY = RADIUS[f.ball_size[b]];
                balls[b].visible = true;
            }
            else {
                const [x, y] = f.balls[b].split(',').map(Number);
                posX = x + 0.5;
                posZ = y + 0.5;
                posY = RADIUS[f.ball_size[b]];
                balls[b].visible = true;
            }
            balls[b].position.set(posX, posY, posZ);
        }

        const ballsAtGoalFinalCheck = Object.entries(f.balls).filter(([_, pos]) => pos === GOAL_POS);
        const ballsAtGoalCount = ballsAtGoalFinalCheck.length;
        const sizesAtGoal = new Set(ballsAtGoalFinalCheck.map(([b]) => f.ball_size[b]));
        const isSnowmanFormed = (ballsAtGoalCount === 3 && sizesAtGoal.has(0) && sizesAtGoal.has(1) && sizesAtGoal.has(2));

        if (character) {
            character.visible = !(isSnowmanFormed && f.character === GOAL_POS);
        }

        if (character && character.visible && f.character) {
            let cx, cz, rotationY = 0;
            if (f.type === 'move_to_ball') {
                const [sx, sy] = f.start.split(',').map(Number);
                const [ex, ey] = f.end.split(',').map(Number);
                cx = sx + 0.5 + f.alpha * (ex - sx);
                cz = sy + 0.5 + f.alpha * (ey - sy);
                rotationY = f.direction === 'left' ? Math.PI : f.direction === 'right' ? 0 : f.direction === 'up' ? Math.PI / 2 : f.direction === 'down' ? -Math.PI / 2 : 0;
                spotlight.position.set(cx, 2, cz);
                spotlight.target.position.set(cx, 0, cz);
            } else if (['move_ball', 'push', 'roll', 'roll_ball'].includes(f.type)) {
                const [bx, by] = f.start.split(',').map(Number);
                const [ex, ey] = f.end.split(',').map(Number);
                cx = bx + 0.5 + f.alpha * (ex - bx);
                cz = by + 0.5 + f.alpha * (ey - by);
                rotationY = f.direction === 'left' ? Math.PI : f.direction === 'right' ? 0 : f.direction === 'up' ? Math.PI / 2 : f.direction === 'down' ? -Math.PI / 2 : 0;
                spotlight.position.set(cx, 2, cz);
                spotlight.target.position.set(balls[f.ball].position.x, balls[f.ball].position.y, balls[f.ball].position.z);
            } else if (f.type === 'move') {
                const [sx, sy] = f.start.split(',').map(Number);
                const [ex, ey] = f.end.split(',').map(Number);
                cx = sx + 0.5 + f.alpha * (ex - sx);
                cz = sy + 0.5 + f.alpha * (ey - sy);
                rotationY = f.direction === 'left' ? Math.PI : f.direction === 'right' ? 0 : f.direction === 'up' ? Math.PI / 2 : f.direction === 'down' ? -Math.PI / 2 : 0;
                spotlight.position.set(cx, 2, cz);
                spotlight.target.position.set(cx, 0, cz);
            } else {
                const [x, y] = f.character.split(',').map(Number);
                cx = x + 0.5;
                cz = y + 0.5;
                spotlight.position.set(cx, 2, cz);
                spotlight.target.position.set(cx, 0, cz);
            }
            character.position.set(cx, 0, cz);
            character.rotation.y = rotationY;
            console.log(`[updateFrame] Character position set to (${cx.toFixed(2)}, 0, ${cz.toFixed(2)}) for frame ${Math.floor(currentFrame)}. Frame type: ${f.type}.`);
        }
    } catch (err) {
        console.error('Error updating frame:', err);
        showPopup('errorPopup', 'errorMessage', `Error updating frame: ${err.message}`);
    }
}

/**
 * Resets the entire scene to its initial state, clearing all dynamic objects and data.
 * The static environment (ground, grid, walls, trees, forbidden icons) remains.
 * @param {boolean} clearFiles - If true, also clears the selected problem and plan files.
 */
function resetScene(clearFiles = true) {
    try {
        console.log('Resetting scene');
        planData = {
            problem: { ...defaultProblemData },
            frames: [],
            isNumeric: false
        };
        currentFrame = 0;
        isPlaying = false;
        currentTime = 0;
        startTime = null;

        if (clearFiles) {
            problemFile = null;
            planFile = null;
            document.getElementById('problemFile').value = '';
            document.getElementById('planFile').value = '';
        }

        playIcon.style.display = 'block';
        pauseIcon.style.display = 'none';
        playPauseText.textContent = 'Play';
        document.getElementById('step').value = 0;
        document.getElementById('step').max = 0;
        document.getElementById('step').disabled = true;
        ['playPause', 'stepForward', 'stepBackward', 'reset'].forEach(id => document.getElementById(id).disabled = true);

        clearDynamicSceneObjects();
        initStaticEnvironment(defaultProblemData);
        renderer.render(scene, camera);
    } catch (err) {
        console.error('Error resetting scene:', err);
        showPopup('errorPopup', 'errorMessage', `Error resetting scene: ${err.message}`);
    }
}

/**
 * Generates and returns a string with plan metrics.
 * @returns {string} - HTML formatted string of metrics.
 */
function getMetrics() {
    const totalSteps = Math.floor(planData.frames.length / SUBSTEPS);
    const planDuration = currentTime.toFixed(2);
    const ballsAtGoal = planData.frames.length > 0 ? Object.values(planData.frames[planData.frames.length - 1].balls).filter(pos => pos === GOAL_POS).length : 0;
    return `
        <strong>Plan Metrics:</strong><br>
        Total Steps: ${totalSteps}<br>
        Plan Duration: ${planDuration} seconds<br>
        Balls at Goal: ${ballsAtGoal}
    `;
}

/**
 * The main animation loop for Three.js.
 */
function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    controls.update();

    if (isPlaying && planData.frames.length > 0 && currentFrame < planData.frames.length - 1) {
        updateFrame(planData.frames[Math.floor(currentFrame)]);
        currentFrame += speed;
        document.getElementById('step').value = Math.floor(currentFrame / SUBSTEPS);
        currentTime = startTime ? (performance.now() - startTime) / 1000 : 0;
        if (currentFrame >= planData.frames.length - 1) {
            isPlaying = false;
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
            playPauseText.textContent = 'Play';
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

    if (mixer && isPlaying && planData.frames[Math.floor(currentFrame)] && ['move', 'move_to_ball', 'move_ball', 'push', 'roll', 'roll_ball'].includes(planData.frames[Math.floor(currentFrame)].type)) {
        mixer.update(delta);
    }

    renderer.render(scene, camera);
}

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
        document.getElementById('problemFile').style.display = 'none';
        document.getElementById('planFile').style.display = 'none';
        document.querySelectorAll('#controlPanel button, #controlPanel label, #controlPanel input[type="range"]').forEach(el => el.style.display = 'none');
        document.getElementById('resizeHandle').style.display = 'none';
    } else {
        document.getElementById('minimizeBtn').textContent = '-';
        document.getElementById('problemFile').style.display = 'block';
        document.getElementById('planFile').style.display = 'block';
        document.querySelectorAll('#controlPanel button, #controlPanel label, #controlPanel input[type="range"]').forEach(el => el.style.display = 'block');
        document.getElementById('resizeHandle').style.display = 'block';
    }
});

document.getElementById('menuBtn').addEventListener('click', () => {
    const menu = document.getElementById('menuDropdown');
    menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
});

document.getElementById('helpBtn').addEventListener('click', () => {
    showPopup('helpPopup', 'helpMessage', document.getElementById('helpMessage').innerHTML);
    document.getElementById('menuDropdown').style.display = 'none';
});
document.getElementById('metricsBtn').addEventListener('click', () => {
    showPopup('metricsPopup', 'metricsMessage', getMetrics());
    document.getElementById('menuDropdown').style.display = 'none';
});
document.getElementById('aboutBtn').addEventListener('click', () => {
    showPopup('aboutPopup', 'aboutMessage', document.getElementById('aboutMessage').innerHTML);
    document.getElementById('menuDropdown').style.display = 'none';
});

document.getElementById('errorClose').addEventListener('click', () => hidePopup('errorPopup'));
document.getElementById('helpClose').addEventListener('click', () => hidePopup('helpPopup'));
document.getElementById('metricsClose').addEventListener('click', () => hidePopup('metricsPopup'));
document.getElementById('aboutClose').addEventListener('click', () => hidePopup('aboutPopup'));

document.getElementById('problemFile').addEventListener('change', selectFiles);
document.getElementById('planFile').addEventListener('change', selectFiles);

document.getElementById('playPause').addEventListener('click', () => {
    if (planData.frames.length === 0) return;
    isPlaying = !isPlaying;
    startTime = performance.now() - currentTime * 1000;
    playIcon.style.display = isPlaying ? 'none' : 'block';
    pauseIcon.style.display = isPlaying ? 'block' : 'none';
    playPauseText.textContent = isPlaying ? 'Pause' : 'Play';
});

document.getElementById('stepForward').addEventListener('click', () => {
    if (planData.frames.length === 0 || currentFrame >= planData.frames.length - 1) return;
    isPlaying = false;
    playIcon.style.display = 'block';
    pauseIcon.style.display = 'none';
    playPauseText.textContent = 'Play';
    currentFrame = Math.min(planData.frames.length - 1, Math.floor(currentFrame / SUBSTEPS + 1) * SUBSTEPS);
    document.getElementById('step').value = Math.floor(currentFrame / SUBSTEPS);
    updateFrame(planData.frames[Math.floor(currentFrame)]);
    currentTime = planData.frames[Math.floor(currentFrame)].time;
    renderer.render(scene, camera);
});

document.getElementById('stepBackward').addEventListener('click', () => {
    if (planData.frames.length === 0 || currentFrame <= 0) return;
    isPlaying = false;
    playIcon.style.display = 'block';
    pauseIcon.style.display = 'none';
    playPauseText.textContent = 'Play';
    currentFrame = Math.max(0, Math.floor(currentFrame / SUBSTEPS - 1) * SUBSTEPS);
    document.getElementById('step').value = Math.floor(currentFrame / SUBSTEPS);
    updateFrame(planData.frames[Math.floor(currentFrame)]);
    currentTime = planData.frames[Math.floor(currentFrame)].time;
    renderer.render(scene, camera);
});

document.getElementById('reset').addEventListener('click', () => {
    if (planData.frames.length === 0) return;
    isPlaying = false;
    currentFrame = 0;
    currentTime = 0;
    startTime = null;
    playIcon.style.display = 'block';
    pauseIcon.style.display = 'none';
    playPauseText.textContent = 'Play';
    document.getElementById('step').value = 0;
    updateFrame(planData.frames[0]);
    renderer.render(scene, camera);
});

document.getElementById('speed').addEventListener('input', e => {
    speed = parseFloat(e.target.value);
});

document.getElementById('snowSpeed').addEventListener('input', e => {
    snowSpeed = parseFloat(e.target.value);
});

document.getElementById('step').addEventListener('input', e => {
    if (planData.frames.length === 0) return;
    isPlaying = false;
    playIcon.style.display = 'block';
    pauseIcon.style.display = 'none';
    playPauseText.textContent = 'Play';
    currentFrame = parseInt(e.target.value) * SUBSTEPS;
    updateFrame(planData.frames[Math.floor(currentFrame)]);
    currentTime = planData.frames[Math.floor(currentFrame)].time;
    renderer.render(scene, camera);
});

window.addEventListener('resize', () => {
    camera.aspect = 0.65 * window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth * 0.65, window.innerHeight - 70);
});

initCoreThreeJs();
initStaticEnvironment(defaultProblemData);