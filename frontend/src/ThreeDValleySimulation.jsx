import { useEffect, useMemo, useRef, useState } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';

const Plot = createPlotlyComponent(Plotly);

const SIZE = 85;
const MAX_ACTIVE_SLIDES = 240;
const VISUAL_BOOST = 1.2;
const REGION_PROFILES = {
    western_ghats: { maxSlopeDeg: 48, ridgeScale: 1.05, peakScale: 1.1, mountainSpread: 1.14, floodSensitivity: 1.02 },
    uttarakhand: { maxSlopeDeg: 63, ridgeScale: 1.2, peakScale: 1.34, mountainSpread: 1.26, floodSensitivity: 0.92 },
    assam: { maxSlopeDeg: 43, ridgeScale: 0.9, peakScale: 0.86, mountainSpread: 1.0, floodSensitivity: 1.16 }
};

function getRegionProfile(regionId) {
    return REGION_PROFILES[regionId] || REGION_PROFILES.western_ghats;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function pseudoNoise(i, j) {
    const value = Math.sin(i * 12.9898 + j * 78.233) * 43758.5453;
    return value - Math.floor(value);
}

function generateCinematicTerrain(regionProfile, size = SIZE) {
    const terrain = [];

    for (let i = 0; i < size; i++) {
        const row = [];
        for (let j = 0; j < size; j++) {
            const dx = (i - size / 2) / size;
            const dy = (j - size / 2) / size;

            const spread = regionProfile.mountainSpread;
            const ridgeScale = regionProfile.ridgeScale;
            const peakScale = regionProfile.peakScale;

            const ridgeNorth = Math.exp(-Math.pow(dy + 0.26 * spread, 2) * (18 / spread)) * (54 * ridgeScale * VISUAL_BOOST);
            const ridgeSouth = Math.exp(-Math.pow(dy - 0.28 * spread, 2) * (17 / spread)) * (52 * ridgeScale * VISUAL_BOOST);
            const peakWest = Math.exp(-((Math.pow(dx + 0.35 * spread, 2) * (30 / spread)) + (Math.pow(dy + 0.06, 2) * 48))) * (86 * peakScale * VISUAL_BOOST);
            const peakEast = Math.exp(-((Math.pow(dx - 0.36 * spread, 2) * (32 / spread)) + (Math.pow(dy - 0.08, 2) * 52))) * (82 * peakScale * VISUAL_BOOST);
            const centerPeak = Math.exp(-((Math.pow(dx, 2) * 60) + (Math.pow(dy + 0.4, 2) * 45))) * (46 * peakScale * VISUAL_BOOST);

            const valley = Math.exp(-Math.pow(dy, 2) * 72) * 66;
            const riverTrench = Math.exp(-Math.pow(dy, 2) * 380) * 14;

            const wave = Math.sin(dx * 20) * 2.9 + Math.cos(dy * 14) * 2.4;
            const texture = (pseudoNoise(i, j) - 0.5) * 3.6;

            const elevation = 84 + ridgeNorth + ridgeSouth + peakWest + peakEast + centerPeak - valley - riverTrench + wave + texture;
            row.push(clamp(elevation, 8, 220));
        }
        terrain.push(row);
    }

    return terrain;
}

function getTerrainExtrema(terrain) {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;

    terrain.forEach((row) => {
        row.forEach((value) => {
            min = Math.min(min, value);
            max = Math.max(max, value);
        });
    });

    return { min, max };
}

function buildSlopeAndFlow(terrain) {
    const slope = terrain.map((row) => row.map(() => 0));
    const flow = terrain.map((row) => row.map(() => ({ vx: 0, vy: 0 })));

    for (let i = 1; i < terrain.length - 1; i++) {
        for (let j = 1; j < terrain[0].length - 1; j++) {
            const dzdx = (terrain[i + 1][j] - terrain[i - 1][j]) / 2;
            const dzdy = (terrain[i][j + 1] - terrain[i][j - 1]) / 2;

            const steepness = Math.sqrt(dzdx * dzdx + dzdy * dzdy);
            slope[i][j] = steepness;

            const vx = -dzdx;
            const vy = -dzdy;
            const mag = Math.sqrt(vx * vx + vy * vy) || 1;
            flow[i][j] = {
                vx: vx / mag,
                vy: vy / mag
            };
        }
    }

    return { slope, flow };
}

function buildRoadGeometry(terrain) {
    const roads = [];
    const center = terrain.length / 2;

    const patterns = [
        { baseOffset: -11, waviness: 0.12, amp: 2.5 },
        { baseOffset: 10, waviness: 0.1, amp: 3.1 },
        { baseOffset: 0, waviness: 0.07, amp: 1.4 }
    ];

    patterns.forEach((pattern) => {
        const x = [];
        const y = [];
        const z = [];

        for (let i = 3; i < terrain.length - 3; i++) {
            const centerLine = center + pattern.baseOffset + Math.sin(i * pattern.waviness) * pattern.amp;
            const col = clamp(Math.round(centerLine), 2, terrain.length - 3);
            x.push(i);
            y.push(col);
            z.push(terrain[i][col] + 1.2);
        }

        roads.push({ x, y, z });
    });

    return roads;
}

function buildBuildingGeometry(terrain, slope) {
    const meshX = [];
    const meshY = [];
    const meshZ = [];
    const meshI = [];
    const meshJ = [];
    const meshK = [];
    const rooftopX = [];
    const rooftopY = [];
    const rooftopZ = [];
    const rooftopSize = [];

    const center = terrain.length / 2;
    const spacing = 7;

    for (let i = 8; i < terrain.length - 8; i += spacing) {
        for (let j = 8; j < terrain.length - 8; j += spacing) {
            if (Math.abs(j - center) < 6) {
                continue;
            }

            const localSlope = slope[i][j];
            if (localSlope > 3.6) {
                continue;
            }

            const seed = pseudoNoise(i * 1.33, j * 1.77);
            if (seed < 0.58) {
                continue;
            }

            const base = terrain[i][j] + 0.4;
            const height = (6 + seed * 12) * VISUAL_BOOST;
            const halfW = 0.5 + seed * 0.35;
            const halfD = 0.5 + seed * 0.3;

            const v = [
                [i - halfW, j - halfD, base],
                [i + halfW, j - halfD, base],
                [i + halfW, j + halfD, base],
                [i - halfW, j + halfD, base],
                [i - halfW, j - halfD, base + height],
                [i + halfW, j - halfD, base + height],
                [i + halfW, j + halfD, base + height],
                [i - halfW, j + halfD, base + height]
            ];

            const baseIndex = meshX.length;
            v.forEach((pt) => {
                meshX.push(pt[0]);
                meshY.push(pt[1]);
                meshZ.push(pt[2]);
            });

            const faces = [
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
                [4, 5, 6], [4, 6, 7]
            ];

            faces.forEach((face) => {
                meshI.push(baseIndex + face[0]);
                meshJ.push(baseIndex + face[1]);
                meshK.push(baseIndex + face[2]);
            });

            rooftopX.push(i);
            rooftopY.push(j);
            rooftopZ.push(base + height);
            rooftopSize.push(4 + seed * 5);
        }
    }

    return {
        meshX,
        meshY,
        meshZ,
        meshI,
        meshJ,
        meshK,
        rooftopX,
        rooftopY,
        rooftopZ,
        rooftopSize
    };
}

function generateRiverFlowVisuals(terrain, waterDepth, t, rainNorm) {
    const flowX = [];
    const flowY = [];
    const flowZ = [];
    const foamX = [];
    const foamY = [];
    const foamZ = [];
    const center = terrain.length / 2;

    for (let lane = -2; lane <= 2; lane++) {
        for (let i = 2; i < terrain.length - 2; i += 2) {
            const sway = Math.sin(i * 0.11 + t * 0.08 + lane) * 1.2;
            const y = clamp(Math.round(center + lane * 1.3 + sway), 1, terrain.length - 2);
            const z = terrain[i][y] + waterDepth[i][y] + 0.4;
            flowX.push(i);
            flowY.push(y);
            flowZ.push(z);

            const foamGate = (Math.sin(i * 0.34 + t * 0.22 + lane * 1.7) + 1) * 0.5;
            if (foamGate > 0.72 && waterDepth[i][y] > 0.9) {
                foamX.push(i + (pseudoNoise(i, lane + t * 0.2) - 0.5) * 0.8);
                foamY.push(y + (pseudoNoise(y, lane + t * 0.15) - 0.5) * 0.7);
                foamZ.push(z + 0.15 + rainNorm * 0.2);
            }
        }
        flowX.push(null);
        flowY.push(null);
        flowZ.push(null);
    }

    return { flowX, flowY, flowZ, foamX, foamY, foamZ };
}

function generateWaterAccumulation(terrain, extrema, rainfall, t, regionProfile) {
    const rainEffect = clamp((rainfall - 15) / 285, 0, 1.2);
    const amplitude = Math.pow(rainEffect, 1.2) * (1.5 + regionProfile.floodSensitivity * 0.7);
    const center = terrain.length / 2;

    if (rainfall <= 0 || rainEffect === 0) {
        const waterDepth = terrain.map((row) => row.map(() => 0));
        return { waterDepth, waterSurface: terrain };
    }

    const waterDepth = terrain.map((row, i) =>
        row.map((height, j) => {
            const lowlandFactor = (extrema.max - height) / (extrema.max - extrema.min + 1e-6);
            const riverMask = Math.exp(-Math.pow((j - center) / (terrain.length * 0.08), 2));
            const basinMask = Math.exp(-((Math.pow(i - terrain.length * 0.58, 2) + Math.pow(j - center, 2)) / (terrain.length * terrain.length * 0.016)));
            const pulse = 0.75 + (Math.sin(i * 0.2 + j * 0.18 + t * 0.16) + 1) * 0.24;

            const depth = Math.max(
                0,
                amplitude * pulse * rainEffect * ((lowlandFactor * 6.5) + (riverMask * 15.0) + (basinMask * 7.0)) - 1.1
            );

            return depth;
        })
    );

    

    const waterSurface = terrain.map((row, i) => row.map((h, j) => h + waterDepth[i][j]));
    return { waterDepth, waterSurface };
}

function generateRainStreaks(rainfall, maxElevation, t) {
    if (rainfall <= 0) {
        return { x: [], y: [], z: [], rainNorm: 0 };
    }
    const rainNorm = clamp(rainfall / 320, 0, 1.25);
    const streaks = Math.floor((180 + rainNorm * 520) * 1.08);
    const x = [];
    const y = [];
    const z = [];

    for (let i = 0; i < streaks; i++) {
        const randomX = (Math.sin((i + 1) * 17.1) * 0.5 + 0.5) * (SIZE - 1);
        const randomY = (Math.cos((i + 1) * 23.9) * 0.5 + 0.5) * (SIZE - 1);
        const phase = (t * 2.2 + i * 5.7) % 44;

        const top = maxElevation + 30 - phase;
        const bottom = top - (5.2 + rainNorm * 5.5);

        x.push(randomX, randomX, null);
        y.push(randomY, randomY, null);
        z.push(top, bottom, null);
    }

    return { x, y, z, rainNorm };
}

function spawnLandslideEvents(activeSlides, slope, flow, terrain, waterDepth, rainfall, t, regionProfile) {

    if (rainfall <= 80) return 0;

    let spawnCount = 0;

    const rainNorm = clamp((rainfall - 70) / 260, 0, 1.35);
    if (rainNorm <= 0.15) {
        return 0;
    }

    const slopeFactor = regionProfile.maxSlopeDeg / 65;
    const attempts = Math.floor(3 + rainNorm * 10);
    const center = SIZE / 2;

    for (let k = 0; k < attempts && activeSlides.length < MAX_ACTIVE_SLIDES; k++) {

        const base = t * 0.013 + k * 1.618;
        const i = clamp(Math.floor((Math.sin(base * 11.7) * 0.5 + 0.5) * (SIZE - 8)) + 4, 4, SIZE - 5);
        const j = clamp(Math.floor((Math.cos(base * 9.2) * 0.5 + 0.5) * (SIZE - 8)) + 4, 4, SIZE - 5);

        const steepness = slope[i][j];
        const steepnessThreshold = 7.1 - slopeFactor * 1.5;

        if (steepness < steepnessThreshold || Math.abs(j - center) < 5) {
            continue;
        }

        const waterInfluence = waterDepth[i][j] / 6;

        const boostedRain = Math.pow(rainNorm, 1.5);

        const triggerChance = clamp(
            boostedRain * (steepness / 11) * (0.68 + slopeFactor * 0.5) * (1 + waterInfluence),
            0,
            0.97
        );

        const noise = pseudoNoise(i + t * 0.1, j + t * 0.1);
        if (noise > triggerChance) continue;

        const velocityScale = 0.26 + (steepness / 14) * 0.34;
        const direction = flow[i][j];

        activeSlides.push({
            x: i,
            y: j,
            z: terrain[i][j] + 2,
            vx: direction.vx * velocityScale,
            vy: direction.vy * velocityScale,
            life: 22 + Math.floor(Math.random() * 14),
            trail: []
        });

        spawnCount++; // ✅ KEY LINE
    }

    if (activeSlides.length > MAX_ACTIVE_SLIDES) {
        activeSlides.splice(0, activeSlides.length - MAX_ACTIVE_SLIDES);
    }

    return spawnCount; // ✅ KEY LINE
}
function updateLandslideEvents(activeSlides, terrain, flow) {
    const alive = [];

    activeSlides.forEach((slide) => {
        const i = clamp(Math.round(slide.x), 1, SIZE - 2);
        const j = clamp(Math.round(slide.y), 1, SIZE - 2);
        const dir = flow[i][j];

        slide.vx = slide.vx * 0.72 + dir.vx * 0.24;
        slide.vy = slide.vy * 0.72 + dir.vy * 0.24;

        slide.x += slide.vx;
        slide.y += slide.vy;

        const ci = clamp(Math.round(slide.x), 1, SIZE - 2);
        const cj = clamp(Math.round(slide.y), 1, SIZE - 2);

        const surface = terrain[ci][cj] + 0.8;
        // 🪨 Erosion effect
        terrain[ci][cj] -= 0.005 + (0.005 * Math.random());
        slide.z = Math.max(surface, slide.z - 1.2);
        slide.life -= 1;

        slide.trail.push({ x: slide.x, y: slide.y, z: slide.z + 0.25 });
        if (slide.trail.length > 6) {
            slide.trail.shift();
        }

        if (
            slide.life > 0 &&
            slide.x >= 1 &&
            slide.x <= SIZE - 2 &&
            slide.y >= 1 &&
            slide.y <= SIZE - 2
        ) {
            alive.push(slide);
        }
    });

    return alive;
}

function landslideTraceData(slides) {
    const markerX = [];
    const markerY = [];
    const markerZ = [];
    const trailX = [];
    const trailY = [];
    const trailZ = [];

    slides.forEach((slide) => {
        markerX.push(slide.x);
        markerY.push(slide.y);
        markerZ.push(slide.z + 0.2);

        slide.trail.forEach((pt) => {
            trailX.push(pt.x);
            trailY.push(pt.y);
            trailZ.push(pt.z);
        });

        trailX.push(null);
        trailY.push(null);
        trailZ.push(null);
    });

    return {
        markerX,
        markerY,
        markerZ,
        trailX,
        trailY,
        trailZ
    };
}

export default function ThreeDValleySimulation({ rainfall, regionId = 'western_ghats' }) {
    const regionProfile = useMemo(() => getRegionProfile(regionId), [regionId]);
    const terrain = useMemo(() => generateCinematicTerrain(regionProfile), [regionProfile]);
    const extrema = useMemo(() => getTerrainExtrema(terrain), [terrain]);
    const { slope, flow } = useMemo(() => buildSlopeAndFlow(terrain), [terrain]);
    const roads = useMemo(() => buildRoadGeometry(terrain), [terrain]);
    const buildings = useMemo(() => buildBuildingGeometry(terrain, slope), [terrain, slope]);

    const [plotData, setPlotData] = useState(null);
    const [lightningFlash, setLightningFlash] = useState(0);
    const [dragMode, setDragMode] = useState('orbit');
    const [stats, setStats] = useState({
        floodCoverage: 0,
        landslideRisk: 0,
        activeSlides: 0,
        rainIntensity: 'Light',
        probabilityValue: 0
    });

    const frameRef = useRef(0);
    const landslidesRef = useRef([]);
    const lightningRef = useRef({
        flash: 0,
        cooldown: 0,
        x: Math.floor(SIZE * 0.5),
        y: Math.floor(SIZE * 0.5)
    });

    useEffect(() => {
        let animationFrame;

        const animate = () => {
            frameRef.current += 1;
            const t = frameRef.current;

            const { waterDepth, waterSurface } = generateWaterAccumulation(terrain, extrema, rainfall, t, regionProfile);
            const { x: rainX, y: rainY, z: rainZ, rainNorm } = generateRainStreaks(rainfall, extrema.max, t);
            const { flowX, flowY, flowZ, foamX, foamY, foamZ } = generateRiverFlowVisuals(terrain, waterDepth, t, rainNorm);
            const spawnedThisFrame = spawnLandslideEvents(
                landslidesRef.current,
                slope,
                flow,
                terrain,
                waterDepth,
                rainfall,
                t,
                regionProfile
            );
            landslidesRef.current = updateLandslideEvents(landslidesRef.current, terrain, flow);
            const landslideTraces = landslideTraceData(landslidesRef.current);

            let wetCells = 0;
            waterDepth.forEach((row) => {
                row.forEach((depth) => {
                    if (depth > 1.2) {
                        wetCells += 1;
                    }
                });
            });

            const floodCoverage = rainfall <= 0 ? 0 : wetCells / (SIZE * SIZE);
            const slopeFactor = regionProfile.maxSlopeDeg / 65;
            const landslideRisk = rainfall <= 80 ? 0 : clamp(
                (spawnedThisFrame / (3 + rainNorm * 10)) *
                (rainNorm + 0.3) *
                (0.8 + slopeFactor * 0.5),
                0,
                1
            );
            const probabilityValue = clamp(
                0.45 * landslideRisk +
                0.35 * floodCoverage +
                0.15 * rainNorm +
                0.05 * slopeFactor,
                0,
                1
            );
            const rainIntensity = rainfall < 60 ? 'Light' : rainfall < 140 ? 'Moderate' : rainfall < 240 ? 'Heavy' : 'Extreme';

            if (lightningRef.current.cooldown > 0) {
                lightningRef.current.cooldown -= 1;
            }

            if (rainfall > 200 && rainNorm > 0.85 && lightningRef.current.cooldown <= 0 && Math.random() < 0.014 * rainNorm) {
                lightningRef.current.flash = 1;
                lightningRef.current.cooldown = 28 + Math.floor(Math.random() * 42);
                lightningRef.current.x = clamp(Math.floor((Math.sin(t * 0.23) * 0.5 + 0.5) * (SIZE - 6)) + 3, 2, SIZE - 3);
                lightningRef.current.y = clamp(Math.floor((Math.cos(t * 0.27) * 0.5 + 0.5) * (SIZE - 6)) + 3, 2, SIZE - 3);
            }
            lightningRef.current.flash *= 0.88;
            const lightningOpacity = clamp(lightningRef.current.flash, 0, 1);
            setLightningFlash(lightningOpacity);

            setStats({
                floodCoverage,
                landslideRisk,
                activeSlides: landslidesRef.current.length,
                rainIntensity,
                probabilityValue
            });

            const cameraOrbit = 1.44 + Math.sin(t * 0.01) * 0.08;

            const terrainSurface = {
                type: 'surface',
                z: terrain,
                colorscale: [
                    [0.0, '#0ea5e9'],
                    [0.1, '#22c55e'],
                    [0.35, '#84cc16'],
                    [0.55, '#facc15'],
                    [0.75, '#a16207'],
                    [1.0, '#f8fafc']
                ],
                showscale: false,
                contours: {
                    z: {
                        show: true,
                        usecolormap: false,
                        color: 'rgba(10, 24, 48, 0.16)',
                        width: 1
                    }
                },
                lighting: {
                    ambient: 0.58,
                    diffuse: 0.95,
                    roughness: 0.82,
                    specular: 0.2,
                    fresnel: 0.04
                },
                lightposition: { x: 120, y: 90, z: 220 }
            };

            const waterTrace = {
                type: 'surface',
                z: waterSurface,
                colorscale: [
                    [0.0, 'rgba(59, 130, 246, 0.35)'],
                    [0.45, 'rgba(56, 189, 248, 0.65)'],
                    [1.0, 'rgba(14, 165, 233, 0.95)']
                ],
                opacity: 0.72,
                showscale: false,
                lighting: {
                    ambient: 0.5,
                    diffuse: 0.82,
                    roughness: 0.2,
                    specular: 1,
                    fresnel: 0.2
                },
                lightposition: { x: 120, y: 120, z: 300 }
            };

            const roadTraces = roads.map((road, idx) => ({
                type: 'scatter3d',
                mode: 'lines',
                x: road.x,
                y: road.y,
                z: road.z,
                line: {
                    color: idx === 2 ? '#facc15' : '#374151',
                    width: idx === 2 ? 4 : 8
                },
                hoverinfo: 'skip',
                showlegend: false
            }));

            const buildingColumnsTrace = {
                type: 'mesh3d',
                x: buildings.meshX,
                y: buildings.meshY,
                z: buildings.meshZ,
                i: buildings.meshI,
                j: buildings.meshJ,
                k: buildings.meshK,
                color: '#f59e0b',
                opacity: 0.82,
                flatshading: true,
                lighting: {
                    ambient: 0.38,
                    diffuse: 0.85,
                    roughness: 0.52,
                    specular: 0.4
                },
                hoverinfo: 'skip',
                showlegend: false
            };

            const buildingRooftopTrace = {
                type: 'scatter3d',
                mode: 'markers',
                x: buildings.rooftopX,
                y: buildings.rooftopY,
                z: buildings.rooftopZ,
                marker: {
                    color: '#fde68a',
                    size: buildings.rooftopSize,
                    opacity: 0.98,
                    line: { color: '#f59e0b', width: 1 }
                },
                hoverinfo: 'skip',
                showlegend: false
            };

            const rainTrace = {
                type: 'scatter3d',
                mode: 'lines',
                x: rainX,
                y: rainY,
                z: rainZ,
                line: {
                    color: rainNorm > 0.8 ? 'rgba(125, 211, 252, 0.72)' : 'rgba(147, 197, 253, 0.55)',
                    width: rainNorm > 0.8 ? 3.1 : 2.2
                },
                hoverinfo: 'skip',
                showlegend: false
            };

            const riverFlowTrace = {
                type: 'scatter3d',
                mode: 'lines',
                x: flowX,
                y: flowY,
                z: flowZ,
                line: {
                    color: 'rgba(56, 189, 248, 0.72)',
                    width: 7
                },
                hoverinfo: 'skip',
                showlegend: false
            };

            const riverFoamTrace = {
                type: 'scatter3d',
                mode: 'markers',
                x: foamX,
                y: foamY,
                z: foamZ,
                marker: {
                    color: 'rgba(226, 232, 240, 0.9)',
                    size: 3.6,
                    opacity: 0.86
                },
                hoverinfo: 'skip',
                showlegend: false
            };

            const landslideTrailTrace = {
                type: 'scatter3d',
                mode: 'lines',
                x: landslideTraces.trailX,
                y: landslideTraces.trailY,
                z: landslideTraces.trailZ,
                line: {
                    color: 'rgba(249, 115, 22, 0.55)',
                    width: 4
                },
                hoverinfo: 'skip',
                showlegend: false
            };

            const landslideMarkerTrace = {
                type: 'scatter3d',
                mode: 'markers',
                x: landslideTraces.markerX,
                y: landslideTraces.markerY,
                z: landslideTraces.markerZ,
                marker: {
                    color: '#ef4444',
                    size: 5,
                    opacity: 0.95,
                    line: { color: '#f97316', width: 1 }
                },
                name: 'Active Landslides',
                showlegend: false
            };

            const strikeX = clamp(Math.round(lightningRef.current.x), 0, SIZE - 1);
            const strikeY = clamp(Math.round(lightningRef.current.y), 0, SIZE - 1);
            const strikeGround = terrain[strikeX][strikeY] + 1;
            const lightningTrace = {
                type: 'scatter3d',
                mode: 'lines',
                x: [
                    strikeX + Math.sin(t * 0.7) * 0.7,
                    strikeX + Math.sin(t * 0.9 + 0.7) * 1.1,
                    strikeX + Math.sin(t * 1.1 + 1.5) * 0.5,
                    strikeX
                ],
                y: [
                    strikeY + Math.cos(t * 0.8) * 0.8,
                    strikeY + Math.cos(t * 0.95 + 0.9) * 1.0,
                    strikeY + Math.cos(t * 1.2 + 1.3) * 0.7,
                    strikeY
                ],
                z: [extrema.max + 32, extrema.max + 20, extrema.max + 9, strikeGround],
                line: {
                    color: `rgba(255, 255, 255, ${0.15 + lightningOpacity * 0.85})`,
                    width: 2 + lightningOpacity * 5
                },
                opacity: lightningOpacity,
                hoverinfo: 'skip',
                showlegend: false
            };

            setPlotData({
                data: [
                    terrainSurface,
                    waterTrace,
                    ...roadTraces,
                    buildingColumnsTrace,
                    buildingRooftopTrace,
                    rainTrace,
                    riverFlowTrace,
                    riverFoamTrace,
                    landslideTrailTrace,
                    landslideMarkerTrace,
                    lightningTrace
                ],
                layout: {
                    title: {
                        text: `Cinematic Hazard Twin | Rainfall ${rainfall} mm`,
                        font: { size: 24, color: '#e2e8f0' }
                    },
                    paper_bgcolor: lightningOpacity > 0.08 ? '#111827' : '#020617',
                    plot_bgcolor: lightningOpacity > 0.08 ? '#0b1220' : '#020617',
                    margin: { l: 0, r: 0, t: 42, b: 0 },
                    scene: {
                        aspectmode: 'manual',
                        aspectratio: { x: 1.35, y: 1, z: 0.9 },
                        dragmode: dragMode,
                        camera: {
                            eye: { x: cameraOrbit, y: 1.44, z: 0.92 },
                            up: { x: 0, y: 0, z: 1 }
                        },
                        xaxis: {
                            visible: false,
                            showbackground: false,
                            showgrid: false,
                            zeroline: false
                        },
                        yaxis: {
                            visible: false,
                            showbackground: false,
                            showgrid: false,
                            zeroline: false
                        },
                        zaxis: {
                            visible: false,
                            showbackground: false,
                            showgrid: false,
                            zeroline: false
                        }
                    }
                }
            });

            animationFrame = requestAnimationFrame(animate);
        };

        animate();
        return () => cancelAnimationFrame(animationFrame);
    }, [terrain, extrema, slope, flow, roads, buildings, rainfall, regionProfile, dragMode]);

    return (
        <div style={{ width: '100%', height: '100vh', position: 'relative', background: 'radial-gradient(circle at 50% 20%, #0b1224 0%, #020617 65%)' }}>
            <div
                style={{
                    position: 'absolute',
                    inset: 0,
                    pointerEvents: 'none',
                    zIndex: 6,
                    opacity: lightningFlash * 0.34,
                    background: 'radial-gradient(circle at 50% 28%, rgba(255,255,255,0.95), rgba(125,211,252,0.22) 45%, rgba(2,6,23,0) 70%)',
                    transition: 'opacity 80ms linear'
                }}
            />
            <div
                style={{
                    position: 'absolute',
                    top: 68,
                    left: 14,
                    background: 'linear-gradient(145deg, rgba(6, 23, 49, 0.9), rgba(4, 12, 28, 0.8))',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    borderRadius: '14px',
                    padding: '14px 16px',
                    color: '#e2e8f0',
                    zIndex: 9,
                    minWidth: '290px',
                    boxShadow: '0 14px 42px rgba(2, 6, 23, 0.65)'
                }}
            >
                <div style={{ fontWeight: 700, letterSpacing: '0.5px', marginBottom: 8, fontSize: 17 }}>Hazard Visual Intelligence</div>
                <div style={{ fontSize: 15, marginBottom: 5 }}>Region Slope Max: <span style={{ color: '#fbbf24', fontWeight: 600 }}>{regionProfile.maxSlopeDeg}°</span></div>
                <div style={{ fontSize: 15, marginBottom: 5 }}>Rainfall: <span style={{ color: '#7dd3fc', fontWeight: 600 }}>{rainfall} mm</span> ({stats.rainIntensity})</div>
                <div style={{ fontSize: 15, marginBottom: 5 }}>Flood Accumulation: <span style={{ color: '#38bdf8', fontWeight: 600 }}>{(stats.floodCoverage * 100).toFixed(1)}%</span></div>
                <div style={{ fontSize: 15, marginBottom: 5 }}>Probability Value: <span style={{ color: '#a78bfa', fontWeight: 600 }}>{stats.probabilityValue.toFixed(3)}</span></div>
                <div style={{ fontSize: 15, marginBottom: 5 }}>Landslide Risk: <span style={{ color: '#f97316', fontWeight: 600 }}>{(stats.landslideRisk * 100).toFixed(1)}%</span></div>
                <div style={{ fontSize: 15 }}>Active Landslides: <span style={{ color: '#f43f5e', fontWeight: 700 }}>{stats.activeSlides}</span></div>
            </div>

            <div
                style={{
                    position: 'absolute',
                    top: 68,
                    right: 80,
                    zIndex: 9,
                    display: 'flex',
                    gap: 8,
                    background: 'rgba(2, 6, 23, 0.65)',
                    border: '1px solid rgba(56, 189, 248, 0.35)',
                    borderRadius: 10,
                    padding: '8px 10px'
                }}
            >
                <button
                    type="button"
                    onClick={() => setDragMode((prev) => (prev === 'orbit' ? 'turntable' : 'orbit'))}
                    style={{
                        background: '#1d4ed8',
                        border: 'none',
                        color: '#fff',
                        borderRadius: 8,
                        padding: '6px 10px',
                        cursor: 'pointer',
                        fontSize: 12,
                        fontWeight: 600
                    }}
                >
                    Rotate: {dragMode === 'orbit' ? 'Orbit' : 'Turntable'}
                </button>
            </div>

            {plotData ? (
                <Plot
                    data={plotData.data}
                    layout={plotData.layout}
                    config={{
                        responsive: true,
                        displaylogo: false,
                        displayModeBar: true,
                        scrollZoom: true
                    }}
                    style={{ width: '100%', height: '100%' }}
                />
            ) : (
                <div style={{ color: '#e2e8f0', padding: 20 }}>Preparing cinematic 3D simulation...</div>
            )}
        </div>
    );
}
