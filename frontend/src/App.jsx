import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap, CircleMarker } from 'react-leaflet';
import { Activity, Droplets, Mountain, AlertTriangle, Layers, Map as MapIcon, Play, Image as ImageIcon } from 'lucide-react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import './App.css';
import ThreeDValleySimulation from './ThreeDValleySimulation';

const Plot = createPlotlyComponent(Plotly);

const MAX_3D_POINTS = 1800;

const REGIONS = [
  { id: 'uttarakhand', name: 'Uttarakhand', center: [30.0, 80.0], zoom: 8 },
  { id: 'western_ghats', name: 'Western Ghats', center: [10.0, 75.5], zoom: 7 },
  { id: 'assam', name: 'Assam', center: [25.0, 90.0], zoom: 8 }
];

const REGION_STRICT_BBOX = {
  uttarakhand: { minLat: 28.4, maxLat: 31.6, minLon: 77.0, maxLon: 81.5 },
  assam: { minLat: 24.0, maxLat: 28.8, minLon: 89.5, maxLon: 96.5 },
  western_ghats: { minLat: 8.0, maxLat: 21.5, minLon: 73.0, maxLon: 78.8 }
};

const MODES = [
  { id: 'historical', name: 'Historical' },
  { id: 'live', name: 'Live' },
  { id: 'dynamic', name: 'Dynamic Input' }
];


const METRICS = [
  {
    id: 'probability',
    name: 'Probability',
    icon: Activity,
    color: '#ef4444'
  },
  {
    id: 'flood',
    name: 'Flood',
    icon: Droplets,
    color: '#3b82f6'
  },
  {
    id: 'landslide',
    name: 'Landslide',
    icon: Mountain,
    color: '#f97316'
  },
  {
    id: 'risk',
    name: 'Composite Risk',
    icon: AlertTriangle,
    color: '#a855f7'
  }
];

const RISK_METRIC_ID = 'risk';

const BACKEND_ORIGIN = 'http://localhost:8345';

const INDIA_BOUNDS = [
  [6.5, 67.5],
  [38.8, 98.8]
];

// Mainland India polygon (lon, lat) for strict client-side filtering fallback.
const INDIA_MAINLAND_POLYGON = [
  [68.1, 23.9], [68.4, 22.0], [69.1, 20.5], [70.3, 19.0], [71.6, 17.2],
  [72.5, 15.5], [73.3, 13.8], [74.0, 12.0], [75.3, 10.2], [76.8, 8.4],
  [78.4, 8.1], [79.8, 9.0], [80.8, 10.5], [81.3, 12.4], [81.8, 14.3],
  [82.5, 16.2], [83.6, 18.2], [85.2, 20.1], [86.8, 21.7], [88.3, 22.5],
  [89.8, 22.4], [91.4, 23.2], [92.8, 24.7], [94.3, 26.4], [95.5, 27.8],
  [96.6, 28.7], [96.9, 29.5], [95.2, 29.9], [93.5, 28.8], [91.8, 27.9],
  [90.0, 27.4], [88.2, 27.7], [86.4, 28.4], [84.7, 29.2], [82.7, 30.4],
  [80.8, 31.2], [78.8, 32.3], [77.1, 33.2], [75.3, 34.1], [73.6, 34.8],
  [72.2, 34.8], [71.1, 33.5], [70.5, 31.8], [69.7, 30.4], [69.0, 28.9],
  [68.5, 27.2], [68.2, 25.5], [68.1, 23.9]
];

function isPointInPolygon(point, polygon) {
  const [x, y] = point;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];

    const intersect = ((yi > y) !== (yj > y))
      && (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-12) + xi);
    if (intersect) inside = !inside;
  }

  return inside;
}

function isPointInIndia(lat, lon) {
  // Quick reject by India-wide bounds first.
  if (lat < 6.0 || lat > 38.5 || lon < 67.5 || lon > 98.5) {
    return false;
  }
  return isPointInPolygon([lon, lat], INDIA_MAINLAND_POLYGON);
}

function filterGeoJsonToIndia(data, regionId) {
  if (!data?.features?.length) {
    return data;
  }

  const strictBounds = REGION_STRICT_BBOX[regionId];

  const filtered = data.features.filter((feature) => {
    const center = getGeometryCenter(feature?.geometry);
    if (!center) {
      return false;
    }
    const [lon, lat] = center;

    if (!isPointInIndia(lat, lon)) {
      return false;
    }

    if (!strictBounds) {
      return true;
    }

    return (
      lat >= strictBounds.minLat &&
      lat <= strictBounds.maxLat &&
      lon >= strictBounds.minLon &&
      lon <= strictBounds.maxLon
    );
  });

  return {
    ...data,
    features: filtered
  };
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function hexToRgb(hex) {
  const cleaned = hex.replace('#', '');
  const expanded = cleaned.length === 3
    ? cleaned.split('').map((char) => char + char).join('')
    : cleaned;

  const numeric = Number.parseInt(expanded, 16);
  return {
    r: (numeric >> 16) & 255,
    g: (numeric >> 8) & 255,
    b: numeric & 255
  };
}

function mixWithWhite(hex, mixRatio) {
  const ratio = clamp01(mixRatio);
  const { r, g, b } = hexToRgb(hex);
  const blended = {
    r: Math.round(255 - (255 - r) * ratio),
    g: Math.round(255 - (255 - g) * ratio),
    b: Math.round(255 - (255 - b) * ratio)
  };

  return `rgb(${blended.r}, ${blended.g}, ${blended.b})`;
}

function formatHazardValue(value) {
  return Number.isFinite(value) ? value.toFixed(3) : 'n/a';
}

function getFeatureLabel(feature) {
  const props = feature?.properties || {};
  return (
    props.name ||
    props.NAME ||
    props.title ||
    props.label ||
    props.district ||
    props.region ||
    feature?.id ||
    'Selected location'
  );
}

function getFeatureKey(feature) {
  const props = feature?.properties || {};
  return props.cell_id ?? feature?.id ?? getFeatureLabel(feature);
}

function buildFocusedPlace(feature, outputMode = 'physics') {
  const center = getGeometryCenter(feature?.geometry);
  if (!center) {
    return null;
  }

  const props = feature?.properties || {};
  const featureKey = getFeatureKey(feature);
  const flood = props.flood ?? props.probability ?? 0;
  const landslide = props.landslide ?? props.stress ?? 0;
  const riskPropertyId = getRiskPropertyId(outputMode);
  const risk = props[riskPropertyId] ?? props.hybrid_risk_score ?? props.ml_risk_score ?? props.physics_risk_score ?? props.risk ?? props.composite_risk ?? Math.max(flood, landslide);

  return {
    featureKey,
    label: getFeatureLabel(feature),
    centerLngLat: center,
    centerLatLng: [center[1], center[0]],
    flood,
    landslide,
    risk,
    riskMode: outputMode,
    riskLabel: 'Hybrid'
  };
}

function getRiskPropertyId(_outputMode) {
  return 'risk';
}

function getDisplayMetricPropertyId(metricId, outputMode) {
  if (metricId === 'risk') {
    return getRiskPropertyId(outputMode);
  }

  return metricId;
}

function MapFocusController({ place }) {
  const map = useMap();

  useEffect(() => {
    if (!place) {
      return;
    }

    map.flyTo(place.centerLatLng, 10.5, { duration: 1.25 });
  }, [map, place]);

  return null;
}

function HazardFocusOverlay({ place, progress, animating }) {
  if (!place) {
    return null;
  }

  const floodWave = clamp01(progress / 0.45);
  const landslideWave = clamp01((progress - 0.38) / 0.54);
  const beat = 0.5 + 0.5 * Math.sin(progress * Math.PI * 2);

  const floodRadius = 10 + Math.max(6, place.flood * 10) * (0.85 + 0.45 * floodWave);
  const landslideRadius = 8 + Math.max(6, place.landslide * 9) * (0.82 + 0.5 * landslideWave);
  const coreRadius = 5 + beat * 4;

  return (
    <>
      <CircleMarker
        center={place.centerLatLng}
        radius={floodRadius}
        pathOptions={{
          color: '#3b82f6',
          weight: 2,
          fillColor: '#3b82f6',
          fillOpacity: 0.12 + 0.22 * floodWave,
          opacity: 0.7
        }}
      />
      <CircleMarker
        center={place.centerLatLng}
        radius={landslideRadius}
        pathOptions={{
          color: '#f97316',
          weight: 2,
          fillColor: '#f97316',
          fillOpacity: 0.08 + 0.2 * landslideWave,
          opacity: 0.75
        }}
      />
      <CircleMarker
        center={place.centerLatLng}
        radius={coreRadius}
        pathOptions={{
          color: animating ? '#ffffff' : '#c9d1d9',
          weight: 1,
          fillColor: animating ? '#ffffff' : '#c9d1d9',
          fillOpacity: 1,
          opacity: 1
        }}
      />
    </>
  );
}

function MapController({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, zoom, { duration: 1.5 });
  }, [center, zoom, map]);
  return null;
}

function collectBounds(coordinates, bounds) {
  if (!Array.isArray(coordinates)) {
    return;
  }

  if (coordinates.length >= 2 && typeof coordinates[0] === 'number' && typeof coordinates[1] === 'number') {
    const [lon, lat] = coordinates;
    bounds.minLon = Math.min(bounds.minLon, lon);
    bounds.maxLon = Math.max(bounds.maxLon, lon);
    bounds.minLat = Math.min(bounds.minLat, lat);
    bounds.maxLat = Math.max(bounds.maxLat, lat);
    return;
  }

  coordinates.forEach((entry) => collectBounds(entry, bounds));
}

function getGeometryCenter(geometry) {
  if (!geometry || !geometry.coordinates) {
    return null;
  }

  const bounds = {
    minLon: Infinity,
    maxLon: -Infinity,
    minLat: Infinity,
    maxLat: -Infinity
  };

  collectBounds(geometry.coordinates, bounds);

  if (!Number.isFinite(bounds.minLon) || !Number.isFinite(bounds.minLat)) {
    return null;
  }

  return [
    (bounds.minLon + bounds.maxLon) / 2,
    (bounds.minLat + bounds.maxLat) / 2
  ];
}

function buildThreeDPlot(geoData, metric, dataRange, propertyId) {
  if (!geoData?.features?.length) {
    return null;
  }

  const features = geoData.features;
  const stride = Math.max(1, Math.floor(features.length / MAX_3D_POINTS));
  const points = [];

  for (let index = 0; index < features.length && points.length < MAX_3D_POINTS; index += stride) {
    const feature = features[index];
    const value = feature?.properties?.[propertyId];

    if (value === undefined || value === null || !feature?.geometry) {
      continue;
    }

    const center = getGeometryCenter(feature.geometry);
    if (!center) {
      continue;
    }

    const range = (dataRange.max - dataRange.min) || 1;
    const normalized = metric.id === 'probability' ? value : (value - dataRange.min) / range;
    const zValue = Math.max(0, normalized) * 100;

    points.push({
      x: center[0],
      y: center[1],
      z: zValue,
      value
    });
  }

  if (!points.length) {
    return null;
  }

  const colorscale = metric.id === 'probability'
    ? 'Reds'
    : metric.id === 'flood'
      ? 'Blues'
      : metric.id === 'landslide'
        ? 'Oranges'
        : 'Purples';

  const zValues = points.map((point) => point.z);

  return {
    data: [
      {
        type: 'scatter3d',
        mode: 'markers',
        x: points.map((point) => point.x),
        y: points.map((point) => point.y),
        z: zValues,
        text: points.map((point) => `${metric.name}: ${point.value.toFixed(4)}`),
        hovertemplate:
          'Longitude %{x:.3f}<br>' +
          'Latitude %{y:.3f}<br>' +
          'Normalized height %{z:.2f}<br>' +
          '%{text}<extra></extra>',
        marker: {
          size: 3,
          color: zValues,
          colorscale,
          opacity: 0.85,
          colorbar: {
            title: `${metric.name} height`
          }
        }
      }
    ],
    layout: {
      title: {
        text: `3D ${metric.name} Hazard View`,
        font: { color: '#fff', size: 18 }
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 0, r: 0, t: 50, b: 0 },
      font: { color: '#c9d1d9' },
      scene: {
        xaxis: {
          title: 'Longitude',
          color: '#c9d1d9',
          backgroundcolor: 'rgba(13, 17, 23, 0.75)',
          gridcolor: 'rgba(255,255,255,0.08)'
        },
        yaxis: {
          title: 'Latitude',
          color: '#c9d1d9',
          backgroundcolor: 'rgba(13, 17, 23, 0.75)',
          gridcolor: 'rgba(255,255,255,0.08)'
        },
        zaxis: {
          title: 'Height',
          color: '#c9d1d9',
          backgroundcolor: 'rgba(13, 17, 23, 0.75)',
          gridcolor: 'rgba(255,255,255,0.08)'
        },
        aspectmode: 'data',
        bgcolor: 'rgba(0,0,0,0)'
      }
    }
  };
}

function resolveThreeDPlot(geoData, metric, dataRange, propertyId) {
  if (!geoData?.features?.length) {
    return null;
  }

  const candidates = [
    metric,
    { id: 'risk', name: 'Composite Risk' },
    { id: 'probability', name: 'Probability' },
    { id: 'flood', name: 'Flood' },
    { id: 'landslide', name: 'Landslide' },
    { id: 'stress', name: 'Stress' }
  ];

  const seen = new Set();
  for (const m of candidates) {
    if (!m?.id || seen.has(m.id)) {
      continue;
    }
    seen.add(m.id);

    const values = geoData.features
      .map((f) => f?.properties?.[m.id === 'risk' ? propertyId : m.id])
      .filter((v) => v !== undefined && v !== null);

    if (!values.length) {
      continue;
    }

    const range = {
      min: Math.min(...values),
      max: Math.max(...values)
    };

    const plot = buildThreeDPlot(geoData, m, range, m.id === 'risk' ? propertyId : m.id);
    if (plot) {
      return plot;
    }
  }

  return null;
}

function App() {
  const [region, setRegion] = useState(REGIONS[0]);
  const [simMode, setSimMode] = useState(MODES[0]); // default live
  const [metric, setMetric] = useState(METRICS[0]);
  const outputMode = 'hybrid'; // Fixed: 70% ML / 30% Physics
  const [rainfallInput, setRainfallInput] = useState(100);

  const [geoData, setGeoData] = useState(null);
  const [plotImages, setPlotImages] = useState([]);
  const [plotsVersion, setPlotsVersion] = useState(Date.now());

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dataRange, setDataRange] = useState({ min: 0, max: 1 });
  const [showPlots, setShowPlots] = useState(false);
  const [show3DView, setShow3DView] = useState(false);
  const [threeDPlot, setThreeDPlot] = useState(null);
  const [riskStats, setRiskStats] = useState(null);
  const [statsFromFile, setStatsFromFile] = useState(false);
  const [legendCutoff, setLegendCutoff] = useState(0.75);
  const [filterMode, setFilterMode] = useState('above');
  const [filterWindow, setFilterWindow] = useState(10);
  const [sparsePulsePhase, setSparsePulsePhase] = useState(0);
  const [focusedPlace, setFocusedPlace] = useState(null);
  const [focusAnimating, setFocusAnimating] = useState(true);
  const [focusProgress, setFocusProgress] = useState(0);

  const [progress, setProgress] = useState({ percent: 0, status: '' });

  useEffect(() => {
    setFocusedPlace(null);
    setFocusAnimating(true);
    setFocusProgress(0);
  }, [region.id, simMode.id]);

  // Auto-load cache strictly for Historical mode
  useEffect(() => {
    if (simMode.id === 'historical') {
      loadHistoricalCachedOutputs(region.id);
    } else {
      setGeoData(null);
      setPlotImages([]);
      setShowPlots(false);
      setShow3DView(false);
      setThreeDPlot(null);
      setRiskStats(null);
      setStatsFromFile(false);
    }
  }, [region, simMode.id]);

  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(async () => {
        try {
          const res = await fetch('/api/progress');
          const data = await res.json();
          setProgress({ percent: data.progress, status: data.status });
        } catch (e) { }
      }, 500);
    } else {
      setProgress({ percent: 0, status: '' });
    }
    return () => clearInterval(interval);
  }, [loading]);

  useEffect(() => {
    if (!geoData) return;

    const displayMetricId = getDisplayMetricPropertyId(metric.id, outputMode);

    let min = Infinity, max = -Infinity;

    geoData.features.forEach(f => {
      const val = f.properties[displayMetricId];
      if (val !== undefined && val !== null) {
        if (val < min) min = val;
        if (val > max) max = val;
      }
    });

    setDataRange({
      min: min === Infinity ? 0 : min,
      max: max === -Infinity ? 1 : max
    });

  }, [metric, outputMode, geoData]);

  useEffect(() => {
    if (!geoData) {
      setThreeDPlot(null);
      setShow3DView(false);
      return;
    }

    const displayMetricId = getDisplayMetricPropertyId(metric.id, outputMode);
    setThreeDPlot(resolveThreeDPlot(geoData, metric, dataRange, displayMetricId));
  }, [geoData, metric, outputMode, dataRange]);

  useEffect(() => {
    if (!focusedPlace || !focusAnimating) {
      setFocusProgress(0);
      return undefined;
    }

    let frameId = 0;
    const duration = 4200;
    const startTime = performance.now();

    const step = (now) => {
      const elapsed = (now - startTime) % duration;
      setFocusProgress(elapsed / duration);
      frameId = window.requestAnimationFrame(step);
    };

    frameId = window.requestAnimationFrame(step);

    return () => window.cancelAnimationFrame(frameId);
  }, [focusedPlace, focusAnimating]);

  const openThreeDView = () => {
    if (!geoData) {
      return;
    }

    const displayMetricId = getDisplayMetricPropertyId(metric.id, outputMode);
    const plot = resolveThreeDPlot(geoData, metric, dataRange, displayMetricId);
    if (!plot) {
      setError('3D view is unavailable because no plottable metric values were found in the loaded GeoJSON.');
      return;
    }

    setThreeDPlot(plot);
    setShow3DView(true);
  };

  useEffect(() => {
    if (statsFromFile) {
      return;
    }

    if (!geoData?.features?.length) {
      setRiskStats(null);
      return;
    }

    const features = geoData.features;
    const gridSize = features.length;

    // Extract all property values
    const probValues = features
      .map(f => f.properties?.probability)
      .filter(v => v !== undefined && v !== null);

    const floodValues = features
      .map(f => f.properties?.flood)
      .filter(v => v !== undefined && v !== null);

    const stressValues = features
      .map(f => {
        const p = f.properties || {};
        return p.stress ?? p.landslide;
      })
      .filter(v => v !== undefined && v !== null);

    const riskValues = features
      .map(f => f.properties?.risk)
      .filter(v => v !== undefined && v !== null);

    // Calculate ranges
    const getRange = (values) => {
      if (!values.length) return { min: 0, max: 0 };
      return {
        min: Math.min(...values),
        max: Math.max(...values)
      };
    };

    const probRange = getRange(probValues);
    const floodRange = getRange(floodValues);
    const stressRange = getRange(stressValues);
    const riskRange = getRange(riskValues);

    // Categorize function
    const categorize = (values) => {
      const total = values.length;
      const low = values.filter(v => v < 0.25).length;
      const medium = values.filter(v => v >= 0.25 && v < 0.5).length;
      const high = values.filter(v => v >= 0.5 && v < 0.75).length;
      const extreme = values.filter(v => v >= 0.75).length;

      return {
        low: { count: low, percent: ((low / total) * 100).toFixed(2) },
        medium: { count: medium, percent: ((medium / total) * 100).toFixed(2) },
        high: { count: high, percent: ((high / total) * 100).toFixed(2) },
        extreme: { count: extreme, percent: ((extreme / total) * 100).toFixed(2) }
      };
    };

    // Normalize values for categorization (0-1 range)
    const normalizeValues = (values, range) => {
      const span = range.max - range.min || 1;
      return values.map(v => (v - range.min) / span);
    };

    if (!probValues.length) {
      setRiskStats(null);
      return;
    }

    const normalizedFlood = normalizeValues(floodValues, floodRange);
    const normalizedStress = normalizeValues(stressValues, stressRange);

    setRiskStats({
      gridSize,
      probRange,
      floodRange,
      stressRange,
      riskRange,
      floodRisk: categorize(normalizedFlood),
      landslideRisk: categorize(normalizedStress),
      overallProbability: categorize(probValues)
    });
  }, [geoData, statsFromFile]);

  async function loadSimulationStatsFromPath(statsPath) {
    try {
      const cacheBust = `t=${Date.now()}`;
      const relativeUrl = `${statsPath}${statsPath.includes('?') ? '&' : '?'}${cacheBust}`;
      let response = await fetch(relativeUrl, { cache: 'no-store' });

      if (!response.ok) {
        const absoluteUrl = `${BACKEND_ORIGIN}${statsPath}${statsPath.includes('?') ? '&' : '?'}${cacheBust}`;
        response = await fetch(absoluteUrl, { cache: 'no-store' });
      }

      if (!response.ok) {
        return false;
      }

      const rawText = await response.text();
      if (!rawText.trim()) {
        return false;
      }

      setRiskStats(JSON.parse(rawText));
      setStatsFromFile(true);
      return true;
    } catch (err) {
      return false;
    }
  }

  async function loadGeoJsonFromPath(geoJsonPath) {
    try {
      const cacheBust = `t=${Date.now()}`;
      const relativeUrl = `${geoJsonPath}${geoJsonPath.includes('?') ? '&' : '?'}${cacheBust}`;
      let response = await fetch(relativeUrl, { cache: 'no-store' });

      if (!response.ok) {
        const absoluteUrl = `${BACKEND_ORIGIN}${geoJsonPath}${geoJsonPath.includes('?') ? '&' : '?'}${cacheBust}`;
        response = await fetch(absoluteUrl, { cache: 'no-store' });
      }

      if (!response.ok) {
        return false;
      }

      const rawText = await response.text();
      if (!rawText.trim()) {
        return false;
      }

      const parsed = JSON.parse(rawText);
      const data = filterGeoJsonToIndia(parsed, region.id);

      let min = Infinity, max = -Infinity;
      const displayMetricId = getDisplayMetricPropertyId(metric.id, outputMode);

      if (data && data.features) {
        data.features.forEach(f => {
          const val = f.properties[displayMetricId];
          if (val !== undefined && val !== null) {
            if (val < min) min = val;
            if (val > max) max = val;
          }
        });
      }
      setDataRange({ min: min === Infinity ? 0 : min, max: max === -Infinity ? 1 : max });
      setGeoData(data);
      setFocusedPlace(null);
      return true;
    } catch (err) {
      return false;
    }
  }

  async function loadGeoJsonWithMl(regId, modeId) {
    try {
      const cacheBust = `t=${Date.now()}`;
      const apiUrl = `${BACKEND_ORIGIN}/api/geojson?region=${regId}&mode=${modeId}&include_ml=true&${cacheBust}`;
      let response = await fetch(apiUrl, { cache: 'no-store' });

      if (!response.ok) {
        const fallbackPath = `/outputs/${regId}_${modeId}.geojson`;
        return loadGeoJsonFromPath(fallbackPath);
      }

      const data = await response.json();

      const displayMetricId = getDisplayMetricPropertyId(metric.id, outputMode);
      let min = Infinity;
      let max = -Infinity;

      if (data && data.features) {
        data.features.forEach((feature) => {
          const val = feature.properties?.[displayMetricId];
          if (val !== undefined && val !== null) {
            if (val < min) min = val;
            if (val > max) max = val;
          }
        });
      }

      setDataRange({ min: min === Infinity ? 0 : min, max: max === -Infinity ? 1 : max });
      setGeoData(data);
      setFocusedPlace(null);
      return true;
    } catch (err) {
      return false;
    }
  }

  async function loadSimulationStatsFromPath(statsPath) {
    try {
      const cacheBust = `t=${Date.now()}`;
      const relativeUrl = `${statsPath}${statsPath.includes('?') ? '&' : '?'}${cacheBust}`;
      let response = await fetch(relativeUrl, { cache: 'no-store' });

      if (!response.ok) {
        const absoluteUrl = `${BACKEND_ORIGIN}${statsPath}${statsPath.includes('?') ? '&' : '?'}${cacheBust}`;
        response = await fetch(absoluteUrl, { cache: 'no-store' });
      }

      if (!response.ok) {
        return false;
      }

      const rawText = await response.text();
      if (!rawText.trim()) {
        return false;
      }

      setRiskStats(JSON.parse(rawText));
      setStatsFromFile(true);
      return true;
    } catch (err) {
      return false;
    }
  }

  async function loadGeoJsonOnly(regId, modeId) {
    setStatsFromFile(false);

    const statsLoaded = await loadSimulationStatsFromPath(`/outputs/${regId}_${modeId}_stats.json`);
    const summaryLoaded = await loadGeoJsonWithMl(regId, modeId);

    if (!summaryLoaded) {
      await loadGeoJsonFromPath(`/outputs/${regId}_${modeId}_summary.geojson`);
    }

    if (!statsLoaded && !summaryLoaded) {
      setStatsFromFile(false);
    }
  }

  async function loadHistoricalCachedOutputs(regId) {
    setStatsFromFile(false);

    // Load stats from the pre-existing static file (always present after historical run).
    await loadSimulationStatsFromPath(`/outputs/${regId}_historical_stats.json`);

    // Load GeoJSON from static files directly — no API processing needed.
    // These files exist on disk after a historical simulation has been run.
    const staticCandidates = [
      `/outputs/${regId}_historical_summary.geojson`,
      `/outputs/${regId}_historical.geojson`
    ];

    for (const geoPath of staticCandidates) {
      const loaded = await loadGeoJsonFromPath(geoPath);
      if (loaded) {
        setPlotImages([
          `/outputs/${regId}_historical_prob_hist.png`,
          `/outputs/${regId}_historical_multi_hazard.png`,
          `/outputs/${regId}_historical_rainfall_scenario.png`
        ]);
        setPlotsVersion(Date.now());
        setShowPlots(true);
        return true;
      }
    }

    return false;
  }

  const runSimulation = async () => {
    setLoading(true);
    setProgress({ percent: 0, status: 'Initializing Backend Engine...' });
    setError(null);
    setPlotImages([]);
    setShow3DView(false);
    setFocusedPlace(null);

    // Historical mode should be instant when cache exists.
    if (simMode.id === 'historical') {
      const loadedFromCache = await loadHistoricalCachedOutputs(region.id);
      if (loadedFromCache) {
        setProgress({ percent: 100, status: 'Loaded historical cache.' });
        setLoading(false);
        return;
      }
    }

    // Try proxy first, then direct backend origin (useful outside Vite dev proxy).
    const payload = {
      region: region.id,
      mode: simMode.id,
      rainfall: Number(rainfallInput)
    };

    const simulateUrls = [
      '/api/simulate',
      `${BACKEND_ORIGIN}/api/simulate`
    ];

    try {
      let lastError = null;
      let response = null;

      for (const url of simulateUrls) {
        try {
          response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          if (response) {
            break;
          }
        } catch (fetchErr) {
          lastError = fetchErr;
        }
      }

      if (!response) {
        throw lastError || new Error('Failed to connect to backend');
      }

      const rawResponse = await response.text();
      let resData = {};

      if (rawResponse.trim()) {
        try {
          resData = JSON.parse(rawResponse);
        } catch {
          resData = {};
        }
      }

      if (!response.ok) {
        const fallbackMessage = response.status
          ? `Simulation failed (${response.status}). Backend may be down.`
          : 'Simulation failed. Backend may be down.';
        throw new Error(resData.detail || fallbackMessage);
      }

      if (resData.inline_geojson) {
        // Dynamic/live: server returned the ready-to-use GeoJSON inline.
        // Apply it directly — no second fetch round-trip needed.
        const data = filterGeoJsonToIndia(resData.inline_geojson, region.id);
        const displayMetricId = getDisplayMetricPropertyId(metric.id, outputMode);
        let min = Infinity, max = -Infinity;
        if (data?.features) {
          data.features.forEach(f => {
            const val = f.properties?.[displayMetricId];
            if (val !== undefined && val !== null) {
              if (val < min) min = val;
              if (val > max) max = val;
            }
          });
        }
        setDataRange({ min: min === Infinity ? 0 : min, max: max === -Infinity ? 1 : max });
        setGeoData(data);
        setFocusedPlace(null);
      } else if (resData.geojson) {
        const loaded = await loadGeoJsonWithMl(region.id, simMode.id);
        if (!loaded) {
          const fallbackCandidates = [
            `/outputs/${region.id}_${simMode.id}_summary.geojson`,
            `/outputs/${region.id}_${simMode.id}.geojson`
          ];

          let fallbackLoaded = false;
          for (const candidate of fallbackCandidates) {
            fallbackLoaded = await loadGeoJsonFromPath(candidate);
            if (fallbackLoaded) break;
          }

          if (!fallbackLoaded) {
            throw new Error(`/outputs/${region.id}_${simMode.id}.geojson: missing`);
          }
        }
      } else {
        await loadGeoJsonOnly(region.id, simMode.id);
      }

      if (Array.isArray(resData.plots) && resData.plots.length > 0) {
        setPlotImages(resData.plots);
        setPlotsVersion(Date.now());
        setShowPlots(true);
      }

      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const generatePlotsOnDemand = async () => {
    setLoading(true);
    setProgress({ percent: 0, status: 'Generating plots...' });
    setError(null);

    const payload = {
      region: region.id,
      mode: simMode.id,
      rainfall: Number(rainfallInput)
    };

    const endpointCandidates = [
      '/api/generate-plots',
      `${BACKEND_ORIGIN}/api/generate-plots`
    ];

    try {
      let response = null;
      let lastError = null;

      for (const endpoint of endpointCandidates) {
        try {
          response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          if (response) {
            break;
          }
        } catch (error) {
          lastError = error;
        }
      }

      if (!response) {
        throw lastError || new Error('Failed to connect to backend');
      }

      const rawResponse = await response.text();
      let resData = {};

      if (rawResponse.trim()) {
        try {
          resData = JSON.parse(rawResponse);
        } catch {
          resData = {};
        }
      }

      if (!response.ok) {
        const fallbackMessage = response.status
          ? `Plot generation failed (${response.status}). Backend may be down.`
          : 'Plot generation failed. Backend may be down.';
        throw new Error(resData.detail || fallbackMessage);
      }

      if (Array.isArray(resData.plots) && resData.plots.length > 0) {
        setPlotImages(resData.plots);
        setPlotsVersion(Date.now());
        setShowPlots(true);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getMetricValue = (feature) => {
    const displayMetricId = getDisplayMetricPropertyId(metric.id, outputMode);
    const raw = feature?.properties?.[displayMetricId];
    if (raw === undefined || raw === null || Number.isNaN(raw)) {
      return null;
    }
    return Number(raw);
  };

  const isProbabilityMetric = metric.id === 'probability';
  const legendMin = isProbabilityMetric ? 0 : dataRange.min;
  const legendMax = isProbabilityMetric ? 1 : dataRange.max;
  const legendSpan = (legendMax - legendMin) || 1;
  const activeCutoffValue = legendMin + legendCutoff * legendSpan;
  const activeWindow = Math.max(0, Number(filterWindow) || 0);
  const rangeUpperValue = activeCutoffValue + activeWindow;
  const valueStep = isProbabilityMetric ? 0.01 : 1;
  const valuePrecision = isProbabilityMetric ? 3 : 1;

  const handleLegendPick = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = clamp01((event.clientX - rect.left) / rect.width);
    setLegendCutoff(ratio);
  };

  const setCutoffByValue = (value) => {
    if (!Number.isFinite(value)) {
      return;
    }

    const clampedValue = Math.max(legendMin, Math.min(legendMax, value));
    const ratio = clamp01((clampedValue - legendMin) / legendSpan);
    setLegendCutoff(ratio);
  };

  const handleCutoffInputChange = (event) => {
    const nextValue = Number(event.target.value);
    setCutoffByValue(nextValue);
  };

  const getNormalizedValue = (value) => clamp01((value - legendMin) / legendSpan);

  const getFeatureStyle = (feature) => {
    const val = getMetricValue(feature);
    const featureKey = getFeatureKey(feature);
    const isFocusedFeature = focusedPlace && String(featureKey) === String(focusedPlace.featureKey);

    if (val === undefined || val === null) {
      return {
        fillColor: '#444',
        fillOpacity: 0.2,
        weight: 0.5,
        color: '#222'
      };
    }

    let normalized;

    normalized = getNormalizedValue(val);

    const fillColor = mixWithWhite(metric.color, normalized);

    return {
      fillColor,
      fillOpacity: isFocusedFeature ? 0.95 : focusedPlace ? 0.04 : 0.9,
      weight: isFocusedFeature ? 1.8 : focusedPlace ? 0.1 : 0.3,
      color: isFocusedFeature ? '#f0f6fc' : focusedPlace ? 'rgba(255,255,255,0.12)' : '#111',
      opacity: focusedPlace && !isFocusedFeature ? 0.15 : 0.9
    };

  };

  const onEachFeature = (feature, layer) => {
    const val = getMetricValue(feature);
    if (val !== null) {
      layer.bindTooltip(`${metric.name}: ${val.toFixed(4)}`, { sticky: true });
    }

    layer.on('click', () => {
      const place = buildFocusedPlace(feature, outputMode);
      if (place) {
        setFocusedPlace(place);
        setFocusAnimating(true);
      }
    });
  };

  const thresholdFilteredGeoData = geoData?.features?.length
    ? {
      ...geoData,
      features: geoData.features.filter((feature) => {
        const metricValue = getMetricValue(feature);
        if (metricValue === null) {
          return false;
        }

        if (filterMode === 'band') {
          return metricValue >= activeCutoffValue && metricValue <= rangeUpperValue;
        }

        return metricValue >= activeCutoffValue;
      })
    }
    : geoData;

  const displayGeoData = focusedPlace && thresholdFilteredGeoData?.features?.length
    ? {
      ...thresholdFilteredGeoData,
      features: thresholdFilteredGeoData.features.filter((feature) => String(getFeatureKey(feature)) === String(focusedPlace.featureKey))
    }
    : thresholdFilteredGeoData;

  const visibleCount = thresholdFilteredGeoData?.features?.length ?? 0;
  const totalCount = geoData?.features?.length ?? 0;
  const sparseMarkers = visibleCount > 0 && visibleCount < 15
    ? (thresholdFilteredGeoData?.features ?? [])
      .map((feature) => {
        const center = getGeometryCenter(feature?.geometry);
        const value = getMetricValue(feature);

        if (!center || value === null) {
          return null;
        }

        return {
          key: String(getFeatureKey(feature)),
          centerLatLng: [center[1], center[0]],
          value
        };
      })
      .filter(Boolean)
    : [];

  useEffect(() => {
    if (visibleCount <= 0 || visibleCount >= 15) {
      setSparsePulsePhase(0);
      return undefined;
    }

    let frameId = 0;
    const duration = 1500;
    const startTime = performance.now();

    const step = (now) => {
      const elapsed = (now - startTime) % duration;
      setSparsePulsePhase(elapsed / duration);
      frameId = window.requestAnimationFrame(step);
    };

    frameId = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(frameId);
  }, [visibleCount]);

  const geoJsonRenderKey = [
    region.id,
    metric.id,
    outputMode,
    simMode.id,
    filterMode,
    activeCutoffValue.toFixed(6),
    activeWindow.toFixed(6),
    focusedPlace?.featureKey ?? 'nofocus'
  ].join('-');

  const statsGridSize = Number.isFinite(riskStats?.gridSize) ? riskStats.gridSize : 0;

  const getRangeBounds = (rangeObj) => ({
    min: Number.isFinite(rangeObj?.min) ? rangeObj.min : 0,
    max: Number.isFinite(rangeObj?.max) ? rangeObj.max : 0
  });

  const getBucketInfo = (groupName, levelName) => {
    const group = riskStats?.[groupName];
    const entry = group?.[levelName];

    if (entry && typeof entry === 'object') {
      const count = Number.isFinite(entry.count)
        ? entry.count
        : (Number.isFinite(entry.value) ? entry.value : 0);
      const percent = entry.percent ?? (statsGridSize > 0 ? ((count / statsGridSize) * 100).toFixed(2) : '0.00');
      return { count, percent: String(percent) };
    }

    const count = Number.isFinite(entry) ? entry : 0;
    const percent = statsGridSize > 0 ? ((count / statsGridSize) * 100).toFixed(2) : '0.00';
    return { count, percent };
  };

  const probRange = getRangeBounds(riskStats?.probRange);
  const floodRange = getRangeBounds(riskStats?.floodRange);
  const stressRange = getRangeBounds(riskStats?.stressRange);
  const riskRange = getRangeBounds(riskStats?.riskRange);

  const floodLow = getBucketInfo('floodRisk', 'low');
  const floodMedium = getBucketInfo('floodRisk', 'medium');
  const floodHigh = getBucketInfo('floodRisk', 'high');
  const floodExtreme = getBucketInfo('floodRisk', 'extreme');

  const landslideLow = getBucketInfo('landslideRisk', 'low');
  const landslideMedium = getBucketInfo('landslideRisk', 'medium');
  const landslideHigh = getBucketInfo('landslideRisk', 'high');
  const landslideExtreme = getBucketInfo('landslideRisk', 'extreme');

  const overallLow = getBucketInfo('overallProbability', 'low');
  const overallMedium = getBucketInfo('overallProbability', 'medium');
  const overallHigh = getBucketInfo('overallProbability', 'high');
  const overallExtreme = getBucketInfo('overallProbability', 'extreme');
  const [show3DSim, setShow3DSim] = useState(false);

  return (
    <div className="app-container">
      {/* Header */}
      <div className="header glass-panel">
        <MapIcon color="#58a6ff" />
        <h1>Digital Twin Hazard Dashboard</h1>
      </div>

      {/* Plots Modal Panel */}
      {showPlots && plotImages.length > 0 && (
        <div className="plots-panel glass-panel">
          <div className="plots-header">
            <h3>Simulation Plots ({simMode.name})</h3>
            <button onClick={() => setShowPlots(false)}>Close</button>
          </div>
          <div className="plots-grid">
            {plotImages.map((img, idx) => (
              <img key={idx} src={`${img}?t=${plotsVersion}`} alt="Simulation Plot" className="plot-img" />
            ))}
          </div>
        </div>
      )}

      {/* 3D View Modal */}
      {show3DView && threeDPlot && (
        <div className="three-d-panel glass-panel">
          <div className="plots-header">
            <h3>3D Hazard View ({metric.name})</h3>
            <button onClick={() => setShow3DView(false)}>Close</button>
          </div>
          <div className="three-d-chart">
            <Plot
              data={threeDPlot.data}
              layout={threeDPlot.layout}
              config={{ responsive: true, displaylogo: false }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
            />
          </div>
        </div>
      )}

      {/* Floating Controls */}
      <div className="floating-panel glass-panel">

        {/* Region */}
        <div className="control-group">
          <label><Layers size={14} style={{ verticalAlign: 'middle', marginRight: 4 }} />Region</label>
          <select
            className="dropdown-select"
            value={region.id}
            onChange={(e) => setRegion(REGIONS.find(r => r.id === e.target.value))}
          >
            {REGIONS.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
          </select>
        </div>

        {/* Mode */}
        <div className="control-group">
          <label>Simulation Mode</label>
          <div className="model-options">
            {MODES.map(m => (
              <button
                key={m.id}
                className="model-btn"
                data-active={simMode.id === m.id}
                onClick={() => setSimMode(m)}
              >
                {m.name}
              </button>
            ))}
          </div>
        </div>

        {/* Dynamic Rainfall Slider */}
        {simMode.id === 'dynamic' && (
          <div className="control-group slider-group">
            <label>Rainfall Input: {rainfallInput} mm</label>
            <input
              type="range"
              min="0" max="500" step="10"
              value={rainfallInput}
              onChange={(e) => setRainfallInput(e.target.value)}
              className="rainbow-slider"
            />
          </div>
        )}

        <button
          className="run-btn"
          onClick={runSimulation}
          disabled={loading}
          style={loading ? { background: `linear-gradient(90deg, #2ea043 ${progress.percent}%, #30363d ${progress.percent}%)` } : {}}
        >
          {loading ? <div className="spinner-small" /> : <Play size={16} />}
          {loading ? `${Math.round(progress.percent)}% - ${progress.status || 'Starting...'}` : 'Run Simulation'}
        </button>

        {plotImages.length > 0 && !showPlots && (
          <button className="view-plots-btn" onClick={() => setShowPlots(true)}>
            <ImageIcon size={16} /> View Rendered Plots
          </button>
        )}

        {(simMode.id === 'live' || simMode.id === 'dynamic') && geoData && !loading && (
          <button className="view-plots-btn" onClick={generatePlotsOnDemand}>
            <ImageIcon size={16} /> Generate Plots
          </button>
        )}
        <button
          className="view-3d-btn"
          onClick={() => setShow3DSim(prev => !prev)}
          disabled={simMode.id !== 'dynamic'}
        >
          🌍 {show3DSim ? 'Hide 3D Simulation' : 'Show 3D Simulation'}
        </button>

        <button
          className="view-3d-btn"
          onClick={openThreeDView}
          disabled={!geoData || loading}
          title={geoData ? 'Open interactive 3D hazard view' : 'Run a simulation to enable 3D view'}
        >
          <Mountain size={16} /> {geoData ? 'View 3D Hazard View' : '3D View (run simulation first)'}
        </button>

        <hr style={{ borderColor: 'var(--panel-border)', width: '100%' }} />

        {/* Risk Statistics */}
        {riskStats && !loading && (
          <details className="risk-stats-details">
            <summary className="risk-stats-summary">Simulation Statistics</summary>
            <div className="risk-stats-container">
              <div className="risk-stats-title">═══ SIMULATION STATISTICS ═══</div>

              <div className="stats-section-small">
                <span>Grid size: <span className="stat-highlight">{statsGridSize.toLocaleString()}</span></span>
              </div>

              <div className="stats-divider">Ranges</div>
              <div className="range-stats">
                <div className="range-row">
                  <span>Probability:</span>
                  <span className="range-val">{probRange.min.toFixed(4)} → {probRange.max.toFixed(4)}</span>
                </div>
                <div className="range-row">
                  <span>Flood:</span>
                  <span className="range-val">{floodRange.min.toFixed(1)} → {floodRange.max.toFixed(1)}</span>
                </div>
                <div className="range-row">
                  <span>Stress:</span>
                  <span className="range-val">{stressRange.min.toFixed(1)} → {stressRange.max.toFixed(1)}</span>
                </div>
                <div className="range-row">
                  <span>Risk:</span>
                  <span className="range-val">{riskRange.min.toFixed(1)} → {riskRange.max.toFixed(1)}</span>
                </div>
              </div>

              <div className="stats-divider">Flood Risk</div>
              <div className="risk-stat-row">
                <span className="risk-label">low:</span>
                <span className="risk-value">{floodLow.count.toLocaleString()} ({floodLow.percent}%)</span>
              </div>
              <div className="risk-stat-row">
                <span className="risk-label">medium:</span>
                <span className="risk-value">{floodMedium.count.toLocaleString()} ({floodMedium.percent}%)</span>
              </div>
              <div className="risk-stat-row">
                <span className="risk-label">high:</span>
                <span className="risk-value">{floodHigh.count.toLocaleString()} ({floodHigh.percent}%)</span>
              </div>
              <div className="risk-stat-row">
                <span className="risk-label">extreme:</span>
                <span className="risk-value">{floodExtreme.count.toLocaleString()} ({floodExtreme.percent}%)</span>
              </div>

              <div className="stats-divider">Landslide Risk</div>
              <div className="risk-stat-row">
                <span className="risk-label">low:</span>
                <span className="risk-value">{landslideLow.count.toLocaleString()} ({landslideLow.percent}%)</span>
              </div>
              <div className="risk-stat-row">
                <span className="risk-label">medium:</span>
                <span className="risk-value">{landslideMedium.count.toLocaleString()} ({landslideMedium.percent}%)</span>
              </div>
              <div className="risk-stat-row">
                <span className="risk-label">high:</span>
                <span className="risk-value">{landslideHigh.count.toLocaleString()} ({landslideHigh.percent}%)</span>
              </div>
              <div className="risk-stat-row">
                <span className="risk-label">extreme:</span>
                <span className="risk-value">{landslideExtreme.count.toLocaleString()} ({landslideExtreme.percent}%)</span>
              </div>

              <div className="stats-divider">Overall Probability</div>
              <div className="risk-stat-row">
                <span className="risk-label">low:</span>
                <span className="risk-value">{overallLow.count.toLocaleString()} ({overallLow.percent}%)</span>
              </div>
              <div className="risk-stat-row">
                <span className="risk-label">medium:</span>
                <span className="risk-value">{overallMedium.count.toLocaleString()} ({overallMedium.percent}%)</span>
              </div>
              <div className="risk-stat-row">
                <span className="risk-label">high:</span>
                <span className="risk-value">{overallHigh.count.toLocaleString()} ({overallHigh.percent}%)</span>
              </div>
              <div className="risk-stat-row">
                <span className="risk-label">extreme:</span>
                <span className="risk-value">{overallExtreme.count.toLocaleString()} ({overallExtreme.percent}%)</span>
              </div>
            </div>
          </details>
        )}

        {/* Hazard Metric */}
        <div className="control-group">
          <label>Hazard Metric Overlay</label>
          <div className="model-options" style={{ gridTemplateColumns: '1fr' }}>
            {METRICS.map(m => {
              const Icon = m.icon;

              return (
                <button
                  key={m.id}
                  className="model-btn"
                  data-active={metric.id === m.id}
                  onClick={() => setMetric(m)}
                  style={{ textAlign: 'left', paddingLeft: '1rem', display: 'flex', gap: '10px' }}
                >
                  <Icon size={16} /> {m.name}
                </button>
              );
            })}
          </div>
        </div>


        {/* Dynamic Legend */}
        {geoData && !loading && (
          <div className="legend-container">
            <div className="legend-title">
              {metric.id === 'risk' ? 'Composite Risk (Hybrid: 70% ML / 30% Physics)' : `${metric.name} Severity`}
            </div>
            <div
              className="legend-scale legend-scale-interactive"
              style={{
                background: `linear-gradient(to right, white, ${metric.color})`
              }}
              onClick={handleLegendPick}
              onMouseMove={(event) => {
                if (event.buttons === 1) {
                  handleLegendPick(event);
                }
              }}
              title="Click or drag to show only higher-intensity cells"
            />
            <div
              className="legend-thumb"
              style={{ left: `calc(${(legendCutoff * 100).toFixed(2)}% - 7px)` }}
            />
            <div className="legend-labels">
              <span>{legendMin.toFixed(2)}</span>
              <span>{legendMax.toFixed(2)}</span>
            </div>
            <div className="legend-filter-readout">
              <span>
                {filterMode === 'band'
                  ? `Showing ${activeCutoffValue.toFixed(valuePrecision)} to ${rangeUpperValue.toFixed(valuePrecision)}`
                  : `Showing >= ${activeCutoffValue.toFixed(valuePrecision)}`}
              </span>
              <span>{visibleCount.toLocaleString()} / {totalCount.toLocaleString()} cells</span>
            </div>
            {visibleCount > 0 && visibleCount < 15 && (
              <div className="legend-filter-readout">
                <span>Sparse mode active</span>
                <span>Animating {visibleCount} cells</span>
              </div>
            )}
            <div className="legend-filter-controls">
              <div className="legend-filter-modes">
                <button
                  type="button"
                  className="model-btn"
                  data-active={filterMode === 'above'}
                  onClick={() => setFilterMode('above')}
                >
                  Above
                </button>
                <button
                  type="button"
                  className="model-btn"
                  data-active={filterMode === 'band'}
                  onClick={() => setFilterMode('band')}
                >
                  Range
                </button>
              </div>
              <div className="legend-filter-inputs">
                <label>
                  Min
                  <input
                    type="number"
                    step={valueStep}
                    value={activeCutoffValue.toFixed(valuePrecision)}
                    min={legendMin}
                    max={legendMax}
                    onChange={handleCutoffInputChange}
                  />
                </label>
                {filterMode === 'band' && (
                  <label>
                    Width
                    <input
                      type="number"
                      step={valueStep}
                      value={activeWindow}
                      min={0}
                      onChange={(event) => setFilterWindow(Number(event.target.value))}
                    />
                  </label>
                )}
              </div>
            </div>
          </div>
        )}

        {error && (
          <div style={{ color: '#ff7b72', fontSize: '0.85rem', marginTop: '1rem' }}>
            <AlertTriangle size={14} style={{ verticalAlign: 'middle' }} /> {error}
          </div>
        )}

        {focusedPlace && (
          <div className="focus-card">
            <div className="focus-card-header">
              <div>
                <div className="focus-card-kicker">Selected place</div>
                <div className="focus-card-title">{focusedPlace.label}</div>
              </div>
              <button
                className="focus-clear-btn"
                onClick={() => {
                  setFocusedPlace(null);
                  setFocusAnimating(true);
                  setFocusProgress(0);
                }}
              >
                Clear
              </button>
            </div>

            <div className="focus-hint">
              Click a polygon on the map to focus a place, then watch flood and landslide rings pulse around it.
            </div>

            <div className="focus-metrics">
              <div className="focus-metric">
                <span>Flood</span>
                <strong>{formatHazardValue(focusedPlace.flood)}</strong>
              </div>
              <div className="focus-metric">
                <span>Landslide</span>
                <strong>{formatHazardValue(focusedPlace.landslide)}</strong>
              </div>
              <div className="focus-metric">
                <span>Risk ({focusedPlace.riskLabel})</span>
                <strong>{formatHazardValue(focusedPlace.risk)}</strong>
              </div>
            </div>

            <div className="focus-actions">
              <button
                className="focus-toggle-btn"
                onClick={() => setFocusAnimating((current) => !current)}
              >
                {focusAnimating ? 'Pause animation' : 'Play animation'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Map Container */}
      <div className="map-container">
        <MapContainer
          center={region.center}
          zoom={region.zoom}
          style={{ width: '100%', height: '100%' }}
          zoomControl={false}
          preferCanvas={true}
          maxBounds={INDIA_BOUNDS}
          maxBoundsViscosity={1.0}
          minZoom={5}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png"
          />
          <MapController center={region.center} zoom={region.zoom} />
          <MapFocusController place={focusedPlace} />
          {displayGeoData && !loading && (
            <GeoJSON
              key={geoJsonRenderKey}
              data={displayGeoData}
              style={getFeatureStyle}
              onEachFeature={onEachFeature}
            />
          )}
          {sparseMarkers.map((marker, index) => {
            const phaseOffset = ((index % 5) * 0.14) % 1;
            const pulse = 0.5 + 0.5 * Math.sin((sparsePulsePhase + phaseOffset) * Math.PI * 2);
            const outerRadius = 10 + pulse * 8;

            return (
              <CircleMarker
                key={`sparse-${marker.key}`}
                center={marker.centerLatLng}
                radius={outerRadius}
                pathOptions={{
                  color: metric.color,
                  weight: 2,
                  fillColor: metric.color,
                  fillOpacity: 0.08 + pulse * 0.18,
                  opacity: 0.95
                }}
              />
            );
          })}
          {sparseMarkers.map((marker) => (
            <CircleMarker
              key={`sparse-core-${marker.key}`}
              center={marker.centerLatLng}
              radius={4}
              pathOptions={{
                color: '#f0f6fc',
                weight: 1,
                fillColor: '#f0f6fc',
                fillOpacity: 1,
                opacity: 1
              }}
            />
          ))}
          {focusedPlace && (
            <HazardFocusOverlay
              place={focusedPlace}
              progress={focusProgress}
              animating={focusAnimating}
            />
          )}
        </MapContainer>
      </div>
      {simMode.id === 'dynamic' && show3DSim && (
  <div style={{
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    zIndex: 999,
    background: '#000'
  }}>
    <button
      onClick={() => setShow3DSim(false)}
      style={{
        position: 'absolute',
        top: 20,
        right: 20,
        zIndex: 1000,
        padding: '10px',
        background: 'red',
        color: 'white',
        border: 'none'
      }}
    >
      Close
    </button>

    <ThreeDValleySimulation rainfall={rainfallInput} regionId={region.id} />
  </div>
)}
    </div>
  );
}

export default App;
