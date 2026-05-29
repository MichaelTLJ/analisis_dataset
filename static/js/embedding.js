const data = window.SDSS_DATA;

const state = {
  visible: new Set(Object.keys(data.classColors)),
  selectedId: null,
  rotation: 0,
  raMin: null,
  raMax: null,
  decMin: null,
  decMax: null,
};

const plot = document.getElementById("embeddingPlot");
const tooltip = document.getElementById("tooltip");
const dataById = new Map();
let renderScheduled = false;

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/d";
  return Number(value).toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function extent(values) {
  const clean = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!clean.length) return [0, 1];
  const lo = clean[Math.floor((clean.length - 1) * 0.02)];
  const hi = clean[Math.floor((clean.length - 1) * 0.98)];
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}

function scale(value, domain, range) {
  return range[0] + ((value - domain[0]) / (domain[1] - domain[0])) * (range[1] - range[0]);
}

function labelFor(key) {
  return data.featureLabels[key] || key;
}

function normalizeCoordinate(value, min, max) {
  if (!Number.isFinite(value)) return null;
  return Math.min(max, Math.max(min, value));
}

function applyBounds(point) {
  if (state.raMin !== null && point.ra < state.raMin) return false;
  if (state.raMax !== null && point.ra > state.raMax) return false;
  if (state.decMin !== null && point.dec < state.decMin) return false;
  if (state.decMax !== null && point.dec > state.decMax) return false;
  return true;
}

function visiblePoints() {
  return data.points.filter((point) => state.visible.has(point.class) && applyBounds(point));
}

const redshiftDomain = extent(data.points.map((point) => Math.log1p(Math.max(point.redshift || 0, 0))));

function depthFor(point) {
  const logged = Math.log1p(Math.max(point.redshift || 0, 0));
  return clamp(scale(logged, redshiftDomain, [0, 1]), 0, 1);
}

function renderHeader() {
  document.getElementById("pointCount").textContent = data.projection.rows_used.toLocaleString();
  document.getElementById("projectionMethod").textContent =
    "Mapa de posiciones reales de photoObj usando RA/DEC y profundidad visual por redshift.";
}

function renderFilters() {
  const filters = document.getElementById("classFilters");
  filters.innerHTML = Object.keys(data.classColors)
    .map((className) => {
      const count = data.points.filter((point) => point.class === className).length;
      return `
        <div class="class-filter">
          <label>
            <input type="checkbox" checked data-class="${className}">
            <span class="swatch" style="background:${data.classColors[className]}"></span>
            ${className}
          </label>
          <span class="count">${count.toLocaleString()}</span>
        </div>
      `;
    })
    .join("");

  filters.querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) state.visible.add(input.dataset.class);
      else state.visible.delete(input.dataset.class);
      scheduleRender();
    });
  });
}

function projectCelestial(point, centerX, centerY, radius) {
  const ra = ((point.ra || 0) * Math.PI) / 180 + state.rotation;
  const dec = ((point.dec || 0) * Math.PI) / 180;
  const depth = depthFor(point);
  const distance = 0.48 + depth * 0.66;
  const x3 = distance * Math.cos(dec) * Math.cos(ra);
  const y3 = distance * Math.sin(dec);
  const z3 = distance * Math.cos(dec) * Math.sin(ra);
  const perspective = 1.18 / (1.74 - z3 * 0.72);

  return {
    x: centerX + x3 * radius * perspective,
    y: centerY - y3 * radius * perspective,
    z: z3,
    depth,
    perspective,
  };
}

function shapeValues(point) {
  const shape = point.shape || {};
  const ellipticity = clamp(shape.ellipticity || 0, 0, 0.86);
  const fracdev = clamp(shape.fracdev || 0, 0, 1);
  const angle = 0.5 * Math.atan2(shape.m_e2 || 0, shape.m_e1 || 0) * 180 / Math.PI;
  const petrotheta = clamp(shape.petrotheta || 1, 0.4, 8);
  return { ellipticity, fracdev, angle, petrotheta };
}

function galaxyGlyph(point, projected, color, selected) {
  const shape = shapeValues(point);
  const base = (4.6 + shape.petrotheta * 0.32 + shape.fracdev * 2.6) * projected.perspective;
  const rx = clamp(base, 4.5, 13);
  const ry = clamp(rx * (1 - shape.ellipticity * 0.72), 2.3, rx);
  const core = clamp(1.8 + shape.fracdev * 2.2, 1.8, 4.2);
  return `
    <g class="object-glyph${selected}" data-id="${point.id}" transform="translate(${projected.x} ${projected.y}) rotate(${shape.angle})">
      <ellipse class="glyph-outline" cx="0" cy="0" rx="${rx}" ry="${ry}" fill="${color}" opacity="${0.28 + projected.depth * 0.34}"></ellipse>
      <ellipse cx="0" cy="0" rx="${rx * 0.48}" ry="${Math.max(ry * 0.42, 1.6)}" fill="${color}" opacity="${0.36 + shape.fracdev * 0.28}"></ellipse>
      <circle cx="0" cy="0" r="${core}" fill="#f8fafc" opacity="0.8"></circle>
    </g>
  `;
}

function starGlyph(point, projected, color, selected) {
  const size = (4.2 + projected.depth * 2.5) * projected.perspective;
  return `
    <g class="object-glyph${selected}" data-id="${point.id}" transform="translate(${projected.x} ${projected.y})">
      <line class="glyph-outline" x1="${-size}" y1="0" x2="${size}" y2="0" stroke="${color}" stroke-width="1.4"></line>
      <line class="glyph-outline" x1="0" y1="${-size}" x2="0" y2="${size}" stroke="${color}" stroke-width="1.4"></line>
      <circle cx="0" cy="0" r="${Math.max(size * 0.45, 2.2)}" fill="${color}" opacity="${0.65 + projected.depth * 0.25}"></circle>
    </g>
  `;
}

function qsoGlyph(point, projected, color, selected) {
  const size = (4.4 + projected.depth * 3.4) * projected.perspective;
  return `
    <g class="object-glyph${selected}" data-id="${point.id}" transform="translate(${projected.x} ${projected.y})">
      <polygon class="glyph-outline" points="0,${-size} ${size},0 0,${size} ${-size},0" fill="${color}" opacity="${0.58 + projected.depth * 0.28}"></polygon>
      <circle cx="0" cy="0" r="${Math.max(size * 0.3, 2)}" fill="#fef08a" opacity="0.86"></circle>
    </g>
  `;
}

function objectGlyph(item) {
  const color = data.classColors[item.point.class];
  const selected = item.point.id === state.selectedId ? " selected" : "";
  if (item.point.class === "GALAXY") return galaxyGlyph(item.point, item.projected, color, selected);
  if (item.point.class === "QSO") return qsoGlyph(item.point, item.projected, color, selected);
  return starGlyph(item.point, item.projected, color, selected);
}

function sceneGrid(centerX, centerY, radius) {
  let markup = `
    <defs>
      <radialGradient id="spaceGlow" cx="35%" cy="25%" r="72%">
        <stop offset="0%" stop-color="rgba(125, 211, 252, 0.18)"></stop>
        <stop offset="60%" stop-color="rgba(15, 23, 42, 0.14)"></stop>
        <stop offset="100%" stop-color="rgba(2, 6, 23, 0.82)"></stop>
      </radialGradient>
    </defs>
  `;

  [-60, -30, 0, 30, 60].forEach((dec) => {
    const y = centerY - Math.sin((dec * Math.PI) / 180) * radius * 0.72;
    const rx = Math.cos((dec * Math.PI) / 180) * radius;
    markup += `<ellipse class="grid-line" cx="${centerX}" cy="${y}" rx="${rx}" ry="${rx * 0.22}"></ellipse>`;
  });

  for (let ra = 0; ra < 180; ra += 30) {
    const squash = Math.abs(Math.cos((ra * Math.PI) / 180));
    markup += `<ellipse class="grid-line" cx="${centerX}" cy="${centerY}" rx="${radius * squash}" ry="${radius * 0.72}"></ellipse>`;
  }

  markup += `<text class="axis-label" x="${centerX}" y="${centerY - radius - 28}" text-anchor="middle">Mapa 3D por ascensión recta y declinación; profundidad visual por redshift</text>`;
  return markup;
}

function renderPlot() {
  const rect = plot.getBoundingClientRect();
  const width = Math.max(rect.width, 460);
  const height = Math.max(rect.height, 500);
  const centerX = width / 2;
  const centerY = height / 2 + 8;
  const radius = Math.min(width, height) * 0.38;

  const items = visiblePoints()
    .filter((point) => Number.isFinite(point.ra) && Number.isFinite(point.dec))
    .map((point) => ({ point, projected: projectCelestial(point, centerX, centerY, radius) }))
    .sort((a, b) => a.projected.z - b.projected.z);

  plot.setAttribute("viewBox", `0 0 ${width} ${height}`);
  plot.innerHTML = sceneGrid(centerX, centerY, radius) + items.map(objectGlyph).join("");
}

function handlePlotEvent(event) {
  const target = event.target.closest(".object-glyph");
  if (!target) return;
  const point = dataById.get(Number(target.dataset.id));
  if (!point) return;
  if (event.type === "click") selectPoint(point.id);
  if (event.type === "mousemove") showTooltip(event, point);
}

function initDataIndex() {
  data.points.forEach((point) => {
    dataById.set(point.id, point);
  });
}

function searchNearestPoint(ra, dec) {
  if (!Number.isFinite(ra) || !Number.isFinite(dec)) return null;
  let nearest = null;
  let minDist = Infinity;
  data.points.forEach((point) => {
    if (!Number.isFinite(point.ra) || !Number.isFinite(point.dec)) return;
    const dra = Math.min(Math.abs(point.ra - ra), 360 - Math.abs(point.ra - ra));
    const ddec = point.dec - dec;
    const dist = Math.sqrt(dra * dra + ddec * ddec);
    if (dist < minDist) {
      minDist = dist;
      nearest = point;
    }
  });
  return nearest;
}

function showTooltip(event, point) {
  tooltip.innerHTML = `
    <strong>${point.class}</strong><br>
    RA: ${fmt(point.ra, 5)}<br>
    DEC: ${fmt(point.dec, 5)}<br>
    Redshift: ${fmt(point.redshift)}
  `;
  tooltip.style.opacity = "1";
  tooltip.style.left = `${event.clientX + 12}px`;
  tooltip.style.top = `${event.clientY + 12}px`;
}

function hideTooltip() {
  tooltip.style.opacity = "0";
}

function selectPoint(pointId) {
  state.selectedId = pointId;
  const point = dataById.get(pointId);
  if (!point) return;
  renderRecord(point);
  renderShape(point);
  renderVector("rawVector", point.raw);
  scheduleRender();
}

function renderRecord(point) {
  const recordPanel = document.getElementById("recordPanel");
  recordPanel.className = "";
  recordPanel.innerHTML = `
    <dl class="record-list">
      <div><dt>Clase</dt><dd>${point.class}</dd></div>
      <div><dt>Grupo</dt><dd>${point.redshiftGroup}</dd></div>
      <div><dt>OBJID</dt><dd>${point.objid}</dd></div>
      <div><dt>RA</dt><dd>${fmt(point.ra, 6)}</dd></div>
      <div><dt>DEC</dt><dd>${fmt(point.dec, 6)}</dd></div>
      <div><dt>Redshift</dt><dd>${fmt(point.redshift)}</dd></div>
      <div><dt>Error RA</dt><dd>${fmt(point.raerr, 8)}</dd></div>
      <div><dt>Error DEC</dt><dd>${fmt(point.decerr, 8)}</dd></div>
      <div><dt>Galáctica</dt><dd>L ${fmt(point.galactic_l, 4)} / B ${fmt(point.galactic_b, 4)}</dd></div>
    </dl>
  `;
}

function renderShape(point) {
  const shape = point.shape || {};
  const values = {
    "Tamaño aparente": shape.petrotheta,
    "Elipticidad": shape.ellipticity,
    "Forma horizontal/vertical": shape.m_e1,
    "Forma diagonal": shape.m_e2,
    "Perfil tipo galaxia elíptica": shape.fracdev,
  };
  renderNamedValues("shapePanel", values);
}

function renderVector(targetId, values) {
  const readable = {};
  Object.entries(values).forEach(([key, value]) => {
    readable[labelFor(key)] = value;
  });
  renderNamedValues(targetId, readable);
}

function renderNamedValues(targetId, values) {
  document.getElementById(targetId).innerHTML = Object.entries(values)
    .map(([key, value]) => `
      <div class="vector-row">
        <span>${key}</span>
        <span>${fmt(value)}</span>
      </div>
    `)
    .join("");
}

function resetSelection() {
  state.selectedId = null;
  document.getElementById("recordPanel").className = "record-empty";
  document.getElementById("recordPanel").textContent = "Selecciona una estrella, galaxia o quásar.";
  document.getElementById("shapePanel").innerHTML = "";
  document.getElementById("rawVector").innerHTML = "";
  scheduleRender();
}

function updateFilters() {
  const raMin = parseFloat(document.getElementById("raMin").value);
  const raMax = parseFloat(document.getElementById("raMax").value);
  const decMin = parseFloat(document.getElementById("decMin").value);
  const decMax = parseFloat(document.getElementById("decMax").value);
  state.raMin = Number.isFinite(raMin) ? normalizeCoordinate(raMin, 0, 360) : null;
  state.raMax = Number.isFinite(raMax) ? normalizeCoordinate(raMax, 0, 360) : null;
  state.decMin = Number.isFinite(decMin) ? normalizeCoordinate(decMin, -90, 90) : null;
  state.decMax = Number.isFinite(decMax) ? normalizeCoordinate(decMax, -90, 90) : null;
  scheduleRender();
}

function clearFilters() {
  state.raMin = state.raMax = state.decMin = state.decMax = null;
  document.getElementById("raMin").value = "";
  document.getElementById("raMax").value = "";
  document.getElementById("decMin").value = "";
  document.getElementById("decMax").value = "";
  scheduleRender();
}

function scheduleRender() {
  if (renderScheduled) return;
  renderScheduled = true;
  requestAnimationFrame(() => {
    renderScheduled = false;
    renderPlot();
  });
}

function initEvents() {
  plot.addEventListener("click", handlePlotEvent);
  plot.addEventListener("mousemove", handlePlotEvent);
  plot.addEventListener("mouseleave", hideTooltip);
  document.getElementById("resetZoom").addEventListener("click", resetSelection);
  document.getElementById("applyFilter").addEventListener("click", updateFilters);
  document.getElementById("clearFilter").addEventListener("click", clearFilters);
  window.addEventListener("resize", scheduleRender);
}

function animate() {
  state.rotation += 0.002;
  scheduleRender();
  requestAnimationFrame(animate);
}

initDataIndex();
renderHeader();
renderFilters();
initEvents();
renderPlot();
requestAnimationFrame(animate);
