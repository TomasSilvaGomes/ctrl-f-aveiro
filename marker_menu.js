const layer = L.layerGroup();
const dotRenderer = L.canvas({ padding: 0.35, tolerance: 4 });
const MIN_POINT_ZOOM = 13;
const ICON_ZOOM = 17;
const points = rows.map(([lat, lng, data]) => ({
    lat, lng, latlng: L.latLng(lat, lng),
    entries: data.map(([id, icon, label, color]) => ({ id, icon, label, color })),
})).sort((left, right) => left.lat - right.lat);
const visible = new Map();
let activePoint = null;
let menu = null;
let detailed = null;
let frame = 0;

function pointsIn(bounds) {
    const south = bounds.getSouth();
    const north = bounds.getNorth();
    const west = bounds.getWest();
    const east = bounds.getEast();
    let first = 0;
    let last = points.length;
    while (first < last) {
        const middle = (first + last) >>> 1;
        if (points[middle].lat < south) first = middle + 1;
        else last = middle;
    }
    const result = [];
    for (let i = first; i < points.length && points[i].lat <= north; i++) {
        const point = points[i];
        if (point.lng >= west && point.lng <= east) result.push(point);
    }
    return result;
}

function selectEntry(point, entry) {
    map.closePopup();
    visible.get(activePoint)?.getElement?.()?.classList.remove("catalog-selected");
    activePoint = point;
    visible.get(point)?.getElement?.()?.classList.add("catalog-selected");
    window.catalogSelectionRevision = (window.catalogSelectionRevision || 0) + 1;
    window.Streamlit.setComponentValue({
        catalog_selection: { id: entry.id, lat: point.lat, lng: point.lng, revision: window.catalogSelectionRevision }
    });
}

function choiceMenu(anchor, entries) {
    const pageSize = 8;
    let page = 0;
    const shell = document.createElement("section");
    shell.className = "catalog-choice";
    shell.setAttribute("aria-label", "Escolher elemento");
    L.DomEvent.disableClickPropagation(shell);
    L.DomEvent.disableScrollPropagation(shell);

    const heading = document.createElement("strong");
    heading.textContent = "Escolher elemento";
    shell.append(heading);
    const grid = document.createElement("div");
    grid.className = "catalog-choice-grid";
    shell.append(grid);
    const footer = document.createElement("div");
    footer.className = "catalog-choice-footer";
    shell.append(footer);

    function action(label, callback, disabled = false) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.disabled = disabled;
        button.addEventListener("click", callback);
        return button;
    }

    function showPage() {
        const start = page * pageSize;
        grid.replaceChildren();
        footer.replaceChildren();
        for (const [point, entry] of entries.slice(start, start + pageSize)) {
            const button = action("", () => selectEntry(point, entry));
            button.className = "catalog-choice-item";
            button.title = entry.label;
            const icon = document.createElement("span");
            icon.className = "catalog-choice-icon";
            icon.setAttribute("aria-hidden", "true");
            icon.innerHTML = icons[entry.icon];
            const text = document.createElement("span");
            text.textContent = entry.label;
            button.append(icon, text);
            grid.append(button);
        }
        if (entries.length > pageSize) {
            footer.append(
                action("Anterior", () => { page -= 1; showPage(); }, page === 0),
                action("Seguinte", () => { page += 1; showPage(); }, start + pageSize >= entries.length),
            );
        }
    }

    showPage();
    menu = L.popup({
        className: "catalog-choice-popup",
        maxWidth: 300,
        autoPanPaddingTopLeft: [75, 30],
        autoPanPaddingBottomRight: [18, 26],
    }).setLatLng(anchor).setContent(shell).openOn(map);
}

function clickPoint(point) {
    const center = map.latLngToContainerPoint(point.latlng);
    const radius = detailed ? 35 : 7;
    const shift = L.point(radius, radius);
    const hit = L.latLngBounds(
        map.containerPointToLatLng(center.subtract(shift)),
        map.containerPointToLatLng(center.add(shift)),
    );
    const hits = pointsIn(hit);
    hits.sort((a, b) => center.distanceTo(map.latLngToContainerPoint(a.latlng)) -
        center.distanceTo(map.latLngToContainerPoint(b.latlng)));
    const entries = hits.flatMap(candidate => candidate.entries.map(entry => [candidate, entry]));
    if (entries.length > 1) choiceMenu(point.latlng, entries);
    else selectEntry(point, point.entries[0]);
}

function pointLayer(point) {
    const first = point.entries[0];
    if (!detailed) {
        return L.circleMarker(point.latlng, {
            renderer: dotRenderer, radius: 2.7, color: "#ffffff", opacity: 0.8, weight: 0.75,
            fillColor: first.color, fillOpacity: 0.93, bubblingMouseEvents: false,
        });
    }
    const stack = point.entries.length > 1 ? " catalog-marker-stack" : "";
    const marker = L.marker(point.latlng, {
        title: stack ? "Escolher elemento" : first.label,
        icon: L.divIcon({
            className: "catalog-map-marker" + (point === activePoint ? " catalog-selected" : ""),
            html: '<div class="catalog-marker-glyph' + stack + '">' + icons[first.icon] + '</div>',
            iconSize: [34, 34],
            iconAnchor: [17, 17],
        }),
    });
    return marker;
}

function refresh() {
    if (!map.hasLayer(layer)) return;
    const zoom = map.getZoom();
    if (zoom < MIN_POINT_ZOOM) {
        layer.clearLayers();
        visible.clear();
        if (map.hasLayer(dotRenderer)) map.removeLayer(dotRenderer);
        if (menu && map.hasLayer(menu)) map.removeLayer(menu);
        return;
    }
    const nextMode = zoom >= ICON_ZOOM;
    if (nextMode !== detailed) {
        layer.clearLayers();
        visible.clear();
        detailed = nextMode;
    }
    const next = new Set(pointsIn(map.getBounds().pad(0.16)));
    for (const [point, marker] of visible) {
        if (!next.has(point)) {
            layer.removeLayer(marker);
            visible.delete(point);
        }
    }
    for (const point of next) {
        if (visible.has(point)) continue;
        const marker = pointLayer(point).on("click", () => clickPoint(point));
        visible.set(point, marker);
        layer.addLayer(marker);
    }
}

function scheduleRefresh() {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(refresh);
}

layer.on("add", () => {
    map.on("moveend zoomend resize", scheduleRefresh);
    scheduleRefresh();
});
layer.on("remove", () => {
    map.off("moveend zoomend resize", scheduleRefresh);
    cancelAnimationFrame(frame);
    layer.clearLayers();
    visible.clear();
    if (map.hasLayer(dotRenderer)) map.removeLayer(dotRenderer);
    if (menu && map.hasLayer(menu)) map.removeLayer(menu);
});
