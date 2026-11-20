const layer = L.layerGroup();
const MIN_POINT_ZOOM = 13;
const ICON_ZOOM = 17;
const OVERVIEW_SPACING = 34;
const visible = new Map();
let points = [];
let pointIndex = new Map();
let searchMarker = null;
let searchRevision;
let activePoint = null;
let menu = null;
let detailed = null;
let frame = 0;

function updateCatalog(nextRows, nextIcons, nextPin) {
    const nextIndex = new Map();
    const previousIcons = icons;
    icons = nextIcons;
    let changed = nextRows.length !== points.length;
    for (const [lat, lng, data] of nextRows) {
        const key = lat + ":" + lng;
        const previous = pointIndex.get(key);
        const unchanged = previous && previous.entries.length === data.length &&
            data.every(([id, icon, label], index) => {
                const entry = previous.entries[index];
                return entry.id === id && entry.icon === icon && entry.label === label &&
                    previousIcons[icon] === nextIcons[icon];
            });
        if (unchanged) {
            nextIndex.set(key, previous);
        } else {
            changed = true;
            const point = previous || {lat, lng, latlng: L.latLng(lat, lng)};
            const priorEntries = point.entries;
            point.entries = data.map(([id, icon, label]) => ({id, icon, label}));
            nextIndex.set(key, point);
            const marker = visible.get(point);
            if (marker && priorEntries) {
                const before = priorEntries[0];
                const after = point.entries[0];
                if (before.label !== after.label ||
                    previousIcons[before.icon] !== nextIcons[after.icon] ||
                    (priorEntries.length > 1) !== (point.entries.length > 1)) {
                    marker.setIcon(pointIcon(point));
                    marker.setTooltipContent(pointHint(point));
                    const element = marker.getElement();
                    if (element) element.setAttribute("aria-label", after.label);
                }
            }
        }
    }
    if (changed) {
        pointIndex = nextIndex;
        points = [...pointIndex.values()].sort((left, right) => left.lat - right.lat);
        if (activePoint && pointIndex.get(activePoint.lat + ":" + activePoint.lng) !== activePoint) {
            activePoint = null;
        }
        if (menu && map.hasLayer(menu)) map.removeLayer(menu);
        scheduleRefresh();
    }
    updateSearchPin(nextPin);
}

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
    window.parent.postMessage({type: "catalog:close-filters"}, "*");
    map.closePopup();
    visible.get(activePoint)?.getElement?.()?.classList.remove("catalog-selected");
    activePoint = point;
    visible.get(point)?.getElement?.()?.classList.add("catalog-selected");
    scheduleRefresh();
    window.catalogSelectionRevision = (window.catalogSelectionRevision || 0) + 1;
    window.Streamlit.setComponentValue({
        catalog_selection: { id: entry.id, lat: point.lat, lng: point.lng, revision: window.catalogSelectionRevision }
    });
}

function dismissMenus() {
    window.parent.postMessage({type: "catalog:close-filters"}, "*");
    map.closePopup();
    if (activePoint) {
        visible.get(activePoint)?.getElement?.()?.classList.remove("catalog-selected");
        activePoint = null;
        scheduleRefresh();
    }
    window.catalogSelectionRevision = (window.catalogSelectionRevision || 0) + 1;
    window.Streamlit.setComponentValue({catalog_dismiss: window.catalogSelectionRevision});
}

function choiceMenu(anchor, entries) {
    const pageSize = 8;
    let page = 0;
    const shell = document.createElement("section");
    shell.className = "catalog-choice";
    shell.setAttribute("aria-label", "Escolher elemento na área");
    L.DomEvent.disableClickPropagation(shell);
    L.DomEvent.disableScrollPropagation(shell);

    const heading = document.createElement("strong");
    heading.textContent = "Escolher elemento na área";
    shell.append(heading);
    const grid = document.createElement("div");
    grid.className = "catalog-choice-grid";
    shell.append(grid);
    const footer = document.createElement("div");
    footer.className = "catalog-choice-footer";
    shell.append(footer);

    function action(label, callback) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.addEventListener("click", event => {
            L.DomEvent.stopPropagation(event);
            callback();
        });
        return button;
    }

    const previous = action("Anterior", () => { page -= 1; showPage(); });
    const next = action("Seguinte", () => { page += 1; showPage(); });
    if (entries.length > pageSize) footer.append(previous, next);

    function showPage() {
        const start = page * pageSize;
        grid.replaceChildren();
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
        previous.disabled = page === 0;
        next.disabled = start + pageSize >= entries.length;
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
    const shift = L.point(OVERVIEW_SPACING, OVERVIEW_SPACING);
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

function pointHint(point) {
    const hint = document.createElement("span");
    hint.textContent = point.entries[0].label;
    return hint;
}

function pointIcon(point) {
    const first = point.entries[0];
    const size = detailed ? 34 : 26;
    const stack = point.entries.length > 1 ? " catalog-marker-stack" : "";
    const overview = detailed ? "" : " catalog-marker-overview";
    return L.divIcon({
        className: "catalog-map-marker" + overview + (point === activePoint ? " catalog-selected" : ""),
        html: '<div class="catalog-marker-glyph' + stack + '">' + icons[first.icon] + '</div>',
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
    });
}

function pointLayer(point) {
    const size = detailed ? 34 : 26;
    return L.marker(point.latlng, {
        alt: point.entries[0].label,
        icon: pointIcon(point),
    }).bindTooltip(pointHint(point), { direction: "top", offset: [0, -size / 2 - 2], opacity: 1, className: "catalog-tooltip" });
}

function spreadIcons(candidates, zoom) {
    const cells = new Map();
    for (const point of candidates) {
        const pixels = map.project(point.latlng, zoom);
        const cell = Math.floor(pixels.x / OVERVIEW_SPACING) + ":" + Math.floor(pixels.y / OVERVIEW_SPACING);
        if (point === activePoint || !cells.has(cell)) cells.set(cell, point);
    }
    return cells.values();
}

function refresh() {
    if (!map.hasLayer(layer)) return;
    const zoom = map.getZoom();
    if (zoom < MIN_POINT_ZOOM) {
        layer.clearLayers();
        visible.clear();
        if (menu && map.hasLayer(menu)) map.removeLayer(menu);
        return;
    }
    const nextMode = zoom >= ICON_ZOOM;
    if (nextMode !== detailed) {
        layer.clearLayers();
        visible.clear();
        detailed = nextMode;
    }
    const candidates = pointsIn(map.getBounds().pad(0.16));
    const next = new Set(detailed ? candidates : spreadIcons(candidates, zoom));
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

function updateSearchPin(pin) {
    if (pin?.revision === searchRevision) return;
    searchRevision = pin?.revision;
    map.off("dragstart", hideSearchPin);
    map.off("dragend", reportSearchPinDismissal);
    searchMarker?.remove();
    searchMarker = null;
    if (!pin) return;
    searchMarker = L.marker(pin.point, {
        interactive: false,
        keyboard: false,
        icon: L.divIcon({
            className: "catalog-search-pin",
            iconSize: [32, 36],
            iconAnchor: [16, 31],
            html: '<span aria-hidden="true"></span>',
        }),
    }).addTo(map);
    map.once("dragstart", hideSearchPin);
}

function hideSearchPin() {
    searchMarker.remove();
    map.once("dragend", reportSearchPinDismissal);
}

function reportSearchPinDismissal() {
    window.Streamlit.setComponentValue({catalog_search_pin_dismissed: searchRevision});
}

layer.on("add", () => {
    map.on("moveend zoomend resize", scheduleRefresh);
    map.on("click", dismissMenus);
    scheduleRefresh();
});
layer.on("remove", () => {
    map.off("moveend zoomend resize", scheduleRefresh);
    map.off("click", dismissMenus);
    updateSearchPin(null);
    cancelAnimationFrame(frame);
    layer.clearLayers();
    visible.clear();
    if (menu && map.hasLayer(menu)) map.removeLayer(menu);
});
