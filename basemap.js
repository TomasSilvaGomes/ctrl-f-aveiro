(async () => {
    const attribution = '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>';
    L.control.attribution({ position: "bottomleft", prefix: false }).addTo(map)
        .addAttribution('<a href="https://www.geonames.org/export/">GeoNames</a>');
    const scale = L.control.scale({ position: "bottomleft", metric: true, imperial: false, maxWidth: 92 });
    const formatMeters = new Intl.NumberFormat("pt-PT");
    scale._updateMetric = function (maxMeters) {
        const meters = this._getRoundNum(maxMeters);
        this._updateScale(this._mScale, formatMeters.format(meters) + " m", meters / maxMeters);
    };
    scale.addTo(map);
    const fallback = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution, maxZoom: map.getMaxZoom(), maxNativeZoom: 19, className: "catalog-raster-tiles",
    });

    function loadScript(url) {
        return new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = url;
            script.onload = resolve;
            script.onerror = reject;
            document.head.append(script);
        });
    }

    const deadline = setTimeout(() => fallback.addTo(map), 2000);
    let vector;
    try {
        const css = document.createElement("link");
        css.rel = "stylesheet";
        css.href = "https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl.css";
        document.head.append(css);
        const [response] = await Promise.all([
            fetch("https://tiles.openfreemap.org/styles/positron"),
            loadScript("https://cdn.jsdelivr.net/npm/maplibre-gl@4.7.1/dist/maplibre-gl.js"),
        ]);
        if (!response.ok) throw new Error("Basemap unavailable");
        await loadScript("https://cdn.jsdelivr.net/npm/@maplibre/maplibre-gl-leaflet@0.0.22/leaflet-maplibre-gl.js");
        const style = await response.json();
        const fills = { water: "#abd9e7", landcover: "#e0ebd2", park: "#dcebcf", building: "#e8dfd3" };
        style.layers = style.layers.filter(layer => (
            layer.type !== "symbol" || layer.id.startsWith("label_") ||
            ["highway-name-major", "highway-name-minor"].includes(layer.id)
        )).map(layer => {
            if (layer.id.startsWith("highway-name")) return { ...layer, minzoom: 16 };
            if (layer.type === "background") layer.paint["background-color"] = "#f8f5ed";
            const fill = fills[layer["source-layer"]];
            if (fill && layer.type === "fill") layer.paint["fill-color"] = fill;
            return layer;
        });
        vector = L.maplibreGL({
            style, attributionControl: false,
            attribution: attribution + ' · <a href="https://openfreemap.org/">OpenFreeMap</a> © OpenMapTiles',
        }).addTo(map);
        vector.getMaplibreMap().once("idle", () => {
            clearTimeout(deadline);
            fallback.remove();
        });
    } catch (error) {
        clearTimeout(deadline);
        if (vector) vector.remove();
        fallback.addTo(map);
    }
})();
