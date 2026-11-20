(() => {
    if (window.catalogMenuMotionInstalled) return;
    window.catalogMenuMotionInstalled = true;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const legendSurface = ".st-key-hud_legend details[open] > div";
    const signalMenu = ".st-key-hud_signals";
    const surfaces = [
        ".st-key-inspector",
        "[data-testid='stPopoverBody']",
        "[data-testid='stDialog'] [role='dialog']",
        legendSurface,
        signalMenu,
    ];
    const selector = surfaces.join(",");
    const filterToggle = ".catalog-filter-toggle";
    const filterPanel = ".st-key-hud_filters";
    const familyTilePrefix = "st-key-subtype_tile_sinalizacao-";
    const familyPanelPrefix = "st-key-signal_family_";
    const visible = new Map();
    const exiting = new Map();
    let frame = 0;
    let signalFamily;

    function closeFilters() {
        signalFamily = undefined;
        const toggle = document.querySelector(`${filterToggle}[open]`);
        if (toggle) toggle.open = false;
    }

    document.addEventListener("click", event => {
        const tile = event.target.closest?.(`[class*="${familyTilePrefix}"]`);
        if (tile) {
            const family = [...tile.classList].find(name => name.startsWith(familyTilePrefix))?.slice(familyTilePrefix.length);
            const menu = document.querySelector(`${signalMenu}:not(.catalog-menu-exit *)`);
            const alreadySelected = family && menu?.querySelector(`.${familyPanelPrefix}${CSS.escape(family)}`);
            if (alreadySelected && signalFamily !== family) {
                event.preventDefault();
                event.stopPropagation();
            }
            signalFamily = family;
            schedule();
        } else if (event.target.closest?.(`${filterPanel} [class*="st-key-subtype_tile_"]`)) {
            signalFamily = undefined;
            schedule();
        }
        if (!event.target.closest?.(`${filterToggle}, ${filterPanel}, ${signalMenu}`)) {
            closeFilters();
        }
    }, true);

    document.addEventListener("keydown", event => {
        if (event.key !== "Escape") return;
        closeFilters();
        if (event.target.closest?.(`${filterPanel}, ${signalMenu}`)) {
            document.querySelector(`${filterToggle} > summary`)?.focus();
        }
    });

    function positionSignalMenu() {
        const menu = document.querySelector(`${signalMenu}:not(.catalog-menu-exit *)`);
        const panel = document.querySelector(filterPanel);
        const contents = menu?.querySelector(".st-key-hud_signal_contents");
        if (!document.querySelector(`${filterToggle}[open]`)) signalFamily = undefined;
        if (!contents) return;
        const family = signalFamily && CSS.escape(signalFamily);
        const selected = family && contents.querySelector(`.${familyPanelPrefix}${family}`);
        if (selected) {
            for (const group of contents.querySelectorAll(`[class*="${familyPanelPrefix}"]`)) {
                group.toggleAttribute("data-signal-family-hidden", group !== selected);
            }
        }
        if (selected && menu.dataset.activeSignalFamily && menu.dataset.activeSignalFamily !== signalFamily) {
            menu.removeAttribute("data-signal-open");
            // Resolve the closed style before reopening on the newly selected tile.
            getComputedStyle(menu).display;
        }
        menu.dataset.activeSignalFamily = selected ? signalFamily : "";
        menu.toggleAttribute("data-signal-open", Boolean(selected));
        if (!selected || !panel?.offsetWidth || !menu.offsetWidth) return;

        const tile = panel.querySelector(`.${familyTilePrefix}${family}`);
        const panelRect = panel.getBoundingClientRect();
        const tileRect = tile?.getBoundingClientRect() || panelRect;
        const edge = 12, gap = 10;
        const width = menu.offsetWidth;
        const maxLeft = window.innerWidth - width - edge;
        const clamp = (number, min, max) => Math.min(Math.max(number, min), Math.max(min, max));
        const anchorX = (tileRect.left + tileRect.right) / 2;
        const anchorY = clamp((tileRect.top + tileRect.bottom) / 2, panelRect.top, panelRect.bottom);
        const contentHeight = Math.min(contents.scrollHeight + menu.offsetHeight - contents.offsetHeight, 352);
        const fitsRight = panelRect.right + gap <= maxLeft;
        const fitsLeft = panelRect.left - gap - width >= edge;
        let side, left, top, height, anchor;
        if (fitsRight || fitsLeft) {
            side = fitsLeft && (!fitsRight || anchorX < (panelRect.left + panelRect.right) / 2) ? "left" : "right";
            left = side === "right" ? panelRect.right + gap : panelRect.left - gap - width;
            height = Math.min(contentHeight, window.innerHeight - 2 * edge);
            top = clamp(anchorY - 22, edge, window.innerHeight - height - edge);
            anchor = clamp(anchorY - top, 17, height - 17);
        } else {
            const below = window.innerHeight - panelRect.bottom - gap - edge;
            const above = panelRect.top - gap - edge;
            side = below >= Math.min(140, contentHeight) || below >= above ? "below" : "above";
            height = Math.max(0, Math.min(contentHeight, side === "below" ? below : above));
            left = clamp(anchorX - width / 2, edge, maxLeft);
            top = side === "below" ? panelRect.bottom + gap : panelRect.top - gap - height;
            anchor = clamp(anchorX - left, 12, width - 12);
        }
        menu.dataset.side = side;
        menu.style.setProperty("--signal-left", `${left}px`);
        menu.style.setProperty("--signal-top", `${top}px`);
        menu.style.setProperty("--signal-anchor", `${anchor}px`);
        menu.style.setProperty("--signal-max-height", `${height}px`);
    }

    window.addEventListener("message", event => {
        if (event.data?.type !== "catalog:close-filters") return;
        if (event.source === document.querySelector(".st-key-map_surface iframe")?.contentWindow) {
            closeFilters();
        }
    });

    function measure(node) {
        const rect = node.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const style = getComputedStyle(node);
        if (style.visibility === "hidden") return;
        const legend = node.matches(legendSurface);
        return {
            rect,
            kind: surfaces.find(surface => node.matches(surface)),
            x: style.getPropertyValue("--menu-enter-x").trim() || "0px",
            y: style.getPropertyValue("--menu-enter-y").trim() || "10px",
            scrollTop: node.scrollTop,
            legend,
        };
    }

    function conceal(record, node) {
        const shell = document.createElement("div");
        shell.className = "stApp catalog-menu-exit";
        if (record.legend) shell.classList.add("st-key-hud_legend");
        shell.inert = true;
        shell.setAttribute("aria-hidden", "true");
        const copy = node.cloneNode(true);
        copy.querySelectorAll("[id]").forEach(element => element.removeAttribute("id"));
        copy.removeAttribute("id");
        copy.querySelectorAll("iframe, video, canvas").forEach(element => element.remove());
        shell.append(copy);
        const {rect} = record;
        const fixed = {
            position: "fixed", inset: "auto", left: rect.left + "px", top: rect.top + "px",
            width: rect.width + "px", height: rect.height + "px", "max-width": "none", "max-height": "none",
            "box-sizing": "border-box", margin: "0", transform: "none", translate: "none", animation: "none",
        };
        for (const [key, value] of Object.entries(fixed)) copy.style.setProperty(key, value, "important");
        document.body.append(shell);
        copy.scrollTop = record.scrollTop;
        const reduced = reducedMotion.matches;
        const keyframes = reduced
            ? [{opacity: 1}, {opacity: 0}]
            : [{opacity: 1, transform: "translate(0, 0)"},
                {opacity: 0, transform: `translate(${record.x}, ${record.y})`}];
        const animation = shell.animate(keyframes, {duration: reduced ? 130 : 180, easing: "ease-in"});
        exiting.set(record.kind, animation);
        const cleanup = () => {
            shell.remove();
            if (exiting.get(record.kind) === animation) exiting.delete(record.kind);
        };
        animation.onfinish = cleanup;
        animation.oncancel = cleanup;
    }

    function refresh() {
        frame = 0;
        positionSignalMenu();
        const next = new Map();
        const kinds = new Set();
        for (const node of document.querySelectorAll(selector)) {
            if (node.closest(".catalog-menu-exit")) continue;
            const record = measure(node);
            if (!record) continue;
            next.set(node, record);
            kinds.add(record.kind);
        }
        for (const [kind, animation] of exiting) {
            if (kinds.has(kind)) animation.cancel();
        }
        for (const [node, record] of visible) {
            if (next.has(node)) continue;
            resize.unobserve(node);
            if (!kinds.has(record.kind)) conceal(record, node);
        }
        for (const node of next.keys()) {
            if (!visible.has(node)) resize.observe(node);
        }
        visible.clear();
        for (const [node, record] of next) visible.set(node, record);
    }

    function schedule() {
        if (!frame) frame = requestAnimationFrame(refresh);
    }

    const resize = new ResizeObserver(schedule);
    const mutations = new MutationObserver(records => {
        const hasSurfaceChange = records.some(({target, addedNodes, removedNodes, type}) => {
            if (target.closest?.(".catalog-menu-exit")) return false;
            if (type === "attributes") {
                return target.closest(".st-key-hud_legend") || target.matches(`${selector}, ${filterToggle}`) ||
                    visible.has(target);
            }
            if (target.closest?.(signalMenu)) return true;
            return [...addedNodes, ...removedNodes].some(node =>
                node instanceof Element && !node.matches(".catalog-menu-exit") &&
                (node.matches(selector) || node.querySelector(selector))
            );
        });
        if (hasSurfaceChange) {
            cancelAnimationFrame(frame);
            refresh();
        }
    });
    mutations.observe(document.body, {
        subtree: true, childList: true, attributes: true, attributeFilter: ["open", "hidden"],
    });
    window.addEventListener("resize", schedule, {passive: true});
    document.addEventListener("scroll", event => {
        const record = visible.get(event.target);
        if (record) record.scrollTop = event.target.scrollTop;
        if (event.target.closest?.(filterPanel)) schedule();
    }, {capture: true, passive: true});
    reducedMotion.addEventListener("change", () => {
        for (const animation of exiting.values()) animation.cancel();
        schedule();
    });
    schedule();
})();
