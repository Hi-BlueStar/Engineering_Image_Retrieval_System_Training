// Global State
let datasetState = {
    seeds: [],
    gt_selections: {},
    distractors: [],
    categories: [],
    stats: {}
};

let currentSeedIndex = 0;
let currentSearchQuery = "";
let csrfToken = "";

// Zoom Modal State
let zoomScale = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let startX = 0;
let startY = 0;
let zoomImageList = [];
let zoomImageIndex = -1;

// DOM Elements
const statV = document.getElementById("stat-v");
const statTSmall = document.getElementById("stat-t-small");
const statTLarge = document.getElementById("stat-t-large");
const seedListContainer = document.getElementById("seed-list-container");
const querySeedImg = document.getElementById("query-seed-img");
const querySeedName = document.getElementById("query-seed-name");
const querySeedClass = document.getElementById("query-seed-class");
const gtsGridContainer = document.getElementById("gts-grid-container");
const gtCount = document.getElementById("gt-count");
const searchInput = document.getElementById("search-input");
const candidatesGridContainer = document.getElementById("candidates-grid-container");
const saveBuildBtn = document.getElementById("save-build-btn");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

// Zoom Modal DOM Elements
const imageModal = document.getElementById("image-modal");
const modalClose = document.getElementById("modal-close");
const modalImg = document.getElementById("modal-img");
const modalContentContainer = document.getElementById("modal-content-container");
const modalSeedImg = document.getElementById("modal-seed-img");
const modalSeedContainer = document.getElementById("modal-seed-container");
const zoomInBtn = document.getElementById("zoom-in-btn");
const zoomOutBtn = document.getElementById("zoom-out-btn");
const zoomResetBtn = document.getElementById("zoom-reset-btn");
const zoomPrevBtn = document.getElementById("zoom-prev-btn");
const zoomNextBtn = document.getElementById("zoom-next-btn");
const zoomGtBtn = document.getElementById("zoom-gt-btn");

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    // Get CSRF token from meta tag
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta) {
        csrfToken = meta.getAttribute("content");
    }
    
    setupEventListeners();
    loadDataset();
});

// Event Listeners Setup
function setupEventListeners() {
    searchInput.addEventListener("input", (e) => {
        currentSearchQuery = e.target.value.toLowerCase().trim();
        filterCandidates();
    });

    saveBuildBtn.addEventListener("click", () => {
        saveAndRebuild();
    });

    // Query seed preview click to zoom
    querySeedImg.addEventListener("click", () => {
        if (querySeedImg.src) {
            const seedPath = datasetState.seeds[currentSeedIndex];
            openZoomModal(seedPath, [seedPath]);
        }
    });

    // Zoom modal events
    modalClose.addEventListener("click", closeZoomModal);
    
    // Close when clicking outside the image
    imageModal.addEventListener("click", (e) => {
        if (e.target === imageModal || e.target === modalContentContainer || e.target === modalSeedContainer) {
            closeZoomModal();
        }
    });

    // Zoom controls buttons
    zoomInBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        zoomScale = Math.min(zoomScale + 0.25, 8.0);
        updateModalImageTransform();
    });
    
    zoomOutBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        zoomScale = Math.max(zoomScale - 0.25, 0.25);
        updateModalImageTransform();
    });
    
    zoomResetBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        zoomScale = 1.0;
        panX = 0;
        panY = 0;
        updateModalImageTransform();
    });

    // Next/Prev navigation buttons
    zoomPrevBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (zoomImageIndex > 0) {
            zoomImageIndex--;
            displayZoomImage();
        }
    });

    zoomNextBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (zoomImageIndex < zoomImageList.length - 1) {
            zoomImageIndex++;
            displayZoomImage();
        }
    });

    // GT Selection Toggle Button inside Zoom Modal
    zoomGtBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (zoomImageIndex < 0 || zoomImageIndex >= zoomImageList.length) return;
        
        const canonicalPath = zoomImageList[zoomImageIndex];
        const seedPath = datasetState.seeds[currentSeedIndex];
        const gts = datasetState.gt_selections[seedPath] || [];
        const isChecked = gts.includes(canonicalPath);
        
        // Toggle selection
        toggleGTSelection(canonicalPath, !isChecked);
        
        // Update button visual state
        updateZoomModalGTButton();
    });

    // Mouse wheel zoom on both containers
    const containers = [modalContentContainer, modalSeedContainer];
    containers.forEach(container => {
        if (!container) return;
        container.addEventListener("wheel", (e) => {
            e.preventDefault();
            const zoomIntensity = 0.1;
            const delta = e.deltaY < 0 ? 1 : -1;
            zoomScale = Math.max(0.25, Math.min(8.0, zoomScale + delta * zoomIntensity));
            updateModalImageTransform();
        }, { passive: false });

        // Drag to pan (mousedown on either container)
        container.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return; // Only left mouse button
            isDragging = true;
            startX = e.clientX - panX;
            startY = e.clientY - panY;
            containers.forEach(c => { if (c) c.style.cursor = "grabbing"; });
        });
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        panX = e.clientX - startX;
        panY = e.clientY - startY;
        updateModalImageTransform();
    });

    window.addEventListener("mouseup", () => {
        if (isDragging) {
            isDragging = false;
            containers.forEach(c => { if (c) c.style.cursor = "grab"; });
        }
    });

    // Keyboard navigation (ArrowLeft, ArrowRight, Escape)
    window.addEventListener("keydown", (e) => {
        if (!imageModal.classList.contains("show")) return;
        
        if (e.key === "ArrowLeft") {
            if (zoomImageIndex > 0) {
                zoomImageIndex--;
                displayZoomImage();
            }
        } else if (e.key === "ArrowRight") {
            if (zoomImageIndex < zoomImageList.length - 1) {
                zoomImageIndex++;
                displayZoomImage();
            }
        } else if (e.key === "Escape") {
            closeZoomModal();
        }
    });
}

// Log status utility
function setStatus(text, state = "online") {
    statusText.textContent = text;
    statusDot.className = "dot";
    
    if (state === "online") {
        statusDot.classList.add("online");
    } else if (state === "working") {
        statusDot.classList.add("working");
    } else if (state === "error") {
        statusDot.classList.add("error");
    }
}

// Extract filename from a path string
function getFilename(pathStr) {
    if (!pathStr) return "";
    return pathStr.substring(pathStr.lastIndexOf('/') + 1);
}

// Extract category directory name from path string
function getCategoryFromPath(pathStr) {
    if (!pathStr) return "";
    const parts = pathStr.split('/');
    if (parts.length > 2) {
        return parts[parts.length - 2];
    }
    return "";
}

// API Call: Load the initial dataset split JSON
async function loadDataset() {
    setStatus("正在載入資料集結構...", "working");
    try {
        const response = await fetch("/api/dataset");
        if (!response.ok) {
            throw new Error(`HTTP 錯誤: ${response.status}`);
        }
        const data = await response.json();
        
        datasetState.seeds = data.seeds || [];
        datasetState.gt_selections = data.gt_selections || {};
        datasetState.distractors = data.distractors || [];
        datasetState.categories = data.categories || [];
        datasetState.stats = data.stats || {};
        
        // Update stats UI
        updateStatsUI();
        
        // Populate sidebar seeds
        renderSeedList();
        
        // Select first seed by default
        if (datasetState.seeds.length > 0) {
            selectSeed(0);
        }
        
        setStatus("系統就緒 (已連線)", "online");
    } catch (err) {
        loggerError(err);
        setStatus(`載入資料集失敗: ${err.message}`, "error");
    }
}

function updateStatsUI() {
    statV.textContent = datasetState.stats.total_v || "0";
    statTSmall.textContent = datasetState.stats.total_t_small || "0";
    statTLarge.textContent = datasetState.stats.total_t_large || "0";
}

// Render Seeds in Sidebar
function renderSeedList() {
    seedListContainer.replaceChildren();
    
    datasetState.seeds.forEach((seedPath, idx) => {
        const seedName = getFilename(seedPath);
        const seedCat = getCategoryFromPath(seedPath);
        const numGts = (datasetState.gt_selections[seedPath] || []).length;
        
        const seedItem = document.createElement("div");
        seedItem.className = `seed-item ${idx === currentSeedIndex ? 'active' : ''}`;
        seedItem.dataset.index = idx;
        
        // Thumbnail image
        const img = document.createElement("img");
        img.className = "seed-item-thumb";
        img.src = `/api/image?path=${encodeURIComponent(seedPath)}`;
        img.alt = "Thumb";
        
        const infoDiv = document.createElement("div");
        infoDiv.className = "seed-item-info";
        
        const nameP = document.createElement("p");
        nameP.className = "seed-item-name";
        nameP.textContent = `${idx + 1}. ${seedName}`;
        
        const metaDiv = document.createElement("div");
        metaDiv.className = "seed-item-meta";
        
        const catSpan = document.createElement("span");
        catSpan.className = "seed-item-class";
        catSpan.textContent = seedCat;
        
        const badgeSpan = document.createElement("span");
        badgeSpan.className = "seed-item-badge";
        badgeSpan.textContent = `${numGts} GTs`;
        
        metaDiv.appendChild(catSpan);
        metaDiv.appendChild(badgeSpan);
        
        infoDiv.appendChild(nameP);
        infoDiv.appendChild(metaDiv);
        
        seedItem.appendChild(img);
        seedItem.appendChild(infoDiv);
        
        seedItem.addEventListener("click", () => {
            selectSeed(idx);
        });
        
        seedListContainer.appendChild(seedItem);
    });
}

// Select a specific Seed to Edit
function selectSeed(index) {
    currentSeedIndex = index;
    
    // Highlight sidebar item
    const items = seedListContainer.querySelectorAll(".seed-item");
    items.forEach((item, idx) => {
        if (idx === index) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });
    
    const seedPath = datasetState.seeds[currentSeedIndex];
    const seedName = getFilename(seedPath);
    const seedCat = getCategoryFromPath(seedPath);
    
    // Update main seed view
    querySeedImg.src = `/api/image?path=${encodeURIComponent(seedPath)}`;
    querySeedName.textContent = seedName;
    querySeedClass.textContent = seedCat;
    
    // Reset search
    searchInput.value = "";
    currentSearchQuery = "";
    
    // Load lists
    updateGTGrid();
    loadCandidatesForSeed(seedPath);
}

// Update the grid showing selected ground truths
function updateGTGrid() {
    gtsGridContainer.replaceChildren();
    
    const seedPath = datasetState.seeds[currentSeedIndex];
    const gts = datasetState.gt_selections[seedPath] || [];
    
    gtCount.textContent = gts.length;
    
    // Update count in sidebar badge
    const activeSidebarItem = seedListContainer.querySelector(".seed-item.active");
    if (activeSidebarItem) {
        const badge = activeSidebarItem.querySelector(".seed-item-badge");
        if (badge) {
            badge.textContent = `${gts.length} GTs`;
        }
    }
    
    if (gts.length === 0) {
        const emptyMsg = document.createElement("p");
        emptyMsg.className = "text-muted";
        emptyMsg.textContent = "尚未勾選任何 Ground Truth 影像。";
        emptyMsg.style.gridColumn = "1 / -1";
        emptyMsg.style.padding = "20px";
        emptyMsg.style.textAlign = "center";
        gtsGridContainer.appendChild(emptyMsg);
        return;
    }
    
    gts.forEach(gtPath => {
        const gtName = getFilename(gtPath);
        const gtCat = getCategoryFromPath(gtPath);
        
        const card = document.createElement("div");
        card.className = "img-card";
        
        const thumbContainer = document.createElement("div");
        thumbContainer.className = "card-thumb-container";
        
        const img = document.createElement("img");
        img.src = `/api/image?path=${encodeURIComponent(gtPath)}`;
        img.alt = gtName;
        
        thumbContainer.appendChild(img);
        
        const nameP = document.createElement("p");
        nameP.className = "card-name";
        nameP.textContent = gtName;
        
        const classP = document.createElement("p");
        classP.className = "card-class";
        classP.textContent = gtCat;
        
        // Remove button overlay
        const removeBtn = document.createElement("button");
        removeBtn.className = "remove-overlay";
        removeBtn.textContent = "✕";
        removeBtn.title = "移除此 Ground Truth";
        removeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            toggleGTSelection(gtPath, false);
        });
        
        card.appendChild(thumbContainer);
        card.appendChild(nameP);
        card.appendChild(classP);
        card.appendChild(removeBtn);
        
        // Click card to open zoom modal
        card.addEventListener("click", () => {
            const seedPath = datasetState.seeds[currentSeedIndex];
            const gts = datasetState.gt_selections[seedPath] || [];
            openZoomModal(gtPath, gts);
        });
        
        gtsGridContainer.appendChild(card);
    });
}

// Fetch images for candidate selection pool using semantic similarity
let currentCandidates = []; // stores { name, canonical_path } of the Top 50 similar images

async function loadCandidatesForSeed(seedPath) {
    candidatesGridContainer.replaceChildren();
    const loadingMsg = document.createElement("p");
    loadingMsg.className = "text-muted";
    loadingMsg.textContent = "正在利用特徵模型檢索 Top 50 最相似影像...";
    candidatesGridContainer.appendChild(loadingMsg);
    
    try {
        const response = await fetch(`/api/candidates?path=${encodeURIComponent(seedPath)}`);
        if (!response.ok) {
            throw new Error(`HTTP 錯誤: ${response.status}`);
        }
        const data = await response.json();
        
        currentCandidates = data.images || [];
        
        // Render
        filterCandidates();
    } catch (err) {
        loggerError(err);
        candidatesGridContainer.replaceChildren();
        const errMsg = document.createElement("p");
        errMsg.className = "text-danger";
        errMsg.textContent = `無法載入相似候選集: ${err.message}`;
        candidatesGridContainer.appendChild(errMsg);
    }
}

// Filters & renders candidates based on search query and current active seed
function filterCandidates() {
    candidatesGridContainer.replaceChildren();
    
    const seedPath = datasetState.seeds[currentSeedIndex];
    const seedCat = getCategoryFromPath(seedPath);
    const gts = datasetState.gt_selections[seedPath] || [];
    
    // Filter out the seed itself, and filter by text search query
    const filtered = currentCandidates.filter(item => {
        const isSeed = item.canonical_path === seedPath;
        if (isSeed) return false;
        
        if (currentSearchQuery) {
            return item.name.toLowerCase().includes(currentSearchQuery);
        }
        return true;
    });
    
    if (filtered.length === 0) {
        const noResults = document.createElement("p");
        noResults.className = "text-muted";
        noResults.textContent = "無符合條件的候選影像。";
        noResults.style.gridColumn = "1 / -1";
        noResults.style.padding = "20px";
        noResults.style.textAlign = "center";
        candidatesGridContainer.appendChild(noResults);
        return;
    }
    
    // Identify if any candidate should be auto-selected (pre-checked) because it is same class or already a GT
    let stateChanged = false;
    filtered.forEach(item => {
        const itemCat = getCategoryFromPath(item.canonical_path);
        const isSameClass = seedCat && itemCat && seedCat !== "converted_images" && seedCat === itemCat;
        const isAlreadyGT = gts.includes(item.canonical_path);
        
        const isChecked = isAlreadyGT || isSameClass;
        
        // If it should be checked but not in the gts array, add it to gts
        if (isChecked && !isAlreadyGT) {
            gts.push(item.canonical_path);
            stateChanged = true;
        }
    });
    
    if (stateChanged) {
        datasetState.gt_selections[seedPath] = gts;
        // Update the GT grid to reflect the added GTs
        updateGTGrid();
    }
    
    filtered.forEach(item => {
        const isChecked = gts.includes(item.canonical_path);
        
        const card = document.createElement("div");
        card.className = `img-card ${isChecked ? 'selected' : ''}`;
        
        const thumbContainer = document.createElement("div");
        thumbContainer.className = "card-thumb-container";
        
        const img = document.createElement("img");
        img.src = `/api/image?path=${encodeURIComponent(item.canonical_path)}`;
        img.alt = item.name;
        
        thumbContainer.appendChild(img);
        
        const nameP = document.createElement("p");
        nameP.className = "card-name";
        nameP.textContent = item.name;
        
        // Checkbox wrapper
        const cbWrapper = document.createElement("label");
        cbWrapper.className = "card-checkbox-wrapper";
        
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = isChecked;
        cb.addEventListener("change", (e) => {
            toggleGTSelection(item.canonical_path, cb.checked);
        });
        
        const cbLabel = document.createElement("span");
        cbLabel.textContent = isChecked ? "已選 GT" : "設為 GT";
        
        cbWrapper.appendChild(cb);
        cbWrapper.appendChild(cbLabel);
        
        card.appendChild(thumbContainer);
        card.appendChild(nameP);
        card.appendChild(cbWrapper);
        
        // Click image thumbnail to zoom
        thumbContainer.addEventListener("click", (e) => {
            e.stopPropagation(); // Avoid triggering selection toggle
            const candidatePaths = currentCandidates.map(c => c.canonical_path);
            openZoomModal(item.canonical_path, candidatePaths);
        });
        
        // Also card click toggles checkbox
        card.addEventListener("click", (e) => {
            // Only toggle if they didn't click checkbox itself directly (handled by change listener)
            if (e.target !== cb && e.target !== cbWrapper && !cbWrapper.contains(e.target)) {
                cb.checked = !cb.checked;
                toggleGTSelection(item.canonical_path, cb.checked);
            }
        });
        
        candidatesGridContainer.appendChild(card);
    });
}

// Add or Remove an image path to the Ground Truth list of the active seed
function toggleGTSelection(canonicalPath, isAdd) {
    const seedPath = datasetState.seeds[currentSeedIndex];
    if (!datasetState.gt_selections[seedPath]) {
        datasetState.gt_selections[seedPath] = [];
    }
    
    let currentGts = datasetState.gt_selections[seedPath];
    
    if (isAdd) {
        if (!currentGts.includes(canonicalPath)) {
            currentGts.push(canonicalPath);
        }
    } else {
        datasetState.gt_selections[seedPath] = currentGts.filter(p => p !== canonicalPath);
    }
    
    // Refresh grids
    updateGTGrid();
    
    // Find candidate card and update its visual style
    const cards = candidatesGridContainer.querySelectorAll(".img-card");
    cards.forEach(card => {
        // We find the checkbox inside to verify the path
        const cb = card.querySelector('input[type="checkbox"]');
        if (cb) {
            // Find parent img card corresponding to this click
            const img = card.querySelector("img");
            if (img && img.src.includes(encodeURIComponent(canonicalPath))) {
                cb.checked = isAdd;
                const labelText = card.querySelector(".card-checkbox-wrapper span");
                if (isAdd) {
                    card.classList.add("selected");
                    if (labelText) labelText.textContent = "已選 GT";
                } else {
                    card.classList.remove("selected");
                    if (labelText) labelText.textContent = "設為 GT";
                }
            }
        }
    });
}

// API Call: Save current selections and trigger dataset rebuilding pipeline
async function saveAndRebuild() {
    setStatus("正在重建資料集分割並建立符號連結...", "working");
    saveBuildBtn.disabled = true;
    
    const payload = {
        gt_selections: datasetState.gt_selections,
        distractors: datasetState.distractors
    };
    
    try {
        const response = await fetch("/api/save", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRF-Token": csrfToken
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || `HTTP 錯誤: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Update stats state
        datasetState.stats.total_v = result.stats.total_v;
        datasetState.stats.total_t_small = result.stats.total_t_small;
        datasetState.stats.total_t_large = result.stats.total_t_large;
        
        updateStatsUI();
        
        setStatus("資料集分割建置成功！符號連結已更新！", "online");
        alert("🎉 資料集建置完成！(dataset_v2 結構就緒)");
    } catch (err) {
        loggerError(err);
        setStatus(`重建資料集失敗: ${err.message}`, "error");
        alert(`❌ 建置資料集失敗: ${err.message}`);
    } finally {
        saveBuildBtn.disabled = false;
    }
}

// Zoom Modal Functions
function updateModalImageTransform() {
    const transformStr = `translate(${panX}px, ${panY}px) scale(${zoomScale})`;
    modalImg.style.transform = transformStr;
    modalSeedImg.style.transform = transformStr;
}

function openZoomModal(targetPath, pathList = []) {
    zoomImageList = pathList && pathList.length > 0 ? pathList : [targetPath];
    zoomImageIndex = zoomImageList.indexOf(targetPath);
    if (zoomImageIndex === -1) {
        zoomImageIndex = 0;
    }
    
    // Reset transforms
    zoomScale = 1.0;
    panX = 0;
    panY = 0;
    updateModalImageTransform();
    
    displayZoomImage();
    
    imageModal.classList.add("show");
}

function displayZoomImage() {
    if (zoomImageIndex < 0 || zoomImageIndex >= zoomImageList.length) return;
    
    const canonicalPath = zoomImageList[zoomImageIndex];
    const imgSrc = `/api/image?path=${encodeURIComponent(canonicalPath)}`;
    
    modalImg.src = imgSrc;
    modalSeedImg.src = querySeedImg.src; // Copy current Query Seed image
    
    // Update GT button status
    updateZoomModalGTButton();
    
    // Update navigation buttons visibility/disabled states
    zoomPrevBtn.disabled = (zoomImageIndex === 0);
    zoomNextBtn.disabled = (zoomImageIndex === zoomImageList.length - 1);
    
    // Style disabled buttons
    zoomPrevBtn.style.opacity = zoomPrevBtn.disabled ? "0.3" : "1";
    zoomNextBtn.style.opacity = zoomNextBtn.disabled ? "0.3" : "1";
}

function updateZoomModalGTButton() {
    if (zoomImageIndex < 0 || zoomImageIndex >= zoomImageList.length) return;
    
    const canonicalPath = zoomImageList[zoomImageIndex];
    const seedPath = datasetState.seeds[currentSeedIndex];
    const gts = datasetState.gt_selections[seedPath] || [];
    const isChecked = gts.includes(canonicalPath);
    
    if (isChecked) {
        zoomGtBtn.textContent = "✅ 已選 GT";
        zoomGtBtn.classList.add("is-gt");
    } else {
        zoomGtBtn.textContent = "⬜ 設為 GT";
        zoomGtBtn.classList.remove("is-gt");
    }
    
    // Disable GT button if we are looking at the seed itself (seed cannot be its own GT)
    if (canonicalPath === seedPath) {
        zoomGtBtn.disabled = true;
        zoomGtBtn.style.opacity = "0.3";
    } else {
        zoomGtBtn.disabled = false;
        zoomGtBtn.style.opacity = "1";
    }
}

function closeZoomModal() {
    imageModal.classList.remove("show");
    modalImg.src = "";
    modalSeedImg.src = "";
    zoomImageList = [];
    zoomImageIndex = -1;
}

function loggerError(err) {
    // Console warnings are fine for error diagnostics, avoiding print of sensitive objects
    console.error("App error:", err.message);
}
