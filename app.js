"use strict";

const byId = (id) => document.getElementById(id);
const dropzone = byId("dropzone");
const fileInput = byId("file");
const runButton = byId("run");
const frame = byId("frame");
const slider = byId("compare");
const download = byId("download");
const modeInputs = document.querySelectorAll("input[name=mode]");
const exampleButtons = document.querySelectorAll(".example");
const examples = { mountain: "山间湖泊", lake: "日光湖面" };
const hints = {
    color: "从重绘缩略图提取颜色，应用到原图。",
    native: "按原尺寸逐块重绘。4K 耗时取决于浏览器和显卡。",
    preview: "缩至长边 352 像素，先看重绘效果。",
};
const modeLabels = { color: "原图调色", native: "逐块重绘", preview: "快速预览" };
let selectedFile = null;
let inputUrl;
let outputUrl;
let busy = false;
let worker;

function updateSplit() {
    frame.style.setProperty("--split", slider.value + "%");
}

function setSplit() {
    slider.value = "50";
    updateSplit();
}

function showPreview(before, after, width, height, caption, filename) {
    byId("before").src = before;
    frame.style.setProperty("--ratio", String(width / height));
    frame.dataset.state = after ? "compare" : "original";
    slider.disabled = !after;
    byId("compare-hint").hidden = !after;
    byId("preview-meta").textContent = caption;

    if (after) {
        byId("after").src = after;
        download.href = after;
        download.download = filename;
        download.removeAttribute("aria-disabled");
    } else {
        byId("after").removeAttribute("src");
        download.removeAttribute("href");
        download.setAttribute("aria-disabled", "true");
    }

    setSplit();
}

function errorMessage(message) {
    byId("error").textContent = message;
    byId("error").hidden = !message;
}

function setBusy(value, message = "正在重绘") {
    busy = value;
    dropzone.disabled = value;
    fileInput.disabled = value;
    runButton.disabled = value;
    modeInputs.forEach((input) => { input.disabled = value; });
    exampleButtons.forEach((button) => { button.disabled = value; });
    byId("viewport").setAttribute("aria-busy", String(value));
    byId("progress-label").textContent = message;
    byId("progress").hidden = !value;
    if (!value) {
        byId("cancel").hidden = true;
        byId("tile-progress").hidden = true;
    }
    runButton.hidden = value || !selectedFile;
}

function releasePhotos() {
    if (inputUrl) URL.revokeObjectURL(inputUrl);
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    inputUrl = undefined;
    outputUrl = undefined;
}

function updateExample(id) {
    exampleButtons.forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.example === id));
    });
}

function loadExample(id) {
    if (busy) return;
    releasePhotos();
    selectedFile = null;
    errorMessage("");
    byId("selected-file").hidden = true;
    runButton.hidden = true;
    byId("drop-label").textContent = "拖入图片";
    byId("choose-label").textContent = "或点击选择文件";
    updateExample(id);
    showPreview(
        "./" + id + "-original.jpg", "./" + id + "-repaint.png", 1440, 840,
        "示例 · " + examples[id] + " · 1440 × 840", examples[id] + "-晴绘.png",
    );
}

async function chooseFile(file) {
    if (!file || busy) return;
    if (!/\.(jpe?g|png|webp)$/i.test(file.name) && !/^image\/(jpeg|png|webp)$/.test(file.type)) {
        errorMessage("请选择 JPG、PNG 或 WebP 图片。");
        return;
    }
    if (!file.size || file.size > 32 * 1024 * 1024) {
        errorMessage("请选择 32 MB 以内的有效图片。");
        return;
    }

    errorMessage("");
    setBusy(true, "正在读取图片");
    const url = URL.createObjectURL(file);
    const picture = new Image();
    picture.src = url;
    try {
        await picture.decode();
    } catch {
        URL.revokeObjectURL(url);
        setBusy(false);
        errorMessage("打不开这张图片，请检查文件后重试。");
        return;
    }

    releasePhotos();
    inputUrl = url;
    selectedFile = { file, width: picture.naturalWidth, height: picture.naturalHeight };
    byId("selected-file").hidden = false;
    byId("file-name").textContent = file.name;
    byId("file-name").title = file.name;
    byId("file-size").textContent = file.size < 1024 * 1024
        ? Math.ceil(file.size / 1024) + " KB" : (file.size / 1024 / 1024).toFixed(1) + " MB";
    byId("drop-label").textContent = "换一张图片";
    byId("choose-label").textContent = "拖入或点击选择";
    updateExample(null);
    showPreview(inputUrl, null, selectedFile.width, selectedFile.height, selectedFile.width + " × " + selectedFile.height);
    setBusy(false);
    generate();
}

function generate() {
    if (!selectedFile || busy) return;
    const { file } = selectedFile;
    const mode = document.querySelector("input[name=mode]:checked").value;
    errorMessage("");
    setBusy(true, "正在准备重绘");
    byId("cancel").hidden = false;
    if (!worker) {
        worker = new Worker("./inference.js", { type: "module" });
        worker.onerror = (event) => {
            console.error("Worker failed to start", event.message);
            worker.terminate();
            worker = undefined;
            errorMessage("无法启动浏览器重绘，请使用新版 Chrome 或 Edge 后重试。");
            setBusy(false);
        };
    }
    worker.onmessage = ({ data }) => {
        if (data.backend) setDevice(data.backend === "webgpu" ? "浏览器 GPU" : "浏览器 CPU");
        if (data.type === "progress") {
            byId("progress-label").textContent = data.message + (data.total ? " · " + data.current + "/" + data.total : "");
            const progress = byId("tile-progress");
            progress.hidden = !data.total;
            if (data.total) {
                progress.max = data.total;
                progress.value = data.current;
            }
            return;
        }
        setBusy(false);
        if (data.type === "error") {
            errorMessage(data.message || "重绘失败，请切换到快速预览后重试。");
            return;
        }
        const { blob, width, height, seconds } = data;
        const previous = outputUrl;
        outputUrl = URL.createObjectURL(blob);
        const caption = width + " × " + height + " · " + modeLabels[mode] + " · " + seconds.toFixed(1) + " 秒";
        const name = file.name.replace(/\.[^.]+$/, "") + "-晴绘.png";
        showPreview(inputUrl, outputUrl, selectedFile.width, selectedFile.height, caption, name);
        if (previous) URL.revokeObjectURL(previous);
    };
    worker.postMessage({ file, mode });
}

function setDevice(label) {
    byId("device").dataset.state = "ready";
    byId("device-label").textContent = label;
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
    const [file] = fileInput.files;
    fileInput.value = "";
    chooseFile(file);
});
runButton.addEventListener("click", generate);
byId("cancel").addEventListener("click", () => {
    if (!worker) return;
    worker.terminate();
    worker = undefined;
    setBusy(false);
});
slider.addEventListener("input", updateSplit);
exampleButtons.forEach((button) => {
    button.addEventListener("click", () => loadExample(button.dataset.example));
});
modeInputs.forEach((option) => {
    option.addEventListener("change", () => {
        byId("mode-hint").textContent = hints[option.value];
    });
});

document.addEventListener("dragover", (event) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = busy ? "none" : "copy";
    if (!busy) dropzone.classList.add("is-dragging");
});
document.addEventListener("dragleave", (event) => {
    if (!event.relatedTarget) dropzone.classList.remove("is-dragging");
});
document.addEventListener("drop", (event) => {
    if (!event.dataTransfer.files.length) return;
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
    if (busy) return;
    if (event.dataTransfer.files.length > 1) {
        errorMessage("请一次拖入一张图片。");
        return;
    }
    chooseFile(event.dataTransfer.files[0]);
});

loadExample("mountain");
if (navigator.gpu) {
    setDevice("优先使用 GPU");
} else {
    setDevice("浏览器 CPU");
}
