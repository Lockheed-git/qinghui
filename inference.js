import * as ort from "./runtime/ort.webgpu.module.js";

const TILE = 480;
const CONTEXT = 64;
const OVERLAP = 160;
const PREVIEW_EDGE = TILE - 2 * CONTEXT;
const model = name => new URL("./model/" + name, import.meta.url).href;

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = {
    mjs: new URL("./runtime/ort-wasm-simd-threaded.jsep.js", import.meta.url).href,
    wasm: new URL("./runtime/ort-wasm-simd-threaded.jsep.wasm", import.meta.url).href,
};
ort.env.logLevel = "error";
ort.env.webgpu.powerPreference = "high-performance";

let backend;
let weights;
let guideSession;
let tileSession;

const report = (message, current = 0, total = 0) => {
    postMessage({ type: "progress", message, current, total, backend });
};

async function prepareGuide() {
    if (guideSession) return;
    report("正在加载模型和浏览器运行时");
    if (!weights) {
        const manifest = await fetch(model("weights.json"), { cache: "force-cache" });
        if (!manifest.ok) throw new Error("模型下载失败，请检查网络后重试。");
        const files = await manifest.json();
        weights = await Promise.all(files.map(async path => {
            const response = await fetch(model(path), { cache: "force-cache" });
            if (!response.ok) throw new Error("模型下载失败，请检查网络后重试。");
            return { path, data: new Uint8Array(await response.arrayBuffer()) };
        }));
    }

    if (globalThis.navigator?.gpu) {
        try {
            guideSession = await createSession("guide", "webgpu");
            backend = "webgpu";
            return;
        } catch (error) {
            console.warn("WebGPU initialization failed", error);
        }
    }
    guideSession = await createSession("guide", "wasm");
    backend = "wasm";
}

function createSession(name, provider) {
    return ort.InferenceSession.create(model("scenimefy-" + name + ".onnx"), {
        executionProviders: provider === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
        externalData: weights,
    });
}

function readPixels(bitmap, longest) {
    const scale = Math.min(1, longest / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, 0, 0, width, height);
    return ctx.getImageData(0, 0, width, height);
}

const paddedSize = length => 2 * CONTEXT + Math.ceil(length / 4) * 4;
const reflect = (coord, length) => {
    if (length === 1) return 0;
    const period = 2 * (length - 1);
    const reflected = ((coord % period) + period) % period;
    return reflected < length ? reflected : period - reflected;
};

function cropTensor(image, x0, y0, width, height) {
    const body = width * height;
    const values = new Float32Array(3 * body);
    const columns = Array.from({ length: width }, (_, n) => reflect(x0 + n - CONTEXT, image.width) * 4);
    for (let y = 0; y < height; ++y) {
        const row = reflect(y0 + y - CONTEXT, image.height) * image.width * 4;
        for (let x = 0; x < width; ++x) {
            const source = row + columns[x];
            const target = y * width + x;
            values[target] = image.data[source] / 127.5 - 1;
            values[target + body] = image.data[source + 1] / 127.5 - 1;
            values[target + 2 * body] = image.data[source + 2] / 127.5 - 1;
        }
    }
    return new ort.Tensor("float32", values, [1, 3, height, width]);
}

function tiles(length) {
    const steps = Math.ceil(Math.max(0, length - TILE) / (TILE - OVERLAP));
    const starts = steps
        ? Array.from({ length: steps + 1 }, (_, i) => 4 * Math.round(i * (length - TILE) / (4 * steps)))
        : [0];
    const size = Math.min(TILE, length);
    const ranges = starts.map((start, i) => ({
        start, size, innerStart: start + (i ? CONTEXT : 0),
        innerEnd: start + size - (i < steps ? CONTEXT : 0),
    }));
    return ranges.map((range, i) => {
        const left = i ? ranges[i - 1].innerEnd - range.innerStart : 0;
        const right = i < steps ? range.innerEnd - ranges[i + 1].innerStart : 0;
        const weights = new Float32Array(range.innerEnd - range.innerStart).fill(1);
        for (let j = 0; j < left; ++j) weights[j] *= Math.sin((j + .5) * Math.PI / (2 * left)) ** 2;
        for (let j = 0; j < right; ++j) weights[weights.length - right + j] *= Math.cos((j + .5) * Math.PI / (2 * right)) ** 2;
        return { ...range, weights };
    });
}

function writeDirect(image, data, width, height) {
    const plane = width * height;
    for (let y = 0; y < image.height; ++y) {
        for (let x = 0; x < image.width; ++x) {
            const source = (y + CONTEXT) * width + x + CONTEXT;
            const out = (y * image.width + x) * 4;
            for (let c = 0; c < 3; ++c) image.data[out + c] = color(data[source + c * plane]);
        }
    }
}

function color(value) {
    if (!Number.isFinite(value)) throw new Error("模型返回了无效颜色，请重试。");
    return (value + 1) * 127.5;
}

function transferColors(image, source, painted) {
    const mapAxis = (size, reduced) => Array.from({ length: size }, (_, i) => {
        const position = Math.max(0, Math.min(reduced - 1, (i + .5) * reduced / size - .5));
        const left = Math.floor(position);
        return [left, Math.min(left + 1, reduced - 1), position - left];
    });
    const xs = mapAxis(image.width, source.width);
    const ys = mapAxis(image.height, source.height);
    const delta = new Int16Array(painted.data.length);
    for (let i = 0; i < delta.length; ++i) delta[i] = painted.data[i] - source.data[i];

    for (let y = 0; y < image.height; ++y) {
        const [y0, y1, dy] = ys[y];
        for (let x = 0; x < image.width; ++x) {
            const [x0, x1, dx] = xs[x];
            const output = (y * image.width + x) * 4;
            const r = image.data[output];
            const g = image.data[output + 1];
            const b = image.data[output + 2];
            let sum = 0;
            let dr = 0;
            let dg = 0;
            let db = 0;
            // Favor guide samples whose original color is on the same side of an edge.
            for (let k = 0; k < 4; ++k) {
                const column = k & 1 ? x1 : x0;
                const row = k & 2 ? y1 : y0;
                const sample = (row * source.width + column) * 4;
                const distance = Math.abs(r - source.data[sample])
                    + Math.abs(g - source.data[sample + 1]) + Math.abs(b - source.data[sample + 2]);
                const positionWeight = (k & 1 ? dx : 1 - dx) * (k & 2 ? dy : 1 - dy);
                const weight = positionWeight / (1 + distance / 45) ** 2;
                sum += weight;
                dr += weight * delta[sample];
                dg += weight * delta[sample + 1];
                db += weight * delta[sample + 2];
            }
            image.data[output] = r + dr / sum;
            image.data[output + 1] = g + dg / sum;
            image.data[output + 2] = b + db / sum;
        }
    }
}

function addTile(horizontal, vertical, tensor, width, height, rgb, totals) {
    const xFirst = Math.max(CONTEXT, horizontal.innerStart);
    const xLast = Math.min(CONTEXT + width, horizontal.innerEnd);
    const yFirst = Math.max(CONTEXT, vertical.innerStart);
    const yLast = Math.min(CONTEXT + height, vertical.innerEnd);
    const plane = horizontal.size * vertical.size;
    for (let y = yFirst; y < yLast; ++y) {
        const row = (y - vertical.start) * horizontal.size;
        const wy = vertical.weights[y - vertical.innerStart];
        for (let x = xFirst; x < xLast; ++x) {
            const weight = wy * horizontal.weights[x - horizontal.innerStart];
            const i = (y - CONTEXT) * width + x - CONTEXT;
            const source = row + x - horizontal.start;
            rgb[i * 3] += tensor[source] * weight;
            rgb[i * 3 + 1] += tensor[source + plane] * weight;
            rgb[i * 3 + 2] += tensor[source + 2 * plane] * weight;
            totals[i] += weight;
        }
    }
}

async function renderTiled(image, reference) {
    if (!tileSession) tileSession = await createSession("tile", backend);
    const { width, height, data } = image;
    const columns = tiles(paddedSize(width));
    const rows = tiles(paddedSize(height));
    const rgb = new Float32Array(width * height * 3);
    const totals = new Float32Array(width * height);
    const total = columns.length * rows.length;
    let completed = 0;
    report("正在分块重绘", completed, total);
    for (const vertical of rows) {
        for (const horizontal of columns) {
            const input = cropTensor(image, horizontal.start, vertical.start, horizontal.size, vertical.size);
            const { result } = await tileSession.run({ image: input, reference });
            try {
                addTile(horizontal, vertical, result.data, width, height, rgb, totals);
            } finally {
                input.dispose();
                result.dispose();
            }
            report("正在分块重绘", ++completed, total);
        }
    }
    for (let i = 0; i < totals.length; ++i) {
        if (totals[i] === 0) throw new Error("有部分图像未完成重绘。");
        for (let c = 0; c < 3; ++c) data[i * 4 + c] = color(rgb[i * 3 + c] / totals[i]);
    }
}

async function run(file, mode) {
    const bitmap = await createImageBitmap(file);
    try {
        if (bitmap.width * bitmap.height > 24_000_000 || Math.max(bitmap.width, bitmap.height) > 8192) {
            throw new Error("图片需在 2400 万像素以内，且长边不超过 8192 像素。");
        }
        await prepareGuide();
        const started = performance.now();
        const guide = readPixels(bitmap, PREVIEW_EDGE);
        const tiled = mode === "native" && (guide.width !== bitmap.width || guide.height !== bitmap.height);
        const width = paddedSize(guide.width);
        const height = paddedSize(guide.height);
        report(tiled ? "正在估算全图色调" : "正在重绘");
        const input = cropTensor(guide, 0, 0, width, height);
        const { result, reference } = await guideSession.run({ image: input });
        let image = guide;
        try {
            if (tiled) {
                image = readPixels(bitmap, Infinity);
                await renderTiled(image, reference);
            } else if (mode === "color") {
                const source = new ImageData(guide.data.slice(), guide.width, guide.height);
                writeDirect(guide, result.data, width, height);
                report("正在给原尺寸照片调色");
                image = readPixels(bitmap, Infinity);
                transferColors(image, source, guide);
            } else {
                writeDirect(image, result.data, width, height);
            }
        } finally {
            input.dispose();
            result.dispose();
            reference.dispose();
        }
        report("正在编码 PNG");
        const canvas = new OffscreenCanvas(image.width, image.height);
        canvas.getContext("2d").putImageData(image, 0, 0);
        const blob = await canvas.convertToBlob({ type: "image/png" });
        postMessage({ type: "result", blob, width: image.width, height: image.height, backend,
            seconds: (performance.now() - started) / 1000 });
    } finally {
        bitmap.close();
    }
}

addEventListener("message", async ({ data }) => {
    try {
        await run(data.file, data.mode);
    } catch (error) {
        console.error(error);
        postMessage({ type: "error", message: error.message });
    }
});
