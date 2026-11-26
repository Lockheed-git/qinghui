// Estimate one bounded, monotonic luminance curve per photo.
const luminance = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
const clamp01 = value => Math.min(1, Math.max(0, value));

function percentile(histogram, count, fraction) {
    const rank = Math.ceil(count * fraction);
    let accumulated = 0;
    for (let i = 0; i < histogram.length; ++i) {
        accumulated += histogram[i];
        if (accumulated >= rank) return i;
    }
    return histogram.length - 1;
}

function imageStats({ data }) {
    const histogram = new Uint32Array(256);
    let count = 0;
    let chroma = 0;
    const step = 4 * Math.max(1, Math.floor(data.length / (4 * 65_536)));
    for (let i = 0; i < data.length; i += step) {
        if (data[i + 3] < 32) continue;
        const r = data[i], g = data[i + 1], b = data[i + 2];
        ++histogram[Math.round(luminance(r, g, b))];
        chroma += Math.max(r, g, b) - Math.min(r, g, b);
        ++count;
    }
    if (!count) return null;
    return {
        middle: percentile(histogram, count, .5),
        high: percentile(histogram, count, .96),
        low: percentile(histogram, count, .04),
        chroma: chroma / count,
    };
}

export function estimateTone(image) {
    const stats = imageStats(image);
    if (!stats) return null;
    const { middle, high, low, chroma } = stats;
    const span = high - low;
    const dark = clamp01((117 - middle) / 85);
    const flat = clamp01((165 - span) / 105);
    if (Math.max(dark, flat) < .035) return null;

    const gamma = 1 - .48 * dark;
    const gain = 1 + .95 * flat + .12 * dark;
    const saturation = 1 + (.21 * dark + .18 * flat) * clamp01((52 - chroma) / 42);
    const curve = Float32Array.from({ length: 256 }, (_, y) => {
        const stretched = clamp01((middle + (y - middle) * gain) / 255);
        return .22 * y + .78 * 255 * stretched ** gamma;
    });
    return { curve, saturation };
}

export function estimateOutputTone(reference, image) {
    const target = imageStats(reference);
    const output = imageStats(image);
    if (!target || !output) return null;
    const { middle, high, low } = output;
    const headroom = 255 - high;
    const lift = Math.min(48, .6 * headroom, Math.max(0, .85 * target.middle - middle));
    const contrast = Math.min(58, .9 * headroom - lift,
        Math.max(0, .8 * (target.high - target.low) - (high - low)));
    if (Math.max(lift, contrast) < 2) return null;

    // Preserve black and white endpoints and apply the added contrast above the median.
    const curve = Float32Array.from({ length: 256 }, (_, y) => {
        const shadow = clamp01(y / Math.max(8, middle));
        const rolloff = clamp01((255 - y) / Math.max(1, headroom));
        const upperMidtone = clamp01((y - middle) / Math.max(1, high - middle));
        return y + (lift * shadow + contrast * upperMidtone) * rolloff;
    });
    return { curve, saturation: 1 };
}

export function applyTone({ data }, tone) {
    if (!tone) return;
    const { curve, saturation } = tone;
    for (let i = 0; i < data.length; i += 4) {
        if (data[i + 3] === 0) continue;
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const y = Math.min(255, luminance(r, g, b));
        const low = Math.floor(y);
        const target = curve[low] + (curve[Math.min(255, low + 1)] - curve[low]) * (y - low);
        const below = y - Math.min(r, g, b);
        const above = Math.max(r, g, b) - y;
        // Compress chroma toward the target luminance when a channel would clip.
        let scale = saturation;
        if (below > 0) scale = Math.min(scale, target / below);
        if (above > 0) scale = Math.min(scale, (255 - target) / above);
        data[i] = target + (r - y) * scale;
        data[i + 1] = target + (g - y) * scale;
        data[i + 2] = target + (b - y) * scale;
    }
}
