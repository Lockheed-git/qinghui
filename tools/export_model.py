"""Export the licensed Scenimefy generator for preview and tiled web inference."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import onnx
from onnx.external_data_helper import set_external_data
import torch
from torch import nn

from scenimefy_model import SceneNorm, load_generator


class GuideNorm(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, image):
        mean = image.mean(dim=(2, 3), keepdim=True)
        centered = image - mean
        scale = torch.rsqrt(centered.square().mean(dim=(2, 3), keepdim=True) + self.eps)
        self.stats = torch.cat((scale.flatten(), (-mean * scale).flatten()))
        return centered * scale


class TileNorm(nn.Module):
    def __init__(self, offset, channels):
        super().__init__()
        self.offset = offset
        self.channels = channels

    def forward(self, image):
        start, count = self.offset, self.channels
        scale = self.stats[start:start + count].view(1, count, 1, 1)
        shift = self.stats[start + count:start + 2 * count].view(1, count, 1, 1)
        return image * scale + shift


def guide_outputs(guide, layers, image):
    output = guide(image)
    return output, torch.cat([layer.stats for layer in layers])


class Guide(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.generator = load_generator(checkpoint, torch.device("cpu"), torch.float32)
        self.layers = []
        for name, layer in list(self.generator.named_modules()):
            if isinstance(layer, SceneNorm):
                norm = GuideNorm(layer.eps)
                self.generator.set_submodule(name, norm)
                self.layers.append(norm)

    def forward(self, image):
        return guide_outputs(self.generator, self.layers, image)


class Tile(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.generator = load_generator(checkpoint, torch.device("cpu"), torch.float32)
        self.layers = []
        self.stats_length = 0
        for name, layer in list(self.generator.named_modules()):
            if isinstance(layer, SceneNorm):
                count = layer.num_features
                norm = TileNorm(self.stats_length, count)
                self.generator.set_submodule(name, norm)
                self.layers.append(norm)
                self.stats_length += 2 * count

    def forward(self, image, reference):
        for layer in self.layers:
            layer.stats = reference
        return self.generator(image)


def externalize(models, destination):
    """Store identical read-only tensors from both graphs once."""
    index = {}
    filenames = []
    shard_limit = 6 * 1024 * 1024
    with ExitStack() as stack:
        weights = None
        for path in models:
            model = onnx.load(path)
            for tensor in model.graph.initializer:
                raw = tensor.raw_data
                if len(raw) <= 1024:
                    continue
                fingerprint = hashlib.sha256(raw).digest()
                if fingerprint not in index:
                    if weights is None or weights.tell() + len(raw) > shard_limit:
                        if weights is not None:
                            weights.close()
                        filenames.append(f"weights-{len(filenames):02d}.bin")
                        weights = stack.enter_context((destination / filenames[-1]).open("wb"))
                    index[fingerprint] = (filenames[-1], weights.tell())
                    weights.write(raw)
                filename, offset = index[fingerprint]
                set_external_data(tensor, filename, offset, len(raw))
                tensor.ClearField("raw_data")
            path.write_bytes(model.SerializeToString())
    (destination / "weights.json").write_text(json.dumps(filenames), encoding="utf-8")
    for path in models:
        onnx.checker.check_model(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "model")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    guide = Guide(args.checkpoint).eval()
    tile = Tile(args.checkpoint).eval()
    image = torch.zeros(1, 3, 64, 72)
    reference = torch.zeros(tile.stats_length)
    dynamic = {"image": {2: "height", 3: "width"}, "result": {2: "height", 3: "width"}}
    filenames = [args.out / "scenimefy-guide.onnx", args.out / "scenimefy-tile.onnx"]
    with torch.no_grad():
        for model, tensors, inputs, outputs, path in (
            (guide, (image,), ["image"], ["result", "reference"], filenames[0]),
            (tile, (image, reference), ["image", "reference"], ["result"], filenames[1]),
        ):
            torch.onnx.export(
                model, tensors, str(path), dynamo=False, opset_version=17,
                input_names=inputs, output_names=outputs, dynamic_axes=dynamic,
            )
    externalize(filenames, args.out)
    print(f"Exported both Scenimefy graphs to {args.out}")


if __name__ == "__main__":
    main()
