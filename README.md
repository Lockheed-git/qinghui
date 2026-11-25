# 晴绘 · Scenimefy 4K

[打开网页](https://lockheed-git.github.io/qinghui/)

拖入 JPG、PNG 或 WebP 即可处理。网页优先使用 WebGPU，也可用 WebAssembly 在 CPU 上运行。图片留在浏览器内，网页从 GitHub Pages 加载模型和运行时。

- **原图调色**是默认模式。程序从重绘缩略图提取局部色彩变化，再应用到原图像素，保留原分辨率和细节。
- **逐块重绘**在原始分辨率上运行模型，从缩略图估算分块共用的归一化统计量。4K 大图在部分浏览器上可能需要较长时间。
- **快速预览**将长边缩至 352 像素以内。

## 模型与资源

- 模型来自 [Scenimefy](https://github.com/Yuxinn-J/Scenimefy) 的官方 Shinkai checkpoint，与上游 [S-Lab License](./LICENSE.md) 一起分发，请按其非商业条款使用。
- 浏览器使用 [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.31.1。运行时及其许可文件位于 `runtime/`。
- 示例使用 The People's Internet 的 [Grand Teton National Park Mountains](https://commons.wikimedia.org/wiki/File:Grand_Teton_National_Park_Mountains.jpg) 和 Georgia Dixon 的 [Mountains by the lake](https://commons.wikimedia.org/wiki/File:Mountains_by_the_lake_(Unsplash).jpg)，两张原始照片均为 CC0。

## 重新导出模型

安装 PyTorch 与 ONNX，下载上游 [Shinkai_net_G.pth](https://github.com/Yuxinn-J/Scenimefy/releases/tag/v0.1.0)，在仓库根目录运行：

```sh
python tools/export_model.py Shinkai_net_G.pth
```

部署静态文件时保留 `model/` 下的两个 ONNX 图、`weights.json` 和它们共用的权重分片。
