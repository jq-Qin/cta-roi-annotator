# CTA ROI 标注工具

轻量级逐切片人工标注工具。原图只读，mask 仅写入单独的输出目录。

## 功能

- PNG、JPG、JPEG、TIF、TIFF，以及 `.dcm`/无扩展名 DICOM。
- 文件名自然排序。
- Polygon、Circle、Freehand Curve、Brush、Eraser、Undo、Clear。
- 右侧显示完整切片列表，可直接点击任意文件跳转，适合非连续抽样标注。
- 红色半透明 mask overlay，可调透明度、可隐藏。
- 切片切换前自动保存已修改内容。
- 复制上一张 mask 后继续微调。
- mask 原子写入，单通道 `uint8`，值域严格为 `{0, 255}`。
- mask 文件名与输入文件名完全相同。为避免 JPG 有损压缩，mask 内容始终使用 PNG 编码；DICOM 的无扩展名文件也保持无扩展名。

## 目录

```text
cta_roi_annotator/
├── cta_annotator/
│   ├── core.py          # 读取、排序、mask 操作、保存与切片状态
│   ├── ui.py            # PySide6 界面和鼠标交互
│   └── main.py          # CLI 入口
├── tests/
│   └── test_core.py     # synthetic 端到端测试
├── requirements.txt
├── README.md
└── run.py
```

## 安装与启动

Python 3.10+：

```powershell
git clone https://github.com/jq-Qin/cta-roi-annotator.git
cd cta-roi-annotator

python -m pip install -r requirements.txt
python run.py
```

也可直接指定目录：

```powershell
python run.py --image-dir path/to/images --mask-dir path/to/masks
```

`mask_dir` 不存在时，CLI 启动会自动创建。图形界面的目录选择器需要先创建或选中目标目录。

## 操作

- Polygon：连续左键点击；双击或 Enter 闭合。可创建多个独立区域。
- Circle：从圆心按住左键向外拖动，松开后填充圆形 ROI。
- Freehand Curve：按住左键沿区域边界画曲线，松开后自动闭合填充。
- Brush / Eraser：按住左键绘制或擦除。
- 切片列表：点击任意文件直接跳转；`○` 未标注、`✓` 已保存、`*` 当前有未保存修改。
- 鼠标滚轮：上一张/下一张。
- Ctrl + 鼠标滚轮：缩放。
- 中键拖动：平移。
- `A` / `D`：上一张/下一张。
- `V`：复制上一张已保存 mask。
- `Ctrl+S`：保存。
- `Ctrl+Z`：撤销。
- `C`：清空。
- `O`：开关 overlay。
- `F`：适合窗口。
- `Esc`：取消尚未闭合的 Polygon。

## 数据安全与格式

- 程序拒绝使用同一个目录作为 `image_dir` 和 `mask_dir`。
- 不对原图调用任何写操作。
- 保存前检查 mask 和当前原图的高宽一致。
- 临时文件完整写入后再原子替换目标 mask，异常不会破坏上一次已保存结果。
- 已有 mask 会被阈值化为 `{0, 255}` 后载入；尺寸不匹配会停止并显示错误。
- DICOM 像素的 rescale slope/intercept、MONOCHROME1 和 Window Center/Width 仅用于正确显示，mask 坐标仍对应原始 Rows × Columns。

## 测试

测试会在临时目录生成三张不同尺寸的 synthetic 图像，不接触真实 CTA 数据：

```powershell
cd cta-roi-annotator
python -m pytest -q
```

覆盖：二值值域、dtype、分辨率、原名映射、原图哈希不变、已有 mask 重载编辑、Copy Previous、Undo/Clear，以及连续切换时的文件对应关系。

## 限制

- 每次选择一个包含单序列切片的目录，不递归混合多个 DICOM 序列。
- 当前支持单帧二维 DICOM；多帧 DICOM 会明确报错。
- 压缩 DICOM 可能需要额外 pixel-data 解码器，未加入默认依赖。
