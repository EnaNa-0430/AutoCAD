# Bitmap To Parametric Vectorization

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 生成示例输入图

```bash
python tools/generate_demo_image.py
```

## 3. 运行主流程

```bash
python main.py --image data/input/demo.png --config config/default.yaml --output data/output
```

## 4. 输出结果

每次运行会在 `data/output/run_YYYYMMDD_HHMMSS/` 下生成：

- `01_preprocess`：二值图与预处理摘要
- `02_edges_contours`：边缘图与轮廓点集
- `03_segmentation`：分段结果与预览图
- `04_fitting`：候选模型拟合结果
- `05_model_selection`：模型选择结果
- `06_topology`：拓扑图结构
- `07_json_svg`：最终 `JSON` 和 `SVG`
- `08_visualization`：原图叠加可视化
- `09_report`：汇总报告
