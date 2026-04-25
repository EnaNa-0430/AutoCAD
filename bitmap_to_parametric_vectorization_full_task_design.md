# 位图到参数化矢量化完整任务设计

## 1. 目标与范围

### 1.1 总体目标
- 输入：位图图像（PNG/JPG，单图或批处理）
- 输出：
  - 参数化结构 `JSON`
  - 矢量图 `SVG`
  - 原图与拟合结果可视化对比图
- 核心原则：在误差可控的前提下，优先采用参数更少的几何模型（最小参数表达）

### 1.2 任务边界
- 支持对象：二维闭合/非闭合轮廓（线段、圆、椭圆、圆弧为主）
- 默认不处理：
  - 大面积纹理语义分割
  - 复杂手绘风格高频抖动（可通过平滑缓解但不保证完美）
  - 三维重建

---

## 2. 端到端架构

```text
输入图像
  -> M1 预处理
  -> M2 边缘与轮廓提取
  -> M3 轮廓分段
  -> M4 候选模型拟合（含RANSAC）
  -> M5 模型选择（误差+复杂度）
  -> M6 拓扑拼接与一致性修正
  -> M7 参数化导出(JSON)
  -> M8 SVG导出
  -> M9 可视化评估与报告
```

---

## 3. 目录与中间结果保留规范

## 3.1 目录结构

```text
project_root/
  data/
    input/
    output/
      run_YYYYMMDD_HHMMSS/
        00_meta/
        01_preprocess/
        02_edges_contours/
        03_segmentation/
        04_fitting/
        05_model_selection/
        06_topology/
        07_json_svg/
        08_visualization/
        09_report/
```

## 3.2 中间结果保存策略
- 每个模块强制落盘 `*_summary.json`（统计信息 + 参数）
- 每个模块可视化输出一张图（便于排查）
- 关键对象保存为 `npy/json/svg`，避免仅保存在内存
- 所有文件名以 `sample_id` 统一前缀，便于追踪

## 3.3 必须保留的中间文件
- `01_preprocess/{sample_id}_binary.png`
- `02_edges_contours/{sample_id}_edges.png`
- `02_edges_contours/{sample_id}_contours.json`
- `03_segmentation/{sample_id}_segments.json`
- `04_fitting/{sample_id}_fit_candidates.json`
- `05_model_selection/{sample_id}_selected_models.json`
- `06_topology/{sample_id}_graph.json`
- `07_json_svg/{sample_id}.json`
- `07_json_svg/{sample_id}.svg`
- `08_visualization/{sample_id}_overlay.png`
- `09_report/{sample_id}_report.json`

---

## 4. 模块详细设计

## M1 图像预处理

### 输入
- 原图 `img_bgr`

### 处理
- 灰度化
- 自适应阈值（可切换 Otsu）
- 开闭运算去噪

### 输出
- `binary.png`
- `preprocess_summary.json`

### 关键参数
- `blur_ksize`
- `adaptive_block_size`
- `adaptive_C`
- `morph_kernel`

---

## M2 边缘检测与轮廓提取

### 输入
- `binary.png`

### 处理
- Canny 边缘
- `findContours` 提取轮廓点集
- 过滤短轮廓（长度阈值）

### 输出
- `edges.png`
- `contours.json`（每条轮廓点序列）
- `contours_summary.json`（轮廓数量、长度分布）

### `contours.json` 结构示例

```json
{
  "sample_id": "demo_001",
  "contours": [
    {
      "contour_id": 0,
      "closed": true,
      "points": [[12.0, 20.0], [13.0, 21.0], [14.0, 21.5]]
    }
  ]
}
```

---

## M3 轮廓分段

### 输入
- `contours.json`

### 处理
- 方法A：离散曲率计算 + 突变点检测
- 方法B：RDP 简化得到关键点
- 融合策略：两种断点并集，再做最小段长约束

### 输出
- `segments.json`
- `segments_preview.png`

### `segments.json` 结构示例

```json
{
  "sample_id": "demo_001",
  "segments": [
    {
      "segment_id": "0_0",
      "contour_id": 0,
      "closed_parent": true,
      "point_indices": [0, 1, 2, 3, 4]
    }
  ]
}
```

---

## M4 候选基元拟合（含RANSAC）

### 输入
- `segments.json`

### 候选模型集合
- 直线（2参数）
- 圆（3参数）
- 椭圆（5参数）
- 圆弧（4参数）
- 退化兜底：折线片段

### 拟合方法
- 直线：最小二乘
- 圆：Kasa/Pratt（二选一，默认 Pratt）
- 椭圆：Fitzgibbon
- 圆弧：先圆拟合，再估计起止角

### RANSAC 机制
- 对每个模型独立进行 `N` 次采样
- 评价指标：内点率 + 几何残差
- 输出最优参数与内点掩码

### 输出
- `fit_candidates.json`
- `fit_debug.png`

### `fit_candidates.json` 结构示例

```json
{
  "segment_id": "0_0",
  "candidates": [
    {
      "type": "line",
      "params": {"a": 1.2, "b": -0.3, "c": 2.1},
      "param_count": 2,
      "rmse": 0.82,
      "inlier_ratio": 0.91
    },
    {
      "type": "circle",
      "params": {"cx": 120.4, "cy": 98.3, "r": 31.2},
      "param_count": 3,
      "rmse": 0.49,
      "inlier_ratio": 0.95
    }
  ]
}
```

---

## M5 模型选择（最小参数原则）

### 输入
- `fit_candidates.json`

### 评分函数
- 通用形式：`score = data_error + lambda * complexity`
- 可选统计准则：
  - `AIC = 2k + n*ln(RSS/n)`
  - `BIC = k*ln(n) + n*ln(RSS/n)`

### 选择规则（建议）
- 先按误差门限过滤（`rmse <= rmse_max`）
- 在可行模型中取最小 `BIC`
- 若差值小于 `delta`，优先参数更少模型

### 输出
- `selected_models.json`
- `selection_summary.json`（各模型入选比例）

---

## M6 拓扑拼接与一致性修正

### 输入
- `selected_models.json`

### 处理
- 邻接段端点吸附（阈值 `snap_eps`）
- 闭合性修正（首尾点距离阈值）
- 共线段合并、同圆弧段合并（可选）

### 输出
- `graph.json`（节点-边结构）
- `topology_preview.png`

---

## M7 参数化 JSON 导出

### 输出结构（目标格式）

```json
{
  "sample_id": "demo_001",
  "primitives": [
    {"type": "circle", "center": [100.0, 100.0], "radius": 50.0},
    {"type": "line", "p1": [0.0, 0.0], "p2": [100.0, 100.0]},
    {"type": "ellipse", "center": [210.0, 120.0], "axes": [60.0, 30.0], "angle": 25.0}
  ],
  "metrics": {
    "global_rmse": 0.73,
    "primitive_count": 3,
    "param_total": 10
  }
}
```

---

## M8 SVG 导出

### 规则
- 按 primitive 类型映射为 SVG 元素
- 统一坐标系（图像坐标到 SVG 坐标）
- 保留 `stroke-width` 与颜色配置

### 输出
- `sample_id.svg`

---

## M9 可视化评估与报告

### 输出
- `overlay.png`：原图灰度 + 拟合红色叠加
- `report.json`：每阶段统计、异常、耗时、最终指标

### 关键指标
- 几何误差：`RMSE`, `Hausdorff distance`
- 复杂度：`primitive_count`, `param_total`
- 效率：总耗时、各模块耗时

---

## 5. 主流程伪代码

```python
def run_pipeline(image_path, cfg, out_dir):
    img = load_image(image_path)

    bin_img, prep_meta = preprocess(img, cfg.preprocess)
    save_stage("01_preprocess", bin_img, prep_meta)

    edges, contours, c_meta = extract_contours(bin_img, cfg.contour)
    save_stage("02_edges_contours", edges, {"contours": contours, **c_meta})

    segments, s_meta = segment_contours(contours, cfg.segment)
    save_stage("03_segmentation", segments, s_meta)

    fit_candidates, f_meta = fit_models(segments, cfg.fit)
    save_stage("04_fitting", fit_candidates, f_meta)

    selected, m_meta = select_models(fit_candidates, cfg.selection)
    save_stage("05_model_selection", selected, m_meta)

    graph, g_meta = topology_refine(selected, cfg.topology)
    save_stage("06_topology", graph, g_meta)

    result_json = export_json(graph, cfg.export)
    result_svg = export_svg(graph, cfg.export)
    save_stage("07_json_svg", {"json": result_json, "svg": result_svg}, {})

    vis_img, rpt = evaluate_and_visualize(img, graph, cfg.eval)
    save_stage("08_visualization", vis_img, {})
    save_stage("09_report", rpt, {})
```

---

## 6. 配置设计（建议）

```yaml
preprocess:
  blur_ksize: 3
  adaptive_block_size: 21
  adaptive_C: 3
  morph_kernel: 3

contour:
  canny_t1: 40
  canny_t2: 120
  min_contour_len: 30

segment:
  curvature_threshold: 0.15
  rdp_epsilon: 2.0
  min_segment_points: 8

fit:
  models: [line, circle, ellipse, arc]
  ransac_iter: 200
  inlier_threshold: 1.5

selection:
  criterion: bic
  rmse_max: 2.5
  lambda_complexity: 0.3
  tie_delta: 0.05

topology:
  snap_eps: 2.0
  close_eps: 3.0
```

---

## 7. 异常与回退策略

- 轮廓提取失败：回退到阈值参数搜索（最多 `K` 次）
- 段点数不足：标记 `invalid_segment`，跳过拟合
- 所有模型失败：回退为折线表示，保证输出可用
- 闭合检测不稳定：双阈值判定（强闭合/弱闭合）

---

## 8. 任务拆解与里程碑

## P1 基线可运行（M1-M3）
- 交付：预处理、轮廓、分段 + 中间文件全落盘
- 验收：至少可在 10 张样本上稳定输出 `segments.json`

## P2 拟合与选择（M4-M5）
- 交付：候选模型拟合 + AIC/BIC 选择
- 验收：输出 `selected_models.json`，并能体现“误差-复杂度”权衡

## P3 拓扑与导出（M6-M8）
- 交付：拓扑修正 + JSON/SVG 导出
- 验收：SVG 可正确显示，JSON 参数字段完整

## P4 评估与优化（M9）
- 交付：可视化、报告、参数调优建议
- 验收：报告包含误差、复杂度、耗时三类指标

---

## 9. 验收标准

- 功能完整：九个模块均可执行并产出结果
- 可追溯：每阶段均存在中间文件和统计摘要
- 可解释：最终模型选择有分数依据（AIC/BIC 或等价评分）
- 可复现：同配置多次运行结果波动在可接受范围内

---

## 10. 推荐技术栈

- 图像处理：OpenCV
- 数值计算：NumPy / SciPy
- 可视化：matplotlib
- SVG 输出：svgwrite
- 数据校验：pydantic（可选）

---

## 11. 后续扩展

- 深度学习分割（U-Net）替代传统预处理
- 图优化（节点-边全局能量最小化）
- 面向 CAD/CAM 的 DXF 输出扩展
