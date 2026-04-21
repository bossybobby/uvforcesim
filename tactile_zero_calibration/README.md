# 视触觉传感器真实零点校准算法

本目录实现了项目 Phase 1：

- 文件名元数据解析与数据加载
- 按 `(xr, yr)` 分组
- 分组探索性统计与名义零点分布图
- 基于稠密光流（Farnebäck）的 UV 位移特征提取
- `main.py` 一键运行入口

## 使用方式

```bash
cd tactile_zero_calibration
python main.py --data_dir ./data --output_dir ./results --downsample 0.5 --verbose
```

## 当前输出（Phase 1）

- `results/group_summary.csv`：每组图像数量、Z 范围、|Fz| 范围
- `results/nominal_zero_hist.png`：名义零点 `z0_abs_xy` 分布
- `results/uv_features.csv`：相邻帧计算得到的 UV 特征
- `results/phase1_report.txt`：运行统计和一致性检查（同组名义零点唯一性）

## 下一步（Phase 2）

- 每组 `UV-|Fz|` 三次多项式拟合
- 估计 `|Fz|=0` 对应的 `Z0_estimated`
- `z0_abs_xy vs Z0_estimated` 对比图和偏差统计
