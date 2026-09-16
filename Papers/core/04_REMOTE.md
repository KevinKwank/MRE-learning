# REMOTE: A Unified Multimodal Relation Extraction Framework with Multilevel Optimal Transport and Mixture-of-Experts

**Venue:** ACM MM 2025  
**Paper:** https://doi.org/10.1145/3746027.3754868  
**Code:** https://github.com/Nikol-coder/REMOTE

## 1. Problem

标准 MNRE 通常抽：

\[
(text\ entity, relation, text\ entity)
\]

MORE 则更关注：

\[
(text\ entity, relation, visual\ object)
\]

现实世界实际上还存在 object-object relation。不同 task 各自训练一个模型，既割裂又重复。

## 2. Unified MRE

REMOTE 定义统一 relation set：

\[
R=\{(e,e,r), (o,o,r), (e,o,r)\}
\]

覆盖：

- text-text
- object-object
- text-object

## 3. Mixture-of-Experts

不同 triplet 的信息需求不同：

- 有的更依赖 text；
- 有的更依赖 image；
- 有的必须跨模态。

MoE 动态选择更相关的 interaction expert，而不是所有样本走同一条融合路径。

## 4. Multilevel Optimal Transport

论文指出 sequential encoder 容易逐层丢失 low-level information，因此通过 multilevel OT 同时保留：

- low-level detail
- high-level semantics

## 5. Dataset contribution

论文同时提出 UMRE，用于覆盖多种 intra-modal / inter-modal triplets。

## 6. Strengths

- 不是单纯换 fusion module，而是改 task scope。
- 统一不同 relation triplet。
- 公开代码。
- MoE 与 task heterogeneity 有直接对应关系。

## 7. Limitations / cost

- pipeline 比标准 MNRE 明显更重。
- visual object detection / annotation 增加数据成本。
- unified setting 带来更复杂的 evaluation 与 error analysis。
