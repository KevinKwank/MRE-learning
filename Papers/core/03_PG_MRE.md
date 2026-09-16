# Prototype-Guided Multimodal Relation Extraction based on Entity Attributes

**Venue:** AAAI 2025  
**Paper:** https://doi.org/10.1609/aaai.v39i24.34795  
**Repository:** https://github.com/zefanZhang-cn/PG-MRE  
**Code status:** 仓库存在，但目前仍为 “code will be available soon”。

## 1. Motivation

MRE 的输入可能本身就不可靠：

- text 中 entity 语义模糊；
- unseen entity 名称无法理解；
- 图像同类关系的视觉模式变化很大；
- 不同关系又可能长得很像。

## 2. LLM as semantic knowledge provider

模型先让 LLM 为 head / tail entity 生成详细 explanation。

这比直接让 LLM 输出 relation 更稳健，因为 LLM 主要负责补充 **entity semantics**。

## 3. Attribute Prototype Module (APM)

把分散的 entity explanation features 聚成 attribute prototypes：

\[
P_a=\{p^a_1,\dots,p^a_k\}
\]

模型通过 prototype bank 表示细粒度 entity attributes。

## 4. Relation Prototype Module (RPM)

attribute representation 再用于指导 visual features，并形成 relation-centric prototypes。

核心目的：

- 降低 intra-class variance
- 增大 inter-class discriminability

## 5. Strengths

- LLM 被用在“知识补充”而不是完全替代 task model。
- prototype learning 很适合处理类内多样性。
- 对 unseen / ambiguous entities 有明确动机。

## 6. Limitations

- 公开仓库暂无完整代码。
- LLM entity explanations 可能带 hallucination。
- prototype assignment / number 对性能可能敏感。
- 最终仍然偏 classification paradigm。

## 7. Why read after ROC

ROC 强调 **relation description**；PG-MRE 强调 **entity attributes + prototypes**。

把两篇放一起看，会自然想到：

> relation semantics 是否也需要多 prototype，而不是单一 global vector？
