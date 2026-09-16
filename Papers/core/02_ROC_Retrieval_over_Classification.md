# Retrieval over Classification: Integrating Relation Semantics for Multimodal Relation Extraction

**Venue:** EMNLP 2025  
**Paper:** https://aclanthology.org/2025.emnlp-main.943/  
**Code:** 未找到公开官方实现（截至 2026-09-16）

## 1. Problem

传统 MRE 用 classifier：

\[
p(r|x) = softmax(W h_{pair})
\]

relation 只是一个 label id，难以表达细粒度关系之间的语义差异。

## 2. Core idea: Retrieval Over Classification (ROC)

ROC 把 relation label 转成自然语言 description，再编码成 relation semantic embedding。

同时构造：

- **Multimodal Entity Pair Encoder**
- **Relational Semantic Encoder**
- **Contrastive Semantic Retrieval**

推理时不是“输出第 k 类”，而是从 relation semantic space 中检索最匹配的关系。

## 3. Structural prior

模型把：

- entity type
- entity position

作为显式先验，帮助缩小候选关系空间。

## 4. Relation description

论文使用 GPT-4o 生成 relation descriptions，并进行人工核验。

这一步很重要：LLM 在这里不是最终 predictor，而是 **relation semantics generator**。

## 5. Strengths

- 明确建模 relation semantics。
- 适合处理语义相近 relation。
- retrieval representation 比纯 label 更可解释。
- 为 zero-shot / open-world extension 留下接口。

## 6. Limitations

- 每个 relation 仍主要对应一个全局 semantic representation。
- relation description 的质量会直接影响 retrieval。
- relation set 仍然是 predefined。
- LLM description 需要质量控制。

## 7. Follow-up questions

自然延伸问题：

- 一个 relation 是否应该有多个 prototype？
- 能否为不同 entity type / visual context 学不同 relation subspace？
- 能否引入 hard-negative descriptions？
- relation description 是否应该 instance-conditioned？
