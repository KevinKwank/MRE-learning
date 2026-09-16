# SMIR — Span-based Multi-grained Information Refinement for Joint Multimodal Entity-Relation Extraction

**Venue:** Information Processing & Management, 2026  
**DOI:** https://doi.org/10.1016/j.ipm.2026.104687  
**Code:** 未找到可信官方公开仓库（截至 2026-09-16）

## Problem

JMERE 里常见 word-pair tagging，但实体实际上是 span。只看 word pair 会弱化实体内部词语之间的相互理解。

## Main idea

SMIR 直接构造 **span representation matrix**，并通过多粒度 refinement 改善 span 信息：

1. global sample refinement
2. local vision refinement
3. local span refinement

之后使用 **span-guided cross-modality fusion**，只聚合和当前 span 更相关的视觉/文本信息。

## Interesting detail

论文结合：

- LLM implicit knowledge
- diffusion-based visual refinement
- span-level multimodal fusion

## Why it matters

这是“换基本建模单元”的工作：从 word pair 转向 span。

如果做 JMERE，它比单纯再加一层 attention 更值得研究。
