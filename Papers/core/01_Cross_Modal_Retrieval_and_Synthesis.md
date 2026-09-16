# Multimodal Relation Extraction with Cross-Modal Retrieval and Synthesis

**Venue:** ACL 2023 Short  
**Authors:** Xuming Hu, Zhijiang Guo, Zhiyang Teng, Irwin King, Philip S. Yu  
**Code link reported in paper:** https://github.com/THU-BPM/MRE

## 1. Research question

原始 sentence-image pair 经常不足以判断复杂关系。能否从外部检索额外的 **textual evidence + visual evidence**，再进行关系判断？

## 2. Core idea

论文把系统拆成两部分：

### Cross-Modal Retrieval

- image / detected object → retrieve textual entities / captions
- sentence → retrieve relevant images

也就是让每个模态去帮助另一个模态补全信息。

### Cross-Modal Synthesis

检索结果并不是直接拼接，而是通过：

1. visual encoder
2. textual encoder
3. cross-modal selection
4. cross-modal consistency

去筛掉不相关 evidence。

## 3. Why it matters

这篇真正重要的不是“用了 retrieval”，而是它很早就清楚指出：

> retrieved evidence 同样会带来 noise。

所以 MRE 的问题逐渐从“多用信息”转向“只用可信信息”。

## 4. Experimental signal

论文在 MNRE 上报告了强于当时多种 text-only / multimodal baseline 的结果；消融也说明 object/image/text evidence、selection 与 consistency 都有贡献。

## 5. Strengths

- 把外部 evidence 引入 MRE，而不是只做原始 pair fusion。
- 视觉与文本双向检索。
- 显式做 evidence selection / consistency。

## 6. Limitations

- retrieval pipeline 成本较高。
- 依赖外部搜索/API，复现环境更复杂。
- 最终仍然是 predefined relation classification。
- retrieval source 与 relation-level evidence 的绑定仍然比较弱。

## 7. What to learn

如果你要研究“视觉证据到底有没有用”，先读这篇，因为它建立了一个很关键的思维：

**evidence acquisition 和 evidence verification 是两回事。**
