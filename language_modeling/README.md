# **Language Modeling**

[A Study on Neural Network Language Modeling论文笔记](paper_reading/paper_reading.md)

[两种简单语言模型](lm_zh)

在三个小型数据集上效果如下：

| Dataset | Algorithm | Train perplexity | Valid perplexity | Test perplexity |
| :-: | :-: | :-: | :-: | :-: |
| ptb | lstm | 37.208 | 102.808 | 96.895 |
| ptb | gated cnn | 11.771 | 12.875 | 11.978 |
| sanguoyanyi | lstm | 40.031 | 147.971 | 141.901 |
| sanguoyanyi | gated cnn | 1.413 | 1.923 | 1.854 |
| weicheng | lstm | 121.062 | 376.828 | 312.213 |
| weicheng | gated cnn | 2.424 | 10.678 | 8.817 |