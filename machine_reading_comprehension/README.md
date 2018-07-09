# **Machine Reading Comprehension**

[2018机器阅读理解技术竞赛第十名代码,Log,模型结果以及验证集,测试集结果](mrc2018/)

[技术报告](hellobot_report.docx)

主要使用了BiDAF模型，首先采用了字向量，词向量，词性向量和Exact Match多特征，然后建立知识库进行段落选择，接着添加了多文章排序任务联合学习，最后的后处理通过多答案投票机制选择最佳答案。
