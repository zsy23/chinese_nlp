# ***中文Embedding近年论文***

# ***Joint Learning of Character and Word Embeddi***

## ***CWE模型***
### CWE模型主要基于CBOW模型。
### 符号说明：$C$：中文character集合，$w$：中文词典，$\bold c_i, \bold w_i$分别表示character $c_i\in C$和word $w_i\in W$的向量表示，$K$：上下文窗口大小。
### 上下文词$x_j$表示为:
$$
\bold x_j = \bold w_j\oplus \frac{1}{N_j}\sum_{k=1}^{N_j}\bold c_k
$$ (1)
### $\bold w_j$是$x_j$的词向量，$N_j$是$x_j$中character的数目，$\bold c_k$是第$k$个character的embedding，$\oplus$表示组合操作。$\oplus$可以是加法或者连接两者操作，但实验中发现加法更好。因此，最终$x_j$表示为：
$$
\bold x_j = \frac{1}{2}(\bold w_j + \frac{1}{N_j}\sum_{k=1}^{N_j}\bold c_k)
$$ (2)
## 其中乘$\frac{1}{2}$很重要，因为他保证了组合词和非组合词的一致性。具体的，在target word的negative samping和hierarchical 中softmax中不用character embedding。

## ***Multiple-Prototype Character Embeddings***
### 中文字一般有多个意思，采用一个中文字多个向量表示的方法解决这个问题，每个向量对应一个意思。
### 有如下三种方法来实现：1）Position-based character embeddings；2）Cluster-based character embeddings；3）Nonparametric cluster-based character embeddings。
### ***Position-based character embeddings***
### 一般字在词中的不同位置表示了不同的含义，因此每个字对应三个字向量$(\bold c^B, \bold c^M, \bold c^E)$，分别表示字在词的开始，中间和结尾。
### 因此公式（2）变为如下：
$$
\bold x_j = \frac{1}{2}(\bold w_j + \frac{1}{N_j}(\bold c_1^B+\sum_{k=2}^{N_j-1}\bold c_k^M + c_{N_j}^E))
$$ (3)
### ***Cluster-based Character Embeddings***
### 通过将字根据它所有出现时的上下文进行聚类来形成一个字的多个意思，对于一个字$c$，聚成$N_c$个类，并且每个类训练一个向量表示。
### 对于一个上下文词$x_j=\{c_1,...,c_N\}$, $c_k^{r_k^{max}}$将被用来表示$\bold x_j$，定义函数$S()$为cos相似度，
$$
r_k^{max} = \arg \max_{r_k}S(\bold c_k^{r_k}, \bold v_{content})
$$ (4)
$$
\bold v_{content}=\sum_{t=j-K}^{j+K}\bold x_t=\sum_{t=j-K}^{j+K}\frac{1}{2}(\bold w_t+\frac{1}{N_j}\sum_{c_u\in x_t}\bold c_u^{most})
$$ (5)
### 其中$c_u^{most}$表示上一次训练中用来表示$\bold x_t$的字向量。得到所有字的最优类$R=\{r_1^{max},..., r_{Nj}^{max}\}$后，
$$
\bold x_j=\frac{1}{2}(\bold w_j+\frac{1}{N_j}\sum_{k=1}^{N_j}\bold c_k^{r_k^{max}})
$$ (6)
### 进一步可以把Position-based和Cluster-based结合起来，不同位置的字训练多个向量，这种被称为position-cluster-based character embeddings。
### ***Nonparametric Cluster-based Character Embeddings***
### 上面介绍的方法需要手动定类的数目，因此提出一种方法能够自动定每个字的类的数目。
### $N_{c_k}$是字$c_k$的类的数目，对于词$x_j$中的字$c_k$，他所属的类$r_k$如下确定：
$$
r_k = \begin{cases}
        N_{c_k} + 1, & if\ S(c_k^{r_k}, \bold v_{content}) \lt \lambda\ for\ all\ r_k.\\
        r_k^{max}, & otherwise
    \end{cases}
$$ (7)

## ***训练的词的选择***
### 有许多字并没有在意思上是字的组合，例如1）single-morpheme multi-character words，琵琶，徘徊等，这些词中的字很少在其他词中出现；2）音译词，沙发（sofa），巧克力（chocolate）等，它们主要是音的组合；3）许多实体词，比如人名，地名和机构名。
## 为了防止用上述方法来推断非组合词，手动建立了一个音译词的词表，并且通过词性标注来识别出实体词，single-morpheme multi-character words对于模型影响很小，因为它们的字一般只在这些词中出现，所以不做特殊处理。

## ***数据集和实验设置***
### 选择人民日报作为训练语料。向量维度200，上下文窗口5，同时用hierarchical softmax和10个词的negative sampling。进行上述的选词操作，使用预训练的字向量。CBOW,Skip-gram和GloVe作为baseline，使用同样的向量维度和默认参数设置。主要通过词相似和词analogy做评价。

## ***词相似***
### 利用wordsim-240和wordsim-296作为评测数据集，其中240中有7对word是oov的，296中有16对word是oov的。
### 计算预测得分和人工得分的Spearman correlation来进行比较。得分通过计算两个词的cos相似度得到。
### CWE中词向量根据公式（1）得到，对于包含oov的词对，baseline方法将其相似度设为0，CWE模型通过它们的字向量得到embedding。
## ![cwe_wordsim](cwe_wordsim.png)
### 1）CWE及其变体效果好于baseline；2）cluster-based（+L,+LP和+N）好于CWE，表明一个字对应多个向量很重要，而位置信息（+P）则不是很重要；3）对于wordsim-240,oov并没有明显影响效果，可能这些词对本身就关系不大。4）对于wordsim-296，oov明显影响baseline效果，但不影响CWE。比如“老虎”和“美洲虎”，baseline设为0明显太低了，而CWE根据character得到更好的结果。5）但是添加字向量有时也会带来副作用，因为CWE可能会因为两个词共有某些字而误判它们的相似度，比如“肥皂剧”和“歌剧”，“电话”和“回话”等。

## ***词analogy***
### 建立了包含1126对analogies的中文数据库，共有三类：1）国家和首都（687组）；2）州／省和城市（175组）；3）家庭相关的词（240组）
## ![cwe_analogy](cwe_analogy.png)
### 1）大多数情况添加了CWE的版本好于原始版本，说明字向量的重要性；2）CWE同样能提高非组合词的效果，比如国家首都中，基本都是实体词，CWE的版本同样更好；3）正如之前文章所述，Skip-gram和GloVe在analogy上比CBOW好

## ***语料大小的影响***
## ![cwe_corpus_size](cwe_corpus_size.png)
### 在语料小的时候，CWE比CBOW效果更好。

## ***Case Study***
## ![cwe_case_study](cwe_case_study.png)
### 上图结果来自CWE+P和CWE+L，其中cluster数目为2。对于每个字，我们列出和它cos相似度最大的词。$x_j$使用公式（4）作为词向量。
### 大多数情况下，CWE+P和CWE+L都能很好区分出字的不同意思，但CWE+L更加鲁棒，基本都很好。

## ***代码地址***
### https://github.com/Leonard-Xu/CWE
#

# ***Component-Enhanced Chinese Character Embeddings***

## ***Component-Enhanced Character Embeddings***
### 该方法主要针对训练字向量。
### 中文字中偏旁部首一般包含了丰富的语义信息，可以用来训练更加好的词向量。手工建立了所有中文字的component list，偏旁部首作为最重要的组件。
### 符号介绍：$D=\{z_1,...,z_N\}$表示在字表$V$上的包含$N$个字的语料。$z$表示中文字，$c$表示上下文字，$e$表示component list，$K$表示向量维度，$T$表示上下文窗口大小，$M$表示每个字考虑的组件个数，$|V|$表示字表的大小。
### 开发了两种component-enhanced character embedding模型，charCBOW和charSkpGram，分别基于CBOW和SkipGram。
### 比如charCBOW根据上下文窗口中的字和相应组件来预测当前字，最大化如下log likelihood：
$$
L = \sum_{z_i^n\in D}\log p(z_i|h_i),
$$
$$
h_i = concatenation(c_{i-T},e_{i-T},...,c_{i+T},e_{i+T})
$$

## ***评价方法***
### 通过词相似和文本分类作为评测任务。用哈工大信息检索研究中心同义词词林扩展版(http://ir.hit.edu.cn/demo/ltp/Sharing_Plan.htm)作为词相似数据集，腾讯新闻(http://www.datatang.com/data/44341 好像失效了)标题作为文本分类数据集。
### 偏旁部首通过在线新华字典(http://xh.5156edu.com)提取，其余组件通过在新华字典中匹配“从（from）+ X”的模式提取，表示该字含有X这个组件。通过在Hong Kong Com- puter Chinese Basic Component Reference (https://www.ogcio.gov.hk/tc/our_work/business/tech_promotion/ccli/cliac/glyphs_guidelines.html)中匹配“X is only seen”模式来扩充component list。每个字最多取两个组件来构建字向量。
### 使用中文维基(http://download.wikipedia.com/zhwiki/)作为训练语料。预处理时，去处纯数字和非中文字符，并且忽视出现次数少于10的词。设置上下文窗口为2，取5个negative sampling，向量维度取50。
## ![cece_result](cece_result.png)
### 词相似用Spearman’s rank correlation作为评价标准。文本分类用标题中字向量的平均作为分类器的输入。和原文描述一致，CBOW比SkipGram差。总的来说char-*模型更好。
#


# ***Improve Chinese Word Embeddings by Exploiting Internal Structure***

## ***SCWE模型***
### 主要受CWE模型启发，通过多语言的方法计算一个词中每个字对于语义的贡献。

## ***Obtain translations of Chinese words and characters***
### 对中文训练语料进行分词和词性标注，其中实体词作为非组合词，计数字在词中出现的次数，其中有些词由很少在其他词中出现的字组成，它们被当作single-morpheme multi-character words，作为非组合词。
### 通过在线翻译工具将中文词和字翻译成英文，对于非组合词，它们不用翻译。

## ***Perform Chinese character sense disambiguation***
### 首先用CBOW在英文语料上训练英文词向量。
## ![scwe_zh2en](scwe_zh2en.png)
### 如上图，“乐”字有些意思间差别很小，它们只是词性不同。中文字有些词性不同但是表达了相同的意思，因此它们被合为一个语义意思。用$Sim()$函数衡量中文词和字之间的相似度，这里使用cos距离。中文字$c$的第$i$和$j$个意思分别为$c^i$和$c^j$。它们的相似度定义如下：
$$
Sim(c^i, c^j)=\max(\cos(v_{x_m},v_{x_n}))
$$ 
$$
s.t.\ x_m\in Trans(c^i),x_n\in Trans(c^j),x_m,x_n\notin stop\_words(en)
$$ (1)
### 其中$Trans(c^i)$表示$c^i$的英文翻译,$stop\_words(en)$表示英文中的停止词。比如上图中的词“音乐”，$c_2$表示第二个字“乐”，$Trans(c_2^3)$表示“乐”的第三种翻译，即{pleasure, enjoyment}，因此$x_m$可以是pleasure或者enjoyment。
### 如果$Sim(c^i,c^j)$大于阈值$\delta$，他们将被合并为一个语义。简单点，我们合并相应英文翻译的词集。我们也可以对于所有$x_m$词向量求平均来表示相似度，但是实验中取最大值效果更好。
### 中文中组合词的意思和其中某个字的意思会很接近，而音译词词的意思和每个字都不同，比如组合词“音乐”和音译词“沙发”。因此，如果$\max(Sim(x_t,c_k))\gt \lambda, c_k\in x_t$，那么$x_t$将被视为组合词，被纳入COMP集合中。这样对于组合词，我们有这样的集合
$$
F=\{(x_t,s_t,n_t)|x_t\in COMP\}
$$ (2)
### 其中
$$
s_t=\{Sim(x_t,c_k)|c_k\in x_t\}
$$
$$
n_t=\{\max_i Sim(x_t,c_k^i)|c_k\in x_t\}
$$ (3)
### 比如“音乐”定义为("音乐"，{Sim("音乐"，“音”)，Sim("音乐"，“乐”)}，{1，1})

## ***Learn word and character vectors with SCWE***
### 对于词$x_t$，在SCWE模型中
$$
\hat{v}_{x_t}=\frac{1}{2}\{v_{x_t}+\frac{1}{N_t}\sum_{k=1}^{N_t}Sim(x_t,c_k)v_{c_k}\}
$$ (4)
## 提出multiple-prototype character embeddings，称为SCWE+M模型利用F集合中每个元素最后一个属性来给不同意思的字分配不同的字向量，这样在SCWE+M中
$$
\hat{v}_{x_t}=\frac{1}{2}\{v_{x_t}+\frac{1}{N_t}\sum_{k=1}^{N_t}Sim(x_t,c_k)v_{c_k^i}\}
$$ (5)

## ***实验设置***
### 利用英文维基(http://download.wikipedia.com/enwiki/)训练CBOW的英文词向量，用中文维基(http://download.wikipedia.com/zhwiki/)作为语料,把纯数字和非中文字符去掉，用ANSJ(https://github.com/NLPchina/ansj_seg)做分词，词性标注和命名体识别。用ICIBA(http://www.iciba.com/)作为英中翻译工具。上下文窗口大小为5，词／字向量维度为100，阈值$\delta$和$\lambda$设为0.5和0.4。

## ***词相似***
### wordsim-240和wordsim-296(Semeval-2012 task 4: evaluating chinese word similarity)作为数据集，Spearman’s rank correlation作为评价方法。
## ![scwe_wordsim](scwe_wordsim.png)
### SCWE和SCWE+M更好，而且在wordsim-296上SCWE和baseline差不多，但是SCWE+M更好，表示multiple-prototype是有用的。

## ***文本分类***
### 用复旦语料（http://www.datatang.com/data/44139）作为数据集。为了防止不平衡，选取十个类别，分为两大组。一组为Fudan-large，每个类别下文章数多于1000，另一组为Fudan-small，每个类别下文章数小于100。每个类别80%用作训练，剩下的用来测试。
###同样把纯数字和中文字符去掉，然后用ANSJ做处理。每篇文章的出版信息被删掉。通过对文章中词的词向量求平均来表示这篇文章，然后用LIBLINEAR训练分类器。
## ![scwe_text_classify](scwe_text_classify.png)
### SCWE更好。

## ***Multiple Prototype of Chinese Charaters***
### 通过PCA可视化词／字向量，然后取“道”和“光”的三个意思和每个意思最接近的词。
## ![scwe_pca](scwe_pca.png)
## ![scwe_nearest](scwe_nearest.png)
### 选一些多义的字，用在线新华字典(http://xh.5156edu.com/)作为区分词中多义字的标准答案。每个词根据词典中的解释有一个序号，然后我们用KNN分类起来验证所有方法。结果如下
## ![scwe_ambiguous](scwe_ambiguous.png)
### SCWE最好

## ***词向量的定量分析***
### 取两个词作为例子，列出它们最近的词，分析CWE和SCWE。
## ![scwe_quan](scwe_quan.png)
### 大多数情况CWE和SCWE返回的词都是一样的。但是CWE中每个字贡献一样，就会导致例如“青蛙”中，有“青”字的会被认为相似，而SCWE不会这样。

## ***参数分析***
### 探讨参数组合词相似阈值$\lambda$和字义区分阈值$\delta$的取值。
### 构建了一个音译词词表，用来评测$\lambda$的取值。
## ![scwe_lambda](scwe_lambda.png)
### $\lambda=0.4$最好
### 利用Multiple Prototype of Chinese Charaters中的数据集，在不同$\delta$下算多义字区分的准确率。
## ![scwe_delta](scwe_delta.png)
### $\delta=0.5$最好

## ***代码链接***
### https://github.com/JianXu123/SCWE
#


# ***How to Generate a Good Word Embedding?***