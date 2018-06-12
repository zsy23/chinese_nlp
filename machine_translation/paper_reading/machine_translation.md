# **近年机器翻译论文笔记**

## **[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)**

序列学习的开山之作，也是神经机器翻译的基本模型。

基本策略是把输入序列通过RNN映射到一个固定长度的向量，然后用另一个RNN从这个固定长度的向量中解码出目标序列，一般称前一个RNN为编码器，后一个RNN为解码器。由于基础RNN很难学习长期的依赖，所以一般采用LSTM。

Seq2Seq模型的训练目标是预测条件概率$p(y_1,...,y_{T'}|x_1,...,x_T)$，其中$x_1,...,x_T$是输入序列，$y_1,...,y_{T'}$是输出序列，其中两个序列的长度$T$和$T'$不同。LSTM需要首先计算输入序列的固定维度的表示$v$，这里用LSTM最后一个隐层状态表示$v$，然后通过另一个LSTM来计算输出序列的概率，其中这个LSTM初始隐层状态设为$v$，公式如下：

$$
p(y_1,...,y_{T'}|x_1,...,x_T)=\prod_{t=1}^{T'}p(y_t|v,y1,...,y_{t-1})
$$ (1)

其中序列中每一个元素的概率用在整个词表上的softmax表示。另外，我们需要设置每个序列以特殊符号“\<EOS>”结尾，这样使得模型可以处理任意长度的序列。整个模型架构如下图所示：

![seq2seq](seq2seq.png)

实际使用的模型有三个注意点：1. 编码器和解码器使用不同的LSTM，这不仅能够增强模型的能力，而且可以同时训练多个语言对。2. 深层LSTM效果比浅层LSTM好，论文中采用了4层LSTM。3. 作者发现将输入序列逆向后在输入模型十分重要。比如$a,b,c$不是对应于$\alpha, \beta, \gamma$，而是要求模型将$c,b,a$映射到$\alpha, \beta, \gamma$，其中$\alpha, \beta, \gamma$是$a, b, c$的翻译。这样的好处是，使得$a$和$\alpha$，$b$和$\beta$更加接近，使得更容易在输入序列和输出序列中建立联系。作者发现这个简单的操作能够大大提升效果。

## **[NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/pdf/1409.0473.pdf)**

作者提出基本编码器-解码器的神经机器翻译架构将输入序列编码成一个固定长度的向量这一做法，是效果提升的瓶颈所在，因此提出让模型自己软搜索和下一个目标词相关的输入序列中的词，而不是硬性的进行词对齐。这其实就是attention注意力模型在机器翻译上的应用，并且作者展示了模型学到的软对齐符合人类常识。

模型整体采用了双向RNN作为编码器，解码器在解码时需要同时搜索输入序列进行软对齐。

首先介绍解码器的通用架构。定义条件概率如下：

$$
p(y_i|y_1,...,y_{i-1},\bold x)=g(y_{i-1}, s_i, c_i)
$$ (1)

其中$s_i$是$i$时刻RNN的隐层状态，计算如下：

$$
s_i=f(s_{i-1}, y_{i-1}, c_i)
$$ (2)

和传统目标序列条件概率不同，目标词$y_i$依赖一个称之为上下文向量的$c_i$。上下文向量$c_i$又依赖annotation序列$(h_1,...,h_{T_x})$，这个annotations序列序列由编码器映射输入序列得到，其中每个annotation$h_i$包含整个输入序列的信息，尤其关注输入序列中第$i$个词周围的信息，annotation的生成在下面介绍。上下文向量$c_i$由annotations的加权求和得到：

$$
c_i = \sum_{j=1}^{T_x}\alpha_{ij}h_j
$$ (3)

其中每个annotation的权重计算如下：

$$
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})}
$$ (4)

其中

$$
e_{ij} = a(s_{i-1}, h_j)
$$ (5)

$e_{ij}$称为对齐模型，评价输出序列中第$i$个词和输入序列中第$j$个词有多匹配。通常对齐模型$a$使用一个前馈神经网络，和模型的其他部分一起参与训练。

$\alpha_{ij}$或者$e_{ij}$表示了annotation$h_i$对于前一个隐层状态$s_{i-1}$决定当前隐层状态$s_i$和输出$y_i$的重要性。直观地，这实现了一个带有注意力的解码器，解码器能够决定需要注意什么。通过注意力机制，编码器不需要将输入序列的所有信息包含到一个长度固定的向量中，输入序列的信息分散到annotation序列中，解码器可以自己选择要使用哪些信息去解码。

下面介绍编码器的通用架构。传统编码器只能够按顺序进行编码，而annotation不仅需要包含之前的信息，也需要包含之后的信息，因此使用双向RNN来做编码。双向RNN中前向RNN按照正常顺序读入输入序列，得到前向隐层状态$(\overrightarrow{h_1},...,\overrightarrow{h_{T_x}})$，后向RNN按照逆序读入输入序列，得到后向隐层状态$(\overleftarrow{h_1},...,\overleftarrow{h_{T_x}})$，最后将前向隐层状态和后向隐层状态连接起来得到词的annotation，$h_j=[\overrightarrow{h_j}^T;\overleftarrow{h_j}^T]^T$，这样annotation $h_j$同时包含当前词之前和之后的信息，并且由于RNN更倾向于表示最近的信息，annotation $h_j$也会更加关注周围词的信息。

整体架构如下图所示：

![joint_align_translate](joint_align_translate.png)

## **[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)**

由于还没有基于注意力的神经机器翻译模型，作者提出两个简单有效的注意力机制：全局注意力，总是考虑输入序列中的所有词；局部注意力，只考虑输入序列中的一部分词。分别如下两图所示：

![global_attention](global_attention.png)

![local_attention](local_attention.png)

这两种模型的共同之处如下，首先都是把解码器stacking LSTM的顶层隐层状态$\bold{h_t}$作为输入，目标是推出上下文向量$\bold{c_t}$，能够捕捉相关源端的信息，用来预测当前的目标词$y_t$。上下文向量$\bold{c_t}$的推导两种方法不一样，之后再介绍。在得到$\bold{c_t}$后，将其和$\bold{h_t}$连接起来得到带有注意力的隐层状态$\tilde{\bold{h_t}}$：

$$
\tilde{\bold{h_t}}=\tanh(\bold{W_c}[\bold{c_t};\bold{h_t}])
$$ (1)

然后$\tilde{\bold{h_t}}$被输入softmax层来预测下一个目标词：

$$
p(y_t|y_{\lt{t}},x)=softmax(\bold{W_s}\tilde{\bold{h_t}})
$$ (2)

下面介绍全局注意力模型。一个和输入序列一样长的对齐向量$\bold{a_t}$用来比较当前目标隐层状态$\bold{h_t}$和所有源隐层状态$\bar{\bold{h_s}}$：

$$
\bold{a_t}(s)=align(\bold{h_t},\bar{\bold{h_s}})=\frac{\exp(score(\bold{h_t},\bar{\bold{h_s}}))}{\sum_{s'}\exp(score(\bold{h_t},\bar{\bold{h_{s'}}}))}
$$ (3)

这里的$score$被称为基于内容的函数，作者提出三种函数：

$$
score(\bold{h_t},\bar{\bold{h_s}})=\begin{cases}
\bold{h_t}^T\bar{\bold{h_s}}, & dot \\
\bold{h_t}^T\bold{W_a}\bar{\bold{h_s}}, & general \\
\bold{v_a}^T\tanh(\bold{W_a}[\bold{h_t};\bar{\bold{h_s}}]), & concat
\end{cases}
$$ (4)

除此之外，作者还提出一种基于位置的函数，只使用目标隐层状态来计算$\bold{a_t}$：

$$
\bold{a_t}=softmax(\bold{W_a}\bold{h_t})
$$ (5)

这种方法所有对齐向量$\bold{a_t}$一样长，对于短的输入句子，只使用$\bold{a_t}$前面的部分，对于长的输入句子，忽视句子的末尾部分。

之后将对齐向量作为权重，上下文向量$\bold{c_t}$是所有源隐层向量的加权平均。

下面介绍局部注意力模型。全局注意力模型需要注意输入句子中的所有词，这使得计算代价太大，并且实际中无法翻译长句子，因此提出局部注意力模型，只关注输入序列的一部分词。对于时刻$t$的目标词，首先生成一个对齐位置$p_t$，之后上下文向量$\bold{c_t}$是一个窗口内源隐层状态的加权平均，这个窗口为$[p_t-D,p_t+D]$，$D$是根据经验选择的。不同于全局注意力模型，此时局部对齐向量$\bold{a_t}$是固定长度的($\in R^{2D+1}$)，下面介绍该模型的两个变种：

单调对齐。简单设置$p_t=t$,假设目标词和源词是近似单调对齐的，这时对齐向量$\bold{a_t}$如公式(3)定义。

预测对齐。通过模型预测对齐位置：

$$
p_t=S \cdot sigmoid(\bold{v_p}^T\tanh(\bold{W_p}\bold{h_t}))
$$ (6)

其中$\bold{W_p},\bold{v_p}$是模型参数，$S$是输入序列长度，$sigmoid$函数导致$p_t\in[0,S]$。为了突出在$p_t$周围对齐，以中心在$p_t$的高斯分布进行采样，此时的对齐向量定义如下：

$$
\bold{a_t}(s)=align(\bold{h_t},\bar{\bold{h_s}})\exp(-\frac{(s-p_t)^2}{2\sigma^2})
$$ (7)

使用公式(3)描述的$align$函数，并且标准差设为$\sigma=\frac{D}{2}$，其中$p_t$是一个实数，$s$是以$p_t$为中心的窗口内的整数。

上面提到的全局注意力和局部注意力都是独立计算的，也是次优的，标准的翻译模型里需要维护一个覆盖率来追踪哪些输入句子里的词已经被翻译过了，因此神经翻译模型里也应该考虑之前的对齐信息。作者因此又提出input-feeding方法，将注意力向量$\tilde{\bold{h_t}}$和下一时刻的输入连接起来一起作为输入，如下图所示：

![input-feeding](input-feeding.png)

这样有两个好处：1. 可以考虑之前的对齐信息；2. 创造了一个水平垂直方向上更深的网络结构。

[代码和资料地址](https://nlp.stanford.edu/projects/nmt/)

## **[On Using Very Large Target Vocabulary for Neural Machine Translation](https://arxiv.org/pdf/1412.2007.pdf)**

神经机器翻译模型在处理大词表时，训练和解码复杂度都会急剧增加。作者提出基于importance sampling的方法，能够在不增加训练复杂度的情况下使用大词表，解码也可以仅使用大词表的一部分来提高效率。

计算下一个目标词softmax概率如下：

$$
p(y_t|y_{\lt t},x)=\frac{1}{Z}\exp(\bold{w_t}^T\phi (y_{t-1},z_t,c_t)+b_t)
$$ (1)

首先考虑公式(1)对数概率的梯度，包括正项和负项两部分：

$$
\nabla \log p(y_t|y_{\lt t},x)=\nabla \varepsilon(y_t)-\sum_{k:y_t\in V}p(y_k|y_{\lt t}, x)\nabla (y_k)
$$ (2)

其中能量$\varepsilon$定义为：

$$
\varepsilon (y_j)=\bold{w_j}^T\phi (y_{j-1},z_j,c_j)+b_j
$$ (3)

梯度的第二项（负项）就是能量的期望梯度：

$$
E_P[\nabla \varepsilon(y)]
$$ (4)

其中$P$表示$p(y|y_{\lt t}, x)$。

主要策略是通过importance sampling，用小部分采样去估计这个期望（梯度的负项）。给定一个分布$Q$和$Q$中的一组样本$V'$，预测公式(4)如下：

$$
E_P[\nabla \varepsilon(y)]\approx \sum_{k:y_k\in V'}\frac{w_k}{\sum_{k':y_{k'}\in V'}}\nabla \varepsilon(y_k)
$$ (5)

其中

$$
w_k=\exp(\varepsilon(y_k)-\log Q(y_k))
$$ (6)

这个方法能够在训练过程中只用大词表的一小部分去计算归一化项，从而大大降低参数更新的计算复杂度。

尽管这个方法大大降低计算复杂度，但是直接使用这个方法并不能保证每次更新的包含多个目标词的句子对所使用的参数数量是可控的，这在使用GPU这样的小内存设备时尤为麻烦。因此，实际中将训练语料分成若干份，每一份在训练前定义大词表的一个子集$V'$。在训练开始前，我们顺序的检查训练语料中的每一个目标句子，然后累计唯一目标词直到达到数量预定的阈值$\tau$。这个累计的词表就会用作这部分训练语料的词表，重复这个过程直到训练集结束。定义第$i$部分训练语料的词表为$V_i'$。

对于每一部分语料有一个预知的分布$Q_i$，$Q_i$对于$V_i'$中每一个词的概率相同，其他词的概率为0:

$$
Q_i(y_k)=\begin{cases}
\frac{1}{|V_i'|}, & if\ y_t\in V_i' \\
0, & otherwise
\end{cases}
$$ (7)

这个分布把公式(5)(6)中的修正项给去掉$-\log Q(y_k)$，使得提出的方法能够正确估计公式(1)的概率为：

$$
p(y_t|y_{\lt t},x)=\frac{\exp(\bold{w_t}^T\phi (y_{t-1},z_t,c_t)+b_t)}{\sum_{k:y_k\in V'}\exp(\bold{w_k}^T\phi(y_{t-1},z_t,c_t)+b_t)}
$$ (8)

值得注意的是使用的分布$Q$使得估计有偏差。

模型训练好后，可以用完整词表去解码，这样更加准确，但也更慢。因此解码时我们也可以只用词表的一小部分，但和训练时候的区别是，此时并不能直接吧把正确的目标词作为词表。最简单的做法是选取最常见的$K$个词，但是这样就不符合用大词表去训练模型的本意了。因此，可以使用现有的词对齐模型在训练语料上去对齐源词和目标词，然后建立字典。有了这个字典后，对于每一个输入句子，把最常见的$K$个词和字典中每个源词对应的$K'$个目标词构成目标词集合。$K$和$K'$的选择根据内存限制和效果要求，称构建出来的这个目标词集合为候选集。

[代码地址](https://github.com/sebastien-j/LV_groundhog)

## [Addressing the Rare Word Problem in Neural Machine Translation](https://arxiv.org/pdf/1410.8206.pdf)

神经机器翻译一个很大的缺点是不能够处理罕见词，一般只有一个很小的词表，并且用一个unk符号表示所有不在字典中的词。作者提出一个有效的方法解决这个问题，训练模型时同时输出词对齐结果，使得系统在目标句子中遇到OOV时能够定位源句子中与其对齐的词，这个信息在之后的后处理过程中用来根据字典翻译目标句子中的OOV。

作者提出在训练模型时能够追踪目标句子里的OOV在源句子中的位置，从而解决罕见词问题。如果知道了目标句子中OOV在源句子里对齐的词，可以在后处理时，把unk换成源句子中对应词的翻译或者这个词本身。

作者提出三种策略，能应用到任何机器翻译模型中。首先一个无监督的对齐器用来做对齐，然后利用这个对齐信息构建字典，用来在后处理过程中进行翻译。如果一个词没有出现在字典中，那么就把这个词直接拷贝到目标句子中。

首先介绍拷贝模型。这个方法使用多个unk符号来表示源语言和目标语言中的罕见词，而不是只用一个unk符号。把源句子中的OOV分别注释为unk1,unk2,unk3...相同的OOV被赋予相同的unk符号。目标句子中的OOV注释方法如下：目标语言中的OOV和源语言中的OOV对齐时，赋予源语言中OOV的unk符号（所以被称为拷贝模型），如果目标语言中OOV和源语言不对齐活着和不是OOV的词对齐，那么源语言中的OOV被赋予特殊的$unk_{\emptyset}$。如下图：

![copyable](copyable.png)

下面介绍PosAll模型。拷贝模型不能够处理目标语言中OOV和源语言中已知词对齐的情况，而这种情况会频繁出现在源语言词表大于目标语言词表时。这就需要能够建立源句子和目标句子间的完全对齐关系。

具体的，只使用一个unk符号，但是在目标语言端，需要在每一个词后面插入一个位置符号$p_d$，这里$d$表示相对位置关系$(d=-7,...,-1,0,1,...,7)$，表示目标语言中位置$j$处的词和源语言中位置$i=j-d$的词对齐，距离太远的对齐词被认为是不对齐的，这些词被注释为特殊null符号$p_n$。如下图：

![PosAll](PosAll.png)

最后介绍PosUnk模型。PosAll模型的缺点在于它使得输入序列的长度加倍了，这使得学习更加困难，也更慢。由于我们的后处理只关心OOV词，所以可以只注释OOV词。PosUnk模型使用$unkpos_d(d=-7,...7)$来标注OOV词，并且$d$表示和对齐词的相对位置关系，同样使用$unkpos_{\emptyset}$表示不对齐的OOV，用unk注释源句子中的所有OOV。如下图：

![PosUnk](PosUnk.png)

## **[Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/pdf/1511.06709.pdf)**

神经机器翻译传统只使用双语语料，但是目标端的单语语料在传统统计翻译模型中能够提升翻译流畅度，作者探究如何在神经机器翻译中使用单语语料。之前的做法是将神经机器翻译模型和语言模型的结合起来，但是作者发现目前编码起啊解码器模型完全可以学到和语言模型一样的信息，并且不需要改变目前的模型架构。通过自动的后向翻译，可以将单语语料构建出一个合成的双语语料。

作者提出两种策略：1. 将目标单语语料和源语言的空句子组合成双语语料。2. 通过将目标单语语料翻译成源语言后，组成双语语料，这种方法作者称为后向翻译(back-translation)。

首先介绍用空句子构成双语语料。这种方法将目标语言的单语语料和源语言的空句子组合构成合成的双语语料，这样上下文向量$c_i$就不包含任何信息，模型完全通过前一个目标词预测当前目标词。这个方法和dropout的思想很像，也可以当作是一种多任务学习模型，当源句子知道时就是翻译任务，当源句子不知道时就是语言模型任务。

训练时把双语语料和单语语料1比1结合起来，并且随机打乱。当处理单语语料时把源句子设为单个\<null>词语，这样就可以用同一个模型同时训练单语语料和双语语料，当处理单语语料时，固定住编码器和注意力模型的参数。但存在一个问题，不能任意增加单语语料的比例，如果单语语料太大，网络将很难学习。

下面介绍后向翻译构成双语语料。用一个训练好的目标到源的机器翻译模型去翻译目标端的单语语料，然后得到的源语言句子和目标端单语语料一起构成合成的双语语料。这样在训练时将双语语料和合成双语语料混合起来，并且不作区分，参数也不需要固定。

## **[Modeling Coverage for Neural Machine Translation](https://arxiv.org/pdf/1601.04811.pdf)**

传统基于注意力的神经机器翻译会忽略之前的对齐信息，导致过度翻译或者欠翻译。为了解决这个问题，作者提出基于覆盖率的神经机器翻译，通过维护一个覆盖率向量来追踪对齐历史注意力信息。覆盖率向量被输进注意力模型，帮助调整将来的注意力，使得神经机器翻译模型多注意没有翻译到的词。

传统神经机器翻译模型不能够判断一个词是否被翻译，从而导致有些词被多次翻译，而有些词却没有被翻译。直接建模覆盖率比较困难，但是可以通过在解码过程中追踪注意力信息来缓解上述问题。最自然的方式就是给每个词的annotation（$h_j$）添加一个覆盖率向量，这个向量初始化为0向量，但是在每次注意力模型后进行更新。这个覆盖率向量被一起输入进注意力模型，帮着调整将来的注意力，使得模型能够更多关注没有被翻译的词。具体结构如下图所示：

![coverage](coverage.png)

正式地，覆盖率模型定义如下：

$$
C_{i,j} = g_{update}(C_{i-1,j},\alpha_{i,j},\Phi(\bold{h_j}),\Psi)
$$ (1)

其中$g_{update}$是解码过程中时刻$i$新注意力$\alpha_{i,j}$产生后更新$C_{i,j}$的函数，$C_{i,j}$是$d$维覆盖率向量，包含了$\bold{h_j}$直到时刻$i$的所有注意力历史，$\Phi(\bold{h_j})$是词相关的特征，$\Psi$是不同覆盖率模型中的辅助输入。

公式(1)是一个通用覆盖率模型的架构，下面将介绍几种具体实现。

首先是基于语言学的覆盖率模型。神经机器翻译中，语言学上的覆盖率指某个词被翻译了百分之多少（软覆盖率），神经机器翻译中每一个目标词是由所有源词在$\alpha_{i,j}$的概率下翻译出来的，换句话说，每一个源词都参与到所有目标词的翻译中，并且时刻$i$参与到目标词$y_j$的概率为$\alpha_{i,j}$。

使用一个标量（$d=1$）来表示每一个源词语言学上的覆盖率，并且用累加操作作为$g_{update}$。初始的覆盖率为0，然后迭代的累加每次注意力模型生成的对齐概率，但是每一次的对齐概率都是由不同的上下文相关权重所归一化的。这样时刻$i$源词$x_j$的覆盖率计算如下：

$$
C_{i,j}=C_{i-1,j}+\frac{1}{\Phi_{j}}\alpha_{i,j}=\frac{1}{\Phi_j}\sum_{k=1}^i\alpha_{k,j}
$$ (2)

其中$\Phi_j$是预定义的权重，表示源词$x_j$预计生成多少个目标词。简单的做法是对于所有源词固定$\Phi=1$，但是实际翻译中每一个词对于最终的翻译结果贡献是不一样的，所以需要给每一个源词赋于不同的$\Phi_j$，理想结果是$\Phi_j=\sum_{i=1}^I\alpha_{i,j}$，其中$I$是解码的总时长。但是这个$\Phi_j$值在解码前是不得而知的。

为了预测$\Phi_j$，作者介绍了fertility的概念。源词$x_j$的fertility指他会生成多少个目标词，作者提出计算fertility $\Phi_j$如下：

$$
\Phi_j=G(x_j|\bold x)=N\cdot \sigma(U_f\bold{h_j})
$$ (3)

其中$N\in R$，是一个预定义的常数，表示一个源词能产生的目标词的最多个数，$\sigma(\cdot)$是一个逻辑斯蒂函数，$U_f\in R^{1\times 2n}$是权重矩阵。这里使用$\bold{h_j}$表示$(x_j|\bold x)$，因为$\bold{h_j}$包含了整个输入序列的信息，并且着重$x_j$周围的信息。因为$\Phi_j$不与$i$有关，所以可以在解码前进行预计算从而降低计算代价。

下面介绍基于神经网络的覆盖率模型。此时$C_{i,j}$是一个向量（$d\gt 1$），$g_{update}(\cdot)$是一个神经网络，实际使用RNN模型，如下图所示：

![RNN_coverage](RNN_coverage.png)

此时覆盖率计算如下：

$$
C_{i,j}=f(C_{i-1,j},\alpha_{i,j},\bold{h_j},\bold{t_{i-1}})
$$ (4)

其中$f(\cdot)$是一个非线性激活函数，$\bold{t_{i-1}}$是辅助输入，用来编码之前的翻译信息，即解码器的隐层状态。$f(\cdot)$可以是简单的非线性激活函数$\tanh$或者是门函数，作者发现使用门函数能够捕捉长距离的依赖，因此采用了GRU模型。

最后，在时刻$i$解码时，时刻$i-1$的覆盖率也会被一起输入进注意力模型，具体如下：

$$
e_{i,j}=a(\bold{t_{i-1}},\bold{h_j},C_{i-1,j})=v_a^T\tanh(W_a\bold{t_{i-1}}+U_a\bold{h_j}+V_aC_{i-1,j})
$$ (5)

其中$V_a\in R^{n\times d}$是注意力的权重矩阵，$n,d$分别是隐层维度和覆盖率维度。

[代码地址](https://github.com/tuzhaopeng/NMT-Coverage)

## **[Dual Learning for Machine Translation](https://arxiv.org/pdf/1611.00179.pdf)**

传统机器翻译需要大量双语语料进行训练，但是人工标注的成本又很高。为了解决训练数据的问题，作者提出一种对偶学习机制，使得神经机器翻译模型能够通过一个对偶学习的游戏自动从没有标注的语料中进行学习。一般机器翻译都可以分为源到目标和目标到源两个方向的翻译任务，主任务和对偶任务可以构成一个闭环，从而可以不借助人工标注的情况下产生有用的反馈信息从而训练翻译模型。在对偶学习机制中，使用一个agent表示主任务，另一个agent表示对偶任务，然后它们通过增强学习互相教对方学习。根据这一过程产生的反馈信息，可以迭代更新两个模型直到收敛作者称这种方法为dual-NMT。

考虑两个单语语料$D_A,D_B$，包含语言$A,B$的句子，但是这两个语料是不对齐的，甚至是不相关的。假设有两个比较弱的翻译模型，分别可以进行从$A$到$B$和从$B$到$A$的翻译，目标是只利用单语语料来提升两个模型的精度。假设$D_A$包含$N_A$个句子，$D_B$包含$N_B$个句子，$P(\cdot|s;\Theta_{AB})$和$P(\cdot|s;\Theta_{BA})$分别表示两个翻译模型，$\Theta_{AB},\Theta_{BA}$是模型参数。

假设有两个训练好的语言模型$LM_A(\cdot),LM_B(\cdot)$（可以通过单语语料训练得到），能够输入一个句子，输出一个数表示这个句子在这种语言中自然的概率。

对偶学习游戏开始时，有$D_A$中的句子$s$，然后中间翻译结果为$s_{mid}$，这个中间步骤产生一个reward $r_1=LM_B(s_{mid})$，表示这个中间翻译结果在语言$B$中多自然。然后用$s_{mid}$重建$s$的对数概率作为重建reward，即$r_2=\log P(s|s_{mid};\Theta_{BA})$。将两个reward线性加权，得到最终的reward $r=\alpha r_1+(1-\alpha)r_2$，其中$\alpha$是超参数。最后可以用策略梯度方法来最大化reward，从而训练模型。

通过对翻译模型$P(\cdot|s;\Theta_{AB})$进行采样得到$s_{mid}$，然后计算期望reward $E[r]$关于参数$\Theta_{AB},\Theta_{BA}$的梯度，根据策略梯度理论，结果如下：

$$
\nabla_{\Theta_{BA}}E[r]=E[(1-\alpha)\nabla_{\Theta_{BA}}\log P(s|s_{mid};\Theta_{BA})]
$$ (1)

$$
\nabla_{\Theta_{AB}}E[r]=E[r\nabla_{\Theta_{AB}}\log P(s_{mid}|s;\Theta_{AB})]
$$ (2)

基于公式(1)(2)，可以采用任意的采样方法去估计期望梯度，但是随机采样会带来大方差，甚至不可靠的结果，因此采用beam search来进行采样，作者通过贪婪算法产生概率最高的$K$个中间翻译结果，然后用均值表示梯度。

对偶学习游戏可以迭代多轮，算法细节如下图：

![dual_algo](dual_algo.png)

## **[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)**

传统机器翻译是构建在一个固定字典上的，但是翻译本身是一个开放字典的人物，之前的工作通过准备一个字典来解决OOV问题。作者提出一个更加简单有效的方法，通过编码罕见词和未知词为子字串来使得机器翻译模型能够处理开放字典的翻译任务。作者讨论了不同的分词方案，包括简单的n-grams模型和基于byte pair encoding压缩算法的分词技术。

方法来源于人类在不认识一个词是可以通过子字来推测它的意思从而进行翻译，因此可以把罕见词分为子字串，使得神经网络能够学习这种翻译技术，从而更好的处理未知词。如果分词太细就会导致输入句子变得太长，所以权衡词表大小和输入大小，只对罕见词进行分词。

这里主要介绍基于byte pair encoding(BPE)的分词技术。BPE算法是一种简单的数据压缩技术，通过迭代地把序列中平凡出现的字节序列替换为没有出现过的单字节达到压缩的效果。作者改进这种方法，通过将字符不断合并达到提取子字的效果。

首先，建立字符表，然后将每一个词表示为字符序列，并且加上一个特殊的结束符号"$\cdot$"，这使得能够从子字序列重建词序列。然后迭代地计数字符对，把最频繁的字符对替换为新的字符，比如字符对('A','B')替换为'AB'，每次合并操作都会构成一个新的n-grams表，这样频繁的字符序列会最终被合并，构成最终的词表。最终词表的大小由初始词表大小加上合并操作次数，而这个合并操作次数可以调节，来调整词表大小和输入大小。构建词表过程中是不考虑跨词的字符序列的。这一过程的python实现如下图：

![bpe_python](bpe_python.png)

作者又分析了两种BPE方案，一种是分别构建源词表和目标词表，另一种是将源河目标结合共同构建一个词表。如果分别构建BPE词表，那么很可能同一个词会被不同地分割，从而使得神经网络很难学习子字之间的关系，所以为了增加词表的连贯性，可以使用联合构建BPE词表。（其他论文中大多采用联合构建BPE词表的方法）

[代码地址](https://github.com/rsennrich/subword-nmt)

## **[Linguistic Input Features Improve Neural Machine Translation](https://arxiv.org/pdf/1606.02892.pdf)**

传统机器翻译模型没有充分使用语言学特征，作者提出使用更多的语言学特征能够大大提升模型效果。作者拓展了编码器的embedding层，能够输入任意特征，除了传统的词特征，还添加了词法特征，词性特征和句法特征等。

传统编码器的隐层状态计算如下：

$$
\overrightarrow h_j=\tanh(\overrightarrow W Ex_j+\overrightarrow U \overrightarrow h_{j-1})
$$ (1)

其中$E\in R^{m\times K_x}$是词向量矩阵，$\overrightarrow W\in R^{n\times m},\overrightarrow U\in R^{n\times n}$是权重矩阵，$m,n$分别是词向量维度和隐层状态维度，$K_x$是源语言词表大小。作者拓展词向量层，是的能够输入任意数目$|F|$的特征：

$$
\overrightarrow h_j=\tanh(\overrightarrow W(||_{k=1}^{|F|}E_k x_{jk})+\overrightarrow U \overrightarrow h_{j-1})
$$ (2)

其中$||$表示向量连接，$E_k\in R^{m_k\times K_k}$是特征向量矩阵，$\sum_{k=1}^{|F|}m_k=m$，$K_k$是第$k$个特征的词表大小。

作者然后分别分析Lemma，子字tag(B-词的开始，I-词的中间，E-词的结尾，O-整个词)，词性特征（这个具体语言具体特征），词性和句法依存特征的作用，并且输入为BPE子字序列，每个子字赋予整个词的特征。例子如下图：

![linguistic](linguistic.png)

[代码地址](https://github.com/rsennrich/nematus)

## **[Incorporating Structural Alignment Biases into an Attentional Neural Translation Model](https://arxiv.org/pdf/1601.01085.pdf)**

机器翻译模型过于简化了，忽视了很多传统模型中关键的inductive biases。作者提出拓展现有机器翻译模型，从词对齐模型中引入structural biases，包括positional bias, Markov conditioning, fertility and agreement over translation directions。

作者认为现有机器翻译模型忽视了IBM模型中的传统对齐模型，Vogel的隐马尔可夫模型等关键部分，因此提出将这些部分作为structural biases结合进现在的机器翻译模型，可以大大提升效果。

首先考虑position bias，这个想法基于源句子中的词和目标句子中相对位置处的词趋向对齐（$\frac{i}{I}\approx\frac{j}{J}$）这一观察，IBM model 2中就包含了这种离散映射。作者通过改变注意力模型归一化前的$f_{ji}$计算公式来包含这一bias：

$$
f_{ji}=\bold v^T \tanh(\bold W^{(ae)}\bold e_i+\bold W^{ah}\bold g_{j-1}+\bold W^{(ap)}\psi(j,i,I))
$$ (1)

$$
\psi(j,i,I)=[\log(1+j),\log(1+i),\log(1+I)]^T
$$ (2)

其中$\bold W^{(ap)}\in R^{A\times 3}$。作者排除了$J$，因为在解码过程中$J$未知。使用$\log(1+\cdot)$函数是为了防止数值不稳定。

接着考虑Markov condition。Markov condition允许模型在知道$i$和$j$对齐后，拓展到$i+1$和$j+1$对齐或者$i$和$j+1$对齐等。Markov condition以类似position bias的方式加到模型中：

$$
f_{ji}=\bold v^T \tanh(...+\bold W^{(am)}\xi_1(\alpha_{j-1};i))
$$ (3)

其中...包含了之前的所有项。把所有注意力向量包含到$\alpha$中比较困难，因此作者简化只考虑周围$k$个位置，

$$
\xi_1(\alpha_{j-1};i) = [\alpha_{j-1,i-k},...,\alpha_{j-1,i},...,\alpha_{j-1,i+k}]^T
$$ (4)

其中$\bold W^{(am)}\in R^{A\times (2k+1)}$。

接着考虑fertility。首先考虑局部fertility，包含以下这个特征：

$$
\xi_2(\alpha_{\lt j};i)=[\sum_{j'\lt j}\alpha_{j',i-k},...,\sum_{j'\lt j}\alpha_{j',i},...,\sum_{j'\lt j}\alpha_{j',i+k}]
$$ (5)

对应的特征权重为$\bold W^{(af)}\in R^{A\times (2k+1)}$。这些求和项表示周围词的fertility。

接下来考虑全局fertility。提出以下计算fertility的模型：

$$
p(f_i|\bold s,i)=G(\mu(e_i),\sigma^2(e_i))
$$ (6)

其中$f_i=\sum_j\alpha_{j,i}$,$G()$为正态分布。

最后把$\sum_i \log p(f_i|\bold s,i)$这一项作为额外的additive项加到训练目标中。

最后考虑Bilingual Symmetry。作者提出同时训练两个翻译方向上的对齐，也就是要优化下面这一项：

$$
L=-\log p(\bold t|\bold s)-\log p(\bold s|\bold t)+\gamma B
$$ (7)

$B$用来连接两个方向上的模型，$B$应该考虑$\alpha^{s\rightarrow t}\in R^{I\times J},\alpha^{t\rightarrow s}\in R^{J\times I}$这两个对齐矩阵，并且使得它们尽可能接近。作者提出迹奖励来达到这个目的：

$$
B=-tr(\alpha^{s\rightarrow t}\alpha^{t\rightarrow s})=\sum_j\sum_i \alpha^{s\rightarrow t}_{i,j}\alpha^{t\rightarrow s}_{j,i}
$$ (8)

如下图所示：

![bilingual_symmetry](bilingual_symmetry.png)

## **[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)**

谷歌介绍自己翻译系统的论文，讨论了很多工程细节。

神经机器翻译系统在训练和推断时都很慢，在大模型大数据下甚至会无法实现，也有学者表示当句子中有罕见词时模型结果不鲁棒，这些都导致现有的神经机器翻译模型无法应用到实际服务中。本文提出了GNMT，谷歌神经机器翻译系统，尝试解决上述问题。这个模型用了很深的LSTM网络，并且编码器和解码器都有8层，并且使用了残差连接和注意力连接。为了提高并行性从而降低训练时间，注意力机制直接把解码器的底层和编码器的顶层连接起来。为了加速最终的翻译速度，在推断时采用低精度运算。为了处理罕见词，将输入和输出分割为有限的子字集，并且很好的平衡了字模型的灵活性和词模型的高效性，从而很好的处理了罕见词，并且最终提升了模型效果。模型采用的束搜索采用了长度归一化和覆盖率惩罚，使得生成的句子覆盖所有源词。为了直接优化BLEU分数，通过增强学习来改进模型，但是发现BLEU分数的提升并不意味着翻译结果的改善。模型整体架构如下图：

![GNMT](GNMT.png)

令$(X,Y)$是源和目标句子对，$X=x_1,...,x_M,Y=y_1,...,y_N$。

首先考虑残差连接。更深的LSTM会带来更好的结果，但是当深到一定程度时会导致模型学习很慢，并且很难学习，可能是因为梯度爆炸和梯度消失等问题。作者发现4层LSTM效果还好，6层就开始效果一般，8层时效果就变差了。根据残差连接在其他模型中的成功，也把残差连接这一结构应用到深层LSTM结构中。具体地说，$LSTM_i,LSTM_{i+1}$分别表示第$i$层和第$i+1$层LSTM，参数分别为$\bold W^i$和$\bold W^{i+1}$，在时刻$i$时，带有残差连接的LSTM结构如下：

$$
\begin{aligned}
\bold c_t^i,\bold m_t^i &= LSTM_i(\bold c_{t-1}^i,\bold m_{t-1}^i,\bold x_t^{i-1};\bold W^i) \\
\bold x_t^i &= \bold m_t^i+\bold x_t^{i-1} \\
\bold c_t^{i+1},\bold m_t^{i+1} &= LSTM_{i+1}(\bold c_{t-1}^{i+1},\bold m_{t-1}^{i+1},\bold x_t^i;\bold W^{i+1})
\end{aligned}
$$ (1)

残差连接能够极大改善反向传播时梯度的流动，从而从能训练特别深的网络。

接着考虑编码器第一层使用双向模型。采用双向RNN来包含之前词和之后词的信息，但为了增加并行性，只在编码器的底层使用双向RNN。

接着考虑模型并行性。同时利用模型并行和数据并行来加速训练。数据并行很直观：用Downpour随机梯度下降算法同时训练n个相同的模型。这n个模型共享一套参数，每个模型异步地更新参数。除此之外，模型并行用来加速每个模型上的梯度计算，编码器和解码器网络在深度方向上分割，并且被放到不同的GPU上运行，一般一个GPU跑一层网络。除了编码器第一层外，其他都是单向的，所以第$i+1$层可以在第$i$层完全结束前开始计算，从而提高了训练速度。softmax层也被分割处理，每一块只负责整个词表的一部分的计算。由于模型并行性要求，所以只在编码器底层使用双向RNN。在模型的注意力模块，选择把编码器的顶层和解码器的底层进行对齐，也是为了模型并行性的考虑，如果把解码器的顶层和编码器的顶层进行对齐操作，那么整个解码器将无法使用多个GPU来实现并行。

接着考虑分词方法。采用子字来解决OOV问题。采用最初用于日韩文分词问题的wordpiece模型来实现分词到子字序列。为了处理任何词，使用训练好的wordpiece模型将词分割为wordpiece，特殊的词边界符号加到句子中，使得能够从wordpiece序列恢复成词序列。下面是一个例子：

![wordpiece](wordpiece.png)

wordpiece模型使用数据驱动的方法，在给定逐步发展的词顶一下，最大化训练语料的类似语言模型的似然。给定训练语料和需要产生的tokens数目$D$，优化目标是选取$D$个wordpiece使得根据选定wordpiece模型分割的wordpieces树木最小。最贪婪的算法类似于《Neural machine translation of rare words with subword units》中的算法，更多算法细节参考《Japanese and Korean voice search》。和原始实现不同的是，只在词开始处添加特殊边界符号，并且根据语料将基础字符限制到一个可调的数目，并且把剩下的词全部映射到未知字符，避免十分罕见的字符干扰wordpieces词表。作者发现8k和32k wordpieces词表在所有试过的语言对中能够同时保持好的BLEU分数和快速解码速度。对于有些罕见的实体词和数字，直接拷贝是个很好的策略，因此对于源语言和目标语言使用一个共享的wordpieces，使得两种语言对于同一个词有相同的分隔方式，从而使得系统能够学习直接拷贝极罕见词。

第二种处理OOV的方法是使用混合字／词模型。保持一个固定大小的词表，，并且把OOV表示成字序列，并且特殊的前缀加到字符前面，这个词缀不仅用来表示字符在词中的位置，并且用来和词表中的字符区分开来，总共有三种前缀，<B>,<M>,<E>分别表示词头，词中间和词尾。这个处理在源句子和目标句子中都要进行。解码时输出的罕见词就是字符序列，并且可以通过前缀恢复出词序列。

下面考虑训练标准。
