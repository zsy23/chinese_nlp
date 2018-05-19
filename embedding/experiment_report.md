# **General Training Parameter**

| Parameter | Setting |
| :-: | :-: |
| corpus | {wiki, sougou_news, dureader, together} |
| model | {skip-gram, cbow} |
| vector dim | {50, 100, 200, 300} |
| context window size | 5 |
| word min count | 5 |
| subsampling | $10^{-4}$ |
| training | no hierarchical softmax, negative sample num 10 |
| iteration | 5(GLoVe 25) |
| initial learning rate | default |

# **Evaluation**

用CWE提供的240.txt，297.txt做word similarity，包含oov词的case直接忽略，不是把score赋为0，用analogy.txt做word analogy，然后用THUCNews做文本分类。

# **Word2Vec**

## **wiki+sougou_news+dureader**

### **Skip-Gram**

```
 ./word2vec -train ../corpus/rm_digit_and_non_chinese/corpus.complete -output ../model/word2vec/wiki_sougou_news_dureader/word2vec_skip_gram_50.vec -size 50 -window 5 -sample 1e-
4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.025 -cbow 0
```

*Word Similarity*

| Dataset | Found | Not Found | Score(Spearman Correlation) |
| :-: | :-: | :-: | :-: |
| 240.txt | 232 | 8 | 0.5270936654831673 |
| 297.txt | 287 | 10 | 0.5345762511845042 |

*Word Analogy*

Mean rank: 指ground truth在最近邻中排第几（理想情况应该是1）

| Category | Total count | Accuracy | Mean rank |
| :-: | :-: | :-: | :-: |
| City | 175 | 0.7257142857142858 | 8.942857142857143 |
| Family | 272 | 0.3860294117647059 | 338.96691176470586 |
| Capital | 506 | 0.5731225296442688 | 36.405138339920946 |
| Total | 953 | 0.5477439664218258 | 117.71773347324239 |



```
./word2vec -train ../corpus/rm_digit_and_non_chinese/corpus.complete -output ../model/word2vec/wiki_sougou_news_dureader/word2vec_skip_gram_100.vec -size 100 -window 5 -sample 1
e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.025 -cbow 0
```

*Word Similarity*

| Dataset | Found | Not Found | Score(Spearman Correlation) |
| :-: | :-: | :-: | :-: |
| 240.txt | 232 | 8 | 0.5494397040353125 |
| 297.txt | 287 | 10 | 0.5512445993563866 |

*Word Analogy*

Mean rank: 指ground truth在最近邻中排第几（理想情况应该是1）

| Category | Total count | Accuracy | Mean rank |
| :-: | :-: | :-: | :-: |
| City | 175 | 0.9542857142857143 | 1.7885714285714285 |
| Family | 272 | 0.5882352941176471 | 30.56985294117647 |
| Capital | 506 | 0.8320158102766798 | 4.899209486166008 |
| Total | 953 | 0.7848898216159497 | 11.654774396642182 |



```
./word2vec -train ../corpus/rm_digit_and_non_chinese/corpus.complete -output ../model/word2vec/wiki_sougou_news_dureader/word2vec_skip_gram_200.vec -size 200 -window 5 -sample 1
e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.025 -cbow 0
```

*Word Similarity*

| Dataset | Found | Not Found | Score(Spearman Correlation) |
| :-: | :-: | :-: | :-: |
| 240.txt | 232 | 8 | 0.5525256610613511 |
| 297.txt | 287 | 10 | 0.5662803562426717 |

*Word Analogy*

Mean rank: 指ground truth在最近邻中排第几（理想情况应该是1）

| Category | Total count | Accuracy | Mean rank |
| :-: | :-: | :-: | :-: |
| City | 175 | 0.9828571428571429 | 1.1657142857142857 |
| Family | 272 | 0.6911764705882353 | 14.727941176470589 |
| Capital | 506 | 0.8893280632411067 | 1.8754940711462451 |
| Total | 953 | 0.8499475341028332 |  5.413431269674711 |



```
./word2vec -train ../corpus/rm_digit_and_non_chinese/corpus.complete -output ../model/word2vec/wiki_sougou_news_dureader/word2vec_skip_gram_300.vec -size 300 -window 5 -sample 1
e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.025 -cbow 0
```

*Word Similarity*

| Dataset | Found | Not Found | Score(Spearman Correlation) |
| :-: | :-: | :-: | :-: |
| 240.txt | 232 | 8 | 0.5620540699478851 |
| 297.txt | 287 | 10 | 0.5801494826378228 |

*Word Analogy*

Mean rank: 指ground truth在最近邻中排第几（理想情况应该是1）

| Category | Total count | Accuracy | Mean rank |
| :-: | :-: | :-: | :-: |
| City | 175 | 0.9771428571428571 | 1.2457142857142858 |
| Family | 272 | 0.7573529411764706 | 8.242647058823529 |
| Capital | 506 | 0.9071146245059288 |  1.6719367588932805 |
| Total | 953 | 0.8772298006295908 |  3.4690451206715633 |



### **CBOW**

```
./word2vec -train ../corpus/rm_digit_and_non_chinese/corpus.complete -output ../model/word2vec/word2vec_cbow_50.vec -size 50 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.05 -cbow 1
```
```
./word2vec -train ../corpus/rm_digit_and_non_chinese/corpus.complete -output ../model/word2vec/word2vec_cbow_100.vec -size 100 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.05 -cbow 1
```
```
./word2vec -train ../corpus/rm_digit_and_non_chinese/corpus.complete -output ../model/word2vec/word2vec_cbow_200.vec -size 200 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.05 -cbow 1
```
```
./word2vec -train ../corpus/rm_digit_and_non_chinese/corpus.complete -output ../model/word2vec/word2vec_cbow_300.vec -size 300 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.05 -cbow 1
```

## **wiki**

### **Skip-Gram**

```
./word2vec -train ../corpus/rm_digit_and_non_chinese/wiki_zh.corpus.complete -output ../model/word2vec/wiki/word2vec_skip_gram_300.vec -size 300 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.025 -cbow 0
```

### **CBOW**

```
./word2vec -train ../corpus/rm_digit_and_non_chinese/wiki_zh.corpus.complete -output ../model/word2vec/wiki/word2vec_cbow_300.vec -size 300 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.05 -cbow 1
```

## **sougou_news**

### **Skip-Gram**

```
./word2vec -train ../corpus/rm_digit_and_non_chinese/sougou_news.corpus.complete -output ../model/word2vec/sougou_news/word2vec_skip_gram_300.vec -size 300 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.025 -cbow 0
```

### **CBOW**

```
 ./word2vec -train ../corpus/rm_digit_and_non_chinese/sougou_news.corpus.complete -output ../model/word2vec/sougou_news/word2vec_cbow_300.vec -size 300 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.05 -cbow 1
```

## **dureader**

### **Skip-Gram**

```
./word2vec -train ../corpus/rm_digit_and_non_chinese/dureader.corpus.complete -output ../model/word2vec/dureader/word2vec_skip_gram_300.vec -size 300 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.025 -cbow 0
```

### **CBOW**

```
./word2vec -train ../corpus/rm_digit_and_non_chinese/dureader.corpus.complete -output ../model/word2vec/dureader/word2vec_cbow_300.vec -size 300 -window 5 -sample 1e-4 -hs 0 -negative 10 -iter 5 -min-count 5 -alpha 0.05 -cbow 1
```

# **GloVe**

# **fastText**

# **CWE**

# **SCWE**

# **JWE**
