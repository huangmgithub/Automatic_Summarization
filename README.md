# Automatic_Summarization
 
### 方法
+ 基于TextRank
+ 基于Sentence Embedding
+ 基于LDA
+ 基于Seq2Seq，transformer（未完成）
  
### 本项目采用提取式摘要生成，方法如下：
* #### SentenceEmbedding
	+ 基于word2vec词向量和SIF来实现句子向量化，向量相似度计算
	+ 关键字提取(TextRank)
	+ 主题判别(LDA)
	+ 标题加权处理
	+ 位置加权处理
	+ KNN平滑处理
* #### TextRank
* #### LDA