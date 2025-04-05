# Sentiment and Topic Analysis of Chinese Hotel Reviews

## Abstract

This project presents a sentiment classification and topic modeling approach for Chinese hotel reviews. Leveraging pretrained word embeddings and deep learning architectures, we classify user reviews into positive or negative sentiments. Additionally, we employ Latent Dirichlet Allocation (LDA) to extract dominant topics from the corpus. The entire pipeline emphasizes the use of modern NLP techniques adapted for Chinese text, aiming to provide insights into consumer opinions.

## Introduction

Customer reviews in the hospitality sector are a valuable source of feedback, containing both explicit sentiments and implicit topics of concern. While English review analysis is well-studied, Chinese language processing poses additional challenges due to segmentation, sparse datasets, and complex grammar. In this study, we explore sentiment prediction using a BiLSTM model trained on a combination of publicly available and privately curated Chinese hotel reviews. 

## Dataset

We used a combined dataset of 4,000 labeled hotel reviews:
- 2,000 positive reviews
- 2,000 negative reviews

The data was sourced from:
- Public hotel review corpus from Professor Tan Songbo
- Privately constructed positive and negative review samples (not published)

## Methodology

### Preprocessing
- Chinese text segmentation using Jieba
- Removal of punctuation and stopwords
- Conversion of text into word index sequences using pretrained embeddings

### Embedding
- Pretrained Zhihu Bigram word vectors (300 dimensions)
- Limited to top 50,000 most frequent tokens for efficiency

### Sentiment Classification
- BiLSTM + LSTM stacked model architecture
- Trained using Keras and TensorFlow
- Evaluation on a 10% held-out test set

### Topic Modeling
- Tokenized texts transformed into Bag-of-Words format
- LDA model trained with 5 topics using Gensim
- Visualization using pyLDAvis

## Results

- Test Accuracy (BiLSTM model): **87%**
- Clear separation of semantic categories in embedding visualization
- Coherent topics extracted via LDA (e.g., room service, food, pricing)

## Observations

- The model performs well on explicit sentiment expressions but struggles with sarcasm and negation
- LDA topics reflect distinct user concerns, especially around cleanliness, service attitude, and location
- Pretrained embeddings significantly improve semantic understanding compared to TF-IDF

## Limitations

- The dataset is relatively small; performance may improve with more labeled data
- Sarcasm and implicit sentiment remain challenging for BiLSTM
- LDA model's topic quality is sensitive to stopword configuration and token granularity

## Future Work

- Incorporate transformer-based models (e.g., BERT or RoBERTa)
- Experiment with domain-specific embeddings (hotel-specific corpus)
- Fine-tune the topic modeling pipeline using coherence scoring and dynamic topic estimation

## Conclusion

This project demonstrates the feasibility and effectiveness of applying deep learning and unsupervised topic modeling to Chinese-language hotel reviews. The techniques used provide a foundation for further NLP research and commercial applications in sentiment-aware recommendation systems and hotel feedback analysis.