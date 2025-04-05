# Hotel Review Sentiment Classification using Word Embeddings

This project investigates sentiment analysis of Chinese hotel reviews using deep learning techniques. We apply pretrained word embeddings and a bidirectional LSTM model to classify reviews as positive or negative. Additionally, topic modeling via Latent Dirichlet Allocation (LDA) is used to explore semantic patterns within the corpus.

## 📚 Data

The dataset combines:
- Publicly available hotel review data provided by Professor Tan Songbo
- Privately curated samples (`positive_samples.txt`, `negative_samples.txt`) containing user-generated reviews (not published due to licensing)

In total, 4,000 labeled reviews were used: 2,000 positive and 2,000 negative.

## 🧠 Methodology

- **Preprocessing**: Chinese tokenization using Jieba and punctuation removal
- **Word Embeddings**: 300-dimensional pretrained vectors from [Zhihu Bigram Embeddings](https://github.com/Embedding/Chinese-Word-Vectors)
- **Model**: BiLSTM sentiment classifier using Keras and TensorFlow
- **Topic Modeling**: Gensim-based LDA with pyLDAvis visualization (planned)

## 🧪 Results

- Classification Accuracy (Test Set): **87.0%**
- Model Architecture: Embedding + BiLSTM + LSTM + Dense
- Training: EarlyStopping + Checkpointing + TensorBoard

## 📁 Project Structure

```bash
hotel-sentiment-analysis/
├── data/                    # Raw and labeled hotel reviews
├── embeddings/              # Pretrained word vectors (Zhihu)
├── notebooks/               # Jupyter Notebooks for each step
├── src/                     # Python modules (tokenizer, model, LDA, etc.)
├── results/                 # Saved models and output visualizations
├── logs/                    # TensorBoard and checkpoint logs
├── reports/                 # Summary and academic observations
├── requirements.txt         # Python dependencies
└── README.md

```

# Data Folder

This folder is used to store datasets and preprocessed resources for training and topic modeling.

## 📚 Data Sources & References

### 1. Chinese Hotel Review Corpus
A portion of the sentiment data was derived from the publicly available hotel review corpus by Professor **Tan Songbo**. This dataset has been widely used for sentiment analysis research in Chinese NLP.  
Original corpus reference:
- [[http://nlp.fudan.edu.cn/data/](http://nlp.fudan.edu.cn/data/](https://tianchi.aliyun.com/dataset/6550)) (or local academic mirror)

### 2. Pretrained Word Embeddings
This project uses the **Zhihu Bigram word vectors** (300-dimensional) provided by the open-source repository:

- GitHub: [https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)
- Specific file used: `sgns.zhihu.bigram.bz2`

These vectors are trained on a large-scale corpus from Zhihu (a Chinese Q&A platform), and provide semantic-rich representations of Chinese words and phrases.


## Required Files (Not Included)

Please prepare the following files manually before running the notebooks:

- `data/positive_samples.txt`
- `data/negative_samples.txt`
- `data/train_pad.npy`
- `data/train_labels.npy`
- `data/embedding_matrix.npy`

## Included

- `stopwords.txt` — A list of common Chinese stopwords for LDA topic filtering.



Note: Due to data licensing, the actual review texts (`positive_samples.txt`, `negative_samples.txt`) and `.npy` files are not included in this repository. Please refer to `data/README.md` for guidance on how to prepare them.
