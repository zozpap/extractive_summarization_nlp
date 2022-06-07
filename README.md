# **Comparison of unsupervised and supervised extractive summarization methods of Natural Language Processing**
<br>

### **Eötvös Loránd University (ELTE) - Institute of Mathematics**
#### **Mathematics Expert in Data Analytics and Machine Learning postgraduate specialization program**


---

#### In my empirical work, I have built and compared different model architectures and they have been applied to summarize the articles. I have focused on extractive summarization, which selects the most informative sentences from the original article. For this problem, both unsupervised and supervised methods have been used, and their results have been compared. In case of the supervised approach, the task can be translated into a classification problem by using cosine similairty measure. The results are evaluated using ROUGE-score and imbalanced classfifcation metrics. Since many articles can be summarized using 3-4 sentences, I have selected the output of the models the most relevant 4 sentences. Reference scores have been determined by the original highlight ROUGE score performance on the articles. The Sequential Gated Recurrent Unit architecture has outperformed the rest of the models. 

---
<br>

#### **Dataset:**

---
#### Subset of the well known **CNN/Daily news dataset**, having many samples, where each of the original articles are paired with their corresponding highlights summary.
<br>

#### **Word embedding:**


#### State-of-the-art sentence transformer model **all-mpnet-base-v2** for embedding by Microsoft. Its embeddings can be used to find sentences with a similar meaning. The original model was trained on a large and diverse dataset of over 1 billion training pairs.
<br>

#### **Applied algorithms:**

---
> **Unsupervised algorithms:**
* Text rank
* Lex rank
* Latent Semantic Analysis
* Luhn rank
* Kullback–Leibler divergence

---

> **Supervised algorithms:**
* Catboost - optimized with HyperOpt
* Multi-layer Perceptron Network
* Long Short-term Memory
* Bidirectional Long Short-term Memory
* Gated Recurrent Unit
<br>

#### **Reference scores**

---

| Reference   score |                  |
|-------------------|------------------|
| Model name        | First 4 sentence |
| Rouge-1-recall    |      0.5823      |
| Rouge-1-precision |      0.4067      |
| Rouge-1-F_score   |      0.4713      |
|                   |                  |
| Rouge-2-recall    |      0.3087      |
| Rouge-2-precision |      0.2053      |
| Rouge-2-F_score   |      0.2413      |
|                   |                  |
| Rouge-L-recall    |      0.5456      |
| Rouge-L-precision |      0.3814      |
| Rouge-L-F_score   |      0.4419      |
<br>

#### **Results**

---

| Unsupervised      |           |          |           |          |                       |
|-------------------|-----------|----------|-----------|----------|-----------------------|
| Model name        | Text Rank | Lex Rank | Luhn Rank | LSA Rank | Kullback-Leibler Rank |
|                   |           |          |           |          |                       |
| Rouge-1-recall    |   0.2630  |  0.2702  |   0.3054  |  0.2150  |         0.2692        |
| Rouge-1-precision |   0.1081  |  0.1417  |   0.1320  |  0.1224  |         0.1542        |
| Rouge-1-F_score   |   0.1478  |  0.1772  |   0.1780  |  0.1512  |         0.1892        |
|                   |           |          |           |          |                       |
| Rouge-2-recall    |   0.0660  |  0.0856  |   0.1011  |  0.0589  |         0.0927        |
| Rouge-2-precision |   0.0249  |  0.0427  |   0.0429  |  0.0340  |         0.0499        |
| Rouge-2-F_score   |   0.0346  |  0.0541  |   0.0577  |  0.0416  |         0.0622        |
|                   |           |          |           |          |                       |
| Rouge-L-recall    |   0.2209  |  0.2289  |   0.2595  |  0.1809  |         0.2335        |
| Rouge-L-precision |   0.0905  |  0.1202  |   0.1122  |  0.1027  |         0.1332        |
| Rouge-L-F_score   |   0.1239  |  0.1502  |   0.1513  |  0.1271  |         0.1637        |
<br>

| Supervised          |          |        |        |        |        |
|---------------------|----------|--------|--------|--------|--------|
| Model name          | CatBoost | MLP    | GRU    | LSTM   | BILSTM |
| Accuracy            |  83.79%  | 83.36% | 84.60% | 84.49% | 84.55% |
| Balanced   accuracy |  60.44%  | 59.33% | 62.42% | 62.14% | 62.31% |
| F1 score            |  0.2923  | 0.2734 | 0.3264 | 0.3215 | 0.3244 |
|                     |          |        |        |        |        |
| Rouge-1-recall      |  0.4604  | 0.4440 | 0.5058 | 0.4784 | 0.4787 |
| Rouge-1-precision   |  0.2806  | 0.2728 | 0.3145 | 0.2987 | 0.2979 |
| Rouge-1-F_score     |  0.3400  | 0.3297 | 0.3795 | 0.3589 | 0.3582 |
|                     |          |        |        |        |        |
| Rouge-2-recall      |  0.1911  | 0.1781 | 0.2297 | 0.2067 | 0.2080 |
| Rouge-2-precision   |  0.1057  | 0.1001 | 0.1294 | 0.1184 | 0.1187 |
| Rouge-2-F_score     |  0.1317  | 0.1242 | 0.1608 | 0.1458 | 0.1463 |
|                     |          |        |        |        |        |
| Rouge-L-recall      |  0.4262  | 0.4109 | 0.4706 | 0.4441 | 0.4449 |
| Rouge-L-precision   |  0.2599  | 0.2525 | 0.2927 | 0.2774 | 0.2769 |
| Rouge-L-F_score     |  0.3149  | 0.3052 | 0.3531 | 0.3333 | 0.3329 |


---
 
