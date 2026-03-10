# Improving the Performance of Sentiment Classification on Imbalanced Datasets With Transfer Learning

Z. XIAO 1, L. WANG1, AND J. Y. DU1,2

1College of Computer Science and Electronic Engineering, Hunan University, Changsha 410082, China 2School of Computer and Information Engineering, Central South University of Forestry and Technology, Changsha 410004, China Corresponding author: J. Y. Du (maxdujiayi@hnu.edu.cn)

This work was supported in part by the Natural Science Foundation of China under Grant 61872129 and Grant 61802444, and in part by the Doctoral Scientific Research Foundation of Central South University of Forestry and Technology under Grant 2016YJ047.

ABSTRACT In recent years, many sentiments classification models, such as deep learning models and traditional machine learning models, claim that they can achieve state-of-the-art performance in sentiment analysis problems. Admittedly, this is based on the premise that the training samples are class balanced. However, in the real world, the training data sets we can get are often imbalanced, which will cause the trained classifier to tend to predict the test samples into a majority, making the recall of minority very low. In order to minimize the influence of the imbalanced data class on the model performance, a transfer learning method based on a convolution neural network is proposed in this paper. First, we use a CNN-based model for pre-training in the class-balanced source domain data set, before transferring the model to the target domain for fine-tuning to improve the recall of minority class; furthermore, we propose a transfer learning-based under-sampling technique, which can under-sample the majority class in the target domain. In the data set after under-sampling, we again fine-tune the pre-trained model, so that the recall and precision of the minority class have been greatly improved. The experiments on real-world data sets show that our proposed under-sampling method has obvious advantages compared with others.

INDEX TERMS Convolutional neural network, sentiment classification, transfer learning, under-sampling.

# I. INTRODUCTION

Data distribution is often imbalanced in the real world [1], such as reviews on tourist attractions by visitors on travel sites, comments on hotel services by guests on lifestyle service websites, and products reviews left by consumers on e-commerce websites. Most of these reviews are positive and only a few are negative, but negative reviews are often more helpful to us because they are more conducive to companies finding defects in their services or products and remedying identified defects. In order to solve the classification problem of imbalanced data sets, many re-sampling techniques, cost-sensitive algorithms, kernel-based techniques, and active learning methods have been proposed [2].

Re-sampling techniques are usually classified as undersampling and over-sampling. Under-sampling method is adopted to remove redundant information in majority classes, so that the number of samples remaining in the set is as close as possible to the proportion of minority classes. In addition, it still covers most of the valid information in the original set. The commonly used methods are random under-sampling and sampling based on the nearest neighbor algorithm (NNA) [3]. But they can potentially lead to loss of information about the majority classes. For minority classes, the over-sampling method is usually employed for randomly copying the minority samples so as to multiply the minority samples. However, this will cause model to overfitting in the minority samples. Another common over-sampling method is SMOTE [4], which uses random linear interpolation between adjacent samples of minority classes to manually generate samples. For text data, in the existing re-sampling methods, whether it be under-sampling or over-sampling, vector space model (VSM) is used to represent samples. It represents text as a point in high dimensional space, thus ignoring the potential correlation between words in the text and the order of statements; therefore, the existing re-sampling technology has serious defects in the text data.

In addition, the reason why classification algorithms cannot perform well on imbalanced data sets lies in not only the imbalance of classes, but also the small number of samples. Although we can under-sample and over-sample data sets to balance the classes of the training data sets, the classification algorithms uses a small amount of data which cannot train a robust model, especially when using a deep learning methods. Transfer learning (TL) can transfer knowledge learned from source domain $( S _ { d } )$ which is similar to target domain $( T _ { d } )$ , so that even if the training data set of $T _ { d }$ is small, we can also train a robust classification model.

In this paper, we use the data set of hotel service reviews $( S _ { d }$ , positive and negative sample ratio is 1:1) to pre-trained a sentiment classification model, and then fine-tune the model on the data set of reviews on tourist attractions $T _ { d }$ , positive and negative sample ratio is 52:1). Compared with the non-transfer model, the classifier’s precision, recall and f1-value of minority class improved from $3 5 \%$ , $34 \%$ , $3 5 \%$ to $42 \%$ , $45 \%$ , $43 \%$ respectively. Furthermore, we use the TL model to classify the majority class samples in the training set of the $T _ { d }$ , discard all the correct samples, take the predicted error samples as the under-sampling data set, and then fine-tune the model again. Finally, the classifier’s precision, recall and f1-value of the model in the $T _ { d }$ of minority class reached $63 \%$ , $64 \%$ and $63 \%$ respectively.

The main contributions of this paper are as follows:

• Proposed a novel transfer learning-based undersampling technique (TL-based-US), which makes use of auxiliary data sets in similar domain for supervised training, and then under-sample when predicting on the target domain. Therefore, it can be said that TL-based-US is a supervised under-sampling technique. Besides, experiments on real-world data sets show that the proposed TL-based-US method outperforms the existing under-sampling method in text sentiment classification. Presented a new fine-tuning framework for sentiment classification task, which contains two different convolution modules. One is transferred from the pre-training model, and the other is retrained by the target domain datasets. Experiments and verify the effectiveness of the framework in solving the problem of sentiment classification in imbalanced datasets.

The rest of the paper is organized as follows: In Part II, we will introduce related work on text classification of imbalanced datasets and the application of transfer learning based on deep neural network (DNN) to text classification. In Part III, we will elaborate on our proposed TL-based-US approach and the entire model framework. In Part IV, the empirical analysis and the comparison of experimental results are presented. Part V concludes our work.

# II. RELATED WORK

# A. IMBALANCED TEXT CLASSIFICATION

The results of [1] and [5] show that the factors affecting the classification performance involve not only the data skew, but also the number of minority class samples, the independence of samples, the subclasses in the category and the overlap of categories. This paper focuses on the class imbalance of samples and the small number of samples in minority class. In view of the problems above, scholars have proposed many solutions, including data level and algorithm level.

At the data level, it focuses on adjusting the original imbalanced class distribution via re-sampling. Shuyang et al. [6] proposed an under-sampling method based on K-means, which simply keeps the samples in the cluster center of majority class and removes redundant samples. Zhang and Mani [7] proposed a KNN-base approach called NearMiss to undersampling. It perform under-sampling of points in the majority class based on their distance to other points in the same class and selects k nearest neighbors in majority class for every point in minority class. Chawla et al. [8] proposed a SMOTE algorithm to over-sampling, for each point p in minority class, it first computes k nearest neighbors, then randomly chooses r points of the neighbors, which r less then k, and then selects a random point along the lines joining p and each of the r selected neighbors, finally adds these synthetic points to the dataset of minority class. Song et al. [9] combine SMOTE and K-Means under-sampling method to solve the within-class imbalanced problem and the between-class imbalanced problem.

At the algorithmic level, it mainly modifies the existing classification algorithms to solve the classification problem of class imbalance. It includes cost-sensitive learning [10], ensemble learning and recognition-based learning [11]. Cost-sensitive learning mainly considers how to train the classifier in the classification when different classification errors will lead to different punishment strength. He et al. [12] express the loss cost as a gradient, and as for disease diagnosis problem, they convert the target from minimizing the number of errors to minimizing the total cost, thereby making the learning process develop with the lowest cost. To solve the problem of under-sampling which causes many samples in majority classes to be ignored, Liu et al. [13] propose two ensemble learning algorithms, EasyEnsemble and Balance-Cascade. EasyEnsemble samples several subsets from the majority class, then trains a AdaBoost ensemble from each training subset (which consists of minority class and above one of subset), and finally combines the above-mentioned classifiers into a meta-ensemble. The BalanceCascade model is a model structure designed to solve unbalanced data sets. By using multiple meta-models to form a serial structure. As a first step, all the data used to train the first metamodel, which is further used for predicting the majority class in the training data, deleting the correct samples of the prediction and leaving the wrong samples of the prediction as under-sampling samples. By analogy, the training meta-model stops until the proportions of the majority class and the minority class are consistent, and finally the predicted results of all the meta-models are synthesized as the final results.

TABLE 1. Explanation of some notations.   

<table><tr><td>Notations</td><td>Explanation</td><td>Notations</td><td>Explanation</td></tr><tr><td>S</td><td>Source domain</td><td>Td</td><td>Target domain</td></tr><tr><td>mini Td</td><td>Subset of random under-sampling in Td</td><td>Td-test</td><td>Test set of Td</td></tr><tr><td>Td-train-pos</td><td>Positive samples in the training set of Td</td><td>Td-train-neg</td><td>Negative samples in the training set of T</td></tr><tr><td>hi</td><td>The ith hidden layer in the neural network</td><td>modeli</td><td>The ith model of training</td></tr><tr><td>Tag-posi</td><td>Positive emotion polarity of predic- tive output of the model</td><td>Tag-negi</td><td>Negative emotion polarity of pre- dictive output of the model</td></tr></table>

# B. THE APPLICATION OF TRANSFER LEARNING IN TEXT CLASSIFICATION

In machine learning and data mining, there are two major assumptions. One is that the amount of training data is large enough, and the other is that the training data and test data must be in the same feature space with the same distribution [14]. However, in the real world, the assumptions above are often untenable, for example, in the field of sentiment classification (SC) mentioned in this paper. It is gratifying to note that TL is not constrained by the aforementioned assumptions. In addition, we note that the flexible and customizable hierarchical structure of DNN and the adjustability of parameters are very beneficial to TL. Integrating DNN with TL has been carried out successfully in many areas of computer vision, such as image and video style transfer [15], [16]. At the same time, a growing number of studies have been conducted recently in natural language processing (NLP) applications.

TL with DNNs in NLP fileds generally involves transfer at both the word embedding feature level and the model structure level.

Word embedding feature level transfer refers to the use of $S _ { d }$ datasets or auxiliary tasks to learn about the low-level features’ expression of $T _ { d }$ , that is, word vectors, and then the adoption of the learned vectors as input to the $T _ { d }$ model. Mccann et al. [17] use a deep LSTM encoder to machine translation model trained attention Seq2Seq model to obtain context word vectors (CoVe). They reported that CoVe can improve the performance of many NLP tasks compared with unsupervised word and character vectors. Abdelwahab and Elmaghraby [18] use the $S _ { d }$ and $T _ { d }$ to train a number of word2vec models, and finally to calculate the average of all word2vec as the word vector in the $T _ { d }$ , the method in the twitter sentiment classification dataset (Twitter2016) to get a good performance. Bollegala et al. [19] train the sentiment sensitive embeddings by selecting common words as pivots word from the source and target domains, as well as combining unigram and bigram features. Yu and Jiang [20] use auxiliary tasks (predicting whether the input sentence contains domain-independent sentiment words) to learn the sentence embedding of the $T _ { d }$ , which works well in cross-domain SC tasks.

Model structure level transfer refers to continuing training by porting some layers of the DNNs model trained in the

$S _ { d }$ into the $T _ { d }$ . This approach, also known as fine-tuning, helps achieve the purpose of transferring higher-level abstract features. Mou et al. [21] conducted TL experiments on three different layers of neural network: input layer, hidden layer and output layer. Experimental results show that fine-tuning the input layer and hidden layer of the model can boost the classification performance of the $T _ { d }$ . Howard and Ruder [22] obtain a transfer model, FitLaM, by pre-training a model on a large general-domain corpus, then iteratively freezing each layer on the neural network on the target task, while fine-tuning the other layers.

# III. OUR APPROACHES

This section details our proposed approach. First, let’s define some notations that we’ll use later. Let us denote the subset of random under-sampling in $T _ { d }$ by mini $T _ { d }$ , and the testset of target domain by $T _ { d }$ -test . let $T _ { d }$ -train-pos and $T _ { d }$ -train-neg represent positive samples and negative samples in the training set of $T _ { d }$ , respectively. $h _ { i }$ stands for a hidden layer in the neural network whilst modeli represents a model of training. Tag-posi and Tag-negi represent the predictive output of the model, which indicates that the predicted results are positive emotion polarity samples and negative ones respectively. The explanation for these notations can be found in Table 1.

Our proposed solution has the following five steps: 1) Use $S _ { d }$ to train CNN classification model model1; 2) Transfer model1 to fine-tune in mini $T _ { d }$ to get mode $l _ { 2 }$ ; 3) Use model2 to classify on the $T _ { d }$ -train-pos, the classification result is Tag-neg1 samples as the under-sampling; 4) Use model1 to fine-tune in Tag-neg1 and $T _ { d }$ -train-neg, get model3; 5) Use model2 classified on $T _ { d }$ -test to obtain the preliminary classification results Tag-pos2 and Tag-neg2, then use model3 classified on Tag-neg2 to obtain Tag-pos3 and Tag-neg3. So the final classification results of $T _ { d }$ -test are Tag-pos2, Tag-pos3 and Tag-neg3. The entire process is shown in Fig. 1

# A. MODEL PRE-TRAINING ON SOURCE DOMAIN

In order to train a model suitable for knowledge transfer, we use a data set with consistent proportions of positive and negative samples (sampled from the data set MinChnCorp [23]) to pre-train the sentiment classification model model1. The accuracy of the model on $S _ { d }$ is $8 8 . 2 \%$ , although in fact, our focus is not on training a high-precision classification model in $S _ { d }$ . We just want to use the pre-train model to extract some features of negative comments in $T _ { d }$ . Therefore, we did not carefully tuning parameters for the pre-train model. In addition, we use the skip-gram architecture to pre-train the word2vec [24] vectors on the Chinese Wikipedia corpus, with a vector dimension of 200. When we pre-training the classification model, we use the CNN-static [25] mode, which keeps the word2vec vectors parameter constant and trains only the remaining parameters.

![](images/5a892c82cee062d384a225b40c1f4e2c05a17372fd6094c5491c085b34998f0f.jpg)  
FIGURE 1. The flow diagram of our approaches.

<!-- FIGURE-DATA: FIGURE 1 | type: diagram -->
> **[Extracted Data]**
> - Flow diagram: 5-stage transfer learning pipeline
> - Components: Source/Target data, Pre-train, Fine-tune, TL-based-US
> **Analysis:** Transfer learning pipeline with pre-training, iterative fine-tuning, and model-based undersampling.
<!-- /FIGURE-DATA -->
In pre-training, the CNN architecture we used was TextCNN [25], whose structure was slightly tweaked. We set up 4 kinds of convolution filter of different sizes, respectively, 2,3,4,5, and the number of channels per filter was 128. Compared with the original model structure, we have added a convolution kernel with a filter size of 2 to extract more fine-grained sentiment features in Chinese. After the convolution layer we connected the batch-normalization (BN) layer to deal with the parameter distribution of the convolution kernel, thus allowing the model to converge more quickly, and improving the recall rates of minority in $T _ { d }$ (which will be introduced later). Then we added the activation layer and the max pooling layer, before integrating with the fully-connected layer and the Softmax layer.

# B. DNN MODEL FINE-TUNING
<!-- FIGURE-DATA: FIGURE 2 | type: diagram -->
> **[Extracted Data]**
> - CNN architecture for text classification
> - Frozen and trainable layers, pooling, concatenation
> **Analysis:** CNN architecture processes word embeddings into predictions.
<!-- /FIGURE-DATA -->

Fine-tuning is a means of TL, which uses the parameters of the pre-trained model as the initialization value of the target task network $T _ { n e t }$ , and the labeled target domain data set to fine-tune the parameters of the $T _ { n e t }$ . In the 2nd and 4th steps of the approaches we propose (as shown in Fig. 1), we fine-tune the model on data of the $T _ { d }$ . In these two steps,given the different purposes of fine-tuning (the second step of fine-tuning is one of operation in our proposed under-sampling method, and the fourth step fine-tuning, is able to promote the f1-value of minority class in the $T _ { d }$ ), so the way of fine-tuning can be different. Thus, this subsection focuses on the fourth step of fine-tuning. We know that the hierarchical layered architecture of DNNs provides flexibility for building models and is therefore well suited for TL. When using a pre-trained model to transfer knowledge to a target task for fine-tuning, we can choose to transfer the embedding $( \mathcal { E } )$ layer, the convolution $( \mathcal { C } )$ layer, the fully connected hidden $( { \mathcal { H } } )$ layer and the output $( \mathcal { O } )$ layer, a layer or layers of one of these [26]. In our experiment, we chose to transfer the $\mathcal { E }$ and $\mathcal { C }$ layers so that we could fully utilize the n-grams features learned by the model in the $S _ { d }$ and make the model more adaptive to the target task by fine-tuning the $\mathcal { H }$ and $\mathcal { O }$ layers. Figure 2 shows the diagram.

![](images/703aa6f5f9e38210b730a79c8bcb4b7af927bd6f2c9f2d252be394e613695599.jpg)  
FIGURE 2. Transfer diagram of step 4.

The light blue convolution layer and the pooling layer in Figure 2 show the transfer from the pre-trained model1, and we freezed its parameters without updating them during training. As shown in Fig 2, besides transferring the $\mathcal { O }$ layer from the pre-trained model, we also trained the $\mathcal { C }$ layer (shown in light brown) to extract the semantic information features for the target task. We know that in the text classification task, CNN extracts the n-gram features and the local order relationship of the statements. The transfer light blue $\mathcal { C }$ layer has four convolution kernels of different sizes, which are 2, 3, 4, and 5, which can extract some domain-independent sentiment features. For the retrained light brown $\mathcal { C }$ layer, we set up three sizes of convolution kernels, 5, 6, and 7, respectively, to learn some context-related long-term dependency features and some domain related sentiment features in the $T _ { d }$ . Then the output of these two parts of the $\mathcal { C }$ layer to undergo the max pooling operation, the pooling output of the two vectors (called $p _ { 1 }$ and $p _ { 2 }$ , respectively) are concatenated together into a long vector. The symbol $\textcircled{+}$ represents the vector concatenate operation.

Assuming that the sentence length is $n$ (padded where necessary), the convolution layer filter size is $w$ , then convolution operation will extract $n - w + 1$ features from the sentence to form a feature map [25].

$$
c = [ c _ { 1 } , c _ { 2 } , \ldots , c _ { n - w + 1 } ]
$$

Here $c _ { i }$ represents the output value of the convolution operation. The max pooling operation is performed on the feature maps and the maximum value $\widetilde { c } = m a x \{ c \}$ on the feature map is obtained as the most important feature. If the convolution layer of the pre-trained model has $s _ { 1 }$ filters of different sizes, the number of kernel filters per size will be $k _ { 1 }$ ; the corresponding numbers in the newly trained convolution layer are $s _ { 2 }$ and $k _ { 2 }$ , respectively. Then the dimensions of the vectors output by the pooling operation are $s _ { 1 } ~ * ~ k _ { 1 }$ and $s _ { 2 } ~ * ~ k _ { 2 }$ , respectively.

$$
\begin{array} { r } { p _ { 1 } = [ \widetilde { c _ { 1 , 1 } } , \widetilde { c _ { 1 , 2 } } , \ldots , \widetilde { c _ { 1 , s _ { 1 } * k _ { 1 } } } ] } \\ { p _ { 2 } = [ \widetilde { c _ { 2 , 1 } } , \widetilde { c _ { 2 , 2 } } , \ldots , \widetilde { c _ { 2 , s _ { 2 } * k _ { 2 } } } ] } \end{array}
$$

The result of pooling vector after being concatenated is $p = [ p _ { 1 } , p _ { 2 } ]$ . We can control the ratio relationship between the number of features of the transfer and the number of those newly extracted by adjusting the values of $s _ { 2 }$ and $k _ { 2 }$ to make the model get the best performance on $T _ { d }$ . After the pooling layer we use Dropout [27] to prevent over-fitting.

# C. TRANSFER LEARNING-BASED UNDER-SAMPLING

Our proposed TL-based-US approach, contains three steps, shown as step1,step2 and step3 in Fig. 1. First we use the $S _ { d }$ pre-training model model1; then we sample in $T _ { d }$ to get the sub-dataset mini $T _ { d }$ with the same ratio of positive and negative samples, and then use model1 to fine-tune on the mini $T _ { d }$ to get model2; Finally, using model2 to classify majority class of the $T _ { d }$ , we classify the wrong samples as under-sampling samples. From this perspective, TL-based-US is a model-based supervised under-sampling method.

Different from step 4 of fine-tuning, we only fine-tune pre-trained model, and no additional training newly $\mathcal { C }$ layer. We freezed the $\mathcal { E }$ layer and the $\mathcal { C }$ layer, and only fine-tune the $\mathcal { H }$ and $\mathcal { O }$ layer. The entire fine-tuning training is similar to INIT [21].

In the third step, we classify the positive comments (majority class) of the $T _ { d }$ data set by the fine-tuned model model2 and find that most of the samples have been correctly classified. For example, the number of positive comments samples in the training set of $T _ { d }$ is 125,331, of which 100,374 are classified as positive comments, 24,957 are classified as negative ones, and the number of wrong samples is similar to the proportion of negative comments in the training set of $T _ { d }$ After analyzing the correctly classified samples we found that most of the comments containing domain-independent positive sentiment words (such as good, beautiful, cost-effective, affordable, convenient, clean, etc.) and others containing domain-related positive sentiment words (such as beautiful scenery, cheap tickets, etc.) had been correctly classified. Intuitively, we can use positive review samples of classification errors as under-sampling samples for majority class. This allows us, on the one hand, to balance the proportion of the majority and minority classes in the target domain training set (from 52:1 to 10:1) and, on the other hand, to make the fine-tuning training in step 4 more responsive to the distribution of the target domain data set.

In the above, we introduced several re-sampling techniques, all of which use the NNA to re-sample. As is known, traditional machine learning algorithms (including NNA) divide sentences into words when dealing with NLP tasks, and then represent all words as Bag of Words (BoW) model, which destroys the order of statements and fails to capture the semantic information between words. To some extent, these disadvantages can be overcome by using the deep learning method, and similarities between different words can be calculated by using word vectors, and the design of convolution kernel enables the model to capture the local orderliness of the statements. In addition, the knowledge learned can be transferred to the $T _ { d }$ using the TL method with the $S _ { d }$ data set. Therefore, the case of using TL-based-US to under-sample majority class is equivalent to using prior knowledge for selecting samples.

# IV. EXPERIMENTS AND RESULTS

In order to find the best model structure and hyperparameters, and further verify the superiority of our proposed method, we carried out multiple sets of comparative experiments on the real-world Chinese sentiment classification data sets. All experiments were divided into two groups. In the first group, we compared the performance of TL-based-US with that of some other existing under-sampling methods. In the second group, we compared our method and two algorithms, BalanceCascade [13] and Multi-model Fusion [28], specifically designed for imbalanced data sets. For all experiments we used a 5-fold cross-validation approach to optimize the model in the development set. And then used $T _ { d }$ -test to verify the final performance of the under-sampling methods or the model.

# A. PERFORMANCE MEASURES

The metrics of classification algorithm include the accuracy rate, error rate, ROU-AUC curve, precision, recall, and f1-value, among which, the accuracy rate and error rate are applicable to the balance data sets, and the ROU-AUC curve is applicable to binary classification problems in unbalanced classes. Because this paper mainly deals with the classification of extremely unbalanced data sets, our goal is to differentiate as many negative comments as possible to help enterprises improve their service quality. Therefore, we need a model with a higher recall rate for negative comments, and at the same time we do not want to mistake positive comments for negative comments. In other words, the model shall ideally have higher precision for identifying negative comments. So f1-value is chosen the performance measure to evaluate the model. However, as the ratio of positive and negative samples is very imbalanced on testset, it adds confusion when using the average f1-value and f1-value of majority. Taking into account the above considerations, we choose f1-value of negative comments as the indicator for evaluating the performance of the model. The calculation of f1-value is described below.

By convention, we refer to the minority classes in the data set as positive(P) and the majority classes as negative (N). As shown in Table 2 of the confusion matrix, TP and TN, respectively, represent the correct number of predictions in positive and negative cases; in contrast, FP and NP represent the number of prediction errors in positive and negative cases, respectively.

TABLE 2. Confusion matrix.   

<table><tr><td></td><td>True Positive</td><td>True Negative</td></tr><tr><td>Classified Positive</td><td>TP</td><td>FP</td></tr><tr><td>Classified Negative</td><td>FN</td><td>TN</td></tr></table>

Precision stands for the number of true positive comments that are classified as positive divided by the number of all classified as positive. The recall is the number of true positive comments classified as positive divided by the number of all true. They are defined as:

$$
\begin{array} { r } { P r e c i s i o n = \frac { T P } { T P + F P } } \\ { R e c a l l = \frac { T P } { T P + F N } } \end{array}
$$

F1-value takes into account both the precision and recall of the classification model. It can be regarded as a weighted average of the model’s precision and recall rate, with a maximum value of 1 and a minimum value of 0. The definition is as follows:

$$
F 1 \ – \nu a l u e = \frac { 2 * R e c a l l * P r e c i s i o n } { R e c a l l + P r e c i s i o n }
$$

# B. DATA SETS

We choose data set of reviews on Chinese hotels, MinChnCorp [23] as the $S _ { d }$ data set, which contains about 1 million pieces of hotel reviews, each with a corresponding rating score ranging between 1-5. The $T _ { d }$ is the data about reviews on tourist attractions1 with a total of 220,000 reviews. Similarly, it has a corresponding score for each review, with the same scoring range as the former. In the two data sets we removed comments with a score of 3 for containing ambiguous sentiment polarity, and deleted redundant, repetitive comments. In addition, the comments rated as 1, 2 were expressed as positive-polarity samples, and the comments with a rating of 4 and 5 were expressed as negative-polarity samples. In addition, predominantly concerned with sentence-level sentiment classification, we deleted comments in two data sets that are longer than 60 words. Finally, all the samples were processed, the final data sample information was retained as shown in Table 3.

TABLE 3. Data set information of $\pmb { S _ { d } }$ and $\pmb { \tau _ { d } }$ .   

<table><tr><td colspan="2"></td><td>MinChnCorp</td><td>Tour-review</td></tr><tr><td rowspan="4">Raw data</td><td>Total number</td><td>1,037,337</td><td>220,000</td></tr><tr><td>positive number</td><td>799,565</td><td>194,790</td></tr><tr><td>negative number</td><td>78,477</td><td>3,246</td></tr><tr><td>number of score 3</td><td>159,295</td><td>21,964</td></tr><tr><td rowspan="3">Cleaned data</td><td>Total number</td><td>53,336</td><td>127,725</td></tr><tr><td>positive number</td><td>27,610</td><td>125,331</td></tr><tr><td>negative number</td><td>25,726</td><td>2,394</td></tr></table>

As can be seen from Table 3, the ratio of samples with positive and negative sentiment polarity in the $T _ { d }$ is 52:1, and there are only 2394 positive(P) samples. We take one-tenth of them as a test set $\mathit { T } _ { d }$ -test) and the rest as a development set. Because the number of minority samples is too small, we over-sampled the minority samples by randomly splicing the negative comments two or two together to get statement $S _ { c o n c a t }$ , prior to randomly removing the individual words in $S _ { c o n c a t }$ and scrambling the order. In this way, the number of negative samples from the $T _ { d }$ data set increased from 2,154 to 6,000.

# C. HYPERPARAMETER SETTING
<!-- FIGURE-DATA: FIGURE 3 | type: plot -->
> **[Extracted Data]**
> - Bar chart: Precision, Recall, F1-value comparison
> - TL-based-US vs other methods
> **Analysis:** TL-based-US achieves highest scores across all metrics.
<!-- /FIGURE-DATA -->

Our model framework involves five steps, in particular, model pre-training (1st step), model fine-tuning (step 2 and 4). Neural network models have been used, and there are some differences in the training parameters of these different stages. Although we mentioned some parameter settings previously, it is necessary for us to describe in detail in this section some of the parameters that need to be noted in all our models. The specific parameter settings are shown in Table 4.

In Table 4, (2, 3, 4, 5) means that convolution kernels of three different sizes are set, and the sizes are 2, 3, 4, and 5, respectively. In addition, we can see that the filter size in the fourth step is (2,3,4,5), (5,6,7), because we also use the migrated convolution layer and the separately trained convolution layer in the target data set. BN refers to the batch normalization regularization technique.

<!-- FIGURE-DATA: FIGURE 4 | type: plot -->
> **[Extracted Data]**
> - Bar chart: Ours vs BalanceCascade vs Multi-model Fusion
> **Analysis:** Ours framework outperforms others in all metrics.
<!-- /FIGURE-DATA -->
# D. EXPERIMENT RESULTS

In order to verify the superiority of our proposed method, we carried out some comparative experiments on the $T _ { d }$ data set. First, we used TextCNN [25] as a classification model to compare the classification effects of non-undersampling (Non-US) and several different under-sampling strategies. The benchmarks for under-sampling methods are random under-sampling (Random-US), NearMiss [7] and repeated edited nearest neighbor under-sampling (RENN-US)2 [29]. Figure 3 shows the precision, recall and f1-value of the various under-sampling methods on the testset of $T _ { d }$ (minority class).

<!-- FIGURE-DATA: FIGURE 5 | type: plot -->
> **[Extracted Data]**
> - Validation accuracy vs regularization techniques
<!-- FIGURE-DATA: FIGURE 6 | type: plot -->
> **[Extracted Data]**
> - F1-value after fine-tuning with different regularization
> **Analysis:** Shows F1 scores for various regularization techniques.
<!-- /FIGURE-DATA -->
> **Analysis:** Shows impact of different regularization techniques on pre-training.
<!-- /FIGURE-DATA -->
![](images/99a75fba6de7b4a20967ad72a58241812c3f762135ae622abc07bd23233834bb.jpg)  
FIGURE 3. Performance measures results for the comparison between TL-based-US and other methods.

As can be seen from Figure 3, ours proposed undersampling method TL-based-US have obvious advantages over other methods. The precision, recall, and f1-value of the test set in the $T _ { d }$ reached $56 \% { , } 5 8 \%$ and $57 \%$ respectively, higher than other methods by over 5 percentage points. Because of the very uneven proportion of positive and negative samples in the data set and the small number of minority samples, the original TextCNN model had quite poor performance. Without under-sampling for majority class in the training data, the precision, recall, and f1-value would have only yielded $3 5 \%$ , $34 \%$ , and $3 5 \%$ , respectively. For the existing under-sampling techniques, the NNA based on measuring distance is mostly employed for deleting the majority of samples which are partially clustered together so as to achieve the goal of under-sampling. The NNA also uses VSM to represent sentences for modeling, which makes the under-sampling of text data undesirable. On the other hand, because most samples of majority class are near majority class, the number of majority class that can be deleted by this method are very limited. For random under-sampling methods, it is clear that there is a risk of losing some of the key information in majority class.

Although we can use the under-sampling to greatly reduce the number of samples in majority classes, sometimes we still can’t balance the sample ratio between different classes in the dataset. For example, in our target dataset, we can reduce the majority and minority ratios from 52:1 to 10:1, but if we sample most of the majority classes, it will result in great loss of information, as a consequence of which, the trained model does not have better generalization ability. Therefore, in addition to under-sampling, it is necessary to design the model to fit the classification of the imbalanced data set. We use TL-based-US to under-sample the target dataset and then compare it with the BalanceCascade [13] and Multi-model Fusion [28] using our proposed model. The experimental results are shown in Figure 4.

TABLE 4. Hyperparameters setting in ours framework.   

<table><tr><td></td><td>step1(pre-training)</td><td>step2(fine-tune)</td><td>step4(fine-tune)</td></tr><tr><td>Embedding size</td><td>200</td><td>200</td><td>200</td></tr><tr><td>max document length</td><td>60</td><td>60</td><td>60</td></tr><tr><td>filter sizes</td><td>(2,3,4,5)</td><td>(2,3,4,5)</td><td>(2,3,4,5),(5,6,7)</td></tr><tr><td>number of filters</td><td>128</td><td>128</td><td>128,64</td></tr><tr><td>full-connect size</td><td>300</td><td>300</td><td>500</td></tr><tr><td>batch size</td><td>64</td><td>64</td><td>128</td></tr><tr><td>learning rate</td><td>0.001</td><td>0.0001</td><td>0.0001</td></tr><tr><td>regularization</td><td>BN</td><td>BN</td><td>Dropout</td></tr></table>

![](images/7e0bc3d4e31cd72f775951cd8dcbcb7a2e982c82a539dc674f9f8bb0e5e8cca8.jpg)  
FIGURE 4. Performance measures comparison of different model frameworks (after under-sampling with TL-based-US).

As shown in Figure 4, our proposed framework outperforms others substantially, with f1-value reaching $63 \%$ which are 5 and 12 percentage points higher than that of BalanceCascade and Multi-model Fusion, respectively. This is due to our TL model being able to efficiently migrate knowledge learned from the $S _ { d }$ to the $T _ { d }$ . However, for Balancecascade, as in [13], we use AdaBoost as its metaclassifier. For Multi-model Fusion, it uses models such as naive Bayes, decision trees, logistic regression, and support vector machines for fusion. These algorithms all use the VSM to characterize the text. However, the VSM tends to ignore the order information of the sentences and cannot express the relationship between the similar words, so it is not very good in for the task of text classification, especially the sentiment classification task.

# E. REGULARIZATION METHODS IN TRANSFER LEARNING

It is found that using batch normalization can make the model converge faster with a higher recall rate for minority class, which we mentioned in Section III-A. In the first step of model pre-training and the second step of model fine-tuning, we use batch normalization as the regularization technique. In the pre-training model, when using batch normalization, the model reaches the convergence state around 3000 steps, and the model needs to train more than 5,800 steps to converge when using dropout, as shown in Figure 5. In step 5 of our proposed approach, the target domain test set is first predicted using model2, and then the samples Tag-neg2, which predicts negative polarity emotion, is predicted again to model3. It can be seen that model2 needs a good recall rate for negative polarity emotion samples to ensure a high recall rate in the end. Table 5 shows the precision, recall, and f1-value of the minority class of model2 in Td-test when the two regularization techniques, batch normalization and dropout, are used in the second step of model fine-tuning.

![](images/93fc232fe990a7d16ef96363f6af9ba3fb3d3aa230dab618decd1f14b2910704.jpg)  
FIGURE 5. Validation accuracy of using different regularization techniques for pre-training.

![](images/a8ea643d73414ff41cddf163109ef7e9fa7006f6cb6190bb33ceca1ce441c221.jpg)  
FIGURE 6. F1-value obtain after fine-tuning in step 4 by using different regularization techniques.

TABLE 5. Precision, Recall, F1-value for different regularization techniques on $\pmb { \tau _ { d } }$ test set in Step 2.   

<table><tr><td>regularization</td><td>Precision</td><td>Recall</td><td>F1-value</td></tr><tr><td>batch normalization</td><td>0.27</td><td>0.81</td><td>0.41</td></tr><tr><td>dropout</td><td>0.46</td><td>0.73</td><td>0.56</td></tr></table>

While the use of batch normalization can improve recall rates for minority class, it can also be seen that the precision is very low, only $27 \%$ . Therefore, in step 4 of the model fine-tuning, we employed dropout as a regularization tool, as described in Section III-A. Applying batch normalization and dropout into the model fine-tuning in step 4, the final f1-value is obtained in the minority class of the $T _ { d }$ test set, as shown in Figure 6.

# V. CONCLUSION

In this paper, we have introduced a transfer learning method for solving the sentiment classification problem in imbalanced datasets. It includes fine-tuning of the pre-trained model and under-sampling of text data based on transfer learning. We have verified the effective performance of this method based on real sentiment classification datasets. It is important to note, however, that our proposed approach is not without its limitations. First, we need a similar external dataset with a sufficient number of samples as the $S _ { d }$ to pre-train the model; second, we only use TL-based-US for sentiment classification and it works after verification; nonetheless, we are not sure if it works for other text classification tasks. It can be concluded that more comprehensive research is needed in this direction in the future.

# REFERENCES

[1] V. N. Chawla, N. Japkowicz, and A. Kotcz, ‘‘Editorial: Special issue on learning from imbalanced data sets,’’ Acm Sigkdd Explor. Newslett., vol. 6, no. 1, pp. 1–6, Jun. 2004.   
[2] H. He and E. A. Garcia, ‘‘Learning from imbalanced data,’’ IEEE Trans. Knowl. Data Eng., vol. 21, no. 9, pp. 1263–1284, Sep. 2009.   
[3] D. L. Wilson, ‘‘Asymptotic properties of nearest neighbor rules using edited data,’’ IEEE Trans. Syst., Man, Cybern., vol. SMC-2, no. 3, pp. 408–421, Jul. 1972.   
[4] Y. Sun, M. S. Kamel, A. K. C. Wong, and Y. Wang, ‘‘Cost-sensitive boosting for classification of imbalanced data,’’ Pattern Recognit., vol. 40, no. 12, pp. 3358–3378, 2007.   
[5] M. V. Joshi, Learning Classifier Models for Predicting Rare Phenomena. Minneapolis, MI, USA: University Minnesota, 2002.   
[6] L. Shuyang, L. Cuihua, J. Yi, L. Chen, and Q. Zou, ‘‘Under-sampling method research in class-imbalanced data,’’ J. Comput. Res. Develop., vol. 48, pp. 47–53, Aug. 2011.   
[7] J. Zhang and I. Mani, ‘‘KNN approach to unbalanced data distributions: A case study involving information extraction,’’ in Proc. ICML, Aug. 2003, p. 126.   
[8] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, ‘‘SMOTE: Synthetic minority over-sampling technique,’’ J. Artif. Intell. Res., vol. 16, no. 1, pp. 321–357, 2002.   
[9] J. Song, X. Huang, S. Qin, and Q. Song, ‘‘A bi-directional sampling based on $K$ -means method for imbalance text classification,’’ in Proc. IEEE/ACIS Int. Conf. Comput. Inf. Sci., Jun. 2016, pp. 1–5.   
[10] X. Yung and Z. H. Zhou, ‘‘The influence of class imbalance on cost-sensitive learning: An empirical study,’’ in Proc. Int. Conf. Data Mining, Jan. 2007, pp. 970–974.   
[11] B. Raskutti, ‘‘ Extreme re-balancing for SVMS: A case study,’’ Acm Sigkdd Explor. Newslett., vol. 6, no. 1, pp. 60–69, Jun. 2004.   
[12] F. He, H. Yang, Y. Miao, and R. Louis, ‘‘A cost sensitive and class-imbalance classification method based on neural network for disease diagnosis,’’ in Proc. Int. Conf. Inf. Technol.Med. Edu., pp. 7–10, Dec. 2017.   
[13] X.-Y. Liu, J. Wu, and Z.-H. Zhou, ‘‘Exploratory undersampling for class-imbalance learning,’’ IEEE Trans. Syst., Man, Cybern. B, Cybern., vol. 39, no. 2, pp. 539–550, Apr. 2009.   
[14] S. J. Pan and Q. Yang, ‘‘A survey on transfer learning,’’ IEEE Trans. Knowl. Data Eng., vol. 22, no. 10, pp. 1345–1359, Oct. 2010.   
[15] J. Johnson, A. Alahi, and F. F. Li, ‘‘Perceptual losses for real-time style transfer and super-resolution,’’ in Computer Vision—ECCV (Lecture Notes in Computer Science), vol. 9906, B. Leibe, J. Matas, N. Sebe, and M. Welling, Eds. 2016, pp. 694–711.   
[16] D. Chen, L. Yuan, J. Liao, N. Yu, and G. Hua, ‘‘Stereoscopic neural style transfer,’’ in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Salt Lake City, UT, USA, Apr. 2018, pp. 6654–6663.   
[17] B. Mccann, J. Bradbury, C. Xiong, and R. Socher. (2017). ‘‘Learned in translation: Contextualized word vectors.’’ [Online]. Available: https://arxiv.org/abs/1708.00107   
[18] O. Abdelwahab and A. Elmaghraby, ‘‘Uofl at semeval-2016 task 4: Multi domain word2vec for twitter sentiment classification,’’ in Proc. Int. Workshop Semantic Eval., 2016, pp. 164–170.   
[19] D. Bollegala, T. Mu, and J. Y. Goulermas, ‘‘Cross-domain sentiment classification using sentiment sensitive embeddings,’’ IEEE Trans. Knowl. Data Eng., vol. 28, no. 2, pp. 398–410, Feb. 2016.   
[20] J. Yu and J. Jiang, ‘‘Learning sentence embeddings with auxiliary tasks for cross-domain sentiment classification,’’ in Proc. Conf. Empirical Methods Natural Lang. Process., Aug. 2016, pp. 236–246.   
[21] L. Mou et al., ‘‘How transferable are neural networks in nlp appications?’’ in Proc. EMNLP, Aug. 2016, pp. 1–9.   
[22] J. Howard and S. Ruder. (2018). ‘‘Universal language model fine-tuning for text classification.’’ [Online]. Available: https://arxiv.org/abs/1801.06146   
[23] Y. Lin, H. Lei, J. Wu, and X. Li, ‘‘An empirical study on sentiment classification of chinese review using word embedding,’’ in Proc. 29th Pacific Asia Conf. Lang., Inf. Comput., Posters, Shanghai, China 2015, pp. 258–266.   
[24] T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean, ‘‘Distributed representations of words and phrases and their compositionality,’’ in Proc. Int. Conf. Neural Inf. Process. Syst., Aug. 2013, pp. 3111–3119.   
[25] Y. Kim. (2014). ‘‘Convolutional neural networks for sentence classification.’’ [Online]. Available: https://arxiv.org/abs/1408.5882   
[26] T. Semwal, P. Yenigalla, G. Mathur, and S. B. Nair, ‘‘A practitioners’ guide to transfer learning for text classification using convolutional neural networks,’’ in Proc. SIAM Int. Conf. Data Mining, San Diego, CA, USA, 2018, pp. 513–521.   
[27] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, ‘‘Dropout: A simple way to prevent neural networks from overfitting,’’ J. Mach. Learn. Res., vol. 15, no. 1, pp. 1929–1958, 2014.   
[28] L. Gan, R. Benlamri, and R. Khoury, ‘‘Improved sentiment classification by multi-modal fusion,’’ in Proc. IEEE Third Int. Conf. Big Data Comput. Service Appl., Apr. 2017, pp. 11–16.   
[29] I. Tomek, ‘‘An experiment with the edited nearest-neighbor rule,’’ IEEE Trans. Syst., Man, Cybern., vol. SMC-6, no. 6, pp. 448–452, Jun. 1967.

![](images/a9b5155901620585262a5fa900d4c1c6c5fefaf9ae50c2eac811431470401e25.jpg)

Z. XIAO received the B.Sc. degree in communication engineering from Hunan University, in 2003, and the Ph.D. degree in computer science from Fudan University, China, in 2009.

He is currently an Assistant Professor with the College of Information Science and Engineering, Hunan University. His research interests include parallel and distributed computing, distributed artificial intelligence, and collaborative computing.

![](images/e86506af94263f098f1da14169658b63dbbf42b655c3c267efd9c1081cf0cc91.jpg)

L. WANG was born in Zhuzhou, Hunan, China, in 1988. He received the bachelor’s degree from the Hunan University of Science and Engineering, in 2012. He is currently pursuing the master’s degree with the College of Computer Science and Electronic Engineering, Hunan University.

Since 2016, he has been carrying out research under the supervision of Mr. X. Zheng. His research interests include deep learning, transfer learning, and natural language processing.

![](images/49ccefab5ee8ee185a30390a5f8cf007e003773ad1d6bdcd2991a6d4652aba33.jpg)

J. Y. DU received the B.Sc., M.Sc., and Ph.D. degrees in computer science from Hunan University, China, in 2015, 2010, and 2004, respectively.

He is currently an Assistant Professor with the Central South University of Forest and Technology, China. His research interests include modeling and scheduling for parallel and distributed computing systems, embedded system computing, cloud computing, parallel system reliability, and parallel algorithms.