# Sentiment Analysis of Imbalanced Tourism News Dataset Using Random Forest with Particle Swarm Optimization and Synthetic Minority Oversampling

Husni   
Department of Informatics   
University of Trunojoyo Madura   
Bangkalan, Indonesia   
husni@trunojoyo.ac.id   
Arif Muntasa   
Department of Informatics   
University of Trunojoyo Madura   
Bangkalan, Indonesia   
arifmuntasa@trunojoyo.ac.id

Vina Angelina Savitri Department of Informatics University of Trunojoyo Madura Bangkalan, Indonesia 190411100170@student.trunojoyo.ac.id

Abstract—This paper reports the results of research on the classification of Indonesian tourism news texts based on their sentiment using the Random Forest method combined with feature selection and sampling techniques. The dataset comprised lots of tourism news related to Madura, East Java. Feature selection in the classification process was performed using Particle Swarm Optimization (PSO) to identify influential features. Subsequently, the dataset was divided into two parts: training and testing data. In the training data, an imbalance in the number of positive and negative classes led to classification results biased towards the majority class. Therefore, this study employed the Synthetic Minority Oversampling Technique (SMOTE) to balance class numbers in the training data. Following that, the classification process utilized the Random Forest method to determine the accuracy of this study. The obtained results revealed an accuracy rate of $91 \%$ as the average accuracy when using the PSO and Random Forest methods. The PSO feature selection method contributed to accelerating computation time compared to not using feature selection methods.

Keywords—imbalanced tourism news, particle swarm optimization, random forest, sentiment analysis, synthetic minority oversampling

# I. INTRODUCTION

Tourism is a major contributor to economic growth, with developing countries increasingly recognizing its potential. For instance, it is noted that tourism can significantly reduce poverty and create job opportunities, with one in nine jobs globally linked to the sector [1]. The multiplier effect of tourism leads to increased income and employment in related sectors, enhancing overall economic stability [2]. The tourism industry is a vital source of foreign exchange [3]. Increased tourism can attract foreign investment, further stimulating economic development and infrastructure improvements. Beyond economic impacts, tourism fosters socio-cultural exchanges and community engagement, which are essential for sustainable development. The integration of local communities in tourism planning ensures that socio-cultural impacts are considered, promoting a more inclusive approach to development.

The impact of freely accessible news on the internet for the tourism sector is profound, as it enhances information dissemination, influences tourist behavior, and promotes destinations effectively. The integration of news content into tourism marketing strategies has become essential for attracting visitors and fostering engagement with cultural heritage [4]. The internet facilitates instantaneous access to a wealth of tourist information, allowing travelers to make informed decisions without intermediaries [5]. News-based websites serve as reliable sources for current information about destinations, significantly influencing tourists' purchasing decisions [6]. Online news influences destination image and tourist behavior, as travelers often rely on news to shape their perceptions of potential destinations. The decision making of potential tourists depends on the perception, opinion or sentiment they hold. This is where a sentiment analysis approach is needed for information about tourism destinations, whether written in the form of news or reviews.

Sentiment analysis of tourism news is crucial for understanding public perceptions and enhancing decisionmaking within the tourism industry [7][8]. By analyzing sentiments expressed in news articles, stakeholders can gauge the popularity of destinations, identify areas for improvement, and tailor marketing strategies effectively. This analysis not only aids in enhancing tourist experiences but also supports local economies [9][10]. Sentiment analysis helps in assessing how tourism news influences public opinion about destinations, which can affect tourist behaviour. Positive sentiment can drive tourism growth, benefiting local economies, as seen in studies focusing on regions like Madura Island, Indonesia [11].

There are many methods of tourism news analysis based on sentiment that are widely used, one of which is the classification approach using decision trees as in [11]. Decision tree algorithm provide a clear visual representation of decision-making processes, making it easier for users to understand how classifications are made. They can effectively handle feature selection, which enhances performance by filtering out irrelevant features [12]. Decision trees can achieve high accuracy rates [13]. However, Decision trees are prone to overfitting, especially with complex datasets, which can lead to poor generalization on unseen data [14]. They can be sensitive to noisy data, which may result in inaccurate classifications, particularly in informal text data [15]. Decision trees may struggle with imbalanced datasets, leading to biased predictions towards the majority class [16].

The shortcomings of decision tree algorithms such as C4.5 can be overcome using ensemble-based approaches from the decision tree family such as Random Forest. Random forest constructs multiple decision trees during training and merges their outputs, which helps in capturing diverse patterns in data. Each tree is trained on a random subset of the data, promoting diversity and reducing variance, which is crucial for effective sentiment classification [17][18]. Studies indicate that random forest can achieve classification accuracy improvements of up to $5 . 6 8 \%$ over traditional methods like C.45 [18]. Ensemble techniques, including random forest, have shown median performance enhancements of $5 . 5 3 \%$ across various sentiment classification datasets [17]. Random forests are less sensitive to noise and outliers compared to C.45, which can lead to more stable predictions in sentiment analysis tasks [19].

Random forest builds more trees than decision trees and this certainly consumes a lot of computing resources and time, especially when the data has a lot of features like in news datasets. Therefore, the approach that must be taken at first is feature reduction or selection to filter the most relevant features. Selecting relevant features can lead to significant improvements in classification accuracy, as demonstrated by various studies that report accuracy gains of up to $9 \%$ with optimized feature sets [20]. By minimizing the number of features, models can operate more efficiently, leading to faster training and prediction times [21]. Fewer features make it easier to understand the model's decision-making process, which is crucial in applications like public opinion analysis [22]. While feature selection is critical for improving sentiment analysis outcomes, it is also important to recognize that improper feature selection can lead to decreased model performance [23]. Selecting irrelevant features may introduce noise, ultimately hindering the model's ability to generalize effectively [24]. We have to select the most relevant features so that the classifier works efficiently and is able to provide the highest accuracy.

Traditional feature selection methods such as information gain, gain ratio, and mutual information have notable disadvantages in text classification, particularly in sentiment analysis. Information gain does not account for redundancy among features, which can lead to the selection of correlated features that do not contribute additional information [25]. Other methods struggle with noisy data, which is prevalent in sentiment analysis that can result in the selection of irrelevant features that degrade model performance. They may not be computationally efficient, especially when dealing with large datasets, leading to longer processing times and increased resource consumption [26]. We can use more sophisticated feature selection methods such as Particle Swarm Optimization (PSO) that has global search capabilities and adaptability make it particularly effective in high-dimensional datasets.

PSO employs a population-based approach, allowing it to explore multiple solutions simultaneously, which can lead to faster convergence compared to IG, which evaluates features individually [27][28]. The introduction of dynamic weight strategies in PSO variants, improves the balance between exploration and exploitation, enhancing overall performance [29]. PSO effectively eliminates redundant and irrelevant features through its grouping mechanisms, as demonstrated in studies where PSO outperformed IG in maintaining classification accuracy while reducing feature sets [28][30]. Hybrid approaches combining PSO with IG have shown significant improvements in classification accuracy, indicating that PSO can complement traditional methods rather than merely replace them [27]]. PSO is scalable to large datasets, making it suitable for big data applications, while traditional methods may become computationally expensive [31]. The ability to adapt PSO algorithms to specific datasets and classification tasks allows for tailored feature selection strategies that can outperform static methods like IG [30].

Another problem that must be addressed in the implementation of Random Forest is the presence of imbalanced data classes. The impact of data class imbalances on sentiment analysis is significant, as it can lead to biased models that favor majority classes, ultimately diminishing the performance for minority sentiments. Various studies have explored methods to mitigate these imbalances, demonstrating that addressing this issue is crucial for improving classification accuracy and ensuring fair representation of all sentiment classes [32][33][34]. The solution to this problem is resampling such as Random Oversampling and Undersampling have shown varying effectiveness, with Random Oversampling yielding better results in certain contexts [33]. One of the most popular sampling techniques is SMOTE (Synthetic Minority Oversampling Technique).

SMOTE offers several advantages over other data class balancing methods, particularly in its ability to generate synthetic samples that enhance minority class representation. SMOTE generates synthetic samples by interpolating between existing minority class instances, which helps to create a more balanced dataset without simply duplicating existing samples. Studies show that models utilizing SMOTE, such as Logistic Regression, achieved high performance metrics (e.g., accuracy of $8 7 . 2 4 \%$ and F1-score of $9 1 . 3 0 \%$ ) in detecting ontime graduation, outperforming other imbalance treatment methods [35]. Unlike some oversampling methods that may introduce noise and overfitting, SMOTE's approach of generating samples based on existing data points can mitigate these issues, especially when combined with noise-reduction techniques [36].

This article reports the results of research on sentimentbased text classification using the Random Forest method on an imbalanced tourism news dataset that has shown low performance in several studies. This problem is overcome by creating synthetic data using SMOTE. The problem of computational time due to the large number of features is solved by applying PSO to select the most relevant features to be included in the Random Forest classifier. The next section will explain all the approaches used in the study: Random Forest, PSO, and SMOTE, then continue with the results and discussion and close with the conclusion.

# II. PROPOSED METHODS

Text pre-processing is the initial process in preparing textbased datasets before the classification process takes place. This process is crucial because online news often contains noise and irrelevant parts, which can affect the accuracy of the classification results [37]. Pre-processing involves several stages for performing text data cleaning [38] namely, case folding, filtering, tokenizing, stop-word removal, and stemming. Term Frequency (TF) is the second process that aims to measure how often a term appears in a document [39]. Table 1 illustrates the approach to calculating Term Frequency (TF) [40].

The third process involves selecting features deemed important and will be used in the classification process. PSO (Particle Swarm Optimization) is one of the feature selection methods that mimics the principle of a flock of birds or fish, where the flock moves randomly in the search space to find the optimal solution to a problem [41][42].

TABLE I. CALCULATION OF TF   

<table><tr><td rowspan=1 colspan=1>Fields</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>1,0</td><td rowspan=1 colspan=1>The binary weight is assigned a value of 1 for termspresent in the vector, while 0 is assigned for terms notpresent in the vector (term frequency is disregarded).</td></tr><tr><td rowspan=1 colspan=1>Tf</td><td rowspan=1 colspan=1>The count of a term appearing in a document.</td></tr><tr><td rowspan=1 colspan=1>1 +log(tf)</td><td rowspan=1 colspan=1>The logarithm of TF to normalize the impact of very highTF.</td></tr><tr><td rowspan=1 colspan=1>r1 −r+tf</td><td rowspan=1 colspan=1>The inverse of the TF value, commonly denoted as r = 1.</td></tr></table>

The initial step in PSO calculation involves randomly initializing parameter values (population, iterations, c1, c2, w, position, and velocity). In PSO, each population contains several particles, and each particle has a position $X _ { i } =$ $( x _ { i 1 } , x _ { i 2 } , \dots , x _ { i D } )$ and velocity $V _ { i } = ( v _ { i 1 } , v _ { i 2 } , \ldots , v _ { i D } .$ . Equation (1) explains that if a position on a particle has a value of 1, the particle is selected to proceed to the next calculation. Conversely, if the position on a particle has a value of 0, it is not selected to proceed to the next calculation [43].

$$
( v _ { i d } ^ { n e w } ) = \frac { 1 } { 1 + e ^ { - v _ { i d } ^ { n e w } } }
$$

$$
i f { \left( r a n d < S ( v _ { i d } ^ { n e w } ) \right) t h e n } x _ { i d } ^ { n e w } = 1 ;
$$

$$
o t h e r w i s e x _ { i d } ^ { n e w } = 0
$$

The second step in the PSO calculation is to evaluate the fitness value of the selected particles using the formula refer to (2) [44]:

$$
\begin{array} { r l } & { f i t n e s s ( X _ { i } ) = k _ { 1 } \times s c o r e + k _ { 2 } } \\ & { ~ \times ~ ( n u m \_ o f \_ f e a t u r e s _ { x } ) ^ { - 1 } } \end{array}
$$

After obtaining the fitness value, the next step is to calculate the Pbest or personal best value for population i using the formula refer to (3):

$$
P b _ { i } = ( P b _ { i 1 } , P b _ { i 2 } , \dots , P b _ { i D } )
$$

The value of Pbest will be updated in each iteration by comparing the highest Pbest value for population i in each iteration. The selected Pbest is the highest Pbest value.

The next calculation process involves determining the value of Gbest or global best. The difference between Pbest and Gbest is that Gbest represents the selection of the single highest value among Pbest in one iteration (this value will continuously change with each iteration and select the highest value). Meanwhile, Pbest is the highest fitness value in one population for each iteration performed. The formula for Gbest refer to (4):

$$
G b = ( G _ { 1 } , G _ { 2 } , \dots , G _ { D } )
$$

Next is the update of position and velocity, or the calculation starts again for the desired number of iterations. The velocity update uses the formula refefer to (5) [45]:

$$
v _ { i j } ^ { t + 1 } = w v _ { i j } ^ { t } + c _ { 1 } r _ { 1 } \big ( P b _ { i j } ^ { t } - x _ { i j } ^ { t } \big )
$$

$$
+ c _ { 2 } r _ { 2 } \big ( G b _ { j } ^ { t } - x _ { i j } ^ { t } \big )
$$

After reaching the optimum solution, the feature selection process is halted, and the selected features proceed to the next stage.

The next process is the SMOTE oversampling technique, which aims to balance classes in the dataset by generating new synthetic data for the minority class [46][47]. The initial step in the formation of synthetic data is initializing the number of nearest neighbors and calculating the distance to the nearest neighbors using the Euclidean Distance formula refer to (6):

$$
d ( p , q ) = \sqrt { \sum _ { i = 1 } ^ { n } ( q _ { i } - p _ { i } ) ^ { 2 } }
$$

Subsequently, forming random samples from the k-nearest neighbors of the minority class [48].

The final process of this study is the Random Forest classification. The initial step involves initializing the number of trees to be created in the calculation process. The tree formation process typically employs bagging techniques, creating random samples of size n with recovery from the data cluster itself. The initial step is to initialize the number of trees to be created in the calculation process. The tree formation process typically involves using the bagging technique, creating random samples of size n with recovery from the data cluster itself [41]. The classification in Random Forest is determined based on the output from all trees, where the most frequently occurring result becomes the final outcome [49]. Random Forest is formed using the Gini index values to determine the separation that will be used as nodes with the following formula refer to (7):

$$
G i n i \left( S \right) = 1 - \sum _ { i = 1 } ^ { k } p i ^ { 2 }
$$

pi is the probability of S belonging to class i. After calculating the Gini value, the next step is to compute the Gini Gain using the following formula refer to (8) & (9):

$$
G i n i \left( { \cal A } , { \cal S } \right) = \sum _ { i = 1 } ^ { n } \frac { | S _ { i } | } { | { \cal S } | } G i n i ( S _ { i } )
$$

$$
G i n i G a i n \left( S \right) = G i n i \left( S \right) - G i n i \left( A , S \right)
$$

Where $\mathrm { S _ { i } }$ is the partition of S caused by attribute A.

# III. RESULT AND DISCUSSION

# A. Dataset

The dataset utilized in this research consists of 200 news articles about Madura tourism, sourced from various online news outlets, including kompas.com. Among these, 131 articles exhibit a positive sentiment, while 69 articles portray a negative sentiment (Fig. 1). The researcher manually collected the dataset, making it exclusive and not available through any public searches, including Kaggle.

# B. Text preprocessing and Term Frequency (TF)

The next stage of the research dataset involves preprocessing to eliminate items deemed irrelevant in the classification process. ne of the sentences that has not undergone preprocessing is, “Mengenal Gili Iyang di Sumenep, Pulau dengan Kadar Oksigen Terbaik Nomor 2 di

Dunia”. After going through all preprocessing stages, the sentence will transform into “[kenal, gili, iyang, sumenep, pulau, kadar, oksigen, baik, nomor, dunia]”. Subsequently, the Term Frequency (TF) process aims to determine how often a feature or word appears in a document.

# C. Classification

This research conducts two experiments to compare which yields better results. The first experiment employs the Random Forest method, along with SMOTE and PSO. Meanwhile, the second experiment excludes the use of PSO. The parameters utilized in this study are outlined in Table 2 [43][50].

TABLE II. VALUE OF PARAMETERS   

<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>C1</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>C2</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>n_estimator</td><td rowspan=1 colspan=1>200</td></tr><tr><td rowspan=1 colspan=1>Population</td><td rowspan=1 colspan=1>10, 20, 30, 40, 50, 60, 70, 80, 90, 100</td></tr><tr><td rowspan=1 colspan=1>max_depth</td><td rowspan=1 colspan=1>10</td></tr></table>

# D. Random Forest, SMOTE, and PSO

Fig. 1 represents the results of 10 experiments in the study using three methods, namely Random Forest, SMOTE due to unbalanced data (Fig. 2), and PSO. The highest accuracy in this experiment is obtained in population 40 with a value of $9 0 \%$ , while the lowest accuracy is found in population 70 with a value of $7 7 . 5 \%$ . The average accuracy obtained from these 10 experiments is $8 3 . 7 5 \%$ . In addition to accuracy calculations, this research also measures the processing time to obtain these accuracy values. The computation time results for this experiment are presented in Fig. 3.

![](images/51f341f82328f3d0cc683f57d3c68a59f0944d658aedbadb51ba3efb04250e82.jpg)  
Fig. 1. Accuracy of Random Forest, SMOTE, and PSO

<!-- FIGURE-DATA: Fig. 1 | type: plot -->
> **[Extracted Data]**
> - Dual-axis chart: bars for population, line for accuracy
<!-- FIGURE-DATA: Fig. 2 | type: diagram -->
> **[Extracted Data]**
> - Pie chart: Positive 65%, Negative 35%
<!-- FIGURE-DATA: Fig. 3 | type: plot -->
> **[Extracted Data]**
> - Line chart: Running time vs Population
> - Peak at population 30: ~0.99s
> **Analysis:** Running time varies with population, peak at population 30.
<!-- /FIGURE-DATA -->
> **Analysis:** Shows unbalanced data distribution - positive class dominates.
<!-- /FIGURE-DATA -->
> - X-axis: Population (1-10), Y1: 10-100, Y2: Accuracy (0.78-0.90)
<!-- FIGURE-DATA: Fig. 4 | type: plot -->
> **[Extracted Data]**
> - Running time vs Population
> - Peak ~0.51s at population 60
> **Analysis:** Running time fluctuates, peak at population 60.
<!-- FIGURE-DATA: Fig. 5 | type: plot -->
> **[Extracted Data]**
> - Accuracy vs Population (1-10)
> - Peak ~95% at populations 5, 6, 10
> **Analysis:** Accuracy fluctuates, peak at populations 5,6,10.
<!-- /FIGURE-DATA -->
<!-- /FIGURE-DATA -->
<!-- FIGURE-DATA: Fig. 7 | type: plot -->
> **[Extracted Data]**
> - Running time: PSO+RF vs PSO+SMOTE+RF
> - PSO+SMOTE+RF more stable and faster
> **Analysis:** PSO+SMOTE+RF is faster than PSO+RF.
<!-- /FIGURE-DATA -->
> **Analysis:** Peak accuracy ~0.90 at population 4. Larger population does not guarantee higher accuracy.
<!-- /FIGURE-DATA -->
![](images/94de1a526c165ea41f7d6877aa57a06e69088a6cfd31999aace88fcf56c76b38.jpg)  
Fig. 2. Percentage of labeled data (unbalanced)

![](images/569ff8685d5b6f79180fc4c88fb86c5fb7d5a90d73d41e03284799e6b417fc0d.jpg)  
Fig. 3. Running time of Random Forest, SMOTE, and PSO

Fig. 3. displays the computation time results for each population in this first experiment. The findings reveal that the longest computation time in this experiment is observed in population 60, taking 0.5064571 seconds, while the fastest computation is in population 40, clocking in at 0.2142687 seconds
<!-- FIGURE-DATA: Fig. 6 | type: plot -->
> **[Extracted Data]**
> - Bar chart: PSO+RF vs PSO+SMOTE+RF accuracy
> - PSO+RF consistently higher, often >0.90
> **Analysis:** PSO+RF outperforms PSO+SMOTE+RF in accuracy.
<!-- /FIGURE-DATA -->

# E. Random Forest and PSO

Fig. 4. presents the accuracy results obtained in the second experiment using only the Random Forest and PSO methods without incorporating the SMOTE oversampling technique. The results indicate that the highest accuracy values are achieved in populations 50, 60, and 100, with a score of $9 5 \%$ Meanwhile, the lowest accuracy is observed in population 30, with a value of $8 2 . 5 \%$ . The average accuracy obtained from the 10 conducted experiments is $91 \%$ .

![](images/fd93298cc60c465cea705fe4c5bfb5b215614a489a941f967187707a9da983da.jpg)  
Fig. 4. Running Time of Random Forest, SMOTE, and PSO

In addition to obtaining accuracy values, this research also calculates the computational processing speed in obtaining these accuracy values. Analyzing the computation time provides additional insights into the efficiency and performance of the proposed algorithm. The results of the computation time in this experiment are presented in Fig. 5, which focuses on comparing the aspects of computational processing speed for each population.

![](images/52cc4db84023d982c8c00acce0e56e01b1550c2f7c3cec911849414f0c07684b.jpg)  
Fig. 5. Accuracy of Random Forest and PSO

Fig. 5. illustrates the computation time results for each population in this first experiment. The findings reveal that the longest computation time in this experiment is observed in population 40, taking 0.9894934 seconds, while the fastest computation is in population 80, clocking in at 0.3483497 seconds.

significantly improved through SMOTE data class balancing where the average accuracy of the classifier before involving SMOTE is $84 \%$ and after reaching $91 \%$ . The presence of PSO and SMOTE does not require significant time during testing although both take time during data preparation before being used by Random Forest.

![](images/0d2fb9f16b91b1ccc897e6c0a3407250861cb3a7b5ebf8344258260f45832d82.jpg)  
Fig. 7. Comparison of Running time Values

# F. Analysis of Result

From both experiments, an overview of the comparison of accuracy results obtained is depicted in Fig. 6.

Fig. 6. depicts the comparison of accuracy values in each experiment. In the figure, it is evident that the use of the PSO and Random Forest methods without adding the SMOTE oversampling technique yields higher accuracy values. The same figure also shows that in no instance did the accuracy value using the SMOTE oversampling technique exceed without adding the SMOTE oversampling. This indicates that the addition of the SMOTE oversampling technique influences accuracy values. Next is the comparison of computational time as presented in Fig. 7.

![](images/73b872b0f264e8e815ccf0c81949294d83de720d7b4be87bfc406aaef8ff2564.jpg)  
Fig. 6. Comparison of Accuracy Values

Fig. 7. shows that the computation time for using the Random Forest and PSO methods produces slightly longer results. However, the time difference obtained is only in a matter of seconds.

# IV. CONCLUSION

The research that has been done shows that the performance of Random Forest and PSO in the application of sentiment analysis of tourism news datasets can be

# REFERENCES

[1] M.R. Khaksar, E. Amir, “The Contribution of Tourism to the Economic Growth of a Country”, International Journal of Current Science Research and Review, 2023, doi: 10.47191/ijcsrr/v6-i7-107   
[2] Y. Jiao, “Impacts of Tourism Development in Developing Countries”, Advances in hospitality, tourism and the services industry (AHTSI) book series, 2023, doi: 10.4018/978-1-6684-6796-1.ch004   
[3] A. Raja, A.P. Venkateswaran, “The contribution of tourism to economic growth in India”, Asian Journal of Research in Marketing, 2022, doi: 10.5958/2277-6621.2022.00005.6   
[4] R. Ramadhani, E. Setiawan, “Pengembangan Situs Web Untuk Promosi Warisan Budaya Lokal dan Pariwisata Berbasis Berita”, Jurnal Teknologi Sistem Informasi, 2024, doi: 10.35957/jtsi.v5i1.7594   
[5] C.M.Q. Ramos, P.M.M Rodrigues, “Os efeitos da internet na actividade turística the effects of the internet on tourism activity, 2011.   
[6] D. Kutlu, E.E. Bayraktar, H. Ayyildiz, “The Effect Of The Online News On Tourism” Gaziantep University Journal of Social Sciences, 2018, doi: 10.21547/JSS.335584   
[7] N.W.S. Saraswati et al., “Enhance sentiment analysis in big data tourism using hybrid lexicon and active learning support vector machine”, Bulletin of Electrical Engineering and Informatics, 2024, doi: 10.11591/eei.v13i5.7807   
[8] V.A. Narayana, B. Pooja, B.A. Goud, “Sentiment Analysis Based on Travelers’ Reviews Using the SVM Model with Enhanced Conjunction Rule-Based Approach”, International Journal for Research in Applied Science and Engineering Technology, 2024, doi: 10.22214/ijraset.2024.59108   
[9] O.A. Célia, M.Q. Ramos, “Sentiment analysis applied to tourism: exploring tourist-generated content in the case of a wellness tourism destination”, International journal of spa and wellness, 2024, doi: 10.1080/24721735.2024.2352979   
[10] M.B. Anley, G. Negashe, A.Y. Ayalew, “Opinion Mining of Tourists' Sentiments: Towards a Comprehensive Service Improvement of Tourism Industry”, 2024, doi: 10.1109/civemsa57781.2023.10231013   
[11] V.A. Savitri, M. Sa’id, H. Husni, A. Muntasa, “A sentiment analysis of madura island tourism news using C4.5 algorithm”, Journal of Soft Computing Exploration, 2024, doi: 10.52465/joscex.v5i1.258   
[12] R. Primartha, B.A. Tama., A. Arliansyah, K.J. Miraswan, “Decision tree combined with PSO-based feature selection for sentiment analysis, 2019, doi: 10.1088/1742-6596/1196/1/012018   
[13] H.H.F. Modwey, E. Elsamani, A.E. Elsamani, “Sentiment Analysis Bank of Khartoum Customers' comments Using a Decision Tree مجلة العلوم الانسانیة والطبیعیة , 31028hnsj10.53796/ :doi , 2022 ,”Classifier   
[14] M. Syamala, N. J. Nalini, “A Filter Based Improved Decision Tree Sentiment Classification Model for RealTime Amazon Product Review Data”, International Journal of Intelligent Engineering and Systems, 2020, doi: 10.22266/IJIES2020.0229.18   
[15] R., B., Saranya., Ramesh, Kesavan., K., N., Devi. (2022). 3. Extremely Randomized Tree Based Sentiment Polarity Classification on Online Product Reviews. Lecture Notes in Computer Science, doi: 10.1007/978-3-031-24094-2_11   
[16] M. Yang, “English Sentiment Analysis and its Application in Translation Based on Decision Tree Algorithm”, International Journal of Maritime Engineering, 2024, doi: 10.5750/ijme.v1i1.1371   
[17] J. Kazmaier et. al., “The power of ensemble learning in sentiment analysis” Expert Systems With Applications, 2022, doi: 10.1016/J.ESWA.2021.115819   
[18] K.G. Subash et al., “Harmony Gradient Boosting Random Forest Machine Learning Algorithms for Sentiment Classification”, 2022, doi: 10.1109/iSSSC56467.2022.10051210   
[19] T. Omran, B. Sharef, C. Grosan and Y. Li, "Ensemble Learning for Sentiment Analysis of Translation-Based Textual Data," 2022 International Conference on Electrical, Computer, Communications and Mechatronics Engineering (ICECCME), Maldives, Maldives, 2022, pp. 1-9, doi: 10.1109/ICECCME55909.2022.9988242.   
[20] R. Gul, M. Bashir. (2024). Feature selection for sentiment analysis using hybrid multiobjective evolutionary algorithm. Journal of Intelligent and Fuzzy Systems, doi: 10.3233/jifs-234615   
[21] E. Edwar et al., “Perbandingan Metode Seleksi Fitur Pada Analisis Sentimen (Studi Kasus Opini PILKADA DKI 2017), Informatics for educators and professional, 2023, doi: 10.51211/itbi.v8i1.2408   
[22] A. Singh., “A Review of: Ensemble Feature Selection Scheme-Based Performance Evaluation of Several Classifiers for Sentiment Analysis”, International Journal For Science Technology And Engineering, 2024, doi: 10.22214/ijraset.2024.63404   
[23] O. Ayana et al., “BSO: Binary Sailfish Optimization for Feature Selection in Sentiment Analysis”, 2023, doi: 10.22541/au.169111986.61513496/v1   
[24] S. Gouthami et al., “Feature Selection based Sentiment Analysis on US Airline Twitter Data”, International Journal on Recent and Innovation Trends in Computing and Communication, 2023, doi: 10.17762/ijritcc.v11i9.9161   
[25] I. Said et al., “A review of feature selection in sentiment analysis using information gain and domain specific ontology”, 2019, doi: 10.19101/IJACR.PID90   
[26] N.I. Izzatie et al., “Feature Selection Methods in Sentiment Analysis: A Review”, 2020, doi: 10.1145/3386723.3387840   
[27] F. Yang etal., “A hybrid feature selection algorithm combining information gain and grouping particle swarm optimization for cancer diagnosis”, PLOS ONE, 2024, doi: 10.1371/journal.pone.0290332   
[28] J. Gao et al., “Information gain ratio-based subfeature grouping empowers particle swarm optimization for feature selection”, Knowledge Based Systems, 2024, doi: 10.1016/j.knosys.2024.111380   
[29] X. Xing et al., “A Novel Binary Particle Swarm Optimization Algorithm for Feature Selection”, 2024, doi: 10.1109/ccdc62350.2024.10588008   
[30] K. Robindro, “Hybrid distributed feature selection using PSO-MI”, 2023, doi: 10.1016/j.dsm.2023.10.003   
[31] P. Inthapong et al., “A Comparison Study on Particle Swarm Optimization (PSO) Algorithms for Data Feature Selection”, Mechanisms and machine science, 2023, doi: 10.1007/978-3-031- 42515-8_52   
[32] C. Suhaeni, H.S. Yong. “ Mitigating Class Imbalance in Sentiment Analysis through GPT-3-Generated Synthetic Sentences”, Applied Sciences, 2023, doi: 10.3390/app13179766   
[33] P.A. Perwira, N.I. Widiastuti” Imbalance Dataset in Aspect-Based Sentiment Analysis on Game Genshin Impact Review”, Jurnal Infotel, 2024, doi: 10.20895/infotel.v16i1.984   
[34] M.R. Raja, J. Arunadevi, “Deep Active Learning Multiclass Classifier for the Sentimental Analysis in Imbalanced Unstructured Text Data”, 2023, doi: 10.1109/icdsaai59313.2023.10452451   
[35] T. Law, et.al., “Ensemble-SMOTE: Mitigating Class Imbalance in Graduate on Time Detection”, Journal of informatics and web engineering, 2024, doi: 10.33093/jiwe.2024.3.2.17   
[36] Y. Zhang, L. Deng, B. Wei, “Imbalanced Data Classification Based on Improved Random-SMOTE and Feature Standard Deviation. Mathematics”, 2024, doi: 10.3390/math12111709   
[37] K. Nugroho et al., “Improving random forest method to detect hatespeech and offensive word,” 2019 Int. Conf. Inf. Commun. Technol. ICOIACT 2019, pp. 514–518, 2019, doi: 10.1109/ICOIACT46704.2019.8938451.   
[38] G. A. Mursianto, M. Falih, M. Irfan, T. Sakinah, and D. Sandya, “Perbandingan Metode Klasifikasi Random Forest dan XGBoost Serta Implementasi Teknik SMOTE pada Kasus Prediksi Hujan,” no. September, pp. 41–50, 2021.   
[39] E. Haddi, X. Liu, and Y. Shi, “The role of text pre-processing in sentiment analysis,” Procedia Comput. Sci., vol. 17, no. December, pp. 26–32, 2013, doi: 10.1016/j.procs.2013.05.005.   
[40] D.D. Palmer, Text preprocessing. 2010. doi: 10.4018/978-1-5225- 4990-1.ch006.   
[41] S. Qaiser and R. Ali, “Text Mining: Use of TF-IDF to Examine the Relevance of Words to Documents,” Int. J. Comput. Appl., vol. 181, no. 1, pp. 25–29, 2018, doi: 10.5120/ijca2018917395.   
[42] M. Lantt, S. Sungt, H. Lowt, and C. Tant, “A Comparati e Study on Term for Te t Categori eighting Schemes ation,” pp. 546–551, 2005.   
[43] R. Primartha, B. Adhi Tama, A. Arliansyah, and K. Januar Miraswan, “Decision tree combined with pso-based feature selection for sentiment analysis,” J. Phys. Conf. Ser., vol. 1196, no. 1, 2019, doi: 10.1088/1742-6596/1196/1/012018.   
[44] H. Xie, L. Zhang, C. P. Lim, Y. Yu, and H. Liu, “Feature selection using enhanced particle swarm optimisation for classification models,” Sensors, vol. 21, no. 5, pp. 1–40, 2021, doi: 10.3390/s21051816.   
[45] X. fang Song, Y. Zhang, D. wei Gong, and X. yan Sun, “Feature selection using bare-bones particle swarm optimization with mutual information,” Pattern Recognit., vol. 112, p. 107804, 2021, doi: 10.1016/j.patcog.2020.107804.   
[46] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic Minority Over-sampling Technique,” J. Artif. Intell. Res., vol. 16, no. Sept. 28, pp. 321–357, 2002, [Online]. Available: https://arxiv.org/pdf/1106.1813.pdf%0Ahttp://www.snopes.com/horro rs/insects/telamonia.asp   
[47] K. Fithriasari, I. Hariastuti, and K. S. Wening, “Handling Imbalance Data in Classification Model with Nominal Predictors,” Int. J. Comput. Sci. Appl. Math., vol. 6, no. 1, p. 33, 2020, doi: 10.12962/j24775401.v6i1.6643.   
[48] M. Zheng, F. Wang, X. Hu, Y. Miao, H. Cao, and M. Tang, “A Method for Analyzing the Performance Impact of Imbalanced Binary Data on Machine Learning Models,” Axioms, vol. 11, no. 11, 2022, doi: 10.3390/axioms11110607.   
[49] D. D. Tran, T. T. S. Nguyen, and T. H. C. Dao, “Sentiment Analysis of Movie Reviews Using Machine Learning Techniques,” Lect. Notes Networks Syst., vol. 235, no. August, pp. 361–369, 2022, doi: 10.1007/978-981-16-2377-6_34.   
[50] D. Wang, D. Tan, and L. Liu, “Particle swarm optimization algorithm : an overview,” Soft Comput., 2017, doi: 10.1007/s00500-016-2474-6.