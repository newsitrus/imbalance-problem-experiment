# Advancing tourism sentiment analysis: a comparative evaluation of traditional machine learning, deep learning, and transformer models on imbalanced datasets

Sawitree Srianan1 $\cdot$ Aziz Nanthaamornphong $^ 2 \textcircled { | \scriptsize { \cdot } \textcircled { | \scriptsize { \cdot } } }$ · Chayanon Phucharoen

Received: 4 March 2025 / Revised: 12 July 2025 / Accepted: 28 July 2025 /   
Published online: 2 August 2025   
$\circledcirc$ The Author(s), under exclusive licence to Springer-Verlag GmbH Germany, part of Springer Nature 2025

# Abstract

Tourism sentiment analysis faces substantial challenges due to class imbalance and the complex linguistic features of user-generated content. This study systematically compares eight sentiment classification models, spanning traditional machine learning (naïve Bayes, support vector machines, logistic regression), deep learning (convolutional neural networks, long short-term memory networks [LSTMs], gated recurrent units [GRUs]), and transformer-based architectures (RoBERTa in two configurations: pretrained and fine-tuned), using a dataset of 505,980 TripAdvisor reviews. We evaluate model performance under imbalanced class conditions and examine the effectiveness of three oversampling techniques—SMOTE, ADASYN, and RandomOverSampler—in mitigating class bias. The results reveal significant performance disparities across architectures. Deep learning models, particularly LSTM $9 1 . 0 6 \%$ accuracy, Cohen’s kappa $= 0 . 6 8 4 6$ ) and GRU $( 9 0 . 8 2 \%$ accuracy, Cohen’s kappa $= 0 . 6 7 8 1$ ), consistently outperform traditional approaches. Fine-tuned RoBERTa achieved the highest performance, with $9 2 . 3 1 \%$ accuracy, a $9 5 . 3 4 \%$ F1- score, and Cohen’s kappa $= 0 . 7 3 2 1$ . Traditional models showed notable limitations; for example, naïve Bayes exhibited strong majority-class bias, despite an accuracy of $8 2 . 3 5 \%$ (Cohen’s kappa $_ { - 0 . 0 0 5 4 }$ ). Among the oversampling methods, SMOTE was the most effective in improving the fairness of traditional models, while Ro-BERTa’s fine-tuning process inherently mitigated class imbalance. A computational analysis highlights key trade-offs: traditional models train quickly but require oversampling, deep learning offers a balanced trade-off between performance and efficiency, and transformer models provide state-of-the-art accuracy at the cost of high computational resources. These findings offer evidence-based guidance for selecting appropriate models for tourism sentiment analysis.

Keywords Sentiment analysis $\cdot$ Natural language processing (NLP) $\cdot$ Class imbalance $\cdot$ Deep learning $\cdot$ RoBERTa $\cdot$ Online review

# 1 Introduction

Sentiment analysis, also known as opinion mining, has become a powerful tool for interpreting consumer opinions and behaviors through textual data (Păvăloaia et al. 2019). In the tourism and hospitality industry, user-generated content (UGC) from platforms such as TripAdvisor, Yelp, and Google Reviews plays a pivotal role in shaping consumer choices and influencing business strategies. Analyzing these reviews provides tourism operators with actionable insights into customer satisfaction, service gaps, and destination management, supporting data-driven decision-making (Ahani et al. 2019). However, the inherent complexity of textual data, including diverse writing styles, informal language, and subtle expressions of sentiment, presents significant challenges for sentiment analysis in this context.

Traditional machine learning (ML) methods, such as naïve Bayes (NB), have been widely used for sentiment classification due to their simplicity, interpretability, and computational efficiency (Mehraliyev et al. 2022). In tourism research, NB has shown reliable performance on small, balanced datasets and is frequently applied in sentiment classification, route recommendation, and customer feedback analysis (Irawan and Nurdiawan 2023; Yuke and Yusuf 2024; Sincharoenkul and Sangkaew 2023). However, NB’s assumption of feature independence often limits its performance in real-world scenarios, where text features tend to be highly interdependent (Taheri et al. 2014; Foo et al. 2022). For instance, the presence of certain correlated words in a review may mislead the model if it fails to consider their co-occurrence (Jiang et al. 2016; Tesoro et al. 2020). Additionally, NB performs poorly on imbalanced datasets—a common trait of tourism reviews, which are typically skewed toward positive sentiment (Chaudhuri and Sahu 2021; Kim and Lee 2022).

These limitations have led to growing interest in deep learning (DL) models, which offer superior performance for processing large-scale, unstructured text data. Architectures such as convolutional neural networks (CNNs), long short-term memory (LSTM) networks, and gated recurrent units (GRUs) have demonstrated improved capabilities in capturing sequential dependencies and contextual relationships in text (Basiri et al. 2021; Kamyab et al. 2021). These models learn feature representations automatically, reducing the need for manual feature engineering and enabling more effective modeling of linguistic and semantic patterns. Nevertheless, while DL models offer theoretical advantages, empirical evidence of their superiority over traditional ML methods in tourism sentiment analysis remains limited under controlled conditions. Building on the advances in DL, transformer-based models have transformed natural language processing (NLP) by leveraging attention mechanisms and bidirectional encoding. Among them, bidirectional encoder representations from transformers (BERT) and its enhanced variant, robustly optimized BERT pretraining approach (RoBERTa), have established new standards for sentiment analysis (Liu et al. 2019). RoBERTa’s architecture improves contextual understanding through extended training, dynamic masking, and robust optimization. While it has achieved notable success in domains such as movie reviews, public opinion, and news sentiment (Bird et al. 2023; Liao et al. 2021; Demir and Bilgin 2023; Tan et al. 2023; Cheruku et al. 2024; Wang 2024), its application to tourism sentiment analysis remains underexplored.

Tourism-related UGC presents unique challenges for sentiment analysis due to its complex linguistic structures, mixed sentiments within individual texts, and pronounced class imbalance, where positive reviews significantly outnumber negative ones. A single review may simultaneously praise certain aspects of a destination while criticizing others, complicating classification efforts (Taecharungroj and Mathayomchan 2019). Unlike retail products, tourism experiences are inherently intangible and cannot be evaluated prior to consumption, making online reviews and word-of-mouth feedback essential for consumer decision-making (Van den Bergh 2022). This experiential nature results in emotionally charged and highly variable feedback, as individual experiences differ widely and lack the standardization seen in product reviews (Jo 2024). Additionally, cultural diversity and the interplay between utilitarian and hedonic values create diverse, subjective interpretations that further differentiate tourism reviews from those in other domains, requiring specialized analytical approaches (Ham et al. 2019). Class imbalance is a particularly critical issue in tourism sentiment analysis, with positive reviews often comprising $70 \mathrm { - } 8 5 \%$ of datasets. This imbalance can skew model predictions, undermining fairness and reliability, and may lead to misguided business decisions, ineffective resource allocation, and failure to address patterns in negative feedback (Lango 2019; Obiedat et al. 2022). Traditional ML approaches often lack robust mechanisms to manage skewed distributions, and the effectiveness of various oversampling techniques, such as the synthetic minority oversampling technique (SMOTE), adaptive synthetic sampling (ADASYN), and RandomOverSampler, remains underexplored across different model architectures in tourism applications. The absence of systematic evaluation of these methods represents a significant methodological gap that limits the robustness and applicability of existing research.

Despite the increasing availability of advanced transformer models and sophisticated balancing techniques, tourism sentiment research remains heavily reliant on traditional ML or limited DL approaches (Mehraliyev et al. 2022; Núñez et al. 2024). This methodological conservatism persists despite growing evidence that traditional models are often insufficient to capture the nuanced, contextual nature of tourismrelated UGC or to mitigate fairness concerns in imbalanced datasets. The continued use of these methods is not merely an academic limitation but a practical barrier to leveraging the full potential of the rapidly expanding volume of online reviews.

The primary objective of this study is to conduct a systematic comparative analysis of sentiment classification approaches across three model categories: traditional ML (NB, SVM, LR), deep learning (CNN, LSTM, GRU), and transformer-based models (RoBERTa). The analysis evaluates their performance under both imbalanced and balanced conditions. To this end, the study addresses the following research questions:

RQ1 How do traditional ML, deep learning, and transformer-based models compare in terms of classification performance, fairness, and reliability for sentiment analysis of imbalanced tourism reviews?

RQ2 To what extent do oversampling techniques (SMOTE, ADASYN, RandomOverSampler) improve model performance and class balance in imbalanced sentiment classification tasks?

RQ3 What are the trade-offs between performance and training efficiency across sentiment classification approaches, and how can these trade-offs inform practical model selection in tourism sentiment analysis?

To answer these questions, the study conducts a comprehensive analysis using 505,980 English-language tourist attraction reviews from TripAdvisor. Evaluation metrics include accuracy, precision, recall, F1-score, balanced accuracy, and Cohen’s kappa, along with training time to assess computational efficiency. The experimental design features standardized preprocessing, controlled train–test splits, and the systematic application of oversampling techniques to ensure fair comparison across model architectures.

# 1.1 Contributions of this study

The key contributions of this study are as follows:

Comprehensive Multi-Architecture Comparison: This research systematically compares eight sentiment classification models across three categories—traditional ML (NB, SVM, LR), deep learning (CNN, LSTM, GRU), and transformerbased models (RoBERTa in two configurations: pretrained and fine-tuned)—under identical experimental conditions in the tourism domain.   
C Systematic Evaluation of Oversampling Techniques: The study assesses the impact of three oversampling methods (SMOTE, ADASYN, RandomOverSampler) across all model types, providing empirical evidence of their effectiveness in improving performance, fairness, and class balance in imbalanced tourism sentiment datasets. Performance–Efficiency Trade-Off Analysis: By evaluating both classification performance and training time, the study highlights the computational costs and benefits of different models, offering practical guidance for deployment in resource-constrained settings. Large-Scale Empirical Validation: Using a dataset of 505,980 TripAdvisor reviews, the study demonstrates the scalability and real-world applicability of advanced models while addressing the class imbalance characteristic of tourism UGC.   
C Evidence-Based Model Selection Guidelines: The findings offer actionable, data-driven recommendations for researchers and practitioners, supporting model selection decisions based on performance goals, fairness requirements, and computational limitations.   
C Methodological Framework for Tourism NLP: Through standardized evaluation protocols and the inclusion of metrics such as Cohen’s kappa and balanced accuracy, the study establishes a robust methodological foundation for future research in tourism sentiment analysis.

Collectively, these contributions address key challenges in tourism sentiment analysis and support the development of more accurate, fair, and efficient data-driven decision-making tools.

# 2 Literature review

# 2.1 Sentiment analysis in the tourism industry: an overview

Online reviews have become integral to the modern tourism industry, significantly influencing travelers’ decisions and shaping business strategies (Sotiriadis 2017). As more tourists rely on platforms such as TripAdvisor, Yelp, and Google Reviews to guide their choices, the influence of UGC continues to grow. These reviews offer valuable insights into the experiences of past visitors, enabling prospective travelers to make informed decisions aligned with their preferences $\left( \mathrm { W u } ~ 2 0 2 4 \right)$ . For businesses, monitoring and responding to reviews is critical: addressing customer concerns and encouraging positive feedback can strengthen reputation and drive increased visitation.

Sentiment analysis has been widely applied in tourism to support customer satisfaction assessments, destination image evaluation, and service improvement. For example, Hu et al. (2019) identified ten major topics linked to customer dissatisfaction across hotel categories in New York Ramos et al. (2022) developed a decision-support system to assess consumer satisfaction using tourist feedback in Portugal’s Algarve region. Sangkaew et al. (2023) applied sentiment analysis to restaurant reviews in Phuket, Thailand, revealing patterns in customer satisfaction and preferences within the local culinary scene. Destination image evaluation is another prominent application of sentiment analysis. Jeng et al. (2019) examined perceptions of Taiwan by analyzing both positive and negative attributes of tourist experiences, demonstrating how satisfaction is linked to the performance and importance of various travel factors. Similarly, Nowacki and Niezgoda (2020) used TripAdvisor reviews to assess the image of four Baltic cities, identifying both cognitive and affective attributes that enhance destination branding. Li et al. (2023) extended this line of research by investigating how congruence between projected and perceived destination images influenced tourists’ evaluations of Hainan, China. More recently, Eberle et al. (2025) advanced this field by emphasizing the role of primary emotions such as joy, anger, and fear beyond traditional sentiment polarity. Their study demonstrated that deep learning methods can effectively detect emotional expressions in UGC, revealing how emotional responses shape tourists’ overall satisfaction and interpretation of experiences. Collectively, these studies illustrate the versatility of sentiment analysis as a tool for improving service quality, understanding consumer perceptions, and enhancing destination marketing.

Early sentiment analysis in tourism primarily relied on traditional ML classifiers and structured text representations, such as bag-of-words (BoW), n-grams, and TF-IDF. Schmunk et al. (2014) demonstrated one of the earliest implementations of this approach in the tourism domain, using both machine learning (e.g., SVM) and dictionary-based techniques to extract sentiment from travel-related UGC. Their work emphasized the practicality of lexicon-based classification for subjectivity detection and highlighted its role in supporting tourism decision-making. These “predefined features” (Semary et al. 2024) required manual selection of linguistic elements thought to convey sentiment. Traditional ML algorithms, including NB, SVMs, and LR, remain widely used in tourism sentiment studies for their ability to handle structured data (Neidhardt et al. 2017; Vargas-Calderón et al. 2021; Mehraliyev et al. 2022; Irawan and Nurdiawan 2023; Sincharoenkul and Sangkaew 2023; Yuke and Yusuf 2024; Sabri et al. 2024; Primasari and Khadija 2024). However, these models often fall short in capturing contextual nuance, limiting classification accuracy and interpretability. To overcome these limitations, DL models have gained popularity in tourism sentiment analysis due to their ability to learn features directly from data. CNNs, LSTMs, and transformer-based architectures like BERT have improved the capacity to model semantic relationships and contextual information in text (Basiri et al. 2021; Kamyab et al. 2021; Paolanti et al. 2021; Viñán-Ludeña and De Campos 2022). Among these, RoBERTa represents an advancement over BERT, incorporating extended training and dynamic masking techniques to achieve state-of-the-art performance in various NLP tasks. Lexicon-based methods, an earlier and computationally efficient approach, are still used in some tourism sentiment studies. These methods rely on predefined dictionaries of sentiment words and are particularly useful in low-resource settings (Nowacki and Niezgoda 2020; Ramos et al. 2022; Alaei et al. 2023; Ni et al. 2024). However, they generally lack sensitivity to context and evolving language patterns (Song et al. 2019). Hybrid models that combine rulebased techniques with ML or DL approaches can enhance classification performance by integrating linguistic rules with data-driven learning (Amirkumar et al. 2024; Ni et al. 2024). Additionally, topic-based sentiment analysis integrates topic modeling techniques, such as Latent Dirichlet Allocation, with sentiment classification, offering insights into the themes and contexts associated with particular sentiments (Ren and Hong 2017; Hu et al. 2019; Taecharungroj and Mathayomchan 2019; Guerrero-Rodríguez et al. 2024).

However, sentiment analysis in tourism continues to face several persistent challenges. A key issue is the quality and volume of UGC, which often includes noise, irrelevant information, or biased opinions that complicate analysis and reduce reliability. Tourism reviews frequently contain mixed sentiments—such as praising a hotel’s location while criticizing its service—making it difficult for models to classify the review accurately when analyzed as a single unit (Sah et al. 2024). Contextual understanding presents another challenge, as sentiment interpretation can vary by cultural background, dialect, or regional expressions (Alaei et al. 2023). Additionally, class imbalance—where positive, negative, and neutral sentiments are unequally represented—can skew model predictions, necessitating techniques such as oversampling or synthetic data generation (Xiao et al. 2019). The multilingual and cross-cultural nature of tourism data further increases complexity, requiring sentiment models that can adapt to diverse linguistic and cultural contexts. These challenges highlight the need for advanced natural language processing techniques and context-aware models specifically tailored to the tourism industry.

# 2.2 Traditional ML in tourism sentiment analysis

Traditional ML algorithms have long served as the foundation for sentiment analysis in tourism, valued for their simplicity, interpretability, and low computational requirements. Common classifiers include NB, SVM, and LR, which typically rely on structured text representations such as BoW, n-grams, or TF-IDF. These models often operate under simplifying assumptions, such as feature independence or linear separability, enabling rapid processing of UGC like online reviews (Mehraliyev et al. 2022; Bianchi and Heo 2021).

NB, in particular, has been widely adopted for its efficiency in model building with minimal preprocessing. Studies have reported classification accuracies of around $70 \%$ using NB for tasks such as route recommendation, satisfaction analysis, and review filtering (Taecharungroj and Mathayomchan 2019). Its probabilistic framework is well-suited to structured, high-frequency textual data. However, NB—and traditional ML models more broadly—exhibit several limitations when applied to real-world tourism datasets. A key limitation is the assumption of feature independence, which rarely holds in natural language. For example, terms such as “location,” “service,” and “cleanliness” frequently co-occur and influence each other’s meaning (Yuke and Yusuf 2024), yet NB treats them as statistically independent. This assumption can lead to inaccurate sentiment interpretation, particularly in reviews containing nuanced or mixed opinions. Another critical issue is the inability to effectively handle class imbalance—a common feature of tourism review platforms where positive sentiments typically dominate. In such cases, models like NB and SVM may exhibit high overall accuracy but poor recall and fairness for minority (e.g., negative) classes (Singgalen 2024). While oversampling techniques such as SMOTE and ADASYN have been proposed to mitigate class imbalance, their effectiveness varies across models. Additionally, traditional ML models are sensitive to data sparsity, which can be especially problematic in niche tourism markets or for low-volume attractions. In these cases, NB may underperform—either overfitting to noise or underfitting due to a lack of representative features (Gómez–Déniz et al. 2024).

Despite these challenges, traditional ML models continue to serve as useful baselines. Their transparency and computational efficiency make them suitable for exploratory analysis, rapid prototyping, and deployment in resource-constrained settings. However, as the volume and complexity of UGC grow, more sophisticated, contextaware models, particularly DL and transformer-based architectures, are increasingly necessary to ensure robust and fair sentiment classification in tourism research.

# 2.3 DL and transformer models in tourism sentiment analysis

DL techniques have transformed NLP by enabling models to automatically learn complex patterns and contextual meaning from unstructured text. These models use layered architectures to extract hierarchical representations of textual data. In tourism sentiment analysis, DL models such as CNNs, LSTMs, and GRUs have shown notable improvements over traditional methods (Basiri et al. 2021; Kamyab et al. 2021).

CNNs are effective at detecting spatial patterns in text and are often used to extract localized features such as n-gram sentiment cues. LSTMs and GRUs, both variants of RNNs, excel at modeling sequential dependencies and preserving long-term context—key for interpreting sentiments embedded in lengthy and nuanced tourism reviews. GRUs, in particular, offer computational advantages over LSTMs by simplifying gating mechanisms while maintaining contextual learning ability, making them efficient alternatives for real-world tourism applications (Viñán-Ludeña and De Campos 2022). Building on the advances of DL, transformer-based models have further elevated sentiment analysis performance. Transformers process all tokens in a sequence simultaneously using self-attention mechanisms, allowing for superior modeling of global dependencies, especially critical in analyzing the long, complex narratives common in tourism reviews.

Among these models, BERT and its enhanced variant, RoBERTa, have set new performance benchmarks. These models leverage bidirectional context encoding, allowing for more accurate interpretation of word meaning based on surrounding text (Devlin et al. 2019). RoBERTa introduces several key improvements over BERT, including the removal of the next-sentence prediction objective, dynamic masking, longer training times, and training on substantially larger datasets—expanding from BERT’s $1 6 \mathrm { G B }$ to $1 6 0 \mathrm { G B }$ using sources such as Common Crawl News, OpenWebText, and Stories (Liu et al. 2019). This extended training allows RoBERTa to generalize from underrepresented linguistic patterns, which is especially beneficial when handling imbalanced datasets with rare or subtle sentiment expressions. RoBERTa’s self-attention mechanism assigns dynamic importance to all input tokens, preserving nuanced linguistic cues that traditional ML and RNN-based models often overlook due to their reliance on sequential or frequency-based heuristics. Additionally, RoBERTa has demonstrated stronger performance than not only BERT but also other transformer-based models such as XLNet and ALBERT in multiple sentiment classification benchmarks (e.g., Liu et al. 2019; Tan et al. 2023; Cheruku et al. 2024). Its demonstrated ability to handle skewed datasets through contextual encoding makes it particularly suitable for tourism-related UGC, which often exhibits sentiment imbalance, linguistic noise, and emotionally mixed content.

Numerous studies have demonstrated RoBERTa’s superior performance in domains such as product reviews, social media analysis, and cross-lingual sentiment classification (Liao et al. 2021; Bird et al. 2023; Demir and Bilgin 2023; Tan et al. 2023; Cheruku et al. 2024; Wang 2024). Its context-aware token representations and exposure to diverse linguistic inputs allow it to manage skewed sentiment distributions more effectively, particularly in UGC that is long, emotionally complex, and rich in mixed sentiment. Despite these advantages, the use of RoBERTa and other transformer models in tourism sentiment analysis remains relatively limited. Traditional models continue to dominate the literature, largely due to their simplicity and lower computational requirements. However, as access to computational resources improves, the tourism research community is increasingly well-positioned to adopt transformer-based models.

By leveraging dynamic masking, large-scale training corpora, and advanced optimization protocols, transformer models like RoBERTa offer a new standard in sentiment analysis. Their ability to capture nuanced language and contextual subtleties makes them particularly well-suited to tourism, where reviews frequently reflect emotionally charged, multi-faceted experiences. Expanding their use in tourism research promises more precise sentiment interpretation and richer insights to support destination management, service improvement, and policy design. However, further research is needed to systematically evaluate their performance across a broader range of tourism datasets and contexts, particularly under varying degrees of class imbalance and linguistic complexity.

# 2.4 Datasets for tourism sentiment analysis

Tourism sentiment research relies heavily on data availability, with several UGC platforms serving as primary sources. Online travel review sites, particularly TripAdvisor, dominate due to their accessibility and comprehensive content (Taecharungroj and Mathayomchan 2019; Hu et al. 2019; Nowacki and Niezgoda 2020; Manurung & Lhaksmana 2023; Sincharoenkul and Sangkaew 2023; Núñez et al. 2024). TripAdvisor hosts millions of reviews on hotels, restaurants, and attractions worldwide, making it a leading platform for analyzing destination image, service sentiment, and customer satisfaction. Other widely used platforms include Yelp, Expedia, and OTAs such as Booking.com, Airbnb, and Ctrip, especially for hospitality and peer-to-peer accommodation reviews (Ham et al. 2019; Vargas-Calderón et al. 2021; Bianchi and Heo 2021; Li et al. 2023; Gómez-Déniz et al. 2024), and Google Reviews for broader traveler feedback (Irawan and Nurdiawan 2023; Primasari and Khadija 2024).

Beyond review platforms, social media has emerged as a valuable complementary source. Travelers frequently share real-time experiences on Twitter, Instagram, and TikTok, and researchers have leveraged hashtags and geotagged posts for sentiment analysis (Kamyab et al. 2021; Almuayqil et al. 2022; Sabri et al. 2024). While social media presents challenges such as brevity and informal language, its immediacy offers timely insights into tourist perceptions. Survey data also plays a role, offering structured, detailed responses, whereas social media captures more spontaneous emotional reactions.

Each source has unique strengths and limitations—structured, detailed reviews from platforms like TripAdvisor support long-term sentiment analysis but may suffer from selection bias, as reviewers often represent extremes in satisfaction (either very positive or very negative) (Taecharungroj and Mathayomchan 2019; Suwitho et al. 2023; Primasari and Khadija 2024). Review platforms may also lack real-time responsiveness, limiting their utility in tracking rapidly evolving sentiments. In contrast, social media captures real-time, spontaneous reactions but typically features short, noisy, and informal content that requires extensive preprocessing. Sentiment on these platforms can also be distorted by viral trends or external events unrelated to actual travel experiences. Researchers must therefore carefully select, or combine, data sources based on the goals of their study. TripAdvisor is well-suited to in-depth, structured sentiment analysis, while platforms like Twitter are better suited for monitoring dynamic, time-sensitive trends. In both cases, it is essential to consider and adjust for the specific limitations associated with each data type.

# 2.5 Key challenges and solutions in tourism sentiment analysis

Tourism sentiment analysis faces several critical challenges that affect both model performance and practical applicability. One of the most persistent issues is class imbalance, where positive reviews significantly outnumber negative ones. This imbalance leads classifiers to overpredict the majority class, reducing recall for minority sentiments and undermining classification fairness (Lango 2019; Obiedat et al. 2022; Pan et al. 2020). Addressing this imbalance is essential for building robust models capable of reliably distinguishing all sentiment categories.

Contextual complexity and semantic nuance also pose major barriers. Tourism reviews often contain idiomatic expressions, metaphorical language, and culturally embedded references—sometimes within a single review. For example, a tourist may praise a destination’s scenery while simultaneously criticizing service quality. These mixed sentiments are particularly difficult for traditional models, especially those based on feature independence assumptions, which are ill-equipped to handle subtle or layered expressions of opinion (Li et al. 2023; Ren and Hong 2017). Another core challenge lies in feature representation. Early techniques such as bag-of-words (BoW) and TF-IDF ignore term dependencies and context. More recent word embedding methods like Word2Vec and GloVe improve representational power, but they still struggle with polysemy and out-of-vocabulary terms (Kamyab et al. 2021; Jiang et al. 2016). These limitations highlight the need for models that can capture deeper semantic relationships and contextual meaning. Multimodal sentiment analysis introduces additional complexity. Integrating textual, visual, and audio inputs requires advanced data fusion methods and standardized frameworks—areas that remain underdeveloped (Zhao et al. 2024; Ni et al. 2024). Moreover, computational efficiency presents practical constraints. DL models such as LSTM and transformerbased architectures like RoBERTa often demand significant resources, limiting their feasibility for real-time applications or deployment in low-resource environments (Liu et al. 2019; Wang 2024).

To address these challenges, researchers have explored a variety of strategies. Resampling techniques such as SMOTE, ADASYN, and random under-sampling have been applied to mitigate class imbalance, improving the detection of minorityclass sentiments (Pan et al. 2020; Lango 2019). These methods help reduce classifier bias and improve fairness metrics such as F1-score and Cohen’s kappa. Transformerbased models like BERT and RoBERTa have significantly advanced sentiment analysis by capturing long-range dependencies and nuanced language structures (Liu et al. 2019; Tan et al. 2023). Their ability to model context makes them particularly effective for analyzing the lengthy, emotionally complex reviews typical of the tourism domain. Hybrid approaches, which combine rule-based and ML methods, offer a compromise between interpretability and performance. These models are particularly valuable when computational resources are limited or when transparency in model decision-making is required (Irawan and Nurdiawan 2023; Taheri et al. 2014).

In the multimodal domain, emerging work has explored combining CNNs for visual inputs with transformers for textual data, enabling richer sentiment inference. Additionally, transfer learning and data augmentation strategies, such as back-translation, synonym replacement, and paraphrasing, have helped address data scarcity by expanding training corpora while minimizing annotation costs (Muizelaar et al. 2024; Kamyab et al. 2021). Collectively, these solutions reflect a growing convergence of technical innovation and domain-specific adaptation. As tourism sentiment analysis continues to evolve, the adoption of more balanced, context-aware, and scalable models will be essential for extracting meaningful insights from increasingly complex UGC.

# 3 Methodology

This study investigates sentiment classification of tourist attraction reviews by comparing traditional ML, DL, and transformer-based models. The methodology follows a structured pipeline consisting of data collection, preprocessing, model implementation, and evaluation. Each stage is described in the following subsections, detailing how the data were acquired, cleaned, and analyzed to assess model performance and reliability.

# 3.1 Dataset

# 3.1.1 Data source

A total of 505,980 English-language tourist attraction reviews were systematically scraped from TripAdvisor using web scraping techniques. The analysis was limited to English reviews to maintain a consistent linguistic framework and avoid the added complexity of multilingual sentiment analysis, such as language-specific tokenization and potential translation errors. While incorporating multilingual content could offer a more comprehensive view of traveler perspectives, it would require advanced NLP pipelines beyond the scope of this study.

# 3.1.2 Cleaning and preprocessing

Data cleaning was essential to ensure the accuracy and reliability of the analysis. Python, along with libraries such as pandas and NumPy, was used throughout this process. The following steps were performed:

1. Removal of Duplicate Rows: Duplicate entries were identified and removed to avoid redundancy in the dataset.   
2. Elimination of Rows with Missing or Invalid Data: Entries lacking either review text or rating information were excluded to ensure completeness and reliability.   
3. Conversion of Data Types: The “rating” column was converted to numeric format for use in statistical analysis, and the “year” column was converted to datetime format to enable temporal trend analysis.

Preprocessing of the textual data was carried out to prepare the reviews for model training and feature extraction. This included tokenization, removal of stopwords, lowercasing, and elimination of punctuation using tools such as NLTK and Python’s re module. These steps standardized the input data and ensured consistency across the dataset, as illustrated in Fig. 1:

1. Lowercasing Text: All text was converted to lowercase to eliminate inconsistencies caused by case sensitivity and to standardize input formatting.   
2. Removing Punctuation: Punctuation marks such as commas, periods, and exclamation points were removed to reduce noise and improve the clarity of textual data.   
3. Removing Extra Spaces: Superfluous whitespace was eliminated to streamline tokenization and ensure cleaner word boundaries.   
4. Lemmatizing and Removing Stop Words: Common stop words (e.g., and, the, is) were removed due to their limited analytical value. Lemmatization was applied to reduce words to their base forms (e.g., swimming to swim), preserving grammatical context and improving analytical accuracy over basic stemming.

These cleaning and preprocessing steps ensured that the dataset was well-structured, consistent, and ready for effective sentiment analysis.

# 3.1.3 Descriptive statistics

The dataset comprises 505,980 English-language tourist attraction reviews collected from TripAdvisor. Each review includes a rating on a five-point scale (1 to 5), representing levels of tourist satisfaction, with 1 indicating the lowest satisfaction and 5 indicating the highest. As illustrated in Fig. 2, the distribution of ratings is heavily skewed toward the upper end of the scale. More than $56 \%$ of reviews received a rating of 5, while ratings of 1 and 2 together make up only about $7 \%$ of the dataset.

![](images/e3c8ffe25559ec996fdeb54a599ac6d74b5d14cddbe2df74a1622dd4fe8325cb.jpg)  
Fig. 1 Data preprocessing steps

<!-- FIGURE-DATA: Fig. 1 | type: diagram -->
> **[Extracted Data]**
> - Text preprocessing pipeline flowchart
<!-- FIGURE-DATA: Fig. 2 | type: plot -->
> **[Extracted Data]**
> - Distribution of ratings chart
> **Analysis:** Shows rating distribution in the dataset.
<!-- /FIGURE-DATA -->
> **Analysis:** Shows sequential steps of text preprocessing in NLP.
<!-- /FIGURE-DATA -->
![](images/1cab10e896582ae8089908c56f6436391795ada1b035edff08c01224c59a20df.jpg)  
Fig. 2 Distribution of ratings

This pronounced imbalance suggests that the majority of reviewers reported highly positive experiences, highlighting a significant class imbalance in the sentiment data.
<!-- FIGURE-DATA: Fig. 3 | type: plot -->
> **[Extracted Data]**
> - Sentiment distribution chart
> **Analysis:** Shows sentiment distribution in the dataset.
<!-- /FIGURE-DATA -->

The reviews were collected between 2010 and 2023 and focus on tourist attractions located in two internationally recognized Southeast Asian Island destinations: Bali (Indonesia) and Phuket (Thailand). These destinations were selected due to their global popularity, high tourist volumes, and the abundance of English-language reviews on TripAdvisor. They also share similar characteristics of being globally recognized coastal destinations (Norris 2025). Their inclusion provides a rich and relevant source of user-generated content for large-scale sentiment analysis. While ratings of 1–3 are categorized as negative and 4–5 as positive, we acknowledge that a rating of 3 may reflect neutral sentiment. This classification follows the approach of Taecharungroj and Mathayomchan (2019), which has been widely adopted in tourism sentiment research. However, the potential for sentiment misclassification is noted as a limitation of this method.

This predominance of positive ratings aligns with previous research, which highlights users’ tendency to share more positive than negative feedback on review platforms (Lango 2019; Obiedat et al. 2022; Pan et al. 2020; Suwitho et al. 2023; Núñez et al. n.d.). Such tendencies are especially pronounced on platforms like TripAdvisor, where users are often more motivated to leave reviews after satisfying experiences. This sentiment skew presents a significant challenge for classification models, particularly in accurately detecting underrepresented negative sentiments.

According to Taecharungroj and Mathayomchan (2019), Sangkaew et al. (2023), and Sincharoenkul and Sangkaew (2023), reviews can be categorized as negative (ratings 1 to 3) or positive (ratings 4 to 5). Following this established criterion, the dataset in this study was labeled accordingly, as illustrated in Fig. 3. Specifically,

![](images/b3cf9d9f130cbe4118cac984e122a356db7897811054fdebd23748c9d8656b0c.jpg)  
Fig. 3 Sentiment distribution

415,478 reviews $( 8 2 . 1 1 \%$ of the dataset) were classified as positive, and 89,989 reviews $( 1 7 . 7 9 \% )$ as negative. This classification method, grounded in prior research, ensures consistency in sentiment labeling and facilitates a systematic analysis of tourist experiences. Although some studies, such as Hu et al. (2019), Almuayqil et al. (2022), and Irawan and Nurdiawan (2023), have employed manual annotation based on human judgment, such approaches are labor-intensive and impractical for largescale datasets. In contrast, rating-based classification offers efficiency and scalability, allowing for the rapid analysis of extensive review corpora with minimal human intervention. However, a known limitation is that numerical ratings do not always align with the sentiment conveyed in the review text, especially in cases involving mixed or context-dependent emotions. Despite this, rating-based labeling remains a widely accepted and practical approach for large-scale sentiment analysis.

The substantial imbalance between positive and negative reviews reinforces the need for effective data balancing strategies, particularly oversampling techniques that improve the representation of the minority class. Without such interventions, classification models are prone to bias toward the majority class, reducing generalizability and fairness. The approaches used to address this issue are discussed in the next section.

# 3.2 Handling imbalanced data

UGC datasets used in sentiment analysis frequently exhibit substantial class imbalance, with positive reviews vastly outnumbering negative ones. This skew can distort model predictions by causing classifiers to favor the majority class, leading to poor performance in identifying minority-class instances (Zhao et al. 2024). Addressing this imbalance is essential for achieving fair and accurate classification—particularly in tourism sentiment analysis, where such distributional skew is common.

To mitigate this issue, multiple oversampling strategies were implemented, including SMOTE, ADASYN, and RandomOverSampler. These techniques were applied exclusively to the training set to avoid data leakage. Each method enhances class balance by either generating synthetic samples or duplicating minority-class instances. SMOTE creates synthetic examples by interpolating between existing minority samples, helping the model generalize more effectively (Xiao et al. 2019). ADASYN builds on SMOTE by generating more synthetic data in regions that are harder to classify, focusing model learning on ambiguous or misclassified areas (Almuayqil et al. 2022). RandomOverSampler, by contrast, simply duplicates existing minorityclass instances without introducing new patterns, offering a straightforward solution at the risk of overfitting.

These techniques were selected based on their established effectiveness and common usage in prior sentiment analysis research (Lango 2019; Obiedat et al. 2022; Manurung & Lhaksmana 2023). SMOTE and ADASYN provide synthetic diversity, while RandomOverSampler serves as a computationally efficient baseline. The comparative effects of these methods were systematically evaluated across all model types to assess their contribution to performance improvements. By applying these oversampling techniques, the training dataset was rendered more balanced, allowing models to learn more equitably from both positive and negative reviews. This preprocessing step is critical for improving the robustness, fairness, and generalizability of sentiment classifiers in the tourism domain.

# 3.3 Model implementation

This study implemented and compared three categories of sentiment classification models: traditional ML, DL, and transformer-based architectures. All models were trained and evaluated using a standardized pipeline to ensure consistency, reproducibility, and fairness across experiments.

# 3.3.1 Traditional ML models

The traditional ML models included NB, LR, and SVM. Each model was trained using term-level TF-IDF vectorization to transform textual data into numerical representations suitable for classification. NB was selected for its simplicity and efficiency, particularly with sparse textual data. LR was chosen for its robustness and interpretability, while SVM was included for its strong performance in high-dimensional text classification tasks. The hyperparameters used for each of these models are summarized in Table 1.

Table 1 Traditional ML model parameters   

<table><tr><td>No.</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>1</td><td>TF-IDF ngram range</td><td>(1, 2)</td></tr><tr><td>2</td><td>TF-IDF max features</td><td>5000</td></tr><tr><td>3</td><td>random _state</td><td>42</td></tr></table>

Table 2 DL model parameters   

<table><tr><td>No.</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>1</td><td>learning rate</td><td>1e-3</td></tr><tr><td>2</td><td>batch size</td><td>64</td></tr><tr><td>3</td><td>number of epochs</td><td>5</td></tr><tr><td>4</td><td>optimizer</td><td>Adam</td></tr><tr><td>5</td><td>loss function</td><td>CrossEntropyLoss</td></tr><tr><td>6</td><td>embedding dimension</td><td>128</td></tr><tr><td>7</td><td>hidden dimension (LSTM/GRU)</td><td>128</td></tr></table>

Table 3 Oversampling technique parameters   

<table><tr><td>No.</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>1</td><td>RandomOverSampler random_state</td><td>42</td></tr><tr><td>2</td><td>ADASYN n_neighbors</td><td>5</td></tr><tr><td>3</td><td>ADASYN random_state</td><td>42</td></tr><tr><td>4</td><td>SMOTE k_neighbors</td><td>5</td></tr><tr><td>5</td><td>SMOTE random_state</td><td>42</td></tr></table>

# 3.3.2 DL models

The DL group included CNN, LSTM, and GRU models. These models were implemented using PyTorch and trained on padded word sequences generated from a custom-built vocabulary. CNN was used to capture local semantic patterns through convolutional filters, while LSTM and GRU were employed to model sequential dependencies and preserve contextual information across sentences. These architectures were selected based on their proven effectiveness in processing variable-length text and capturing complex linguistic structures. The specific hyperparameters used for the DL models are listed in Table 2.

# 3.3.3 Oversampling techniques

Each model group was evaluated using identical training and test splits, both with and without the application of oversampling techniques to address class imbalance in the dataset. This approach ensured a fair comparison of model performance under balanced and imbalanced conditions. The specific parameters used for each oversampling technique are detailed in Table 3.

# 3.3.4 Transformer-based model (RoBERTa)

For the transformer-based architecture, the RoBERTa model was employed. RoBERTa was selected for its strong ability to capture complex contextual dependencies using self-attention mechanisms. The model was pretrained on a large general-purpose corpus and then fine-tuned on the tourism review dataset to adapt it to the specific sentiment classification task. To evaluate its flexibility and effectiveness, both zero-shot inference and task-specific fine-tuning were conducted. The hyperparameters used during fine-tuning are summarized in Table 4.

This multilevel implementation strategy enables a comprehensive comparison of model performance, providing insights into the trade-offs among computational efficiency, training complexity, and classification effectiveness.

# 3.4 Model evaluation metrics

To evaluate the performance of the sentiment classification models, we employed several metrics that together offer a comprehensive assessment of model effectiveness. These metrics are particularly important given the class imbalance present in the dataset. Each metric is defined below, along with its corresponding formula:

Accuracy measures the proportion of correct predictions (both true positives and true negatives) out of all predictions:

$$
\mathrm { A c c u r a c y } = { \frac { \mathrm { T P } + \mathrm { T N } } { \mathrm { T P } + \mathrm { T N } + \mathrm { F P } + \mathrm { F N } } }
$$

While widely used, accuracy can be misleading in imbalanced datasets, as it may reflect majority-class dominance.

● Balanced Accuracy calculates the average recall for both classes, helping to prevent bias toward the majority class:

$$
\mathrm { B a l a n c e d ~ A c c u r a c y } = { \frac { 1 } { 2 } } \left( { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } + { \frac { \mathrm { T N } } { \mathrm { T N } + \mathrm { F P } } } \right)
$$

Where:

Table 4 Fine-tuned RoBERTa model parameters   

<table><tr><td>No.</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>1</td><td>base model</td><td>cardiffnlp/twitter-roberta-base-sentiment</td></tr><tr><td>2</td><td>maximum length of input sequence</td><td>512</td></tr><tr><td>3</td><td>batch size</td><td>64</td></tr><tr><td>4</td><td>number of epochs</td><td></td></tr><tr><td>5</td><td>learning rate</td><td>2e-5</td></tr><tr><td>6</td><td>dropout prob- ability (encoder/ pooler)</td><td>0.1</td></tr><tr><td>7</td><td>dropout probabil- 0.1 ity (attention)</td><td></td></tr><tr><td>8</td><td>optimizer</td><td>AdamW</td></tr></table>

$\mathrm { T P } =$ True Positives

TN $=$ True Negatives

FP $=$ False Positives

FN $=$ False Negatives

This metric ensures fair evaluation of both majority and minority classes.

● Precision evaluates the proportion of true positives among all instances predicted as positive:

$$
\mathrm { P r e c i s i o n } = \ { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } }
$$

Precision is especially important when the cost of false positives is high.

● Recall (also known as sensitivity) measures the proportion of actual positives correctly identified by the model:

$$
\mathrm { R e c a l l } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } }
$$

Recall is critical in contexts where missing positive instances has significant consequences.

● F1-score provides a harmonic mean of precision and recall, balancing the two when they are in tension:

$$
\mathrm { F 1 - S c o r e = 2 \times \frac { \ P r e c i s i o n \times \ R e c a l l } { P r e c i s i o n + \ R e c a l l } }
$$

The F1-score is particularly useful when dealing with imbalanced datasets, as it accounts for both false positives and false negatives.

Cohen’s kappa accounts for the agreement that could occur by chance, offering a more robust measure of classification reliability:

$$
\mathrm { K } = { \frac { \mathrm { p _ { 0 } - p _ { e } } } { 1 - \mathrm { p _ { e } } } }
$$

Where:

${ \mathfrak { p } } _ { 0 } =$ the observed agreement, $\mathsf { p } _ { \mathrm { e } } =$ the expected agreement by chance

Kappa values range from $^ { - 1 }$ (complete disagreement) to 1 (perfect agreement), with 0 indicating chance-level agreement.

Interpretation guidelines are as follows.

$< 0 . 0 0$ : Poor agreement (worse than chance)   
0.00–0.20: Slight agreement   
0.21–0.40: Fair agreement   
0.41–0.60: Moderate agreement   
0.61–0.80: Substantial agreement   
0.81–1.00: Almost perfect agreement

By applying this set of metrics, we aim to identify the most suitable model for classifying tourism-related reviews. This approach ensures a well-rounded and statistically grounded evaluation of each model’s performance.

# 3.5 Experimental design

The experimental methodology follows a controlled comparative framework designed to isolate the effects of model architecture and class imbalance handling techniques. The dataset was split into $70 \%$ training and $30 \%$ testing sets, using random_state $= 4 2$ to ensure reproducibility. This ratio was selected to provide sufficient training data for complex models, particularly transformer-based architectures, while preserving a substantial test set for robust evaluation.

To address the substantial class imbalance common in tourism sentiment data— where positive reviews typically outnumber negative ones—three oversampling strategies were systematically evaluated: SMOTE, which generates synthetic samples through interpolation; ADASYN, which performs density-based adaptive sampling; and RandomOverSampler, which duplicates minority-class samples as a baseline. This multi-strategy approach allows for a comprehensive analysis of how different data balancing methods affect model performance across architectural types, moving beyond simple duplication methods that may lead to overfitting.

Each model group was evaluated under controlled conditions, both with and without oversampling, to assess performance impacts. Traditional ML models were implemented using TF-IDF vectorization with an n-gram range of (1, 2) for feature extraction. DL models were built with standardized architectures and custom vocabularies. The RoBERTa model was tested in two configurations: zero-shot inference using the CardiffNLP/twitter-roberta-base-sentiment pretrained model and task-specific fine-tuning on the tourism dataset to assess the benefits of domain adaptation.

The study relied on several key Python libraries to streamline various stages of the implementation and evaluation process:

● PyTorch and TorchText for DL model implementation and text preprocessing. Transformers (Hugging Face) for integration of RoBERTa pretrained models and fine-tuning capabilities. Scikit-learn for traditional machine learning algorithms and evaluation metrics. Imbalanced-learn for applying SMOTE, ADASYN, and RandomOverSampler to address class imbalance. Pandas and NumPy for data manipulation and numerical computations.   
TensorFlow for verifying GPU availability and assessing system compatibility.

<!-- FIGURE-DATA: Fig. 4 | type: plot -->
> **[Extracted Data]**
> - Traditional ML vs DL comparison on imbalanced data
> **Analysis:** Compares performance metrics across models.
<!-- /FIGURE-DATA -->
To optimize processing on macOS systems, hardware acceleration was enabled using Metal Performance Shaders (MPS), with automatic fallback to CPU processing when GPU resources were unavailable. This ensured compatibility and scalability across different computing environments, particularly for handling large-scale UGC datasets.

<!-- FIGURE-DATA: Fig. 5 | type: plot -->
> **[Extracted Data]**
> - Oversampling techniques effect on ML models
<!-- FIGURE-DATA: Fig. 6 | type: plot -->
> **[Extracted Data]**
> - Oversampling techniques effect on DL models
> **Analysis:** Shows oversampling impact on deep learning.
<!-- /FIGURE-DATA -->
> **Analysis:** Shows RandomOverSampler, ADASYN, SMOTE impact.
<!-- /FIGURE-DATA -->
Model performance was evaluated using a comprehensive set of metrics: accuracy, precision, recall, F1-score, balanced accuracy, and Cohen’s kappa coefficient. Training time was recorded for all models to enable a comparative analysis of computational efficiency. In addition, confusion matrix analysis was conducted to provide a detailed view of classification behavior across model types and oversampling conditions.

# 4 Results

This section presents a comprehensive comparative analysis of sentiment classification models applied to TripAdvisor reviews. The focus is on traditional ML models (NB, LR, and SVM), DL models (CNN, LSTM, GRU), and three oversampling techniques (SMOTE, ADASYN, and RandomOverSampler). Each model is evaluated using multiple performance metrics—accuracy, precision, recall, F1-score, balanced accuracy, Cohen’s kappa, and training time—under both imbalanced and balanced data conditions. Confusion matrices are also used to visualize misclassification patterns.

# 4.1 Performance on imbalanced dataset

All models in this study outperformed the majority-class baseline $( 8 2 . 1 1 \%$ accuracy, $\kappa { = } 0$ , balanced accuracy $\mathbf { \bar { \rho } } = 0$ ). Figure 4 illustrates the performance comparison between traditional ML and DL models on the imbalanced dataset. Among the traditional ML models, LR demonstrated the most balanced and robust performance, achieving $9 0 . 2 5 \%$ accuracy, $9 1 . 1 3 \%$ precision, $9 7 . 6 6 \%$ recall, and an F1-score of $9 4 . 2 8 \%$ . It also achieved a Cohen’s kappa of 0.6150, indicating substantial agreement, and a balanced accuracy of $7 6 . 7 3 \%$ , reflecting strong fairness across class predictions. SVM produced comparable F1 performance but showed a much lower kappa value $\scriptstyle ( \kappa = 0 . 1 0 4 1$ ), indicating overfitting to the majority class. Its balanced accuracy of $5 3 . 3 1 \%$ highlights a limited ability to classify minority-class sentiment correctly. NB, despite achieving perfect recall for the minority class, performed poorly on fairness-related metrics—recording $\kappa { = } 0 . 0 0 5 4$ and a balanced accuracy of $5 0 . 1 7 \%$ —suggesting extreme bias toward the dominant class and limited discriminatory power across sentiment categories.

![](images/101a77f9b1bfd2f437161910e407062ba39429d93110e7aeabfc3525a4d98a43.jpg)  
Fig. 4 Comparison of traditional ML and DL models on imbalanced dataset across performance metrics

In contrast, the DL models yielded consistently stronger and more reliable performance across all evaluation metrics. LSTM recorded the highest accuracy and Cohen’s kappa $_ { \kappa = 0 . 6 8 4 6 }$ ), along with a balanced accuracy of $8 3 . 0 9 \%$ . GRU and CNN followed closely, with kappa values of 0.6781 and 0.6591, respectively, and balanced accuracy scores exceeding $8 1 \%$ . Unlike traditional models, DL models maintained more equitable classification performance across sentiment classes, even under imbalanced conditions.

While LR emerged as the strongest performer among the traditional ML models, it was outperformed by all DL models in terms of fairness and class balance. SVM’s high recall did not translate into reliable classification due to its low kappa score, and NB failed to meaningfully detect minority-class sentiment. These findings underscore the importance of using evaluation metrics beyond accuracy when working with real-world imbalanced datasets. In this context, DL models, particularly LSTM and GRU, prove to be more consistent and reliable solutions for sentiment classification in the tourism domain under skewed class distributions.
<!-- FIGURE-DATA: Fig. 7 | type: plot -->
> **[Extracted Data]**
> - RoBERTa pretrained vs fine-tuned comparison
> **Analysis:** Shows performance difference between configurations.
<!-- /FIGURE-DATA -->

![](images/e747b5094b39c3971583cacc65470181690a1b5400eb89b5f66095f24cbe4219.jpg)  
Fig. 5 Effect of oversampling techniques (RandomOverSampler, ADASYN, and SMOTE) on traditional ML model performance metrics

![](images/949581ba7f43371efa5ecf7924c2d7570e2605f86389205b2b39390efb9e8788.jpg)  
Fig. 6 Effect of oversampling techniques (RandomOverSampler, ADASYN, and SMOTE) on DL model performance metrics

<!-- FIGURE-DATA: Fig. 8 | type: plot -->
> **[Extracted Data]**
> - Confusion matrices comparison
<!-- FIGURE-DATA: Fig. 9 | type: plot -->
> **[Extracted Data]**
> - Training time comparison
> **Analysis:** Shows computational time across models.
<!-- /FIGURE-DATA -->
> **Analysis:** Shows before/after SMOTE and RoBERTa variants.
<!-- /FIGURE-DATA -->
# 4.2 Impact of oversampling technique on model performance

Figures 5 and 6 illustrate the impact of three oversampling techniques—RandomOverSampler, ADASYN, and SMOTE—on the performance of traditional ML and DL models.

For traditional models (Group 3), SMOTE produced the most significant improvements. SVM, for example, showed a dramatic increase in Cohen’s kappa (from 0.1041 to 0.8464) and a substantial boost in balanced accuracy, indicating greatly improved fairness in class prediction. Logistic Regression also benefited considerably, achieving a kappa score of 0.8237. Notably, NB, which previously demonstrated near-zero agreement, reached a kappa of 0.7692 and a balanced accuracy of

$8 6 . 6 7 \%$ , suggesting effective mitigation of class bias. While RandomOverSampler and ADASYN also improved model performance, their impact was less consistent— particularly for NB. Overall, SVM remained the most stable and responsive performer among the traditional models across all oversampling methods, especially when paired with SMOTE.

Among DL models (Group 4), oversampling improved fairness and class balance rather than significantly increasing accuracy, which was already high. For example, under RandomOverSampler, GRU achieved a balanced accuracy of $8 3 . 5 6 \%$ and a kappa score of 0.6696. LSTM and CNN performed similarly, though with minor metric variations. ADASYN also led to measurable improvements, although CNN showed a slight dip in overall accuracy $( 8 8 . 5 0 \% )$ despite maintaining high precision—highlighting ADASYN’s sensitivity to class boundaries. Once again, SMOTE provided the most consistent gains. Both GRU and LSTM maintained substantial agreement $( \mathrm { K } { \approx } 0 . 6 7 )$ and balanced accuracy above $8 3 \%$ , reinforcing their ability to classify both sentiment classes equitably even under imbalance. CNN saw modest improvement but continued to lag slightly behind in agreement metrics.

In summary, SMOTE outperformed the other techniques in improving fairness (Cohen’s kappa) and class balance across all model types, particularly for those most affected by imbalance, such as NB and SVM. Its interpolation-based sampling produced synthetic data distributions that better reflected class decision boundaries without introducing the noise often associated with RandomOverSampler or the density sensitivities of ADASYN. These results suggest that even robust DL models benefit from oversampling, especially when fairness, minority-class performance, and classification reliability are prioritized.

# 4.3 Transformer-based model performance: RoBERTa performance

Figure 7 presents the comparative performance of the RoBERTa model under two configurations: pretrained and fine-tuned. The fine-tuned variant consistently outperformed all previously evaluated models, including traditional ML models, DL models, and those enhanced with oversampling techniques.

The pretrained RoBERTa model, applied without task-specific adaptation, demonstrated strong recall but showed limitations in fairness and balance. With a Cohen’s kappa of 0.4424 and a balanced accuracy of $6 8 . 8 9 \%$ , the model exhibited a moderate bias toward the majority class—similar to the behavior observed in NB under imbalanced conditions. This result indicates that while pretrained transformer models offer generalizable capabilities, they may underperform when faced with highly skewed datasets unless appropriately fine-tuned.

In contrast, the fine-tuned RoBERTa model showed substantial performance improvements. Cohen’s kappa increased to 0.7321, and balanced accuracy rose to $8 5 . 9 2 \%$ , reflecting a significant reduction in class bias while maintaining high levels of precision and recall. These results place fine-tuned RoBERTa above all previously tested models in terms of fairness, consistency, and overall classification reliability. The superior performance of the fine-tuned model highlights the critical role of task-specific adaptation when applying transformer-based architectures to real-world sentiment analysis, particularly in domains like tourism, where reviews frequently contain mixed sentiments and fairness in classification is essential.

![](images/cb9591c104b457da96ab01488f71a2efe7ec83676db3d9fa75fcebaef7e6f93f.jpg)  
Group 5: RoBERTa - Performance Metrics   
Fig. 7 Performance comparison of RoBERTa: Pretrained vs. Fine-tuned configurations

# 4.4 Confusion matrix analysis

To complement the metric-based evaluation, confusion matrices were analyzed to provide deeper insights into each model’s classification behavior, particularly in distinguishing between positive and negative sentiment classes under imbalanced and balanced (SMOTE-applied) conditions. The confusion matrices before and after SMOTE, as well as for both pretrained and fine-tuned RoBERTa models, are summarized in Fig. 8.

Traditional ML models exhibited a clear bias toward the majority class prior to oversampling. For example, NB identified 124,794 true positives but only 89 true negatives, misclassifying 26,757 positive reviews—demonstrating extreme sensitivity to class imbalance. SVM and LR performed better, though they remained skewed: SVM correctly classified only 1,805 negative instances, while LR captured 14,978. After applying SMOTE, all traditional models showed notable improvements in detecting the minority class. For NB, true negatives increased from 89 to 23,097, while false positives dropped significantly. SVM’s true negatives rose to 22,355, confirming that SMOTE substantially rebalanced its predictions. These results indicate that SMOTE improved minority-class detection without significantly inflating false positives—a crucial balance between recall and precision.

DL models, in contrast, handled imbalance more effectively even without oversampling. For instance, GRU produced a relatively balanced result with 19,198 true negatives and 118,711 true positives, while CNN exhibited a mild positive bias, resulting in 9,048 false positives. SMOTE led to marginal but meaningful improvements across DL models. GRU increased its true negatives to 19,848 and reduced false positives to 7,193. LSTM and CNN followed similar trends—less dramatic than in traditional ML models but still indicative of SMOTE’s fine-tuning effect. This suggests that oversampling serves to refine already resilient DL classifiers rather than fundamentally altering their behavior.

![](images/4dc60678f0fca04e9304e0662741ff349da662d148bbff753cdba163dca4d9d1.jpg)  
Fig. 8 Confusion matrices of all models before and after SMOTE, and RoBERTa Pretrained vs. Fine-tuned

![](images/009f855788fa8eaf7266920234b434b49b1a1ee3406a7e5e6ef8e8e85379c0ca.jpg)  
Fig. 9 Training time comparison across models

For RoBERTa, the confusion matrices illustrate the critical impact of fine-tuning. The pretrained version achieved high recall (119,114 true positives), but this came at the cost of 15,600 false positives, reflecting a limited ability to reject incorrect positive predictions. Fine-tuning RoBERTa significantly improved model balance across all outcomes: true negatives nearly doubled to 20,522, and false positives dropped to 6,475, highlighting a much-improved precision–recall trade-off. These shifts align with RoBERTa’s earlier performance metrics, reinforcing its robustness under finetuned conditions.

Among all oversampling methods, SMOTE again emerged as the most reliable. It consistently enhanced minority-class detection across model families without significantly degrading precision. Unlike ADASYN, which introduced variability, or RandomOverSampler, which risked overfitting through duplication, SMOTE generated synthetic samples that more accurately reflected decision boundaries. This balance is especially valuable in tourism sentiment analysis, where fairness and interpretability are as important as accuracy.

# 4.5 Training time and efficiency

While accuracy, fairness, and agreement metrics are critical in model evaluation, computational efficiency and training time are also essential—especially for largescale or real-time deployment in tourism sentiment systems. Table 4; Fig. 9 summarize the training durations for all model groups, highlighting clear trade-offs between performance and computational cost across traditional ML, DL, oversampling techniques, and transformer-based models.

Among all model types, traditional ML classifiers were the fastest to train. NB completed training in under a second (0.38 s), while SVM and LR followed at 1.28 s and 30.71 s, respectively, making them highly suitable for resource-constrained environments. However, as previously discussed, these models struggled with class imbalance unless paired with balancing strategies. Oversampling modestly increased training time for traditional models. For example, applying SMOTE raised LR training time from 30.71 to 71.81 s, while NB and SVM still trained in under 3 s. ROS and ADASYN incurred similar computational costs, indicating that the performance gains from oversampling come at a relatively low computational expense.

In contrast, DL models required significantly more time due to their complex architectures. CNN, LSTM, and GRU models trained in approximately 3,900 to $6 , 1 0 0 \mathrm { ~ s ~ }$ , with LSTM taking the longest (6,079.11 s) due to its sequential processing steps. These models offered improved performance under class imbalance but required GPU acceleration or cloud infrastructure for efficient use. Oversampling further extended DL training times: GRU with ADASYN trained for 10,929.06 s, and LSTM with SMOTE required $1 1 , 1 3 0 . 5 6 \mathrm { ~ s ~ }$ , reflecting the increased batch sizes and longer convergence cycles introduced by synthetic data generation. Despite the added cost, oversampling enabled DL models to achieve the most reliable and fair results, justifying the time investment.

The pretrained RoBERTa model offered a strong baseline with minimal inference cost, requiring only 1,025 s. However, fine-tuning RoBERTa was by far the most computationally expensive task, taking 110,829.76 s $( \sim 3 0 . 8 \mathrm { h } )$ . This high cost is due to its large parameter space and task-specific optimization. Nevertheless, the finetuned model outperformed all others in terms of accuracy, fairness, and class balance, illustrating a clear trade-off between performance and scalability.

In conclusion, model selection must weigh fairness and effectiveness against time and resource constraints. Traditional ML models are ideal for rapid deployment, DL models provide stronger baseline performance with moderate training costs, and finetuned transformers like RoBERTa deliver state-of-the-art results, albeit at a significantly higher computational expense.

# 5 Discussion

This study presents a comprehensive evaluation of sentiment classification models applied to tourism reviews, comparing traditional machine learning (ML) models, deep learning (DL) architectures, and transformer-based models under both imbalanced and balanced data conditions. The discussion is organized around the study’s core research questions, interpreting the findings with regard to model performance and their practical implications for real-world applications in tourism sentiment analysis. Based on the comprehensive results presented, several key findings emerge from this comparative analysis of sentiment classification models applied to TripAdvisor reviews. The study was guided by three core research questions.

First (RQ1), the results show that DL models, particularly LSTM and GRU, consistently outperformed traditional ML models in terms of accuracy, fairness, and agreement, especially under imbalanced data conditions. While traditional models like logistic regression performed relatively well, others such as Naive Bayes exhibited strong bias toward the majority class. The fine-tuned RoBERTa model outperformed all other models in overall performance, demonstrating substantial agreement and balanced precision–recall. These findings confirm the superiority of transformerbased architectures when fine-tuned on task-specific data.

Second (RQ2), the use of oversampling techniques, especially SMOTE, significantly improved performance across both traditional and DL models. SMOTE yielded the most balanced results, enhancing recall and Cohen’s kappa while reducing false positives. It was particularly effective in improving the reliability of weaker baseline models such as Naive Bayes under class imbalance, thus promoting more robust and balanced sentiment classification.

Third (RQ3), the analysis of training time and efficiency revealed notable tradeoffs. Traditional ML models had the fastest training times but required oversampling to achieve competitive performance. DL models provided stronger baseline accuracy at a moderate computational cost. Although RoBERTa delivered the best overall results, it required significantly more training time—highlighting the need to balance computational resources with desired performance outcomes.

# 5.1 Impact of imbalanced data on model performance

The results addressing RQ1 demonstrate that imbalanced data significantly affects classification performance and reliability, particularly for traditional ML models. Among these, NB exhibited a strong bias toward the majority class. Prior to applying SMOTE, NB achieved perfect recall $( 1 0 0 \% )$ but extremely low Cohen’s kappa (0.0054) and balanced accuracy $( 5 0 . 1 7 \% )$ , indicating near-random agreement despite superficially high recall. This finding supports previous research (Lango 2019; Mehraliyev et al. 2022) highlighting NB’s limitations under skewed distributions—especially its reliance on feature independence and frequency-based assumptions. Other studies have similarly noted its difficulty in detecting subtle or mixed sentiments common in tourism reviews (Obiedat et al. 2022; Pan et al. 2020; Suwitho et al. 2023; Núñez et al. 2024; Foo et al. 2022).

By contrast, DL models—particularly LSTM and GRU—demonstrated stronger baseline performance. LSTM achieved $9 1 . 0 6 \%$ accuracy and a Cohen’s kappa of 0.6846, indicating substantial agreement. These results underscore the advantages of recurrent neural networks in modeling sequential text and mitigating the effects of imbalance without relying on external correction methods.

Most notably, the fine-tuned RoBERTa model delivered the best overall performance, achieving $9 2 . 3 1 \%$ accuracy, a $9 5 . 3 4 \%$ F1-score, and a Cohen’s kappa of 0.7321. Unlike traditional models, RoBERTa leverages contextual embeddings and bidirectional transformer architecture (Liu et al. 2019; Bird et al. 2023; Liao et al. 2021), allowing it to capture nuanced linguistic features. The confusion matrix showed that RoBERTa significantly reduced both false negatives and false positives, leading to more balanced and fair sentiment classification. These findings are consistent with prior research on transformer-based models $\mathrm { \sf W u } 2 0 2 4$ ; Bird et al. 2023), confirming that contextual approaches are more resilient to class imbalance and better suited for complex user-generated content such as tourism reviews.

# 5.2 Effectiveness of SMOTE in addressing class imbalance

In response to RQ2, the findings show that oversampling techniques, particularly SMOTE, significantly enhance classification outcomes for both traditional and DL models. For NB, SMOTE increased Cohen’s kappa from 0.0054 to 0.5849 and raised balanced accuracy to $8 5 . 5 8 \%$ , confirming its effectiveness in correcting class bias. These improvements align with previous research emphasizing SMOTE’s strength in improving recall for minority classes (Xiao et al. 2019; Pan et al. 2020), even if gains sometimes come at the expense of precision.

NB, which previously failed to detect negative sentiment reliably, benefited the most from SMOTE. This suggests that even weaker classifiers can become viable with appropriate class-balancing techniques. As expected, a precision–recall tradeoff was observed: although recall improved, precision declined due to an increase in false positives, consistent with findings by Xiao et al. (2019).

For DL models, the improvements were more modest but still notable. LSTM and GRU already demonstrated strong baseline performance, yet SMOTE further enhanced their fairness by slightly increasing true negative rates and reducing false positives. These adjustments indicate that SMOTE can help refine decision boundaries, even in more robust models.

Compared to RandomOverSampler and ADASYN, SMOTE produced more stable and consistent improvements. While RandomOverSampler risks overfitting by duplicating samples, and ADASYN can introduce noise through variable sampling, SMOTE’s interpolative approach more effectively generates balanced training sets with smoother class boundaries. This helped reduce classification errors, particularly in minority sentiment detection.

# 5.3 The role of fine-tuning and pretrained performance

In response to RQ2, this study examined the extent to which model performance can be enhanced through fine-tuning rather than external data-level resampling. Unlike traditional classifiers, which benefited significantly from SMOTE and other oversampling techniques, RoBERTa did not require such preprocessing. Fine-tuning with domain-specific labeled data alone led to substantial improvements in both precision and recall, as well as in Cohen’s kappa and balanced accuracy.

These findings support the argument that transformer models, when properly adapted to the task, can address class imbalance intrinsically through advanced representation learning (Demir and Bilgin 2023). While prior research has emphasized oversampling as a remedy for weaker classifiers, this study demonstrates that modellevel optimization,—specifically, fine-tuning—is a more scalable and generalizable approach for high-performing architectures such as RoBERTa.

# 5.4 Practical implications for tourism sentiment analysis

The findings have clear implications for sentiment classification in tourism and further address RQ1 and RQ2. While traditional models like NB are computationally efficient, their reliability under class imbalance is limited without additional support. DL models such as GRU and LSTM strike a favorable balance between performance and training time, making them practical for scalable tourism applications.

RoBERTa, although computationally demanding, produced the most accurate and reliable sentiment predictions. This makes it especially well-suited for tasks requiring high precision, such as policymaking, tourism demand forecasting, or managing online reputation. In real-world settings, a hybrid strategy may be most effective: lightweight models can handle real-time sentiment classification, while fine-tuned RoBERTa models can be reserved for batch processing or high-stakes decision-making.

Tourism operators and policymakers can leverage sentiment insights to guide service improvements and strategic planning. For example, simpler ML models could be deployed to monitor general sentiment trends, while high-performance models like RoBERTa could be used to analyze critical reviews or emerging issues in detail. However, accurate sentiment classification is a prerequisite for deeper analyses. Misclassifications at this stage can distort topic modeling and feature extraction, ultimately leading to misguided managerial decisions or ineffective interventions.

Ensuring high-precision sentiment classification is therefore essential for the credibility and utility of downstream applications. Moreover, aggregated sentiment trends can support sustainable tourism development by aligning infrastructure investments, service enhancements, and marketing strategies with actual visitor experiences and expectations.

# 5.5 Limitations and future research directions

With regard to RQ3, the training time comparison reveals a clear trade-off between model performance and computational efficiency. While traditional ML models such as NB and SVM train quickly, their performance, especially under class imbalance, is limited. DL models offer better outcomes but at a higher time cost. RoBERTa delivered the best results, but required over 110,000 s to fine-tune, highlighting the need to align model selection with practical constraints, including hardware capabilities, latency requirements, and budget.

A further limitation of this study is its reliance on document-level sentiment classification. Tourism reviews often contain both positive and negative elements, and analyzing sentiment at the document level may obscure these mixed signals. This limitation likely contributed to NB’s poor performance, as it lacks the capacity to differentiate conflicting sentiments within a single review.

Future research could improve classification reliability by adopting sentence-level sentiment analysis with manually labeled data. Although such annotation is resourceintensive and inherently subjective, particularly in tourism contexts where individual perceptions vary widely, it could enable multi-label or fine-grained sentiment modeling, providing more nuanced and interpretable insights into tourist experiences. Additional directions include exploring lightweight transformer models such as DistilBERT or TinyBERT to balance performance with resource demands. Investigating multilingual sentiment classification and emotion detection in tourism UGC could also extend the reach of this research, especially in cross-cultural and international tourism contexts.

# 6 Conclusion

This study presented a comprehensive comparative analysis of sentiment classification approaches for tourism reviews, evaluating nine models across traditional ML, DL, and transformer-based architectures using 505,980 TripAdvisor reviews. It addressed key gaps in tourism sentiment analysis by examining model performance, fairness, and computational efficiency under imbalanced data conditions.

The results reveal substantial performance differences across model types. DL models, particularly LSTM $( 9 1 . 0 6 \%$ accuracy; Cohen’s kappa: 0.6846) and GRU $9 0 . 8 2 \%$ accuracy; kappa: 0.6781), consistently outperformed traditional approaches in managing imbalanced sentiment data. Traditional models showed notable limitations, especially Naive Bayes, which suffered from severe majority-class bias. Despite an $8 2 . 3 5 \%$ accuracy, its near-zero Cohen’s kappa (0.0054) and $5 0 . 1 7 \%$ balanced accuracy highlighted misleading results. In contrast, the fine-tuned RoBERTa model achieved top-tier performance— $9 2 . 3 1 \%$ accuracy, $9 5 . 3 4 \%$ F1-score, and 0.7321 Cohen’s kappa—demonstrating the effectiveness of contextual embeddings in capturing the linguistic complexity of tourism reviews.

SMOTE proved to be the most effective oversampling technique, significantly enhancing fairness across all model categories. In traditional models, it transformed underperforming classifiers into viable tools; for instance, NB’s kappa increased from 0.0054 to 0.7692. Notably, RoBERTa did not require such preprocessing—its fine-tuning process inherently addressed class imbalance, suggesting that modellevel optimization offers a more scalable and generalizable solution.

The computational analysis underscored key trade-offs. Traditional models trained quickly (0.38–30.71 s) but required oversampling to achieve reliability. DL models offered a favorable balance between performance and resource requirements (3,900– 6,100 s), while RoBERTa achieved state-of-the-art results at a significant computational cost (110,829.76 s).

These findings provide actionable insights for tourism practitioners. Traditional models paired with SMOTE may be suitable for resource-constrained environments; DL models offer a balance between efficiency and effectiveness; and transformerbased models are best reserved for high-stakes or mission-critical applications. The study also reinforces the value of incorporating fairness metrics alongside accuracy when evaluating sentiment classification systems.

Future research should explore sentence-level sentiment analysis, lightweight transformer variants, and multilingual applications. As the tourism industry increasingly turns to data-driven decision-making, the adoption of robust and fair sentiment analysis models is essential for extracting actionable insights from user-generated content and enhancing responsive service delivery in the global tourism ecosystem.

Acknowledgements This work was supported by the Digital Science for Economy, Society, Human Resources Innovative Development and Environment project funded by Reinventing Universities & Research Institutes under grant no. 2046735, Ministry of Higher Education, Science, Research and Innovation, Thailand.

Author contributions S.S. wrote the main manuscript text, performed the data analysis. A.N. provided the concept and edited the manuscript. C.P. provided methodology and edited the manuscript.

Data availability No datasets were generated or analysed during the current study.

# Declarations

Competing interests The authors declare no competing interests.

# References

Ahani A, Nilashi M, Yadegaridehkordi E, Sanzogni L, Tarik AR, Knox K, Ibrahim O (2019) Revealing customers’ satisfaction and preferences through online review analysis: the case of Canary Islands hotels. J Retailing Consumer Serv 51:331–343. https://doi.org/10.1016/j.jretconser.2019.06.014   
Alaei A, Wang Y, Bui V, Stantic B (2023) Target-oriented data annotation for emotion and sentiment analysis in tourism related social media data. Future Internet 15(4):150. https://doi.org/10.3390/​f​i​1 5040150   
Almuayqil SN, Humayun M, Jhanjhi NZ, Almufareh MF, Khan NA (2022) Enhancing sentiment analysis via random majority under-sampling with reduced time complexity for classifying tweet reviews. Electronics 11(21). https://doi.org/10.3390/electronics11213624   
Amirkumar M, Orynbekova K, Talasbek A, Ayazbayev D, Cankurt S (2024) Comparative effectiveness of rule-based and machine learning methods in sentiment analysis of Kazakh Language texts. Sci J Astana IT Univ 16–27. https://doi.org/10.37943/17RHPH9724   
Basiri M, Nemati S, Abdar M, Cambria E, Acharrya U (2021) ABCDM: an attention-based bidirectional CNN-RNN deep model for sentiment analysis. Future Generation Comput Syst 115:279–294. https: //doi.org/10.1016/j.future.2020.08.005   
Bianchi G, Heo CY (2021) A bayesian statistics approach to hospitality research. Curr Issues Tourism 24(22):3141–3150. https://doi.org/10.1080/13683500.2021.1896486   
Bird JJ, Ekárt A, Faria DR (2023) Chatbot interaction with artificial intelligence: human data augmentation with T5 and Language transformer ensemble for text classification. J Ambient Intell Humaniz Comput 14:3129–3144. https://doi.org/10.1007/s12652-021-03439-8   
Catelli R, Bevilacqua L, Mariniello N, di Carlo VS, Magaldi M, Fujita H, Esposito M (2022) Cross lingual transfer learning for sentiment analysis of Italian tripadvisor reviews. Expert Syst Appl 209:118246. https://doi.org/10.1016/j.eswa.2022.118246   
Chaudhuri A, Sahu TP (2021) Feature weighting for Naïve Bayes using multi-objective artificial bee colony algorithm. Int J Comput Sci Eng 24(1):74–88. https://doi.org/10.1504/IJCSE.2021.113655   
Cheruku R, Hussain K, Kavati I, Manne S (2024) Sentiment classification with modified RoBERTa and recurrent neural networks. Multimedia Tools Appl 83:29399–29417. https://doi.org/10.1007/s1104 2-023-16833-5   
Cutler A, Condon DM (2023) Deep lexical hypothesis: identifying personality structure in natural Language. J Personal Soc Psychol 125(1):173–197. https://doi.org/10.1037/pspp0000443   
Demir E, Bilgin M (2023) Sentiment analysis from Turkish news texts with BERT-based language models and machine learning algorithms. In: Proceedings of the 2023 8th International Conference on Computer Science and Engineering (UBMK); IEEE; p. 1–4. https://doi.org/10.1109/UBMK59864. 2023.10286719   
Devlin J, Chang MW, Lee K, Toutanova K (2019) BERT: pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Vol. 1, p. 4171– 86. https://doi.org/10.18653/v1/N19-1423   
Eberle T, Fuchs M, Höpken W (2025) Detecting emotions in user generated content and their influence on tourist satisfaction. In: Nixon L, Tuomi A, O’Connor P (Eds.). Information and Communication Technologies in Tourism 2025. Vol. ENTER 2025. Springer. https://doi.org/10.1007/978-3-031-83705-0_17   
Erfani S, Rajasegarar S, Karunasekera S, Leckie C (2016) High-dimensional and large-scale anomaly detection using a linear one-class SVM with deep learning. Pattern Recogn 58:121–134. https://doi. org/10.1016/j.patcog.2016.03.028   
Foo LK, Chua SL, Ibrahim N (2022) Attribute weighted Naïve Bayes classifier. Computers Mater Continua 71(1):1945–1957. https://doi.org/10.32604/cmc.2022.022011   
Gómez-Déniz E, Martel-Escobar M, Vázquez Polo FJ (2024) A bayesian model for online customer reviews data in tourism research: A robust analysis. Cogent Bus Manage 11(1). https://doi.org/10.10 80/23311975.2024.2363592   
Guerrero-Rodríguez R, Álvarez-Carmona MÁ, Aranda R et al (2024) Big data analytics of online news to explore destination image using a comprehensive deep-learning approach: A case from Mexico. Inform Technol Tourism 26:147–182. https://doi.org/10.1007/s40558-023-00278-5   
Ham J, Lee K, Kim T, Koo C (2019) Subjective perception patterns of online reviews: A comparison of utilitarian and hedonic values. Inf Process Manag 56:1439–1456. https://doi.org/10.1016/J.IPM.20 19.03.011   
Hu N, Zhang T, Gao B, Bose I (2019) What do hotel customers complain about? Text analysis using structural topic model. Tour Manag 72:417–426. https://doi.org/10.1016/j.tourman.2019.01.002   
Irawan B, Nurdiawan O (2023) Naive Bayes and wordcloud for sentiment analysis of halal tourism in Lombok Island, Indonesia. Innov Res Inf (Innovatics) 5(1). https://doi.org/10.37058/innovatics.v5 i1.6675   
Jeng CR, Snyder AT, Chen CF (2019) Importance–performance analysis as a strategic tool for tourism marketers: the case of taiwan’s destination image. Tourism Hospitality Res 19(1):112–125. https://d oi.org/10.1177/146735841770488   
Jiang L, Li C, Wang S, Zhang L (2016) Deep feature weighting for Naive Bayes and its application to text classification. Eng Appl Artif Intell 52:26–39. https://doi.org/10.1016/j.engappai.2016.02.002   
Jo Y (2024) Comprehensive examination of online reviews divergence over time and platform types. Int J Hospitality Manage. https://doi.org/10.1016/j.ijhm.2023.103647   
Kamyab M, Liu G, Adjeisah M (2021) Attention-based CNN and Bi-LSTM model based on TF-IDF and glove word embedding for sentiment analysis. Appl Sci. https://doi.org/10.3390/app112311255   
Kim T, Lee J-S (2022) Exponential loss minimization for learning weighted Naïve Bayes classifiers. IEEE Access 10:22724–22736. https://doi.org/10.1109/ACCESS.2022.3155231   
Lango M (2019) Tackling the problem of class imbalance in multi-class sentiment classification: an experimental study. Found Comput Decis Sci 44(2):151–178. https://doi.org/10.2478/fcds-2019-0009   
Li Y, He Z, Li Y, Huang T, Liu Z (2023) Keep it real: assessing destination image congruence and its impact on tourist experience evaluations. Tour Manag 97:104736. https://doi.org/10.1016/j.tourma n.2023.104736   
Liao W, Zeng B, Yin X, Zhang C (2021) An improved aspect-category sentiment analysis model for text sentiment analysis based on RoBERTa. Appl Intell 51:3522–3533. https://doi.org/10.1007/s10489-0 20-01964-1   
Liu Y, Ott M, Goyal N, Du J, Joshi M, Chen D, Levy O, Lewis M, Zettlemoyer L, Stoyanov V (2019) RoBERTa: a robustly optimized BERT pretraining approach. arXiv [Preprint]; arXiv:1907.11692. https://doi.org/10.48550/arXiv.1907.11692   
Manurung KA, Laksana KM (2023) Sentiment analysis of tourist attraction review from tripAdvisor using CNN and LSTM. Int J Inf Commun Technol 9(1):73–85. https://doi.org/10.21108/ijoict.v9i1.756   
Mehraliyev F, Chan ICC, Kirilenko AP (2022) Sentiment analysis in hospitality and tourism: A thematic and methodological review. Int J Contemp Hospitality Manage 34(1):46–77. https://doi.org/10.1108 /IJCHM-02-2021-0132   
Muizelaar H, Haas M, van Dortmont K et al (2024) Extracting patient lifestyle characteristics from Dutch clinical text with BERT models. PREPRINT (Version 1). Res Square. https://doi.org/10.21203/rs.3 .rs-3831694/v1   
Neidhardt J, Rümmele N, Werthner H (2017) Predicting happiness: user interactions and sentiment analysis in an online travel forum. Inform Technol Tourism 17:101–119. https://doi.org/10.1007/s40558-0 17-0079-2   
Ni WS, Saraswati IKGDP, Sudarma M, Sukarsa IM (2024) Enhance sentiment analysis in big data tourism using hybrid lexicon and active learning support vector machine. Bull Electr Eng Inf 13(5):3663– 3674. https://doi.org/10.11591/eei.v13i5.7807   
Norris K (2025) Bali vs Phuket: an honest comparison in 2025. Waytostay. https://waytostay.com/bali-v s-phuket/   
Nowacki M, Niezgoda A (2020) Identifying unique features of the image of selected cities based on reviews by tripadvisor portal users. Scandinavian J Hospitality Tourism 20(5):503–519. https://doi.o rg/10.1080/15022250.2020.1833362   
Núñez JCS, Gómez-Pulido JA, Ramírez RR (2024) Machine learning applied to tourism: A systematic review. Wiley Interdisciplinary Reviews: Data Min Knowl Discovery 14(5):e1549. https://doi.org/ 10.1002/widm.1549   
Obiedat R, Qaddoura R, Al-Zoubi A, Al-Qaisi L, Harfoushi O, Alrefai M, Faris H (2022) Sentiment analysis of customers’ reviews using a hybrid evolutionary SVM-based approach in an imbalanced data distribution. IEEE Access 10:22260–22273. https://doi.org/10.1109/ACCESS.2022.3149482   
Pan T, Zhao J, Wu W, Yang J (2020) Learning imbalanced datasets based on SMOTE and Gaussian distribution. Inf Sci 512:1214–1233. https://doi.org/10.1016/j.ins.2019.10.048   
Paolanti M, Mancini A, Frontoni E et al (2021) Tourism destination management using sentiment analysis and geo-location information: A deep learning approach. Inform Technol Tourism 23:241–264. https://doi.org/10.1007/s40558-021-00196-4   
Păvăloaia V, Teodor E, Fotache D, Danileț M (2019) Opinion mining on social media data: sentiment analysis of user preferences. Sustainability 11(16):4459. https://doi.org/10.3390/SU11164459   
Primasari I, Khadija MA (2024) Opinion mining of tourism village in Magelang based on Google Reviews data. In: Proceedings of the 2024 International Conference on Data Science and Its Applications (ICoDSA); IEEE; p. 189–94. https://doi.org/10.1109/ICoDSA62899.2024.10652083   
Ramos CM, Cardoso PJ, Fernandes HC, Rodrigues JM (2022) A decision-support system to analyse customer satisfaction applied to a tourism transport service. Multimodal Technol Interact 7(1). https://d oi.org/10.3390/mti7010005   
Ren G, Hong T (2017) Investigating online destination images using a topic-based sentiment analysis approach. Sustainability 9(10) Article 1765. https://doi.org/10.3390/su9101765   
Sabri NM, Subki M, Bahrin SNA, U. F. M., Puteh M (2024) Post-pandemic tourism: sentiment analysis using support vector machine based on TikTok data. Int J Adv Comput Sci Appl 15(2). https://doi.or g/10.14569/IJACSA.2024.0150234   
Sah R, Sengupta S, Kandpal V (2024) The changing face of tourism: smart tourism design and social media analytics. In: Challenges in Information, Communication and Computing Technology. CRC, p. 801–5. https://doi.org/10.1201/9781003559092-138   
Sangkaew N, Nanthaamornphong A, Phucharoen C (2023) Understanding tourists’ perception toward local gourmet consumption in the creative City of gastronomy: factors influencing consumer satisfaction and behavioral intentions. J Qual Assur Hospitality Tourism 1–28. https://doi.org/10.1080/1 528008X.2023.2247159   
Sayeed MS, Mohan V, Muthu KS (2023) BERT: A review of applications in sentiment analysis. HighTech Innov J 4(2):453–462. https://doi.org/10.28991/hij-2023-04-02-015   
Schmunk S, Höpken W, Fuchs M, Lexhagen M (2014) Sentiment analysis: extracting decision-relevant knowledge from UGC. In: Information and Communication Technologies in Tourism 2014: Proceedings of the International Conference in Dublin, Ireland, January 21–24, 2014, 253–265. https://doi.o rg/10.1007/978-3-319-03973-2_19   
Semary A, Ahmed N, Amin K, Pławiak P, Hammad M (2024) Enhancing machine learning-based sentiment analysis through feature extraction techniques. PLoS ONE 19(2):e0294968. https://doi.org/10 .1371/journal.pone.0294968   
Sincharoenkul K, Sangkaew N (2023) Mitigating tourism seasonality: an explanatory sequential analysis of tripadvisor on temple experiences. A case study of Phuket. Int J Tourism Policy 13(3):230–247. https://doi.org/10.1504/IJTP.2023.130808   
Singgalen YA (2024) Sentiment classification of over-tourism issues in responsible tourism content using Naïve Bayes classifier. J Comput Syst Inf (JoSYC) 5(2):275–285. https://doi.org/10.47065/josyc.v 5i2.4904   
Song M, Park H, Shin K (2019) Attention-based long short-term memory network using sentiment lexicon embedding for aspect-level sentiment analysis in Korean. Inf Process Manag 56:637–653. https://do i.org/10.1016/j.ipm.2018.12.005   
Sotiriadis M (2017) Sharing tourism experiences in social media. Int J Contemp Hospitality Manage 29:179–225. https://doi.org/10.1108/IJCHM-05-2016-0300   
Suwitho S, Mustika H, Pradhani FA (2023) Impact of tourist satisfaction attributes on behavior of sharing tourism experience on social media. Jurnal Manajemen Strategi Bisnis Dan Kewirausahaan 17(2):171–171. https://doi.org/10.24843/matrik:jmbk.2023.v17.i02.p05   
Taecharungroj V, Mathayomchan B (2019) Analysing tripadvisor reviews of tourist attractions in phuket, Thailand. Tour Manag 75:550–568. https://doi.org/10.1016/j.tourman.2019.06.020   
Taheri S, Yearwood J, Mammadov M, Seifollahi S (2014) Attribute weighted Naive Bayes classifier using a local optimization. Neural Comput Appl 24(5):995–1002. https://doi.org/10.1007/s00521-012-13 29-z   
Tan KL, Lee CP, Lim KM (2023) RoBERTa-GRU: A hybrid deep learning model for enhanced sentiment analysis. Appl Sci 13(6):3915. https://doi.org/10.3390/app13063915   
Tesoro JC, Buen MJM, Sullera RC Jr, Aborde MV (2020) A semantic approach of the Naïve Bayes classification algorithm. Int J Adv Trends Comput Sci Eng 9(3):3287–3294. https://doi.org/10.30534/ij atcse/2020/125932020 Van den Bergh J (2022) Online reviews in tourism and hospitality industry: a meta-analytical perspective. J Global Bus Advancement 15(4):420. https://doi.org/10.1504/jgba.2022.130443 Vargas-Calderón V, Ochoa M, Castro Nieto A, G. Y., et al (2021) Machine learning for assessing quality of service in the hospitality sector based on customer reviews. Inform Technol Tourism 23:351–379. https://doi.org/10.1007/s40558-021-00207-4 Viñán-Ludeña M, De Campos L (2022) Discovering a tourism destination with social media data: BERTbased sentiment analysis. J Hospitality Tourism Technol. https://doi.org/10.1108/jhtt-09-2021-0259 Wang F (2024) Comparative evaluation of sentiment analysis methods: from traditional techniques to advanced deep learning models. Appl Comput Eng 105(1):23–29. https://doi.org/10.54254/2755-2   
721/105/2024tj0056 Wu Q-M (2024) The influence of online reviews on the purchasing decisions of travel consumers. Sustainability 16(8):3213. https://doi.org/10.3390/su16083213 Xiao Z, Wang L, Du, JY (2019) Improving the performance of sentiment classification on imbalanced datasets with transfer learning. IEEE Access 7:28281–28290. https://doi.org/10.1109/ACCESS.20   
19.289209 Yu Y, Si X, Hu C, Zhang J (2019) A review of recurrent neural networks: LSTM cells and network architectures. Neural Comput 31:1235–1270. https://doi.org/10.1162/neco_a_01199 Yuke W, Yusuf RN (2024) Sentiment analysis of reviews of tourist attractions in the lake Toba area using the Naïve Bayes method. J Comput Networks. Architecture and High-Performance Computinghttps:// doi.org/10.47709/cnahpc.v6i3.4287 Zhao H, Yang M, Bai X, Liu H (2024) A survey on multimodal aspect-based sentiment analysis. IEEE Access 12:12039–12052. https://doi.org/10.1109/ACCESS.2024.3354844

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

# Authors and Affiliations

Sawitree Srianan1 $\cdot$ Aziz Nanthaamornphong $2 \textcircled { \scriptsize { \parallel } }$ · Chayanon Phucharoen3

Aziz Nanthaamornphong aziz.n@phuket.psu.ac.th

Sawitree Srianan annesawitree1407@gmail.com

Chayanon Phucharoen chayanon.p@phuket.psu.ac.th

College of Digital Science, Prince of Songkla University, Songkhla, Thailand

Present address: College of Computing, Prince of Songkla University, Phuket, Thailand

3 Faculty of Hospitality and Tourism, Prince of Songkla University, Phuket, Thailand