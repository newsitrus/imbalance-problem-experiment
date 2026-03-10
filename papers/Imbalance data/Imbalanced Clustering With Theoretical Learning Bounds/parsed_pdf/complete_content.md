# Imbalanced Clustering With Theoretical Learning Bounds

Jing Zhang , Hong Tao , and Chenping Hou , Member, IEEE

Abstract—Imbalanced clustering, where the number of samples varies in different clusters, has arisen from many real data mining applications. It has gained increasing attention. Nevertheless, due to its unsupervised nature, imbalanced clustering is more challenging than its supervised counterpart, i.e., imbalanced classification. Furthermore, existing imbalanced clustering methods are empirically designed and they often lack solid theoretical guarantees, e.g., the excess risk estimation. To solve these important but rarely studied problems, we first propose a novel $\boldsymbol { k }$ -Means algorithm for imbalanced clustering problem with Adaptive Cluster Weight (MACW), together with its excess clustering risk bound analysis. Inspired by this theoretical result, we further propose an improved algorithm called Imbalanced Clustering with Theoretical Learning Bounds (ICTLB). It refines the weights and encourages the optimal trade-off among per-cluster weights by optimizing the excess clustering risk bound. A theoretically-principled justification of ICTLB is provided for verification. Comprehensive experiments on many imbalanced datasets verify the effectiveness of ICTLB in solving cluster imbalanced problems.

Index Terms—Clustering, excess risk, imbalanced data, learning bound.

# I. INTRODUCTION

MBALANCED data learning frequently occurs in many real-world tasks [1], e.g., credit card fraud [2] and machine vision [3], where the imbalanced data indicates that one class compared with other classes has a small number of samples. Most of the research topics for imbalanced data learning often require supervised learning on labeled training data [4]. However, the label information is not easily available in many machine learning applications, including bioinformatics [5], [6], software malfunction identification [7], and text analysis [8], [9], etc. For example, in the area of network security, normal network samples are at a high proportion of all data points, but people should pay more attention to attacked networks. It is well known that the labels of attacked networks are hard to define due to diversity, and the above task is an imbalanced clustering problem, where the clusters are called majority clusters that contain more samples than minority clusters. Compared with imbalanced classification, which applies data-level methods and algorithm-level methods to alleviate imbalance [10], [11], imbalanced clustering may face several challenges. First, the label information and the cluster size are unknown. Second, a common phenomenon is that the uniform effect exists in several clustering algorithms, such as $k$ -means algorithm [12], [13]. Concretely, $k$ -means clustering may cause a poor performance because it divides the data points belonging to majority clusters into the minority clusters, tending to partition different clusters to be relatively uniform sizes, although the cluster sizes of imbalanced data are different. In this case, it is worth investigating the imbalanced clustering, which has gained increasing attentions in data mining applications.

In the literature, to solve the imbalanced clustering problem, several methods have been proposed to avoid the uniform effect of $k$ -means. For instance, an imbalanced clustering algorithm which is helpful to prevent the uniform effect, is proposed to apply multicenters to represent each cluster instead of one single center [14]. Later on, the ensemble algorithm based on supervised learning is transferred to imbalanced clustering [15]. In a recent work [16], a self-adaptive multiprototype-based competitive learning which is designed for imbalanced clustering, utilizes multiple subclusters to represent each cluster by an automatic adjustment of the number of subclusters. Nevertheless, the existing imbalanced clustering methods lack a solid theoretical guarantee. Specifically, these imbalanced methods fail to give excess clustering risk analysis to understand the clustering performance further. In addition, there are no studies on the improvement of imbalanced clustering performance from the perspective of excess clustering risk. Therefore, imbalanced clustering in theory and method has yet to be well investigated in previous studies. To overcome the above drawbacks, our primary goal is to study the excess clustering risk of imbalanced clustering methods based on the framework of $k$ -means; meanwhile we hope to design a novel model that is in expectation closer to the given imbalanced data distribution by the theoretical analysis.

To achieve the above goal, we first propose a novel $k$ -Means algorithm for imbalanced clustering problem with Adaptive Cluster Weight (MACW). Concretely, the cluster weight indicates the different importance of the clusters. The proposed MACW can adaptively achieve better trade-offs between the size of majority clusters and minority clusters. To further understand MACW, we present an excess clustering risk bound of MACW based on the popular excess risk techniques. Note that this bound only depends on the maximum cluster weight across all clusters, the total number of samples, and the number of clusters. Thus, this bound cannot clearly indicate the influenced factors of clustering results for each cluster.

Based on the above analysis, we study the excess clustering risk of each cluster to obtain a uniform excess clustering risk bound. According to this bound, we propose a novel Imbalanced Clustering with Theoretical Learning Bounds (ICTLB), which allows us to combine the re-weighting strategy with $k$ -means in a more efficient way. Specifically, the design of cluster weight based on the excess clustering risk bound encourages the majority clusters to have smaller weights in the cost function. Meanwhile, an optimal trade-off between per-cluster weights of ICTLB method is given by optimizing the learning bound. In a word, although MACW method has good performance in most cases, we propose ICTLB method to improve the performance further, which is more suitable for dealing with imbalanced clustering problems.

The main contributions of this work are summarized below:

- We propose a novel method, i.e., MACW, combining an adaptive cluster weight with the objective function of $k \mathrm { . }$ - means. Its excess clustering risk bound is presented.   
Motivated by the theoretical results of MACW, we present an improved model called ICTLB by optimizing the uniform excess clustering risk bound.   
C We evaluate our method ICTLB with comprehensive experiments across various imbalanced clustering benchmarks. The proposed ICTLB outperforms previous methods and achieves good clustering performance on several real-world datasets.

The remainder of this article is organized as follows. We review the related works in Section II. Several novel results and a theoretical analysis are given in Section III. In Section IV, the effectiveness of the proposed method is shown by experiments. Section V concludes this article. Several related proofs are provided in Section VI.

# II. RELATED WORK

In this section, we briefly introduce the closely related works.

Many clustering methods have been presented to solve various real problems [17], [18], [19], [20], where $k$ -means algorithm is one of the popular clustering techniques because of its efficiency. In the $k$ -means clustering, it starts with the initial cluster centers and generates the optimal cluster centers by minimizing the sum of squared errors. Some variants of $k$ -means algorithm have been widely studied in [21], [22], [23]. People have investigated some factors that influence the performance of $k$ -means, such as initial centers [24], [25], noise in the data [26], and high dimensionality [27], [28]. In particular, imbalanced property of the data is another important factor to influence the clustering performance of $k$ -means. Specifically, some imbalanced clustering methods are proposed to avoid the uniform effect which appears in $k$ -means algorithm [14], [15], [16], [29]. In addition, a deep clustering method called Deep Embedded Clustering (DEC) is proposed to simultaneously improve feature representation and clustering assignment and discuss the performance on imbalanced data [30]. Based on the density distance, the density- $k$ -means $^ { + + }$ is proposed to deal with imbalanced data [31]. However, these imbalanced clustering methods cannot provide excess clustering risk bounds. Overall, imbalanced clustering has been widely noticed over recent years but has yet to be well investigated.

Next, we review works of theoretical analysis for $k$ -means algorithm. The theoretical studies on $k$ -means or its kernel version are proposed can be found in [32], [33], [34]. Specifically, if Gaussian projections are generated, an excess clustering risk bound of $k$ -means algorithm is given in [32]. Later on, the authors prove that sampling $\sqrt { n }$ Nyström landmarks can decrease the computational costs of kernel $k$ -means [33]. The excess clustering risk bounds of these two methods are linearly dependent on $k$ and thus do not match with the theoretical lower bound [35]. To overcome this issue, people propose an optimal excess clustering risk bound of kernel $k$ -means [34]. Other clustering methods based on theoretical properties are given in [36], [37], [38]. Nevertheless, the theoretical results of previous works cannot be adapted to analyze the proposed model in this article directly. Therefore, the theoretical analysis of data-dependent distribution is sparse for imbalanced clustering. Based on the theoretical perspective of excess risk, our proposed methods focus on improving the objective function of $k$ -means to solve imbalanced clustering problems, which has never been given before.

# III. MAIN APPROACH

In this section, we first introduce the problem setting in this article. Next, for the simple improvements of $k$ -means algorithm, a theoretical discussion of the improved cost function is provided. Further, we give the excess clustering risk bound of the proposed method and prove related results. After that, we propose a fine-grained excess clustering risk bound and give an improved model by optimizing this learning bound.

# A. Problem Setting

We define $\mathbb { Q }$ as a unknown distribution from the input space $\mathcal { X }$ . Let $S = \{ \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { n } \}$ be a set of $n$ samples generated i.i.d. from $\mathbb { Q }$ = 1 n. The emprirical distribution is denoted by $\begin{array} { r } { \mathbb { Q } _ { n } ( \mathbf { x } ) = \frac { 1 } { n } } \end{array}$ if $\mathbf { x } \in S$ , otherwise 0. Let $\mathcal { H }$ n( ) = ndenote a separable Hilbert space, and let $\langle \cdot , \cdot \rangle$ and $\| \cdot \|$ denote the inner product and the associated norm on $\mathcal { H }$ , respectively. Suppose the sample $\| \mathbf { x } \| \leq 1$ , for any $\mathbf { x } \in \mathcal { X }$ 1. The commonly used notations in this paper are listed in Table I.

We focus on imbalanced clustering problems solved by the framework of $k$ -means due to it in this work. To demonstrate our problem setting, we now introduce the uniform effect in $k$ -means algorithm [13].

The goal of $k$ -means clustering is to obtain a partition $\Theta =$ $( \theta _ { 1 } , \ldots , \theta _ { k } )$ of the given data into $k$ Θ =clusters, each characterized ( 1 k)by its cluster center $\mathbf { c } _ { r }$ . Here, we denote a set $\theta _ { r }$ associated with a cluster center $\mathbf { c } _ { r }$ as

$$
\theta _ { r } : = \left. i : r = \underset { j = 1 , \ldots , k } { \arg \operatorname* { m i n } } \left\| \mathbf { x } _ { i } - \mathbf { c } _ { j } \right\| ^ { 2 } \right. .
$$

TABLE I NOTATIONS   

<table><tr><td>Notations</td><td>Descriptions</td></tr><tr><td>Q Qn</td><td>The unknown distribution on X The empirical distribution on X</td></tr><tr><td>S</td><td>The set of n samples</td></tr><tr><td>n</td><td>The number of samples The size of i-th cluster</td></tr><tr><td>ni k</td><td>The number of classes</td></tr><tr><td>H</td><td>The Hilbert space</td></tr><tr><td>xi</td><td></td></tr><tr><td>C</td><td>The i-th sample</td></tr><tr><td>cr</td><td>The collection of k centers</td></tr><tr><td>θr</td><td>The r-th cluster center</td></tr><tr><td>Φ(C, Q)</td><td>The sample set θr associated with a cluster center cr</td></tr><tr><td></td><td>The expected squared norm criterion</td></tr><tr><td>Φ(C, Qn)</td><td></td></tr><tr><td>Φ*()</td><td>The empirical squared norm criterion</td></tr><tr><td>∆(Cn)</td><td>The optimal clustering risk The excess clustering risk</td></tr></table>

Define $\mathbf { C } = [ \mathbf { c } _ { 1 } , \hdots , \mathbf { c } _ { k } ] \in \mathcal { H } ^ { k }$ as a collection of $k$ centers. Ac-= [ 1 k]cording to the introduction of [13], it is shown that the objective function of $k$ -means is written as

$$
H _ { k } = \sum _ { r = 1 } ^ { k } \sum _ { i \in \theta _ { r } } \left\| \mathbf x _ { i } - \mathbf c _ { r } \right\| ^ { 2 } ,
$$

where c   ∈ r xir , for $1 \leq r \leq k$ , and $n _ { r }$ is the number of r = i θsamples in the set $\theta _ { r }$ 1. The aim of $k$ r-means is to minimize the robjective function (1). In the reference [13], when the number of clusters $k = 2$ , the minimization of $H _ { 2 }$ of (1) can be equivalent = 2 2to the maximization of the following function:

$$
H ^ { ( 2 ) } = 2 n _ { 1 } n _ { 2 } \left\| \mathbf { c } _ { 1 } - \mathbf { c } _ { 2 } \right\| ^ { 2 } .
$$

Since $H ^ { ( 2 ) } > 0$ , if we isolate the effect of $\| \mathbf { c } _ { 1 } - \mathbf { c } _ { 2 } \| ^ { 2 }$ with $\mathbf { c } _ { 1 } \neq$ $\mathbf { c } _ { 2 }$ 0, the maximization of $H ^ { ( 2 ) }$ 1 2 1 =is equivalent to the maximization 2of $n _ { 1 } n _ { 2 }$ , which results in $n _ { 1 } = n _ { 2 } = n / 2$ . This indicates the 1 2uniform effect of $k$ 1 = 2 = 2-means clustering. The uniform effect also happens with multiple clusters [13].

As discussed above, if we apply $k$ -means algorithm to divide imbalanced data directly, it is highly likely to obtain a poor clustering performance due to its uniform effect. Therefore, this motivates us to study how to design an imbalanced clustering method to alleviate uniform effect of $k$ -means and improve clustering results.

# B. Theoretical Motivations

Based on the analysis of the objective function for $k$ -means algorithm, we propose an adaptive cluster weight learning for the imbalanced clustering in this part.

According to the above descriptions in Section III-A, it is important but rarely considered about how to alleviate the uniform effect of $k$ -means algorithm for the imbalanced clustering problem. In order to overcome this problem, we first propose a novel strategy to add an adaptive cluster weight to the objective function (1)

$$
\hat { H } _ { k } = \sum _ { r = 1 } ^ { k } \sum _ { i \in \theta _ { r } } \| \mathbf { x } _ { i } - \mathbf { c } _ { r } \| ^ { 2 } \lambda _ { r } ^ { \alpha } ,
$$

where $\lambda _ { r }$ should satisfy: $\textstyle \sum _ { r = 1 } ^ { k } \lambda _ { r } = 1 , 0 < \lambda _ { r } < 1$ , and $\alpha$ is a r rconstant. The cluster weight $\lambda _ { r } ^ { \alpha }$ r = 1 0 r 1refers to different importance of the clusters in (2).

Next, we will give a detailed explanation of why such a weight design is reasonable. Meanwhile, we provide an intuitive inference about how to design an appropriate weight to adapt to an imbalanced clustering scenario. Specifically, we first denote

$$
P _ { k } = \sum _ { r = 1 } ^ { k } d \left( \theta _ { r } , \theta _ { r } \right) \lambda _ { r } ^ { \alpha } + 2 \sum _ { 1 \leq i < j \leq k } d \left( \theta _ { i } , \theta _ { j } \right) \lambda _ { i } ^ { \alpha } \lambda _ { j } ^ { \alpha } ,
$$

where $d ( \theta _ { p } , \theta _ { q } ) = \sum _ { i \in \theta _ { p } } \sum _ { j \in \theta _ { q } } \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| ^ { 2 }$ , and recall that $k$ ( p q) = i θ j θ iis the number of clusters. Obviously, $P _ { k }$ jis a constant for the given weights and data. Let $\begin{array} { r } { n = \sum _ { r = 1 } ^ { k } n _ { r } } \end{array}$ be the total number =of samples in the provided data.

To facilitate the discussions, we consider two clusters, i.e., $k = 2$ . Based on (3) with $k = 2$ , we have

$$
P _ { 2 } = d \left( \theta _ { 1 } , \theta _ { 1 } \right) \lambda _ { 1 } ^ { \alpha } + d \left( \theta _ { 2 } , \theta _ { 2 } \right) \lambda _ { 2 } ^ { \alpha } + 2 d \left( \theta _ { 1 } , \theta _ { 2 } \right) \lambda _ { 1 } ^ { \alpha } \lambda _ { 2 } ^ { \alpha } .
$$

It is obvious that P is a constant. Substituting c   ∈ r xir into (2), we can obtain

$$
\begin{array} { l } { { \hat { H } } _ { 2 } = \displaystyle \frac { 1 } { 2 n _ { 1 } } \sum _ { i , j \in \theta _ { 1 } } \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| ^ { 2 } { \boldsymbol { \lambda } } _ { 1 } ^ { \alpha } + \displaystyle \frac { 1 } { 2 n _ { 2 } } \sum _ { i , j \in \theta _ { 2 } } \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| ^ { 2 } { \boldsymbol { \lambda } } _ { 2 } ^ { \alpha } } \\ { \displaystyle } \\ { \displaystyle = \displaystyle \frac { 1 } { 2 } \sum _ { r = 1 } ^ { 2 } \frac { d \left( \theta _ { r } , \theta _ { r } \right) } { n _ { r } } { \boldsymbol { \lambda } } _ { r } ^ { \alpha } . } \end{array}
$$

Denote

$$
{ \hat { \cal H } } ^ { ( 2 ) } = - n _ { 1 } n _ { 2 } \lambda _ { 1 } ^ { \alpha } \lambda _ { 2 } ^ { \alpha } { \cal R } ,
$$

where

$$
R = \left[ { \frac { d \left( \theta _ { 1 } , \theta _ { 1 } \right) } { n _ { 1 } ^ { 2 } } } + { \frac { d \left( \theta _ { 2 } , \theta _ { 2 } \right) } { n _ { 2 } ^ { 2 } } } - 2 { \frac { d \left( \theta _ { 1 } , \theta _ { 2 } \right) } { n _ { 1 } n _ { 2 } } } \right] .
$$

Then we can re-formulate $\hat { H } _ { 2 }$ in (4), given by

$$
\hat { H } _ { 2 } = - \frac { \hat { H } ^ { ( 2 ) } } { 2 n } + \frac { P _ { 2 } } { 2 n } ,
$$

where $n = n _ { 1 } + n _ { 2 }$ . Furthermore, one can see that

$$
\frac { 2 d \left( \theta _ { 1 } , \theta _ { 2 } \right) } { n _ { 1 } n _ { 2 } } = \frac { d \left( \theta _ { 1 } , \theta _ { 1 } \right) } { n _ { 1 } ^ { 2 } } + \frac { d \left( \theta _ { 2 } , \theta _ { 2 } \right) } { n _ { 2 } ^ { 2 } } + 2 \left\| \mathbf { c } _ { 1 } - \mathbf { c } _ { 2 } \right\| ^ { 2 } .
$$

Thus, substituting (8) into (6), we have

$$
R = - 2 \left\| \mathbf { c } _ { 1 } - \mathbf { c } _ { 2 } \right\| ^ { 2 } .
$$

Combining (5) and (9), we finally obtain

$$
\hat { H } ^ { ( 2 ) } = 2 ( \lambda _ { 1 } ^ { \alpha } n _ { 1 } ) ( \lambda _ { 2 } ^ { \alpha } n _ { 2 } ) \left\| \mathbf { c } _ { 1 } - \mathbf { c } _ { 2 } \right\| ^ { 2 } .
$$

Here, since $P _ { 2 }$ is a constant, minimizing the objective function 2(7) is equivalent to the maximization of objective function (10). Note $\hat { H } ^ { ( 2 ) } > 0$ for $\mathbf { c } _ { 1 } \neq \mathbf { c } _ { 2 }$ . Thus if we isolate the effect of $\| \mathbf { c } _ { 1 } - \mathbf { c } _ { 2 } \| ^ { 2 }$ 0 1 = 2, the maximization of $\hat { H } ^ { ( 2 ) }$ implies the maximiza-1tion of $( \lambda _ { 1 } ^ { \alpha } n _ { 1 } ) ( \lambda _ { 2 } ^ { \alpha } n _ { 2 } )$ , which leads to $\lambda _ { 1 } ^ { \alpha } n _ { 1 } = \lambda _ { 2 } ^ { \alpha } n _ { 2 }$ . Here, if $\lambda _ { 1 } = \lambda _ { 2 } = 1 / 2$ ( 2 2), we have $n _ { 1 } = n _ { 2 }$ 1 1 = 2 2, which leads to the uniform 1 = 2effect of $k$ 1 2 1 = 2-means. Thus, without loss of generality, let the number of minority cluster and majority cluster be $n _ { 1 }$ and $n _ { 2 }$ , respectively. To obtain $n _ { 1 } < n _ { 2 }$ , we can make $\lambda _ { 1 } ^ { \alpha } > \lambda _ { 2 } ^ { \alpha }$ to guarantee $\lambda _ { 1 } ^ { \alpha } n _ { 1 } = \lambda _ { 2 } ^ { \alpha } n _ { 2 }$ . Based on a similar idea in [13], 1 1 = 2 2the clustering problem with multiple clusters also has the same property. Inspired by these observations, we can obtain that adding the adaptive cluster weight in the objective function of $k$ -means tends to make a larger cluster weight for the minority cluster and a smaller one for the majority cluster. This conclusion is similar to the idea of imbalanced classification, i.e., we need to focus more on minority classes because of less information about them. Thus, to some extent, this strategy can bring advantages to alleviate the uniform effect of $k$ -means for the imbalanced clustering.

In summary, since there are no accurate labels for samples in the imbalanced clustering, we do not know the importance of each cluster adding into the proposed objective function (2). In this case, the adaptive cluster weight learning is proposed. This method can adapt varied cluster sizes and alleviate the uniform effect of $k$ -means. However, this imbalanced clustering method lacks the excess risk analysis. More importantly, we hope to further improve imbalanced clustering performance by the investigation of theoretical analysis. Then, to design a more effective model with the statistical properties, we first study the excess clustering risk of this proposed method in Section III-C.

# C. Excess Clustering Risk Bound of MACW

To facilitate the theoretical analysis of the proposed method, we rewrite the above model, called $k$ -Means algorithm for imbalanced clustering problem with Adaptive Cluster Weight (MACW). To better understand the performance of MACW, we provide an excess clustering risk bound of it. This is a novel result because very few studies focus on the imbalanced clustering methods from the perspective of excess clustering risk.

For ease of theoretical analysis, we give an equivalent objective function of $k$ -means algorithm given in (2). To be specific, the empirical squared norm criterion of $k$ -means can also be described as

$$
\Phi ( \mathbf { C } , \mathbb { Q } _ { n } ) : = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } _ { r = 1 , \ldots , k } ( l \left\| \mathbf { x } _ { i } - \mathbf { c } _ { r } \right\| ^ { 2 } )
$$

over all possible choices of cluster centers $\mathbf { C } \in \mathcal { H } ^ { k }$ [32]. Based on the discussion in Problem (2), we can use a similar strategy to combine a cluster weight with Problem (11) by providing the following cost function:

$$
\Phi ( { \bf C } , \mathbb { Q } _ { n } ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } _ { r = 1 , \ldots , k } ( l \left\| { \bf x } _ { i } - { \bf c } _ { r } \right\| ^ { 2 } \lambda _ { r } ^ { \alpha } ) .
$$

Recall that $\lambda _ { r }$ should satisfy: $\begin{array} { r } { \sum _ { r = 1 } ^ { k } \lambda _ { r } = 1 , 0 < \lambda _ { r } < 1 } \end{array}$ , and $\alpha$ r r=1 r = 1 0 r 1is a constant. Here we should highlight one point, the cluster weight can be given from the training data in supervised learning with label information. However, we focus on solving the imbalanced clustering problem without label information. Therefore, we can use adaptive cluster weight learning in (12) to adapt varied cluster sizes and alleviate the uniform effect of $k$ -means algorithm.

Input: The dataset ${ \cal { S } } = \{ { \bf { x } } _ { i } \} _ { i = 1 } ^ { n }$ .   
= i i=1Initialize: Generate an initial center $\mathbf { C } ^ { 0 }$ . Give an initial $\lambda _ { r } = 1 / k \left( r = 1 , \ldots , k \right)$ .   
r = 1 ( = 1 )Output: The result label of $k$ clusters.   
1: for $i = 1 , \ldots , n$ do   
2: $\theta _ { r } = \left\{ i : r = \underset { j = 1 , \ldots , k } { \arg \operatorname* { m i n } } \left( \| \mathbf { x } _ { i } - \mathbf { c } _ { j } \| ^ { 2 } \lambda _ { j } ^ { \alpha } \right) \right\} .$   
3: end for   
4: Update $\lambda _ { r }$ $( r = 1 , \ldots k )$ by (13).   
5: r ( = 1 Update the center by $\textstyle \sum _ { i \in \theta _ { r } } \mathbf { x } _ { i } / | \theta _ { r } |$ .   
6: i θ i r Repeat 1-5 until the condition is met.

Similar to Problem (2), the cost function (12) can be expressed as

$$
\Phi ( \mathbf { C } , \mathbb { Q } _ { n } ) : = \frac { 1 } { n } \operatorname* { m i n } _ { \mathbf { C } \in \mathcal { H } ^ { k } } \sum _ { r = 1 } ^ { k } \sum _ { i \in \theta _ { r } } \left( l \left\| \mathbf { x } _ { i } - \frac { 1 } { \left| \theta _ { r } \right| } \sum _ { s \in \theta _ { r } } \mathbf { x } _ { s } \right\| ^ { 2 } \lambda _ { r } ^ { \alpha } \right) ,
$$

where $\left| \theta _ { r } \right|$ denotes the size of the set $\theta _ { r }$ . In this case, we redefine $\theta _ { r }$ rwith a center $\mathbf { c } _ { r }$ rin Problem (12) as

$$
\theta _ { r } : = \left. i : r = \underset { j = 1 , \ldots , k } { \arg \operatorname* { m i n } } \left( \| \mathbf { x } _ { i } - \mathbf { c } _ { j } \| ^ { 2 } \lambda _ { j } ^ { \alpha } \right) \right. .
$$

In Problem (12), the value $\lambda _ { r }$ ${ \bf \Delta } ^ { r } = 1 , \ldots , k$ can be computed by

$$
\lambda _ { r } = \frac { ( \sum _ { \mathbf { x } _ { i } \in \theta _ { r } } l \left\| \mathbf { x } _ { i } - \mathbf { c } _ { r } \right\| ^ { 2 } ) ^ { \frac { 1 } { 1 - \alpha } } } { \sum _ { r = 1 } ^ { k } ( \sum _ { \mathbf { x } _ { i } \in \theta _ { r } } l \left\| \mathbf { x } _ { i } - \mathbf { c } _ { r } \right\| ^ { 2 } ) ^ { \frac { 1 } { 1 - \alpha } } } .
$$

The detailed optimization process of the cost function (12) (i.e., MACW) is shown in Algorithm 1.

Next, we provide the excess clustering risk bound of the proposed MACW. Define the emprirical risk minimizer as

$$
\mathbf { C } _ { n } : = \arg \operatorname* { m i n } _ { \mathbf { C } \in \mathcal { H } ^ { k } } \Phi ( \mathbf { C } , \mathbb { Q } _ { n } ) .
$$

The performance of a clustering method, given via the collection $\mathbf { C } = [ \mathbf { c } _ { 1 } , \dots , \mathbf { c } _ { k } ] \in \mathcal { H } ^ { k }$ of cluster centers, is usually measured = [ 1 k]via the expected squared norm criterion or expected clustering risk

$$
\Phi ( \mathbf { C } , \mathbb { Q } ) : = \int \operatorname* { m i n } _ { r = 1 , \ldots , k } l \left( \left\| \mathbf { x } - \mathbf { c } _ { r } \right\| ^ { 2 } \lambda _ { r } ^ { \alpha } \right) d \mathbb { Q } ( \mathbf { x } ) .
$$

For a $\mathbf { C } \in \mathcal { H } ^ { k }$ , we denote

$$
g _ { \mathbf { C } } ( \mathbf { x } ) = ( g _ { \mathbf { c } _ { 1 } } ( \mathbf { x } ) , \ldots , g _ { \mathbf { c } _ { k } } ( \mathbf { x } ) )
$$

as a $k$ -valued function of the collection $\mathbf { C }$ , where $g _ { \mathbf { c } _ { r } } ( \mathbf { x } ) =$ $l \| \mathbf { x } - \mathbf { c } _ { r } \| ^ { 2 } \lambda _ { r } ^ { \alpha }$ with $r = 1 , \ldots k$ . Define $\phi : \mathbb { R } ^ { k }  \mathbb { R }$ ( ) =as a minr rimum function. Let $\phi ( g _ { \mathbf { C } } ( \mathbf { x } ) ) = \operatorname* { m i n } ( g _ { \mathbf { c } _ { 1 } } , \dots , g _ { \mathbf { c } _ { k } } )$ . Thus we ( ( )) = min( )can rewrite the expected and empirical clustering risk as

$$
\Phi ( \mathbf { C } , \mathbb { Q } ) : = \int \phi ( g _ { \mathbf { C } } ( \mathbf { x } ) ) d \mathbb { Q } ( \mathbf { x } )
$$

and

$$
\Phi ( \mathbf { C } , \mathbb { Q } _ { n } ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \phi ( g _ { \mathbf { C } } ( \mathbf { x } _ { i } ) ) .
$$

Then, the optimal clustering risk is defined as

$$
\Phi ^ { * } ( \mathbb { Q } ) : = \operatorname* { i n f } _ { \mathbf { C } \in \mathcal { H } ^ { k } } \Phi ( \mathbf { C } , \mathbb { Q } ) .
$$

In this article, we focus on bounding the excess clustering risk of the proposed method defined by [32]

$$
\Delta ( \mathbf { C } _ { n } ) : = \mathbb { E } _ { S \sim \mathbb { Q } } [ \Phi ( \mathbf { C } _ { n } , \mathbb { Q } ) ] - \Phi ^ { * } ( \mathbb { Q } ) .
$$

Here, the subscript $\boldsymbol { S }$ will be omitted if the input dataset $\boldsymbol { s }$ is not ambiguous. Next, we derive the excess clustering risk bound for the MACW method in Theorem 1.

Theorem $^ { l }$ : If any $\mathbf { x } \in \mathcal { X }$ , $\| \mathbf { x } \| \leq 1$ , and there is a constant $T$ , for $\forall \delta \in ( 0 , 1 )$ 1, with probability at least $1 - \delta$ , we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Phi \left( \mathbf { C } _ { n } , \mathbb { Q } \right) \right] - \Phi ^ { * } ( \mathbb { Q } ) } \\ & { \quad \leq 4 T \tilde { \lambda } \Bigg [ 2 + \left( 1 + o ( 1 ) \right) \sqrt { \frac { 2 } { \pi } } \Bigg ] \sqrt { \frac { k } { n } } \log ^ { 2 } ( 2 \sqrt { 2 \hat { \lambda } n } ) } \\ & { \quad \quad + 4 \sqrt { \frac { 2 } { n } \log \left( \frac { 1 } { \delta } \right) } , } \end{array}
$$

where $\hat { \lambda } = \mathrm { m a x } _ { r = 1 , \ldots , k } \lambda _ { r } ^ { \frac { \alpha } { 2 } }$ and $\tilde { \lambda } = \operatorname* { m a x } _ { r = 1 , \dots , k } \lambda _ { r } ^ { \alpha }$

= maxr=1,...,k r = maxr=1,...,k rFrom Theorem 1, the excess clustering risk bound of MACW is related to the maximum of $\lambda _ { r } ^ { \frac { \alpha } { 2 } }$ and $\lambda _ { r } ^ { \alpha }$ , which are defined for r rthe analysis of this bound. More importantly, Theorem 1 is a novel result of MACW for imbalanced clustering. The further analysis can be found in Section III-D.

# D. Analysis

We prove Theorem 1 based on the three proposed lemmas in this part.

We first give some notations used in the proof of Theorem 1. Define a family of $k$ -valued functions $\mathcal { G } _ { \mathbf { C } }$ as

$$
\mathcal G _ { \mathbf C } = \left\{ g _ { \mathbf C } = ( g _ { \mathbf c _ { 1 } } , \dots , g _ { \mathbf c _ { k } } ) : \mathbf C \in \mathcal H ^ { k } \right\} .
$$

Recall that $\phi : \mathbb { R } ^ { k }  \mathbb { R }$ is a minimum function, that is, for any $\beta = ( \beta _ { 1 } , \ldots , \beta _ { k } ) \in \mathbb { R } ^ { k }$ ,

$$
\phi ( \beta ) = \operatorname* { m i n } ( \beta _ { 1 } , . . . , \beta _ { k } ) .
$$

Let $\mathcal { F } _ { \mathbf { C } }$ denote a “minimum” family of the functions $\mathcal { G } _ { \mathbf { C } }$

$$
\begin{array} { r } { \mathcal { F } _ { \mathbf { C } } = \left\{ f _ { \mathbf { C } } = \phi \circ g _ { \mathbf { C } } \ | \ g _ { \mathbf { C } } \in \mathcal { G } _ { \mathbf { C } } , f _ { \mathbf { C } } ( \mathbf { x } ) = \phi \left( g _ { \mathbf { C } } ( \mathbf { x } ) \right) \right\} . } \end{array}
$$

To prove Theorem 1, we give a definition of Clustering Rademacher Complexity (abbreviated to CRC) [34] and prove three lemmas in the following.

Definition $^ { l }$ . (CRC): Let $S = \{ \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { n } \} \in { \mathcal { X } } ^ { n }$ be $n$ samples from $\mathcal { X }$ . Let $\mathcal { F } _ { \mathbf { C } }$ = 1 ndenote a family of functions given by in (19). Then, the definition of the clustering empirical Rademacher complexity of $\mathcal { F } _ { \mathbf { C } }$ w.r.t. $\boldsymbol { s }$ is denoted as

$$
\Re _ { n } \left( \mathcal { F } _ { \mathbf { C } } \right) = \mathbb { E } _ { \pmb { \sigma } } \left[ \operatorname* { s u p } _ { f _ { \mathbf { C } } \in \mathcal { F } _ { \mathbf { C } } } \left| \sum _ { i = 1 } ^ { n } \sigma _ { i } f _ { \mathbf { C } } \left( \mathbf { x } _ { i } \right) \right| \right] ,
$$

where $\sigma _ { 1 } , \ldots , \sigma _ { n }$ are independent random variables with equal 1 nprobability of generating values $+ 1$ or $- 1$ . The expectation of $\Re _ { n } ( \mathcal { F } _ { \mathbf { C } } )$ is defined by $\Re ( \mathcal { F } _ { \mathbf { C } } ) = \mathbb { E } [ \Re _ { n } ( \mathcal { F } _ { \mathbf { C } } ) ]$ .

n( ) ( ) = [ n( )]Inspired by the upper bound of Rademacher complexity [39], we present a new bound of CRC for the proposed MACW given by Lemma 1.

Lemma $^ { l }$ : If $\| \mathbf { x } \| \leq 1$ , for any $\mathbf { x } \in \mathcal { X }$ , then, for any ${ \boldsymbol { S } } =$ $\left\{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \right\} \in { \mathcal { X } } ^ { n }$ 1, there exists a constant $T > 0$ such that

$$
\Re _ { n } \left( \mathcal { F } _ { { \bf C } } \right) \le T \sqrt { k } \operatorname* { m a x } _ { r } \hat { \Re } _ { n } \left( \mathcal { G } _ { { \bf C } _ { r } } \right) \log ^ { 2 } ( 2 \sqrt { 2 \hat { \lambda } n } ) ,
$$

where $\mathcal { F } _ { \mathbf { C } }$ is defined in (19), $\mathcal G _ { \mathbf C _ { r } }$ is a family of the output coordinate $r$ of $\mathcal { G } _ { \mathbf { C } }$ , $\begin{array} { r } { \hat { \mathfrak { R } } _ { n } ( \mathcal G _ { \mathbf { C } _ { r } } ) = \operatorname* { s u p } _ { \mathcal { S } \in \mathcal { X } ^ { n } } \mathfrak { R } _ { n } ( \mathcal G _ { \mathbf { C } _ { r } } ) } \end{array}$ , and $\hat { \lambda } =$ λ 2 .

axr=1,...,k rWe provide the proof of Lemma 1 in Section VI-A. To prove Theorem 1, we further propose the following two lemmas.

Lemma 2: If $\| \mathbf { x } \| \leq 1$ , for any $\mathbf { x } \in \mathcal { X }$ , then, for any ${ \boldsymbol { S } } =$ $\left\{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \right\} \in { \mathcal { X } } ^ { n }$ and $\mathbf { C } \in \mathcal { H } ^ { k }$ , we have

$$
\operatorname* { m a x } _ { r } \hat { \Re } _ { n } ( \mathcal G _ { { \bf C } _ { r } } ) \le \tilde { \lambda } \left[ 2 + ( 1 + o ( 1 ) ) \sqrt { \frac { 2 } { \pi } } \right] \sqrt { n } ,
$$

where $\tilde { \lambda } = \operatorname* { m a x } _ { r = 1 , \dots , k } \lambda _ { r } ^ { \alpha }$ . Recall that the $\hat { \Re } _ { n } ( \mathcal G _ { \mathbf { C } _ { r } } ) =$ $\operatorname* { s u p } _ { S \in \mathcal { X } ^ { n } } \Re _ { n } ( \mathcal { G } _ { \mathbf { C } _ { r } } )$ =.

p n( )The proof of Lemma 2 is provided in Section VI-B.

Lemma 3: For $\forall \delta \in ( 0 , 1 )$ , with a probability at least $1 - \delta$ , there exists a constant $T > 0$ , we have

$$
\begin{array} { r l r } {  { \Re ( \mathcal { F } _ { \mathbf { C } } ) \le T \tilde { \boldsymbol { \lambda } } [ 2 + ( 1 + o ( 1 ) ) \sqrt { \displaystyle \frac { 2 } { \pi } } ] \sqrt { k n } \log ^ { 2 } ( 2 \sqrt { 2 \hat { \lambda } n } ) } } \\ & { } & { \qquad + \sqrt { 2 n \log ( \displaystyle \frac { 1 } { \delta } ) } . } \end{array}
$$

Based on Lemma 2, we prove Lemma 3 in Section VI-C. Next, we provide the proof of Theorem 1 by the three lemmas.

Proof: First, we can give the following equation

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Phi \left( \mathbf { C } _ { n } , \mathbb { Q } \right) \right] - \Phi ^ { * } ( \mathbb { Q } ) } \\ & { = \mathbb { E } \left[ \Phi \left( \mathbf { C } _ { n } , \mathbb { Q } \right) - \Phi \left( \mathbf { C } _ { n } , \mathbb { Q } _ { n } \right) \right] + \mathbb { E } \left[ \Phi \left( \mathbf { C } _ { n } , \mathbb { Q } _ { n } \right) \right] - \Phi ^ { * } ( \mathbb { Q } ) . } \end{array}
$$

Since $\mathbf { C } _ { n }$ is the optimal with respect to $\Phi ( \mathbf { C } , \mathbb { Q } _ { n } )$ in (14), we ncan obtain

$$
\Phi \left( \mathbf { C } _ { n } , \mathbb { Q } _ { n } \right) \leq \Phi \left( \mathbf { C } ^ { * } , \mathbb { Q } _ { n } \right) .
$$

Based on (21) and Inequality (22), we have

$$
\begin{array} { r l r } {  { \mathbb { E } [ \Phi ( \mathbf { C } _ { n } , \mathbb { Q } ) ] - \Phi ^ { * } ( \mathbb { Q } ) } } \\ & { } & { \leq \mathbb { E } [ \Phi ( \mathbf { C } _ { n } , \mathbb { Q } ) - \Phi ( \mathbf { C } _ { n } , \mathbb { Q } _ { n } ) ] + \mathbb { E } [ \Phi ( \mathbf { C } ^ { * } , \mathbb { Q } _ { n } ) ] - \Phi ^ { * } ( \mathbb { Q } ) } \\ & { } & { \leq \mathbb { E } \operatorname* { s u p } _ { \mathbf { C } \in \mathcal { H } ^ { k } } [ \Phi ( \mathbf { C } , \mathbb { Q } ) - \Phi ( \mathbf { C } , \mathbb { Q } _ { n } ) ] } \\ & { } & { + \operatorname* { s u p } _ { \mathbf { C } \in \mathcal { H } ^ { k } } \mathbb { E } [ \Phi ( \mathbf { C } , \mathbb { Q } _ { n } ) - \Phi ( \mathbf { C } , \mathbb { Q } ) ] . } \end{array}
$$

According to Jensen’s inequality, it is easy to

$$
\begin{array} { r l } & { \underset { { \bf C } \in \mathcal { H } ^ { k } } { \operatorname* { s u p } } \mathbb { E } \left[ \Phi \left( { \bf C } , \mathbb { Q } _ { n } \right) - \Phi ( { \bf C } , \mathbb { Q } ) \right] } \\ & { \leq \mathbb { E } \underset { { \bf C } \in \mathcal { H } ^ { k } } { \operatorname* { s u p } } \left[ \Phi \left( { \bf C } , \mathbb { Q } _ { n } \right) - \Phi ( { \bf C } , \mathbb { Q } ) \right] ) . } \end{array}
$$

Obviously, by Inequality (23) and Inequality (24), we can obtain

$$
\mathbb { E } \left[ \Phi \left( \mathbf { C } _ { n } , \mathbb { Q } \right) \right] - \Phi ^ { * } ( \mathbb { Q } ) \leq 2 \mathbb { E } \operatorname* { s u p } _ { \mathbf { C } \in \mathcal { H } ^ { k } } \left[ \Phi \left( \mathbf { C } , \mathbb { Q } _ { n } \right) - \Phi ( \mathbf { C } , \mathbb { Q } ) \right] .
$$

The same conclusion of Inequality (25) can be found in [34]. Let $\{ \mathbf { x } _ { 1 } ^ { \prime } , \ldots , \mathbf { x } _ { n } ^ { \prime } \}$ and $\left\{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { n } \right\}$ be different samples set from 1 n 1 nthe same distribution. By the similar idea in [40], we can estimate the following upper bound by

$$
\begin{array} { r l } { { } } & { { \mathbb { E } } \underset { { \mathbf { C } } \in \mathcal { H } ^ { k } } { \operatorname* { s u p } } \left[ \Phi \left( { \mathbf { C } } , \mathbb { Q } _ { n } \right) - \Phi ( { \mathbf { C } } , \mathbb { Q } ) \right] } \\ { ~ } & { { } = { \mathbb { E } } \underset { f _ { { \mathbf { C } } } \in \mathcal { F } _ { { \mathbf { C } } } } { \operatorname* { s u p } } { \mathbb { E } } \left[ \Phi \left( { \mathbf { C } } , \mathbb { Q } _ { n } \right) - \cfrac { 1 } { n } \underset { i = 1 } { \overset { n } { \sum } } f _ { { \mathbf { C } } } ( { \mathbf { x } } ^ { \prime } ) \right] } \\ { ~ } & { { } \leq { \mathbb { E } } \underset { f _ { { \mathbf { C } } } \in \mathcal { F } _ { { \mathbf { C } } } } { \operatorname* { s u p } } \left[ \Phi \left( { \mathbf { C } } , \mathbb { Q } _ { n } \right) - \cfrac { 1 } { n } \underset { i = 1 } { \overset { n } { \sum } } f _ { { \mathbf { C } } } ( { \mathbf { x } } ^ { \prime } ) \right] } \end{array}
$$

by Jensen s inequality

$$
\begin{array} { r l } & { = \mathbb { E } \displaystyle \operatorname* { s u p } _ { f \in \mathcal { S } \subset \mathbb { C } _ { C } } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ f _ { \mathbf { C } } ( \mathbf { x } ) - f _ { \mathbf { C } } \left( \mathbf { x } ^ { \prime } \right) \right] \right] } \\ & { = \mathbb { E } \displaystyle \operatorname* { s u p } _ { f \in \mathcal { F } \mathbb { C } _ { C } } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \left[ f _ { \mathbf { C } } ( \mathbf { x } ) - f _ { \mathbf { C } } \left( \mathbf { x } ^ { \prime } \right) \right] \right] } \\ & { \leq \mathbb { E } \displaystyle \operatorname* { s u p } _ { f \in \mathcal { S } \subset \mathcal { F } _ { C } } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } f _ { \mathbf { C } } ( \mathbf { x } ) \right] + \mathbb { E } \displaystyle \operatorname* { s u p } _ { f \in \mathcal { S } \mathbb { C } } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } f _ { \mathbf { C } } \left( \mathbf { x } ^ { \prime } \right) \right] } \\ & { = 2 \mathbb { E } \displaystyle \operatorname* { s u p } _ { f \in \mathcal { F } \mathbb { C } _ { C } } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } f _ { \mathbf { C } } ( \mathbf { x } ) \right] = \frac { 2 } { n } \Re { ( \mathcal { F } _ { \mathbf { C } } ) } . } \end{array}
$$

Then, there exists an equivalent form given by

$$
\mathbb { E } \operatorname* { s u p } _ { { \mathbf { C } } \in { \mathcal { H } } ^ { k } } \left[ \Phi \left( { \mathbf { C } } , \mathbb { Q } _ { n } \right) - \Phi ( { \mathbf { C } } , \mathbb { Q } ) \right] \leq \frac { 2 } { n } \Re \left( { \mathcal { F } } _ { { \mathbf { C } } } \right) .
$$

Hence, based on Inequalities (25) and (26), we obtain

$$
\mathbb { E } \left[ \Phi \left( \mathbf { C } _ { n } , \mathbb { Q } \right) \right] - \Phi ^ { * } ( \mathbb { Q } ) \leq \frac { 4 } { n } \Re \left( \mathcal { F } _ { \mathbf { C } } \right) .
$$

According to Inequality (27) and Lemma 3, we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \Phi \left( \mathbf { C } _ { n } , \mathbb { Q } \right) \right] - \Phi ^ { * } ( \mathbb { Q } ) } \\ & { \quad \le 4 T \tilde { \lambda } \Bigg [ 2 + ( 1 + o ( 1 ) ) \sqrt { \frac { 2 } { \pi } } \Bigg ] \sqrt { \frac { k } { n } } \log ^ { 2 } ( 2 \sqrt { 2 \hat { \lambda } n } ) } \\ & { \quad + 4 \sqrt { \frac { 2 } { n } \log \left( \frac { 1 } { \delta } \right) } . } \end{array}
$$

This proves the statement in Theorem 1.

# E. Further Results

In this part, we first give a fine-grained excess clustering risk bound of MACW. Next, we propose a novel Imbalanced Clustering with Theoretical Learning Bounds (ICTLB) by optimizing this bound.

1) Fine-Grained Excess Clustering Risk Bound: The excess clustering risk bound of Theorem 1 is a novel and typical result based on the popular excess clustering risk techniques, when the training distribution is the same as the test distribution. Note that the bound in Theorem 1 is oblivious to the cluster distribution, and only involves the maximum cluster weight across all clusters, the number of clusters, and the total number of samples. However, this bound cannot clearly indicate the influential critical factors of the clustering performance of each cluster for imbalanced clustering. This conclusion inspires us to further investigate the excess clustering risk bound given in Theorem 1. Inspired by the similar idea in imbalanced classification [41], we aim to extend the excess clustering risk bound to the setting with balanced test distribution by considering the cluster weight of each cluster. As we will see, the more fine-grained bound below helps us design a better cost function customized to the imbalanced clustering. In the following, the informal and simplified version of Theorem 1 is proposed to bound the expected excess clustering risk of the cost function (12) in Theorem 2. To prove Theorem 2, we first give some definitions. Let

$$
\Phi _ { r } ( \mathbf { C } , \mathbb { Q } _ { n _ { r } } ) = \frac { 1 } { | \theta _ { r } | } \sum _ { i \in \theta _ { r } } \operatorname* { m i n } _ { r = 1 , \ldots , k } ( \left\| \mathbf { x } _ { i } - \mathbf { c } _ { r } \right\| ^ { 2 } \lambda _ { r } ^ { \alpha } )
$$

denote the empirical squared loss on samples from one cluster $r$ , where $\left| \theta _ { r } \right|$ is the size of the set $\theta _ { r }$ . According to the definition rof CRC provided in (20), let $\hat { \Re } ^ { ( r ) } ( \mathcal { F } _ { \bf C } )$ denote the empirical (Rademacher complexity of its cluster $r$

$$
\hat { \mathfrak { R } } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) = \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { \mathbf { c } \in \mathcal { H } _ { k } } \sum _ { i \in \theta _ { r } } \sigma _ { i } \operatorname* { m i n } _ { r = 1 , \ldots , k } ( \left\| \mathbf { x } _ { i } - \mathbf { c } _ { r } \right\| ^ { 2 } \lambda _ { r } ^ { \alpha } ) \right] ,
$$

where $\{ \sigma _ { i } \} _ { i = 1 } ^ { | \theta _ { r } | }$ are independent variables of $\{ - 1 , + 1 \}$ . The i i=expectation of $\hat { \Re } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } )$ is $\Re ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) = \mathbb { E } [ \hat { \Re } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) ]$ 1. We will give more details in the proof.

Theorem 2 (Informal and simplified version of Theorem $I$ ): If $\| \mathbf { x } \| \leq 1$ , for any $\mathbf { x } \in \mathcal { X }$ , then for any data sequence ${ \boldsymbol { S } } \in { \mathcal { X } }$ , 1there exists a constant $P$ such that

$$
\mathbb { E } \left[ \Phi \left( \mathbf { C } _ { n } , \mathbb { Q } \right) \right] - \Phi ^ { * } ( \mathbb { Q } ) \leq \sum _ { r = 1 } ^ { k } \frac { 4 \lambda _ { r } ^ { \alpha } P } { \sqrt { \left| \theta _ { r } \right| } } + \frac { 4 } { k } \sum _ { r = 1 } ^ { k } \sqrt { \frac { 2 \log \left( \frac { k } { \delta } \right) } { \left| \theta _ { r } \right| } } .
$$

Proof: We will prove the excess clustering risk separately for each cluster $r$ and then union bound over all clusters. The set $\theta _ { r }$ with i.i.d. samples are generated from the conditional distribution $\mathbb { Q } _ { r }$ . By (16), we let $\Delta _ { r } ( \mathbf { C } _ { n } )$ denote the expected excess clustering risk over all $\mathbf { C }$ drawn from $\mathbb { Q } _ { r }$ . From [40], we use the expected excess clustering risk bound given in Inequality (27) and Inequality (46) to get with probability $1 - \delta / k$ , for all possible values of $\lambda _ { r } ^ { \alpha } > 0$ and $\mathbf { C } \in \mathcal { H } ^ { k }$ ,

$$
\Delta _ { r } ( \mathbf { C } _ { n } ) \leq \frac { 4 } { \left| \theta _ { r } \right| } \Re ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) \leq \frac { 4 } { \left| \theta _ { r } \right| } \hat { \Re } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) + 4 \sqrt { \frac { 2 \log \left( \frac { k } { \delta } \right) } { \left| \theta _ { r } \right| } } ,
$$

In this case, since we have the setting with balanced test distribution from $\mathbb { Q }$ , we can compute a uniform excess clustering risk bound by $\begin{array} { r } { \Delta ( \mathbf { \bar { C } } _ { n } ) = \frac { 1 } { k } \sum _ { r = 1 } ^ { \tilde { k _ { \ r } } } \Delta _ { r } ( \mathbf { C } _ { n } ) } \end{array}$ . Then, we could union

excess clustering risk bound over all clusters and average (31) to get the following result:

$$
\Delta ( \mathbf { C } _ { n } ) \leq \frac { 1 } { k } \sum _ { r = 1 } ^ { k } \left( \frac { 4 } { \left| \theta _ { r } \right| } \hat { \Re } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) + 4 \sqrt { \frac { 2 \log \left( \frac { k } { \delta } \right) } { \left| \theta _ { r } \right| } } \right)
$$

In order to further prove Theorem 2, we mainly prove the following Lemma 4.

Lemma $^ { 4 }$ : If $\| \mathbf { x } \| \leq 1$ , for any $\mathbf { x } \in \mathcal { X }$ , then for any data sequence $S \in { \mathcal { X } }$ 1, there exists a constant $P$ such that the following statement holds.

$$
\frac { 1 } { k } \sum _ { r = 1 } ^ { k } \frac { 1 } { | \theta _ { r } | } \hat { \mathcal { R } } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) \leq \sum _ { r = 1 } ^ { k } \frac { \lambda _ { r } ^ { \alpha } P } { \sqrt { | \theta _ { r } | } } .
$$

where the definition of $\hat { \mathfrak { R } } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } )$ can be found in (29).

( )The proof of Lemma 4 is in Section VI-D. Based on Inequality (32) and Inequality (33), we have completed the proof of Theorem 2.

2) ICTLB Method: We design an improved model called ICTLB by optimizing the fine-grained excess clustering risk bound given in Theorem 2 in this part.

Without loss of generality, if $\lambda _ { 1 } = \cdot \cdot \cdot = \lambda _ { k }$ , the MACW method is equivalent to $k$ 1 = = k-means algorithm, which has the uniform effect. This algorithm might divide a part of the samples of majority clusters into minority clusters. From the perspective of excess clustering risk, this might lead to a poor generalization performance of majority clusters in the imbalanced clustering. Therefore, the excess clustering risk bound in (30) for each cluster suggests that if we wish to improve the excess clustering risk bound of majority clusters (those with small $| \theta _ { j } |$ ’s compared jwith the real number due to the uniform effect of $k$ -means), we should aim to enforce smaller cluster weights $\lambda _ { j } ^ { \alpha }$ for them. jHowever, enforcing smaller weights for majority clusters may hurt the clustering performance of the minority clusters. What is the optimal trade-off between the weights of different clusters? It may be difficult to answer the general case, but the optimal trade-off for the binary clustering problem can be obtained below.

For the clusters $k = 2$ , our goal is to optimize the learning = 2bound presented in (30), which is simplified to (by removing the constant of numerator $P$ and the low order term $\sqrt { \frac { 2 \log ( \frac { k } { \delta } ) } { | \theta _ { r } | } } )$

$$
\lambda _ { 1 } ^ { \alpha } \sqrt { \frac { 1 } { \vert \theta _ { 1 } \vert } } + \lambda _ { 2 } ^ { \alpha } \sqrt { \frac { 1 } { \vert \theta _ { 2 } \vert } } .
$$

Obviously, it is difficult to understand the optimal weights of (34) in clustering problems. Nevertheless, we could figure out the relative scales between $\lambda _ { 1 } ^ { \alpha }$ and $\lambda _ { 2 } ^ { \alpha }$ with $\lambda _ { 1 } + \lambda _ { 2 } = 1$ . Assume $\lambda _ { 1 } , \lambda _ { 2 } > 0$ 1 2minimize (34), we find that any $\lambda _ { 1 } ^ { \prime } = \lambda _ { 1 } - \delta$ and $\lambda _ { 2 } ^ { \prime } = \lambda _ { 2 } + \delta$ 0(with $\delta \in ( - \lambda _ { 2 } , \lambda _ { 1 } ) )$ 1 = 1 could be realized with a 2 = 2 +shifted bias term. Thus, if $\lambda _ { 1 }$ 2and $\lambda _ { 2 }$ )are optimal, they need to satisfy

$$
\lambda _ { 1 } ^ { \alpha } \sqrt { \frac { 1 } { \left| \theta _ { 1 } \right| } } + \lambda _ { 2 } ^ { \alpha } \sqrt { \frac { 1 } { \left| \theta _ { 2 } \right| } } \geq ( \lambda _ { 1 } - \delta ) ^ { \alpha } \sqrt { \frac { 1 } { \left| \theta _ { 1 } \right| } } + ( \lambda _ { 2 } + \delta ) ^ { \alpha } \sqrt { \frac { 1 } { \left| \theta _ { 2 } \right| } } .
$$

The above inequality implies that $\lambda _ { 1 } ^ { \alpha } = W | \theta _ { 1 } | ^ { \frac { \alpha } { 2 ( 1 - \alpha ) } }$ and $\lambda _ { 2 } ^ { \alpha } =$ $W | \theta _ { 2 } | ^ { \frac { \alpha } { 2 ( 1 - \alpha ) } }$ for some constant $W$ 1 =. Here $\alpha \neq 1$ 2 =. Please see a 2detailed derivation in Section VI-E.

Inspired by the trade-off between the cluster weight for the two clusters, we extend this to multi-clusters clustering so that the improved cost function of MACM (i.e., ICTLB) is proposed by

$$
\Phi ( { \bf C } , \mathbb { Q } _ { n } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } _ { r = 1 , \ldots , k } ( l \left\| { \bf x } _ { i } - { \bf c } _ { r } \right\| ^ { 2 } \lambda _ { r } ^ { \alpha } ) ,
$$

where

$$
\lambda _ { r } ^ { \alpha } = W | \theta _ { r } | ^ { \frac { \alpha } { 2 ( 1 - \alpha ) } } , r = 1 , \dots , k
$$

and $W$ is a hyper-parameter to be tuned. The optimization of Problem (35) is similar to Algorithm 1, and the difference is that we update $\lambda _ { r }$ by (36). Without loss of generality, here we let $\alpha = - 1 / 2$ r. Overall, both MACM and ICTLB methods can be = 1 2applied to solve the imbalanced clustering problem. The main difference of these two methods is that the cluster weight is calculated by diffrent ways. Concretely, MACW is proposed to combine an adaptive cluster weight with the objective function of $k$ -means algorithm, whose cluster weights are computed by (13). Compared with the MACM method, the ICTLB method can refine weight design by optimizing the excess clustering risk bound. The cluster weights of ICTLB are computed by (36).

# IV. EXPERIMENTS

In this section, we provide experiments to evaluate the proposed methods on several real imbalanced datasets. The experimental results could demonstrate the effectiveness of our proposed methods in designing cluster weights.

# A. Dataset Description

In this article, we consider 14 imbalanced datasets collected from two Web sites, i.e., KEEL1 and $\mathrm { U C I } ^ { 2 }$ . We use Imbalance Ratio (abbreviated to IR) to denote the ratio between sample sizes of the most frequent and least frequent cluster, i.e., $\mathrm { I R } =$ $\begin{array} { r } { \frac { \operatorname* { m a x } _ { i \in [ k ] } \{ n _ { i } \} } { \operatorname* { m i n } _ { i \in [ k ] } \{ n _ { i } \} } } \end{array}$ , where $[ k ] = \{ 1 , \dots , k \}$ and $n _ { i }$ is the size of $i$ =th mincluster.

Table II summarizes the characteristics of 14 imbalanced datasets. They are generated from the original datasets via sampling from the original clusters or merging some clusters. In particular, Table II is ordered according to the IR of each dataset. Different imbalance ratios range from low to high values. Furthermore, the IR is generally larger than 5 in our selected datasets, which are selected to reflect the common cases of imbalanced data in real-world scenarios. To meet our theoretical assumption of the proposed method, the feature vectors are normalized, and their $L _ { 2 }$ norm equals 1.

TABLE II CHARACTERISTICS OF THE 14 REAL-WORLD DATASETS   

<table><tr><td>Datasets</td><td>Feature</td><td>Sample Size</td><td>Cluster Size</td><td>IR</td></tr><tr><td>glass1</td><td>9</td><td>214</td><td>76,138</td><td>1.82</td></tr><tr><td>contraceptive</td><td>9</td><td>1473</td><td>333,511,629</td><td>1.89</td></tr><tr><td>glass0</td><td>9</td><td>214</td><td>70,144</td><td>2.06</td></tr><tr><td>haberman</td><td>3</td><td>306</td><td>81,225</td><td>2.78</td></tr><tr><td>wpbc</td><td>33</td><td>198</td><td>47,151</td><td>3.21</td></tr><tr><td>newthyroid</td><td>5</td><td>215</td><td>30:35:150</td><td>5.00</td></tr><tr><td>balance</td><td>4</td><td>625</td><td>49:288:288</td><td>5.88</td></tr><tr><td>glass6</td><td>9</td><td>214</td><td>29,185</td><td>6.38</td></tr><tr><td>ecoli3</td><td>7</td><td>336</td><td>35,301</td><td>8.6</td></tr><tr><td>pageblocks0</td><td>9</td><td>5472</td><td>559, 4913</td><td>8.78</td></tr><tr><td>vowel0</td><td>10</td><td>988</td><td>90,898</td><td>9.98</td></tr><tr><td>balance_uni</td><td>4</td><td>625</td><td>49,576</td><td>11.79</td></tr><tr><td>dermatology6</td><td>34</td><td>358</td><td>20,338</td><td>16.9</td></tr><tr><td>shuttle-2vs5</td><td>9</td><td>3316</td><td>49, 3267</td><td>66.67</td></tr></table>

# B. Comparison Methods

We mainly compare our method with the most related algorithms. Since we study imbalanced clustering based on the framework of $k$ -means algorithm, we mainly compare our method with related methods of $k$ -means algorithm for imbalanced data. All compared algorithms are listed:

- $k$ -means [42]: A popular clustering algorithm is widely applied for clustering analysis due to its efficiency. $k$ -means $^ { + + }$ [30]: This algorithm is a classic clustering algorithm to improve $k$ -means. density- $k$ -means $^ { + + }$ [31]: This algorithm based on density distance deals with imbalanced data.   
C DEC [29]: Deep Embedded Clustering (DEC) algorithm applies deep neural networks to learn cluster assignments and feature representations simultaneously. IDEC [43]: Improved Deep Embedded Clustering (IDEC) algorithm is proposed to optimize cluster labels assignment and learn features that are suitable for clustering with local structure preservation.   
SMCL [16]: This algorithm is called Self-adaptive Multiprototype-based Competitive Learning (SMCL) for imbalanced clusters. MACW: This method is provided in Problem (12), called $k$ -Means algorithm for imbalanced clustering problem with Adaptive Cluster Weight (MACW).   
- ICTLB: Our proposed method is called Imbalanced Clustering with Theoretical Learning Bounds (ICTLB).

Note that density- $k$ -mean $^ { + + }$ , DEC and SMCL are three clustering methods to deal with imbalanced clustering problems. Especially, SMCL is specifically designed for imbalanced clustering problem within the framework of $k$ -means-type competitive learning. Concretely, SMCL uses multiple subclusters to represent each cluster by a self-adaptive multiprototype mechanism. Then, SMCL merges the subclusters into the final clusters with a novel separation measure. Therefore, the majority cluster contains more subclusters and the minority cluster contains less subclusters. SMCL is recently presented to achieve a good clustering performance for imbalanced data. DEC, which is based on deep neural networks, has defined a centroid-based probability distribution and minimize its KL divergence to simultaneously improve feature representation and clustering assignment, which could alleviate the negative effects of cluster imbalance by learning fitted feature representation. In addition, IDEC is an improved method of DEC to take care of data structure preservation. Both MACW and ICTLB are proposed to solve imbalanced clustering problems. To some extent, MACW is a basic method to help design a better method for imbalanced clustering. Hence, we need to compare the performance of MACW with ICTLB. To obtain a better initialization for the imbalanced clustering problemm, we generate an initial center $\mathbf { C } ^ { 0 }$ by the imbalanced clustering method, i.e., density- $k$ -means++.

# C. Evaluation Metrics

In all experiments, four commonly metrics are used to evaluate the clustering performance of all compared algorithms: 1) accuracy (ACC); 2) F-score; 3) Recall; 4) DCV.

To be specific, ACC is widely applied to evaluate the clustering results. More importantly, since we focus on imbalanced datasets, F-score and Recall are crucial indexes for evaluating the performance of clustering methods on imbalanced datasets. Moreover, A higher value of these three metrics (ACC, F-score, Recall) indicates a better clustering performance. According to [44], the evaluation metrics DCV is presented for clustering on imbalanced datasets. Simultaneously, the paper [14] indicates: the smaller value of DCV does not necessarily indicate a good performance, but the larger value of DCV indicates a poor performance.

# D. Experimental Results

In this experiment, we study the performance comparison of the proposed methods (i.e., MACW and ICTLB) with the benchmarks. All methods handle imbalanced datasets with different imbalance ratios, and we want to find the most robust method. More importantly, we focus on the performance improvement of ICTLB in the scenario of imbalanced clustering. Here, we run each algorithm ten times, giving the mean of all clustering results.

1) Experimental Results on Several Datasets: In this part, we evaluate the proposed ICTLB on 14 imbalanced datasets. Tables III, IV, V, and VI show the ACC, F-score, Recall, and DCV results, respectively. In Tables III–V, the best results are given in bold. In Table VI, the smallest results are given in bold. We can obtain the following conclusions from Tables III–V:

- For dataset contraceptive, our method ICTLB achieves the best performance on ACC among all compared methods, and the Recall value of it is very close to IDEC. The density-$k$ -mean $^ { + + }$ achieve the best performance on F-score. For dataset haberman, our method ICTLB has the best performance on ACC and F-score, but its Recall is lower than $k$ -means and SMLC. The MACW shows relative performance compared with ICTLB, especially on Recall. - For dataset wpbc, our method ICTLB and MACW show the best performance on ACC, F-score and Recall. For dataset newthyroid, our method ICTLB delivers the best performance on ACC, F-score and Recall. The MACW also achieves high ACC, F-score and Recall.

TABLE III THE AVERAGE ACC FOR ALL DATASETS   

<table><tr><td>Datasets</td><td>k-means</td><td>k-means++</td><td>density-k-means++</td><td>DEC</td><td>IDEC</td><td>SMCL</td><td>MACW</td><td>ICTLB</td></tr><tr><td>contraceptive</td><td>0.428</td><td>0.427</td><td>0.439</td><td>0.422</td><td>0.429</td><td>0.376</td><td>0.455</td><td>0.468</td></tr><tr><td>haberman</td><td>0.758</td><td>0.513</td><td>0.680</td><td>0.732</td><td>0.603</td><td>0.742</td><td>0.729</td><td>0.755</td></tr><tr><td>wpbc</td><td>0.520</td><td>0.505</td><td>0.767</td><td>0.621</td><td>0.562</td><td>0.748</td><td>0.783</td><td>0.783</td></tr><tr><td>newthyroid</td><td>0.754</td><td>0.842</td><td>0.586</td><td>0.777</td><td>0.888</td><td>0.698</td><td>0.860</td><td>0.893</td></tr><tr><td>balance</td><td>0.506</td><td>0.491</td><td>0.464</td><td>0.596</td><td>0.554</td><td>0.461</td><td>0.480</td><td>0.642</td></tr><tr><td>ecoli3</td><td>0.726</td><td>0.726</td><td>0.533</td><td>0.674</td><td>0.723</td><td>0.827</td><td>0.798</td><td>0.893</td></tr><tr><td>pageblocks0</td><td>0.939</td><td>0.930</td><td>0.888</td><td>0.932</td><td>0.935</td><td>0.928</td><td>0.922</td><td>0.940</td></tr><tr><td>vowel0</td><td>0.715</td><td>0.715</td><td>0.895</td><td>0.743</td><td>0.701</td><td>0.745</td><td>0.746</td><td>0.911</td></tr><tr><td>balance_uni</td><td>0.501</td><td>0.554</td><td>0.898</td><td>0.533</td><td>0.800</td><td>0.891</td><td>0.898</td><td>0.914</td></tr><tr><td>dermatology6</td><td>0.981</td><td>0.946</td><td>0.763</td><td>0.965</td><td>0.971</td><td>0.972</td><td>0.922</td><td>0.992</td></tr><tr><td>shuttle2vs5</td><td>0.957</td><td>0.717</td><td>0.977</td><td>0.720</td><td>0.723</td><td>0.722</td><td>0.985</td><td>0.985</td></tr></table>

The best result of all methods on each dataset is highlighted in bold.

TABLE IV THE AVERAGE F-SCORE FOR ALL DATASETS   

<table><tr><td>Datasets</td><td>k-means</td><td>k-means++</td><td>density-k-means++</td><td>DEC</td><td>IDEC</td><td>SMCL</td><td>MACW</td><td>ICTLB</td></tr><tr><td>contraceptive</td><td>0.367</td><td>0.367</td><td>0.514</td><td>0.373</td><td>0.367</td><td>0.499</td><td>0.464</td><td>0.427</td></tr><tr><td>haberman</td><td>0.729</td><td>0.548</td><td>0.621</td><td>0.740</td><td>0.623</td><td>0.707</td><td>0.729</td><td>0.764</td></tr><tr><td>wpbc</td><td>0.563</td><td>0.566</td><td>0.779</td><td>0.638</td><td>0.566</td><td>0.761</td><td>0.783</td><td>0.783</td></tr><tr><td>newthyroid</td><td>0.744</td><td>0.798</td><td>0.618</td><td>0.766</td><td>0.851</td><td>0.732</td><td>0.818</td><td>0.853</td></tr><tr><td>balance</td><td>0.438</td><td>0.434</td><td>0.529</td><td>0.532</td><td>0.494</td><td>0.572</td><td>0.592</td><td>0.532</td></tr><tr><td>ecoli3</td><td>0.703</td><td>0.703</td><td>0.619</td><td>0.675</td><td>0.701</td><td>0.830</td><td>0.768</td><td>0.894</td></tr><tr><td>pageblocks0</td><td>0.932</td><td>0.923</td><td>0.885</td><td>0.923</td><td>0.927</td><td>0.920</td><td>0.917</td><td>0.933</td></tr><tr><td>vowel0</td><td>0.724</td><td>0.724</td><td>0.895</td><td>0.747</td><td>0.714</td><td>0.749</td><td>0.752</td><td>0.911</td></tr><tr><td>balance_uni</td><td>0.630</td><td>0.640</td><td>0.900</td><td>0.635</td><td>0.825</td><td>0.892</td><td>0.899</td><td>0.914</td></tr><tr><td>dermatology6</td><td>0.979</td><td>0.945</td><td>0.754</td><td>0.963</td><td>0.967</td><td>0.969</td><td>0.922</td><td>0.991</td></tr><tr><td>shuttle2vs5</td><td>0.959</td><td>0.741</td><td>0.977</td><td>0.743</td><td>0.745</td><td>0.741</td><td>0.744</td><td>0.985</td></tr></table>

The best result of all methods on each dataset is highlighted in bold.

TABLE V THE AVERAGE RECALL FOR ALL DATASETS   

<table><tr><td>Datasets</td><td>k-means</td><td>k-means++</td><td>density-k-means++</td><td>DEC</td><td>IDEC</td><td>SMCL</td><td>MACW</td><td>ICTLB</td></tr><tr><td>contraceptive</td><td>0.374</td><td>0.374</td><td>0.359</td><td>0.371</td><td>0.377</td><td>0.349</td><td>0.359</td><td>0.367</td></tr><tr><td>haberman</td><td>0.661</td><td>0.608</td><td>0.659</td><td>0.620</td><td>0.617</td><td>0.661</td><td>0.624</td><td>0.624</td></tr><tr><td>wpbc</td><td>0.631</td><td>0.628</td><td>0.640</td><td>0.630</td><td>0.641</td><td>0.635</td><td>0.656</td><td>0.656</td></tr><tr><td>newthyroid</td><td>0.602</td><td>0.684</td><td>0.702</td><td>0.636</td><td>0.763</td><td>0.583</td><td>0.714</td><td>0.767</td></tr><tr><td>balance</td><td>0.502</td><td>0.497</td><td>0.431</td><td>0.604</td><td>0.564</td><td>0.438</td><td>0.429</td><td>0.495</td></tr><tr><td>ecoli3</td><td>0.890</td><td>0.890</td><td>0.814</td><td>0.858</td><td>0.888</td><td>0.802</td><td>0.919</td><td>0.812</td></tr><tr><td>pageblocks0</td><td>0.912</td><td>0.889</td><td>0.838</td><td>0.911</td><td>0.909</td><td>0.897</td><td>0.869</td><td>0.914</td></tr><tr><td>vowel0</td><td>0.830</td><td>0.830</td><td>0.835</td><td>0.833</td><td>0.831</td><td>0.834</td><td>0.827</td><td>0.837</td></tr><tr><td>balance_uni</td><td>0.855</td><td>0.849</td><td>0.854</td><td>0.854</td><td>0.855</td><td>0.851</td><td>0.852</td><td>0.855</td></tr><tr><td>dermatology6</td><td>0.987</td><td>0.904</td><td>0.954</td><td>0.952</td><td>0.988</td><td>0.996</td><td>0.892</td><td>0.983</td></tr><tr><td>shuttle2vs5</td><td>0.971</td><td>0.974</td><td>0.971</td><td>0.974</td><td>0.974</td><td>0.974</td><td>0.971</td><td>0.971</td></tr></table>

The best result of all methods on each dataset is highlighted in bold.

TABLE VI THE AVERAGE DCV FOR ALL DATASETS   

<table><tr><td>Datasets</td><td>k-means</td><td>k-means++</td><td>density-k-means++</td><td>DEC</td><td>IDEC</td><td>SMCL</td><td>MACW</td><td>ICTLB</td></tr><tr><td>contraceptive</td><td>0.115</td><td>0.116</td><td>1.281</td><td>0.029</td><td>0.178</td><td>0.936</td><td>0.886</td><td>0.540</td></tr><tr><td>haberman</td><td>0.333</td><td>0.619</td><td>0.240</td><td>0.607</td><td>0.532</td><td>0.231</td><td>0.527</td><td>0.693</td></tr><tr><td>wpbc</td><td>0.500</td><td>0.430</td><td>0.443</td><td>0.339</td><td>0.604</td><td>0.600</td><td>0.657</td><td>0.657</td></tr><tr><td>newthyroid</td><td>0.596</td><td>0.411</td><td>0.324</td><td>0.508</td><td>0.289</td><td>0.309</td><td>0.362</td><td>0.279</td></tr><tr><td>balance</td><td>0.583</td><td>0.626</td><td>1.061</td><td>0.521</td><td>0.560</td><td>0.449</td><td>1.012</td><td>0.202</td></tr><tr><td>ecoli3</td><td>0.758</td><td>0.758</td><td>1.069</td><td>0.840</td><td>0.767</td><td>0.101</td><td>0.539</td><td>0.286</td></tr><tr><td>pageblocks0</td><td>0.064</td><td>0.111</td><td>0.164</td><td>0.044</td><td>0.050</td><td>0.073</td><td>0.158</td><td>0.058</td></tr><tr><td>vowel0</td><td>0.395</td><td>0.395</td><td>0.206</td><td>0.311</td><td>0.443</td><td>0.309</td><td>0.272</td><td>0.252</td></tr><tr><td>balance_uni</td><td>1.181</td><td>0.935</td><td>0.240</td><td>1.060</td><td>0.899</td><td>0.136</td><td>0.155</td><td>0.195</td></tr><tr><td>dermatology6</td><td>0.024</td><td>0.138</td><td>0.672</td><td>0.032</td><td>0.059</td><td>0.079</td><td>0.095</td><td>0.024</td></tr><tr><td>shuttle2vs5</td><td>0.105</td><td>0.751</td><td>0.017</td><td>0.741</td><td>0.734</td><td>0.738</td><td>0.040</td><td>0.040</td></tr></table>

The smallest result of all methods on each dataset is highlighted in bold.

![](images/c9ca09e52765bacab54b3585be8d4079a08d6bc2a5bb65071a607f65abd80edc.jpg)  
Fig. 1. Three metrics of various methods for imbalanced versions of Glass dataset.

<!-- FIGURE-DATA: Fig. 1 | type: plot -->
> **[Extracted Data]**
> - 3 grouped bar charts: ACC, F-score, Recall
> - 8 methods: k-means, k-means++, density-k-means, DEC, IDEC, SMCL, MACW, ICTLB
> - 3 datasets: glass1 (IR=1.82), glass0 (IR=2.06), glass6 (IR=6.38)
> **Analysis:** ICTLB consistently achieves highest scores across all metrics and datasets. Performance advantage increases with higher imbalance ratio.
<!-- /FIGURE-DATA -->
- For dataset balance, our method ICTLB achieves the best performance on ACC. However, it is worse than MACW on F-score and DEC on Recall. For dataset ecoli3, our method ICTLB has the best performance on ACC, F-score. ICTLB is worse than MACW on Recall. For dataset pageblocks0, although ICTLB performs the best performance on ACC and Recall, it can be shown in Table III that other compare methods also perform good results except for density- $k$ -means $^ { + + }$ .   
- For dataset vowel0, our method ICTLB gets the best performance on ACC, F-score and Recall. The other compared methods get good results on Recall.   
C For dataset balance_uni, our method ICTLB gets the best performance on ACC and Recall, and IDEC gets the same performance with ICTLB on Recall. However, density- $k$ - means $^ { + + }$ performs the best performance on F-score, For dataset dermatology6, our method ICTLB achieves the best performance on ACC and F-score, meanwhile the performance of ICTLB is close to SMCL on Recall. For dataset shuttle2vs5, our method ICTLB achieves the best performance on ACC and F-score. Although $k \mathrm { . }$ - means $^ { + + }$ , DEC, IDEC and SMCL get the best perfor-++mance on Recall, the performance of ICTLB is closed to theirs.

Besides, from Tables III–V, although the performance of MACW is not better than that of ICTLB in most cases, MACW achieves good clustering results in most cases. In summary, the proposed ICTLB is stable across different datasets. Specifically, under the different metrics, the proposed ICTLB achieves the best performance on most cases. Therefore, applying the theoretical properties of MACW to design the ICTLB method is effective. Furthermore, the results of Table VI show a small DCV does not necessarily indicate good performance in several cases, but a large DCV indicates poor performance in most cases. For example, the DCV of SMCL is small on the ecoli3 dataset, but its Recall is poor. Meanwhile, we observe that the DCV of density- $k$ -means $^ { + + }$ is large to show that ACC, F-score and Recall are poor.

2) Glass Dataset With Different Imbalance Ratios: In this part, we study the clustering performance of the proposed ICTLB on the Glass dataset with different imbalance ratios.

Here we select glass1, glass0, and glass6, which are the imbalanced versions of the Glass dataset. More details of these three datasets are introduced in Table II. We report ACC, F-score, and Recall of various methods for imbalanced versions of Glass dataset in Fig. 1. According the experimental results on these three datasets, our method ICTLB has the best ACC, Recall. For F-score values, ICTLB method obtains slightly worse performance on glass1 compared with the best one, but it gets the best result on glass0 and glass6. Overall, our method ICTLB is stable, although the Glass dataset has different imbalance ratios.

# E. Parameter Study

In this part, we study the optimal parameter combination of the proposed ICTLB. Concretely, we need to focus on the impact of relevant parameters in the objective function (35). For the weight of each cluster in (35), the hyper-parameter $W$ corresponding to each cluster is selected from the set $\{ 0 . 1 , 0 . 2 , 0 . 3 , 1 , 2 , 3 \}$ .

0 1 0 2 0 3 1 2 3In our experiments, we find the optimal parameter combination based on the ACC measure. For example, as illustrated in Fig. 2, we plot the ACC value as a function of the hyperparameter $W$ on haberman, glass6 and pageblocks0 datasets. In Fig. 2, it has a good probability to get best clustering results when we fix one constant $W = 2$ and tune another constant $W$ . = 2Furthermore, if there are more than two parameters to be tuned, we can fix some parameters and tune few parameters to get the good clustering performance. In general, ICTLB method can obtain good performance by selecting the appropriate constant $W$ .

# $F .$ Applications to the Diagnosis of Breast Cancer

In this part, we apply the proposed methods to the aided diagnosis of breast cancer. In the early diagnosis of breast cancer, whose goal is to divide many patients into two clusters (highly suspected patients and lowly suspected patients). According to the hospital database, only a tiny percentage of patients suffer from breast cancer, and a large portion of people are healthy. Therefore, dividing many patients into the above two clusters without label information is an imbalanced clustering problem. If we ignore imbalanced information when we face imbalanced clustering, which might result in performance degradation. So, we could use imbalanced clustering methods to help the early diagnosis of breast cancer.

![](images/4122b82d74631c8782adbaa26302f43eea37540e641df0ae9d5e0dd828ee1ee0.jpg)  
Fig. 2. Parameter sensitivity of ICTLB on ACC with three datasets.
<!-- FIGURE-DATA: Fig. 2 | type: plot -->
> **[Extracted Data]**
> - 3 heatmaps: haberman, glass6, pageblocks0
> - Parameters: W ∈ [0.1, 0.2, 0.3, 1, 2, 3]
> - Peak ACC: Haberman~0.75, Glass6~0.96, Pageblocks0~0.94
> **Analysis:** Performance sensitive to parameters. Optimal W=2. Different datasets need different parameter tuning.
<!-- /FIGURE-DATA -->

TABLE VIITHE BEST RESULT OF ALL METHODS ON EACH DATASET IS HIGHLIGHTED INBOLD  

<table><tr><td>Datasets</td><td>Methods</td><td>ACC</td><td>F-score</td><td>Recall</td></tr><tr><td rowspan="7">WDBC</td><td>k-means</td><td>0.885</td><td>0.819</td><td>0.777</td></tr><tr><td>k-means++</td><td>0.885</td><td>0.819</td><td>0.777</td></tr><tr><td>density-k-means</td><td>0.588</td><td>0.638</td><td>0.528</td></tr><tr><td>DEC</td><td>0.870</td><td>0.803</td><td>0.758</td></tr><tr><td>IDEC</td><td>0.881</td><td>0.814</td><td>0.769</td></tr><tr><td>SMCL</td><td>0.880</td><td>0.812</td><td>0.769</td></tr><tr><td>MACW</td><td>0.873</td><td>0.806</td><td>0.754</td></tr><tr><td rowspan="8">BCC</td><td>TCTLB</td><td>0.896</td><td>0.829</td><td>0.811</td></tr><tr><td>k-means</td><td>0.714</td><td>0.724</td><td>0.611</td></tr><tr><td>k-means++</td><td>0.714</td><td>0.724</td><td>0.611</td></tr><tr><td>density-k-means</td><td>0.500</td><td>0.541</td><td>0.606</td></tr><tr><td>DEC</td><td>0.724</td><td>0.734</td><td>0.593</td></tr><tr><td>IDEC</td><td>0.557</td><td>0.587</td><td>0.658</td></tr><tr><td>SMCL</td><td>0.742</td><td>0.707</td><td>0.657</td></tr><tr><td>MACW TCTLB</td><td>0.543 0.771</td><td>0.542 0.770</td><td>0.606 0.635</td></tr></table>

Here, we evaluate the proposed methods on Wisconsin Diagnostic Breast Cancer (WDBC) and Breast Cancer Coimbra (BCC) datasets extracted from $\mathrm { U C I ^ { 2 } }$ . The IR of WDBC is 1.68. The numbers of each class from BCC are almost equal, however, we know that the cancer data usually has a larger imbalance ratio in practice. Thus, we randomly choose 52 healthy people and 18 cancer patients from BCC dataset, and then the IR is 2.88. In Table VII, ACC, F-score and Recall are employed to evaluate the compared methods. It can be seen that TCTLB obtains desirable results in most cases.

# V. CONCLUSION AND FUTURE WORK

In this article, we have investigated imbalanced clustering problems. We are inspired by theoretical analysis to propose two novel methods. We first propose MACW method, which combines an adaptive cluster weight with the objective function of $k$ -means. Meanwhile, we provide the excess clustering risk bound of MACW. Then, based on theoretical results, we further present ICTLB method by optimizing excess clustering risk bound and give a theoretically-principled justification of

ICTLB. The proposed ICTLB obtains significantly performance improvement on many real-world imbalanced datasets.

A limitation of this work is that the theoretical analysis of ICTLB’s success is not fully clear. We leave this as a direction for future work. Moreover, this article only provides the excess clustering risk bound of $k$ -means on imbalanced clustering. Thus, studying the theoretical properties of other clustering methods is also interesting for further study.

# VI. PROOFS

# A. Proof of Lemma 1

Proof: In order to prove Lemma 1, we first propose the following Lemma 5.

Lemma 5. ( $L _ { \infty }$ Contraction Inequality, presented in $I 3 9 J )$ : Let $\psi : \mathbb { R } ^ { k }  \mathbb { R }$ be L-Lipschitz w.r.t. the $L _ { \infty }$ norm, that is $\| \psi ( \mathbf { u } ) - \psi ( \mathbf { u } ^ { \prime } ) \| _ { \infty } \leq L \cdot \| \mathbf { u } - \mathbf { u } ^ { \prime } \| _ { \infty } , \forall \mathbf { u } , \mathbf { u } ^ { \prime } \in \mathbb { R } ^ { k }$ . Define $\mathcal { G } \subseteq$ $\{ g : \mathcal { X }  \mathbb { R } ^ { k } \}$ . If $\operatorname* { m a x } \{ | \psi ( g ( \mathbf { x } ) ) | , \| g ( \mathbf { x } ) \| _ { \infty } \} \leq \varepsilon$ , and there is :a constant $T > 0$ max ( ( )) ( )satisfying the following inequality

$$
\mathfrak { R } _ { n } ( \psi \circ \mathcal { G } ) \leq T \cdot L \sqrt { k } \operatorname* { m a x } _ { r } \hat { \mathfrak { R } } _ { n } \left( \mathcal { G } _ { r } \right) \log ^ { \frac { 3 } { 2 } + p } \left( \frac { \varepsilon n } { \operatorname* { m a x } _ { r } \hat { \mathfrak { R } } _ { n } ( \mathcal { G } _ { r } ) } \right) ,
$$

where $p$ is any positive value,

$$
\Re _ { n } ( \psi \circ \mathcal { G } _ { \mathbf { C } } ) = \mathbb { E } _ { \pmb { \sigma } } \left[ \operatorname* { s u p } _ { \pmb { \sigma } \in \mathcal { G } } \left| \sum _ { i = 1 } ^ { n } \sigma _ { i } \psi \left( g \left( \mathbf { x } _ { i } \right) \right) \right| \right] ,
$$

and $\begin{array} { r } { \hat { \mathfrak { R } } _ { n } ( \mathcal G _ { r } ) = \operatorname* { s u p } _ { \mathcal S \in \mathcal X ^ { n } } \mathfrak { R } _ { n } ( \mathcal G _ { r } ) } \end{array}$ , where $\mathcal { G } _ { r }$ is a family of the n( r) = suoutput coordinate $r$ of $\mathcal { G }$ .

Based on Lemma 5, we give more details to prove Lemma 1 in the following. First, according to the minimum function defined in (18), we have $\phi ( \mathbf { u } ) = \operatorname* { m i n } ( u _ { 1 } , \dots , u _ { k } )$ , which is easy to ( ) = min( 1 k)verify that it is 1-Lipschitz continuous w.r.t. the $L _ { \infty }$ -norm, that is $\forall \mathbf { u } , \mathbf { u } ^ { \prime } \in \mathbb { R } ^ { k } , | \phi ( \mathbf { u } ) - \phi ( \mathbf { u } ^ { \prime } ) | \leq \| \mathbf { u } - \mathbf { u } ^ { \prime } \| _ { \infty }$ . To be specific, we (can suppose that $\phi ( \mathbf { u } ) \geq \phi ( \mathbf { u } ^ { \prime } )$ without loss of generality. By the definition of $\phi$ ( ) ( ), we can obtain $\phi ( \mathbf { u } ^ { \prime } ) = u _ { j } ^ { \prime }$ , where

$$
j = \underset { i = 1 , . . . , k } { \arg \operatorname* { m i n } } u _ { i } ^ { \prime } .
$$

Then, we have

$$
\left| \phi ( \mathbf { u } ) - \phi \left( \mathbf { u } ^ { \prime } \right) \right|
$$

$$
\begin{array} { r l } & { \mathbf { \phi } = \phi ( \mathbf { u } ) - \phi \left( \mathbf { u } ^ { \prime } \right) } \\ & { \mathbf { \phi } = \phi ( \mathbf { u } ) - u _ { j } ^ { \prime } } \\ & { \leq u _ { j } - u _ { j } ^ { \prime } } \\ & { \leq \left. \mathbf { u } - \mathbf { u } ^ { \prime } \right. _ { \infty } . } \end{array}
$$

Second, we need to verify the next condition that $\operatorname* { m a x } \{ | \phi ( g _ { \mathbf { C } } ( \mathbf { x } ) ) | , \| g _ { \mathbf { C } } ( \mathbf { x } ) \| _ { \infty } \}$ is bounded by a constant. max ( ( )) ( )Specifically, recall that the definition of (15): $g _ { \mathbf { C } } ( \mathbf { x } ) =$ $( g _ { \mathbf { c } _ { 1 } } ( \mathbf { x } ) , \hdots , g _ { \mathbf { c } _ { k } } ( \mathbf { x } ) )$ , where $g _ { \mathbf { c } _ { r } } ( \mathbf { x } ) = l \| \mathbf { x } - \mathbf { c } _ { r } \| ^ { 2 } \lambda _ { r } ^ { \alpha } , \| \mathbf { x } \| \leq 1$ (and $\| \mathbf { c } _ { r } \| \leq 1$ (. For $\forall \mathbf { x } \in \mathcal { X }$ ( ), there is

$$
g _ { \mathbf { c } _ { r } } ( \mathbf { x } ) = l \left\| \mathbf { x } - \mathbf { c } _ { r } \right\| ^ { 2 } \lambda _ { r } ^ { \alpha } \leq 2 \lambda _ { r } ^ { \frac { \alpha } { 2 } } ( \left\| \mathbf { x } \right\| + \left\| \mathbf { c } _ { r } \right\| ) \leq 4 \lambda _ { r } ^ { \frac { \alpha } { 2 } } ,
$$

so we have

$$
| \phi \left( g _ { \mathbf { C } } ( \mathbf { x } ) \right) | = | \operatorname* { m i n } ( g _ { \mathbf { c } _ { 1 } } , \dots , g _ { \mathbf { c } _ { k } } ) | \le 4 \bar { \lambda } ,
$$

where $\begin{array} { r } { \overline { { \lambda } } = \operatorname* { m i n } _ { r = 1 , \ldots , k } \lambda _ { r } ^ { \frac { \alpha } { 2 } } } \end{array}$ . On the other hand, it can be obtained

$$
\| g _ { \mathbf { C } } ( \mathbf { x } ) \| _ { \infty } = \operatorname* { m a x } _ { r } | g _ { \mathbf { c } _ { r } } ( \mathbf { x } ) | \leq 4 \hat { \lambda } ,
$$

where

$$
\hat { \lambda } = \operatorname* { m a x } _ { r = 1 , \ldots , k } \lambda _ { r } ^ { \frac { \alpha } { 2 } } .
$$

Thus, we have

$$
\operatorname* { m a x } \left\{ \left| \phi \left( g _ { \mathbf { C } } ( \mathbf { x } ) \right) \right| , \left\| g _ { \mathbf { C } } ( \mathbf { x } ) \right\| _ { \infty } \right\} \leq 4 \hat { \lambda } ( \hat { \lambda } \mathrm { { i s } ~ a ~ c o n s t a n t { ) } . }
$$

In a result, it can be seen that $\phi ( u )$ is 1-continuous w.r.t. the $L _ { \infty }$ - norm and $\operatorname* { m a x } \{ | \phi ( f _ { \mathbf { C } } ( \mathbf { x } ) ) | , \| f _ { \mathbf { C } } ( \mathbf { x } ) \| _ { \infty } \} \le 4 \hat { \lambda }$ . Then, based on maxLemma 5 with $L = 1$ , $p = 1 / 2$ ( ), and $\varepsilon = 4 \hat { \lambda }$ 4in our problem, we can draw

$$
\mathfrak { R } _ { n } \left( \mathcal { F } _ { \mathbf { C } } \right) \leq T \sqrt { k } \operatorname* { m a x } _ { r } \hat { \mathfrak { R } } _ { n } \left( \mathcal { G } _ { \mathbf { C } _ { r } } \right) \log ^ { 2 } \left( \frac { 4 \hat { \lambda } n } { \operatorname* { m a x } _ { r } \hat { \mathfrak { R } } _ { n } \left( \mathcal { G } _ { \mathbf { C } _ { r } } \right) } \right) .
$$

For any $j$ , according to the definition of $\hat { \Re } _ { n } ( \mathcal G _ { \mathbf C _ { r } } )$ , we note that

$$
\begin{array} { r l } { \widehat { \Phi } _ { \mathrm { n } } \left( \mathcal { G } _ { \mathrm { C } _ { r } } \right) } & { = \displaystyle \operatorname* { s u p } _ { S \in \mathbb { R } ^ { n } } \mathfrak { R } _ { \mathrm { n } } \left( \mathcal { G } _ { \mathrm { C } , r } \right) } \\ & { = \displaystyle \operatorname* { s u p } _ { S \in \mathbb { R } ^ { n } } \mathbb { E } _ { \sigma } \Bigg [ \displaystyle \operatorname* { s u p } _ { \theta \in \mathcal { S } _ { \mathrm { C } } } \left| \displaystyle \sum _ { j = 1 } ^ { n } \sigma _ { i } g _ { \mathrm { C } } \left( \mathbf { x } _ { i } \right) \right| \Bigg ] } \\ & { \geq \displaystyle \operatorname* { s u p } _ { S \in \mathcal { S } } \mathbb { E } _ { \sigma } \Bigg [ \displaystyle \operatorname* { s u p } _ { \theta \in \mathcal { S } _ { \mathrm { C } } } \left| \displaystyle \sum _ { j = 1 } ^ { n } \sigma _ { i } g _ { \mathrm { C } } \left( \mathbf { x } \right) \right| \Bigg ] } \\ & { \geq \displaystyle \operatorname* { s u p } _ { S \in \mathcal { S } } \mathbb { E } _ { \sigma } \mathbb { E } _ { \sigma } \Bigg [ \displaystyle \sum _ { j = 1 } ^ { n } \sigma _ { j } g _ { \mathrm { C } } \left( \mathbf { x } \right) \Bigg | \Bigg ] . } \end{array}
$$

Recall that $\sigma _ { 1 } , \ldots , \sigma _ { n }$ is a sequence of independent Rademacher 1variables. Let $\mathcal { H }$ ndenote a Hilbert space with $\| \cdot \|$ being the associated norm. For $\eta _ { 1 } , \dotsc , \eta _ { n } \in { \mathcal { H } }$ , based on Lemma 24 (a) of [45], we know that

Thus, the inequality (40) can be re-written as

$$
\hat { \mathfrak { R } } _ { n } \left( \mathcal { G } _ { \mathbf { C } _ { r } } \right) \geq \frac { \sqrt { 2 } } { 2 } \sqrt { n } \operatorname* { s u p } _ { \mathbf { x } \in \mathcal { X } , g _ { \mathrm { c } } \in \mathcal { G } _ { \mathbf { C } _ { r } } } \sqrt { | g _ { \mathrm { c } } ( \mathbf { x } ) | } .
$$

Define

$$
\begin{array} { r l } & { w _ { r } = \underset { \mathbf { x } \in \mathcal { X } } { \operatorname* { s u p } } ~ \underset { g _ { \mathbf { c } } \in \mathcal { G } _ { \mathbf { C } _ { r } } } { \operatorname* { s u p } } \left| g _ { \mathbf { c } } ( \mathbf { x } ) \right| ; } \\ & { w = \operatorname* { m a x } \left\{ w _ { r } , r = 1 , \dots , k \right\} . } \end{array}
$$

By the definition of (42), we can re-form the inequality (41) as

$$
\hat { \mathscr { R } } _ { n } \left( \mathcal { G } _ { { \bf C } _ { r } } \right) \geq \frac { \sqrt { 2 } } { 2 } \sqrt { w _ { r } n } .
$$

Based on (37), we obtain $\mathrm { m a x } _ { r }$ $| g _ { \mathbf { c } _ { r } } ( \mathbf { x } ) | \leq 4 \hat { \lambda }$ , i.e., $w \leq 4 \hat { \lambda }$ by maxr ( ) 4the definition in (42), where the definition of $\hat { \lambda }$ 4ˆis given in (38). Hence, we have

$$
\operatorname* { m a x } _ { r } \hat { \mathcal { { R } } } _ { n } \left( \mathcal { G } _ { { \mathbf C } _ { r } } \right) \geq \frac { \sqrt { 2 w n } } { 2 } \geq \sqrt { 2 \hat { \lambda } n } .
$$

Thus, we can obtain

$$
\frac { 4 \hat { \lambda } n } { \operatorname* { m a x } _ { r } \hat { \Re } _ { n } \left( \mathcal G _ { { \bf C } _ { r } } \right) } \leq 2 \sqrt { 2 \hat { \lambda } n } .
$$

Submitting this into (39) proves the result in Lemma 1.

# B. Proof of Lemma 2

Proof: Obviously, for any $\mathcal { S } \in \mathcal { X } ^ { n } , \mathbf { C } \in \mathcal { H } ^ { k }$ and $r \in$ $\{ 1 , \ldots , k \}$ and a fixed constant $\lambda _ { r } ^ { \alpha } > 0$ , we have

$$
\begin{array} { r l } { \mathcal { R } _ { 1 } ( G _ { 1 } ) = } & { \mathbb { E } _ { \varphi _ { 1 } } \frac { \varphi _ { 1 } } { \varphi _ { 1 } } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg ] } \\ & { - \mathbb { E } _ { \varphi _ { 1 } } \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg ] ^ { 2 } \Bigg ] } \\ & { - \mathbb { E } _ { \varphi _ { 1 } } \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg ] ^ { 2 } \Bigg ] } \\ & { - \mathbb { E } _ { \varphi _ { 1 } } \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg ] ^ { 2 } \Bigg ] } \\ & { - \mathbb { E } _ { \varphi _ { 1 } } \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg ] ^ { 2 } \Bigg ] ^ { 2 } \Bigg ] } \\ & { - \mathbb { E } _ { \varphi _ { 1 } } \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg ] ^ { 2 } \Bigg ] ^ { 2 } \Bigg ] } \\ &  \leq 2 \mathbb { R } _ { \varphi _ { 1 } } \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac { \sin \theta } { \sin \theta } \Bigg [ \frac   \end{array}
$$

We denote $\begin{array} { r } { A = 2 \mathbb { E } _ { \pmb { \sigma } } \operatorname* { s u p } _ { \mathbf { c } \in \mathcal { H } } | \sum _ { j = 1 } ^ { n } \pmb { \sigma } _ { j } \left. \mathbf { x } _ { j } , \mathbf { c } \right. \lambda _ { r } ^ { \alpha } | } \end{array}$ and $B =$ $\begin{array} { r } { \mathbb { E } _ { \pmb { \sigma } } \operatorname* { s u p } _ { \mathbf { c } \in \mathcal { H } } | \sum _ { j = 1 } ^ { n } \pmb { \sigma } _ { j } \| \mathbf { c } \| ^ { 2 } \lambda _ { r } ^ { \alpha } | , } \end{array}$ j=1 j j. First, we have

$$
\mathbb { E } _ { \pmb { \sigma } } \left[ \left| \sum _ { i = 1 } ^ { n } \sigma _ { i } \eta _ { i } \right| \right] \geq \frac { \sqrt { 2 } } { 2 } \sqrt { \sum _ { i = 1 } ^ { n } \| \eta _ { i } \| ^ { 2 } } .
$$

$$
A = 2 \mathbb { E } _ { \pmb { \sigma } } \operatorname* { s u p } _ { \mathbf { c } \in \mathcal { H } } \left| \left. \sum _ { j = 1 } ^ { n } \pmb { \sigma } _ { j } \mathbf { x } _ { j } , \mathbf { c } \lambda _ { r } ^ { \alpha } \right. \right|
$$

$$
\begin{array} { r l } & { \leq 2 \lambda _ { r } ^ { \alpha } \mathbb { E } _ { \sigma } \Bigg \{ \Bigg \vert \displaystyle \sum _ { | \alpha | = 1 } ^ { n } \sigma _ { \beta } x _ { \alpha } \Bigg \} \Bigg \vert \left( \mathrm { s i n c e ~ | \boldsymbol { c } | \leq 1  } } \\ & \right){ \leq 2 \lambda _ { r } ^ { \alpha } \Bigg [ \mathbb { E } _ { \sigma } \Bigg \vert \displaystyle \sum _ { | \alpha | = 1 } ^ { n } \sigma _ { \beta } x _ { \alpha } \Bigg \vert \Bigg \vert ^ { 2 } \Bigg ] ^ { 1 / 2 } } \\ & { = 2 \lambda _ { r } ^ { \alpha } \Bigg [ \mathbb { E } _ { \sigma } \displaystyle \sum _ { i , j = 1 } ^ { n } \sigma _ { i } ^ { 2 } ( \mathbf { x } _ { i } ^ { \alpha } \mathbf { x } _ { j } ) ^ { 2 } \Bigg ] ^ { 1 / 2 } } \\ & { \leq 2 \lambda _ { r } ^ { \alpha } \Bigg [ \mathbb { E } _ { \sigma } \displaystyle \sum _ { i , j = 1 } ^ { n } \sigma _ { i } \sigma _ { j } ( \mathbf { x } _ { i } ^ { \alpha } \mathbf { x } _ { j } ) ^ { 2 } \Bigg ] ^ { 1 / 2 } . } \end{array}
$$

Since the independent characteristic of any $\sigma _ { i }$ , there is $\mathbb { E } [ \pmb { \sigma } _ { i } \pmb { \sigma } _ { j } ] = \mathbb { E } [ \pmb { \sigma } _ { i } ] \mathbb { E } [ \pmb { \sigma } _ { j } ] = 0 , \mathrm { i f } i \neq j ; \mathbb { E } [ \pmb { \sigma } _ { i } \pmb { \sigma } _ { j } ] = 1 , \mathrm { i f } i = j .$ [ i j] = [Meanwhile, by $\| \mathbf { x } _ { j } \| \leq 1$ 0 =, we obtain

$$
A \leq 2 \lambda _ { r } ^ { \alpha } \left[ \sum _ { j = 1 } ^ { n } \| \mathbf { x } _ { j } \| ^ { 2 } \right] ^ { 1 / 2 } \leq 2 \lambda _ { r } ^ { \alpha } \sqrt { n } .
$$

Then, it can be seen that

$$
B \leq \lambda _ { r } ^ { \alpha } \mathbb { E } _ { \sigma } \left. \sum _ { j = 1 } ^ { n } \sigma _ { j } \right. .
$$

Denote $\begin{array} { r } { M = \mathbb { E } _ { \pmb { \sigma } } | \sum _ { j = 1 } ^ { n } \pmb { \sigma } _ { j } | } \end{array}$ . By simple computation,

$$
\begin{array} { l } { { \displaystyle M = \sum _ { j = 0 } ^ { \lfloor \frac { n } { 2 } \rfloor } 2 ( n - 2 j ) \frac { 1 } { 2 ^ { n } } C _ { n } ^ { n - j } } } \\ { ~ } \\ { { \displaystyle ~ = \frac { 1 } { 2 ^ { n - 1 } } \sum _ { j = 0 } ^ { \lfloor \frac { n } { 2 } \rfloor } ( n - 2 j ) C _ { n } ^ { n - j } } } \\ { ~ } \\ { { \displaystyle ~ = \frac { 1 } { 2 ^ { n - 1 } } ( U - V ) } , } \end{array}
$$

$$
= V + \lfloor { \frac { n } { 2 } } + 1 \rfloor C _ { n } ^ { \lfloor { \frac { n } { 2 } } + 1 \rfloor } .
$$

Then, we have

$$
\begin{array} { l } { { \displaystyle M = \frac { 1 } { 2 ^ { n - 1 } } ( U - V ) } } \\ { { \displaystyle \quad = \frac { 1 } { 2 ^ { n - 1 } } \lfloor \frac { n } { 2 } + 1 \rfloor C _ { n } ^ { \lfloor \frac { n } { 2 } + 1 \rfloor } } } \\ { { \displaystyle \quad = \frac { n } { 2 ^ { n } } ( 1 + o ( 1 ) ) C _ { n } ^ { \lfloor \frac { n } { 2 } + 1 \rfloor } } } \\ { { \displaystyle \quad = ( 1 + o ( 1 ) ) \sqrt { \frac { 2 n } { \pi } } , } } \end{array}
$$

since C n2 +1 $C _ { n } ^ { \lfloor { \frac { n } { 2 } } + 1 \rfloor } = ( 1 + o ( 1 ) ) \sqrt { \frac { 2 } { \pi n } } 2 ^ { n }$ by the Stirling’s approximation. Hence, $\begin{array} { r } { \operatorname* { l i m } _ { n \to \infty } M = \sqrt { \frac { 2 n } { \pi } } . } \end{array}$ 2n . Thus, we can obtain

$$
B \leq \lambda _ { r } ^ { \alpha } ( 1 + o ( 1 ) ) \sqrt { \frac { 2 n } { \pi } } .
$$

So, Inequality (43) is denoted as

$$
\Re _ { n } ( \mathcal G _ { { \bf C } _ { r } } ) \le \lambda _ { r } ^ { \alpha } \left[ 2 + ( 1 + o ( 1 ) ) \sqrt { \frac 2 \pi } \right] \sqrt { n } .
$$

Finally, since $\begin{array} { r } { \hat { \mathfrak { R } } _ { n } ( \mathcal G _ { \mathbf { C } _ { r } } ) = \operatorname* { s u p } _ { \mathcal { S } \in \mathcal { X } ^ { n } } \mathfrak { R } _ { n } ( \mathcal G _ { \mathbf { C } _ { r } } ) } \end{array}$ , we get

$$
\begin{array} { r l } & { \underset { r } { \operatorname* { m a x } } \hat { \mathcal { \ R } } _ { n } ( \mathcal { G } _ { \mathbf { C } _ { r } } ) \leq \tilde { \lambda } \left[ 2 + \left( 1 + o ( 1 ) \right) \sqrt { \frac { 2 } { \pi } } \right] \sqrt { n } } \\ & { ( \tilde { \lambda } = \underset { r = 1 , \ldots , k } { \operatorname* { m a x } } \lambda _ { r } ^ { \alpha } ) . } \end{array}
$$

This proves the statement in Lemma 2.

C. The Proof of Lemma 3

where $\begin{array} { r } { U = \sum _ { j = 0 } ^ { \lfloor \frac { n } { 2 } \rfloor } ( n - j ) C _ { n } ^ { n - j } } \end{array}$ and $\textstyle V = \sum _ { j = 0 } ^ { \lfloor { \frac { n } { 2 } } \rfloor } j C _ { n } ^ { n - j }$ . One =can see that

$$
\begin{array} { r l } & { v \mathcal { U } = \displaystyle \sum _ { j = 0 } ^ { \lfloor \frac { 3 } { 2 } \rfloor } ( n - j ) \mathcal { C } _ { n } ^ { \alpha - j } } \\ & { = \displaystyle \sum _ { j = 0 } ^ { \lfloor \frac { 3 } { 2 } \rfloor } \frac { n ( n - 1 ) \cdots ( n - j + 1 ) } { j ! } ( n - j ) } \\ & { = \displaystyle \sum _ { j = 0 } ^ { \lfloor \frac { 3 } { 2 } \rfloor } \frac { n ( n - 1 ) \cdots ( n - j + 1 ) ( n - j ) } { ( j + 1 ) ! } ( j + 1 ) } \\ & { = \displaystyle \sum _ { j = 0 } ^ { \lfloor \frac { 3 } { 2 } \rfloor } ( j + 1 ) \mathcal { C } _ { n } ^ { \alpha - j + 1 } } \\ & { = \displaystyle \sum _ { j = 0 } ^ { \lfloor \frac { 3 } { 2 } \rfloor } ( j ) \mathcal { C } _ { n } ^ { \alpha - j + 1 } } \\ & { = \displaystyle \sum _ { j = 1 } ^ { \lfloor \frac { 3 } { 2 } \rfloor } \mathcal { C } _ { n } ^ { \alpha } ( \ln { j } - j + 1 ) } \end{array}
$$

Proof: According to [40], with probability $1 - \delta$ , we have

$$
\Re \left( \mathcal { F } _ { \mathbf { C } } \right) \leq \Re _ { n } \left( \mathcal { F } _ { \mathbf { C } } \right) + \sqrt { 2 n \log \left( \frac { 1 } { \delta } \right) } .
$$

Thus, we can obtain

$$
\Re \left( \mathcal { F } _ { \mathbf { C } } \right) \leq T \sqrt { k } \operatorname* { m a x } _ { r } \hat { \Re } _ { n } \left( \mathcal { G } _ { \mathbf { C } _ { r } } \right) \log ^ { 2 } ( 2 \sqrt { 2 \hat { \lambda } n } ) + \sqrt { 2 n \log \left( \frac { 1 } { \delta } \right) }
$$

$$
\begin{array} { r l } {  { \leq T \tilde { \lambda } [ 2 + ( 1 + o ( 1 ) ) \sqrt { \frac { 2 } { \pi } } ] \sqrt { k n } \log ^ { 2 } ( 2 \sqrt { 2 \hat { \lambda } n } ) } } \\ & { + \sqrt { 2 n \log ( \frac { 1 } { \delta } ) } ( \mathrm { v i a L e m m a \it 2 } ) . } \end{array}
$$

This proves the statement in Lemma 3.

# D. The Proof of Lemma 4

Proof: When $k = 1$ , by the definition of $\Re _ { n } ( \mathcal { G } _ { \mathbf { C } _ { r } ) }$ and (45), we can obtain

$$
\mathbb { E } _ { \sigma } \operatorname* { s u p } _ { \mathbf { c } \in \mathcal { H } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } g _ { \mathbf { c } } \left( \mathbf { x } _ { i } \right) \leq \frac { \lambda _ { 1 } ^ { \alpha } \left[ 2 + \left( 1 + o ( 1 ) \right) \sqrt { \frac { 2 } { \pi } } \right] } { \sqrt { n } } .
$$

When $k = 2$ , we can get

$$
\begin{array} { l } { \displaystyle \frac 1 2 \sum _ { r = 1 } ^ { 2 } ( \frac { 1 } { \left| \theta _ { r } \right| } \widehat { \mathfrak { R } } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) ) } \\ { = \displaystyle \frac 1 2 \mathbb { E } \operatorname* { s u p } _ { ( \mathbf { c } _ { 1 } , \mathbf { c } _ { 2 } ) \in \mathcal { H } ^ { 2 } } \sum _ { r = 1 } ^ { 2 } \frac { 1 } { \left| \theta _ { r } \right| } \sum _ { i \in \theta _ { r } } \sigma _ { i } \left[ \operatorname* { m i n } _ { j = 1 , 2 } g _ { \mathbf { c } _ { j } } \left( \mathbf { x } _ { i } \right) \right] } \\ { = \displaystyle \frac 1 2 \mathbb { E } \operatorname* { s u p } _ { ( \mathbf { c } _ { 1 } , \mathbf { c } _ { 2 } ) \in \mathcal { H } ^ { 2 } } \sum _ { r = 1 } ^ { 2 } \frac { 1 } { 2 \left| \theta _ { r } \right| } \sum _ { i \in \theta _ { r } } \sigma _ { i } \left[ g _ { \mathbf { c } _ { 1 } } \left( \mathbf { x } _ { i } \right) + g _ { \mathbf { c } _ { 2 } } \left( \mathbf { x } _ { i } \right) \right. } \end{array}
$$

$$
- | g _ { \mathbf { c } _ { 1 } } ( \mathbf { x } _ { i } ) - g _ { \mathbf { c } _ { 2 } } ( \mathbf { x } _ { i } ) | ]
$$

$$
\begin{array} { l } { { \displaystyle \leq \frac { 1 } { 4 } \mathbb { E } \operatorname* { s u p } _ { { \bf c } \in \mathcal { H } } \left[ \frac { 1 } { \left| \boldsymbol { \theta } _ { 1 } \right| } \sum _ { i \in \boldsymbol { \theta } _ { 1 } } \sigma _ { i } \left( 4 g _ { c } \left( { \bf x } _ { i } \right) \right) + \frac { 1 } { \left| \boldsymbol { \theta } _ { 2 } \right| } \sum _ { i \in \boldsymbol { \theta } _ { 2 } } \sigma _ { i } \left( 4 g _ { c } \left( { \bf x } _ { i } \right) \right) \right] } \ }  \\ { { \displaystyle = \mathbb { E } \operatorname* { s u p } _ { { \bf c } \in \mathcal { H } } \left[ \frac { 1 } { \left| \boldsymbol { \theta } _ { 1 } \right| } \sum _ { i \in \boldsymbol { \theta } _ { 1 } } \sigma _ { i } g _ { c } \left( { \bf x } _ { i } \right) + \frac { 1 } { \left| \boldsymbol { \theta } _ { 2 } \right| } \sum _ { i \in \boldsymbol { \theta } _ { 2 } } \sigma _ { i } g _ { c } \left( { \bf x } _ { i } \right) \right] . } } \end{array}
$$

By (47), we have

$$
\frac { 1 } { 2 } \sum _ { r = 1 } ^ { 2 } ( \frac { 1 } { \vert \theta _ { r } \vert } \hat { \mathfrak { R } } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) ) \leq \left[ \frac { \lambda _ { 1 } ^ { \alpha } P } { \sqrt { \vert \theta _ { 1 } \vert } } + \frac { \lambda _ { 2 } ^ { \alpha } P } { \sqrt { \vert \theta _ { 2 } \vert } } \right] ,
$$

where $P = [ 2 + ( 1 + o ( 1 ) ) \sqrt { \frac { 2 } { \pi } } ]$ . Based on the recurrence rule πand the result of the arguments presented for $k = 1 , k = 2$ , we could get: for any $( a _ { 1 } , \ldots , a _ { k } ) \in \mathbb { R } ^ { k }$ ,

$$
\begin{array} { l } { \operatorname* { m i n } \left( a _ { 1 } , \dots , a _ { k } \right) } \\ { = \operatorname* { m i n } \left( \operatorname* { m i n } \left( a _ { 1 } , \dots , a _ { \lfloor k / 2 \rfloor } \right) , \operatorname* { m i n } \left( a _ { \lfloor k / 2 \rfloor + 1 } , \dots , a _ { k } \right) \right) . } \end{array}
$$

Thus, one can see that

$$
\frac { 1 } { k } \sum _ { r = 1 } ^ { k } \left( \frac { 1 } { \vert \theta _ { r } \vert } \hat { \Re } ^ { ( r ) } ( \mathcal { F } _ { \mathbf { C } } ) \right) \leq \sum _ { r = 1 } ^ { k } \frac { \lambda _ { r } ^ { \alpha } P } { \sqrt { \vert \theta _ { r } \vert } } .
$$

This proves the statement in Lemma 4.

# E. The Proof of Theorem 3

It will be shown that in the case of $k = 2$ , we are always pos-= 2sible to design the weights by optimizing the excess clustering risk bound in Theorem 2.

Theorem 3: For the cluster $k = 2$ , let $\mathcal { F } _ { \mathbf { C } }$ be a family of $k$ - = 2valued functions. Assume some clustering centers $\mathbf { C } \in \mathcal { H } ^ { k }$ in cost function can achieve a total sum of weights $\lambda _ { 1 } ^ { \prime } + \lambda _ { 2 } ^ { \prime } = 1$ with $\lambda _ { 1 } ^ { \prime } , \lambda _ { 2 } ^ { \prime } > 0$ 1. Then there exists a clustering center $\mathbf { C } _ { n } ^ { \star } \in \mathcal { H } ^ { k }$

in cost function with weights

$$
\lambda _ { 1 } ^ { \star } = \frac { 1 } { 1 + ( \frac { \left| \theta _ { 1 } \right| } { \left| \theta _ { 2 } \right| } ) ^ { \frac { 1 } { 2 ( 1 - \alpha ) } } } \ a n d \ \lambda _ { 2 } ^ { \star } = \frac { 1 } { 1 + ( \frac { \left| \theta _ { 2 } \right| } { \left| \theta _ { 1 } \right| } ) ^ { \frac { 1 } { 2 ( 1 - \alpha ) } } } ,
$$

which with probability $1 - \delta$ gets the optimal excess clustering 1risk guarantee for Theorem 2

$$
\Delta ( \mathbf { C } _ { n } ^ { \star } ) \leq \operatorname* { m i n } _ { \lambda _ { 1 } + \lambda _ { 2 } = 1 } 4 \left( \lambda _ { 1 } ^ { \alpha } P \sqrt { \frac { 1 } { \left| \theta _ { 1 } \right| } } + \lambda _ { 2 } ^ { \alpha } P \sqrt { \frac { 1 } { \left| \theta _ { 2 } \right| } } \right) + 2 \sum _ { r = 1 } ^ { k } \tau ( | \theta _ { r } | ) ,
$$

where $\mathbf { C } _ { n } ^ { \star }$ is generated by $\Phi ( \mathbf { C } , \mathbb { Q } _ { n } )$ related to $\lambda _ { 1 } ^ { \star }$ and $\lambda _ { 2 } ^ { \star }$ , and $\begin{array} { r } { \tau ( | \theta _ { r } | ) = \sqrt { \frac { 2 \log ( \frac { 2 } { \delta } ) } { | \theta _ { r } | } } } \end{array}$

θProof: We can apply Theorem 2 to obtain with probability $1 - \delta$ the excess clustering risk bound

$$
\Delta ( \mathbf { C } _ { n } ^ { \star } ) \leq 4 \lambda _ { 1 } ^ { \star \alpha } P \sqrt { \frac { 1 } { \vert \theta _ { 1 } \vert } } + 4 \lambda _ { 2 } ^ { \star \alpha } P \sqrt { \frac { 1 } { \vert \theta _ { 2 } \vert } } + 2 \sum _ { r = 1 } ^ { k } \tau ( \vert \theta _ { r } \vert ) .
$$

To see that $\lambda _ { 1 } ^ { \star } , \lambda _ { 2 } ^ { \star }$ indeed solve

$$
\operatorname* { m i n } _ { \lambda _ { 1 } + \lambda _ { 2 } = 1 } \lambda _ { 1 } ^ { \alpha } \sqrt { \frac { 1 } { \vert \theta _ { 1 } \vert } } + \lambda _ { 2 } ^ { \alpha } \sqrt { \frac { 1 } { \vert \theta _ { 2 } \vert } } ,
$$

we could substitute $\lambda _ { 2 } = 1 - \lambda _ { 1 }$ into Problem (48) and set the 2 = 1 1derivative of this objective function to 0, getting

$$
\lambda _ { 1 } ^ { \alpha - 1 } \sqrt { \frac { 1 } { \vert \theta _ { 1 } \vert } } - ( 1 - \lambda _ { 1 } ) ^ { \alpha - 1 } \sqrt { \frac { 1 } { \vert \theta _ { 2 } \vert } } = 0 .
$$

We can solve the above equation to obtain ${ \lambda } _ { 1 } ^ { \star }$ and $\lambda _ { 2 } ^ { \star }$ .

# REFERENCES

[1] H. He and E. A. Garcia, “Learning from imbalanced data,” IEEE Trans. Knowl. Data Eng., vol. 21, no. 9, pp. 1263–1284, Sep. 2009.   
[2] A. Somasundaram and S. Reddy, “Parallel and incremental credit card fraud detection model to handle concept drift and data imbalance,” Neural Comput. Appl., vol. 31, no. 1, pp. 3–14, 2019.   
[3] T. Vo, T. Nguyen, and C. Le, “A hybrid framework for smile detection in class imbalance scenarios,” Neural Comput. Appl., vol. 31, no. 12, pp. 8583–8592, 2019.   
[4] F. Thabtah, S. Hammoud, F. Kamalov, and A. Gonsalves, “Data imbalance in classification: Experimental evaluation,” Inf. Sci., vol. 513, pp. 429–441, 2020.   
[5] D. Jiang, C. Tang, and A. Zhang, “Cluster analysis for gene expression data: A survey,” IEEE Trans. Knowl. Data Eng., vol. 16, no. 11, pp. 1370–1386, Nov. 2004.   
[6] Q. Zou, G. Lin, X. Jiang, X. Liu, and X. Zeng, “Sequence clustering in bioinformatics: An empirical study,” Brief. Bioinf., vol. 21, no. 1, pp. 1–10, 2020.   
[7] A. Rodriguez and A. Laio, “Clustering by fast search and find of density peaks,” Science, vol. 344, no. 6191, pp. 1492–1496, 2014.   
[8] S. Shehata, F. Karray, and M. Kamel, “An efficient concept-based mining model for enhancing text clustering,” IEEE Trans. Knowl. Data Eng., vol. 22, no. 10, pp. 1360–1371, Oct. 2010.   
[9] N. Zhong, Y. Li, and S.-T. Wu, “Effective pattern discovery for text mining,” IEEE Trans. Knowl. Data Eng., vol. 24, no. 1, pp. 30–44, Jan. 2012.   
[10] B. Krawczyk, “Learning from imbalanced data: Open challenges and future directions,” Prog. Artif. Intell., vol. 5, no. 4, pp. 221–232, 2016.   
[11] J. M. Johnson and T. M. Khoshgoftaar, “Survey on deep learning with class imbalance,” J. Big Data, vol. 6, no. 1, pp. 1–54, 2019.   
[12] J. MacQueen, “Classification and analysis of multivariate observations,” in Proc. 5th Berkeley Symp. Math. Statist. Probability, 1967, pp. 281–297.   
[13] J. Wu, “The uniform effect of k-means clustering,” in Advances in K-Means Clustering. Berlin, Germany: Springer, 2012, pp. 17–35.   
[14] J. Liang, L. Bai, C. Dang, and F. Cao, “The -means-type algorithms versus imbalanced data distributions,” IEEE Trans. Fuzzy Syst., vol. 20, no. 4, pp. 728–745, Aug. 2012.   
[15] Y. Yang and J. Jiang, “Hybrid sampling-based clustering ensemble with global and local constitutions,” IEEE Trans. Neural Netw. Learn. Syst., vol. 27, no. 5, pp. 952–965, May 2016.   
[16] Y. Lu, Y.-M. Cheung, and Y. Y. Tang, “Self-adaptive multiprototype-based competitive learning approach: A k-means-type algorithm for imbalanced data clustering,” IEEE Trans. Cybern., vol. 51, no. 3, pp. 1598–1612, Mar. 2021.   
[17] J. Shi and J. Malik, “Normalized cuts and image segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 22, no. 8, pp. 888–905, Aug. 2000.   
[18] X. Chen, X. Xu, J. Z. Huang, and Y. Ye, “TW-k-means: Automated twolevel variable weighting clustering algorithm for multiview data,” IEEE Trans. Knowl. Data Eng., vol. 25, no. 4, pp. 932–944, Apr. 2013.   
[19] F. Gilbert, P. Simonetto, F. Zaidi, F. Jourdan, and R. Bourqui, “Communities and hierarchical structures in dynamic social networks: Analysis and visualization,” Social Netw. Anal. Mining, vol. 1, no. 2, pp. 83–95, 2011.   
[20] D. Xu and Y. Tian, “A comprehensive survey of clustering algorithms,” Ann. Data Sci., vol. 2, no. 2, pp. 165–193, 2015.   
[21] G. Tzortzis and A. Likas, “The minmax k-means clustering algorithm,” Pattern Recognit., vol. 47, no. 7, pp. 2505–2516, 2014.   
[22] S. Chakraborty and S. Das, “k- means clustering with a new divergencebased distance metric: Convergence and performance analysis,” Pattern Recognit. Lett., vol. 100, pp. 67–73, 2017.   
[23] L. Zhang and A. Amini, “Label consistency in overfitted generalized -means,” in Proc. Adv. Neural Informat. Process. Syst., 2021, vol. 34, pp. 7965–7977.   
[24] M. Yedla, S. R. Pathakota, and T. Srinivasa, “Enhancing k-means clustering algorithm with improved initial center,” Int. J. Comput. Sci. Inf. Technol., vol. 1, no. 2, pp. 121–125, 2010.   
[25] M. Erisoglu, N. Calis, and S. Sakallioglu, “A new algorithm for initial cluster centers in k-means algorithm,” Pattern Recognit. Lett., vol. 32, no. 14, pp. 1701–1705, 2011.   
[26] S. Im, M. M. Qaem, B. Moseley, X. Sun, and R. Zhou, “Fast noise removal for k-means clustering,” in Proc. Int. Conf. Artif. Intell. Statist., PMLR, 2020, pp. 456–466.   
[27] L. Jing, M. K. Ng, and J. Z. Huang, “An entropy weighting k-means algorithm for subspace clustering of high-dimensional sparse data,” IEEE Trans. Knowl. Data Eng., vol. 19, no. 8, pp. 1026–1041, Aug. 2007.   
[28] D. Napoleon and S. Pavalakodi, “A new method for dimensionality reduction using k-means clustering algorithm for high dimensional data set,” Int. J. Comput. Appl., vol. 13, no. 7, pp. 41–46, 2011.   
[29] J. Xie, R. Girshick, and A. Farhadi, “Unsupervised deep embedding for clustering analysis,” in Proc. Int. Conf. Mach. Learn., PMLR, 2016, pp. 478–487.   
[30] D. Arthur and S. Vassilvitskii, “k-mean $^ { + + }$ : The advantages of careful seeding,” in Proc. 18th Ann. ACM-SIAM Symp. Discrete Algorithms, 2007, pp. 1027–1035.   
[31] L. Fan, Y. Chai, and Y. Li, “A density-based k-means $^ { + + }$ algorithm for imbalanced datasets clustering,” in Proc. Chin. Intell. Syst. Conf., Springer, 2019, pp. 37–43.   
[32] G. Biau, L. Devroye, and G. Lugosi, “On the performance of clustering in hilbert spaces,” IEEE Trans. Inf. Theory, vol. 54, no. 2, pp. 781–790, Feb. 2008.   
[33] D. Calandriello and L. Rosasco, “Statistical and computational trade-offs in kernel k-means,” in Proc. Adv. Neural Informat. Process. Syst., 2018, vol. 31, pp. 9379–9389.   
[34] Y. Liu, “Refined learning bounds for kernel and approximate -means,” in Proc. Adv. Neural Informat. Process. Syst., 2021, vol. 34, pp. 6142–6154.   
[35] P. L. Bartlett, T. Linder, and G. Lugosi, “The minimax distortion redundancy in empirical quantizer design,” IEEE Trans. Inf. Theory, vol. 44, no. 5, pp. 1802–1813, Sep. 1998.   
[36] A. Georgogiannis, “Robust k-means: A theoretical revisit,” in Proc. Adv. Neural Informat. Process. Syst., 2016, vol. 29, pp. 2891–2899.   
[37] E. S. Laber and L. Murtinho, “On the price of explainability for some clustering problems,” in Proc. Int. Conf. Mach. Learn., PMLR, 2021, pp. 5915–5925.   
[38] S. Li and Y. Liu, “Sharper generalization bounds for clustering,” in Proc. Int. Conf. Mach. Learn., PMLR, 2021, pp. 6392–6402.   
[39] D. J. Foster and A. Rakhlin, “vector contraction for rademacher complexity,” 2019, arXiv:1911.06468.   
[40] P. L. Bartlett and S. Mendelson, “Rademacher and Gaussian complexities: Risk bounds and structural results,” J. Mach. Learn. Res., vol. 3, pp. 463– 482, 2002.   
[41] K. Cao, C. Wei, A. Gaidon, N. Arechiga, and T. Ma, “Learning imbalanced datasets with label-distribution-aware margin loss,” in Proc. Adv. Neural Informat. Process. Syst., 2019, vol. 32, Art. no. 140.   
[42] K.-I. Kojima, “Proceedings of the fifth berkeley symposium on mathematical statistics and probability,” Amer. J. Hum. Genet., vol. 21, no. 4, pp. 407–408, 1969.   
[43] X. Guo, L. Gao, X. Liu, and J. Yin, “Improved deep embedded clustering with local structure preservation,” in Proc. 26th Int. Joint Conf. Artif. Intell., 2017, pp. 1753–1759.   
[44] H. Xiong, J. Wu, and J. Chen, “K-means clustering versus validation measures: A data-distribution perspective,” IEEE Trans. Syst., Man, Cybern., B. (Cybern.), vol. 39, no. 2, pp. 318–331, Apr. 2009.   
[45] Y. Lei, Ü. Dogan, D.-X. Zhou, and M. Kloft, “Data-dependent generalization bounds for multi-class classification,” IEEE Trans. Inf. Theory, vol. 65, no. 5, pp. 2995–3021, May 2019.

Jing Zhang received the BS degree from Sichuan Normal University, Chengdu, China, in 2017, and the MS degree from the National University of Defense Technology, Changsha, China, in 2019. She is currently working toward the PhD degree with the National University of Defense Technology, Changsha, China. Her research interests include machine learning, system science and data mining.

![](images/8940b3434b89c4d86ffe66f8d502a98a8ce8ee04e0231388fa3e852a8177cf6f.jpg)

Hong Tao received the PhD degree from the National University of Defense Technology, Changsha, China, in 2019. She is currently a lecturer with the College of Liberal Arts and Science of the same university. Her research interests include machine learning, system science, and data mining.

![](images/9faace5a04962cabf3eab60854bf688868b75a8018b6926c2476efa3d0f9a6a7.jpg)

![](images/f14eaedba1855ab0b48e2835728d9924a1cb67b6070041314a9c9feaa886193b.jpg)

Chenping Hou (Member, IEEE) received the PhD degree from the National University of Defense Technology, Changsha, China, in 2009. He is currently a full professor with the Department of Systems Science of the same university. He has authored more than 80 peer-reviewed papers in journals and conferences, such as the IEEE Transactions on Pattern Analysis and Machine Intelligence, IEEE Transactions on Neural Networks and Learning Systems, IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), IEEE Transactions on Image Processing, the IJCAI and AAAI. His current research interests include machine learning, data mining, and computer vision.