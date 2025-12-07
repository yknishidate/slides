モンテカルロボリュームレンダリングにおける中心的な操作の一つは、透過率推定です。光線に沿ったある区間が与えられたとき、その区間を吸収や散乱を受けることなく通過する光の割合を推定することが目標となります。単純なアプローチは、不偏なレイマーチングを用いて光学的深さτを推定し、その後exp(−τ)を透過率の推定値として用いることです。しかし、この戦略はイェンセンの不等式により、透過率を系統的に過大評価してしまいます。一方、既存の不偏透過率推定器は、高分散に悩まされるか、ランダムな決定に依存するコストがかかるため、SIMDアーキテクチャにはあまり適していません。そこで本稿では、単純なアプローチと比較してバイアスを大幅に低減し、決定論的で低コストなバイアス付き透過率推定器を提案します。層化ジッタリングサンプリングを用いたレイマーチングによって得られる光学的深さの推定値が、ほぼ正規分布に従うことを観察しました。そこで、このような2つの推定値（2つの異なる乱数系列を使用）に基づいて、exp(−τ)の唯一の最小分散不偏推定器（UMVU推定器）を適用します。バイアスは、入力が正規分布に従うという仮定が破られた場合にのみ発生します。さらに、分散を考慮した重点サンプリングスキームを用いて、バイアスと分散を低減します。この基礎となる理論は、光学的深さの任意解析関数の推定に用いることができます。この一般化を用いて、多重重点サンプリング（MIS）重みを推定し、2つの積分器を導入します。1つはバイアス付きMIS重みを用いた不偏MIS、もう1つはより効率的ですがバイアスのあるMISと透過率推定の組み合わせです。

# 1 Introduction

霧、蒸気、煙などの散乱媒体における物理ベースのボリュームレンダリングは困難です。このような媒体が存在する場合、2つの表面相互作用間の光線に沿った放射輝度は一定ではなくなります。代わりに、光の吸収、散乱、および放出を考慮するために、光線に沿った積分を推定する必要があります。これらの効果は、多くの点で表面レンダリングと同様に処理できます。例えば、単方向パストレーサーを使用するなどです。大きな違いは、距離サンプリングと透過率推定がレイトレーシングの代わりとなる点です[Novák et al. 2018]。距離サンプリングは次のパス頂点までの距離を決定し、透過率推定は特定の光線セグメントに沿ってボリュームを通過する光の割合を計算します。不均一な（つまり空間的に変化する）媒体の場合、これらの2つの操作は、レンダラーがボリュームデータと相互作用する主要な方法であるため、ボリュームレンダリング全体のコストに大きく影響します。

本研究における主な貢献は、透過率推定のための効率的かつGPUに優しいソリューションです。光線セグメントに沿った減衰を $\mu(t)$ で表します。ここで𝑡∈[0,𝑡max]⊂Rは光線パラメータです。透過率は

$$
T := T(t_{max}) := \exp(-\tau), where \tau(t_{max}) := \int_0^{t_{max}} \mu(t) dt
$$

積分τは光学的深さと呼ばれる。透過率Tを推定する素朴なアプローチは、τの不偏モンテカルロ推定値Xを計算することである。例えば、一様ジッターサンプリングを用いたレイマーチングを用いる方法などがある[Kettunen et al. 2021; Pauly et al. 2000]。そして、exp(−X)を透過率の推定値として使用する。しかしながら、この戦略は透過率を系統的に過大評価してしまう。なぜなら、イェンセンの不等式により、期待値は以下の関係を満たすからである。

$$
E(\exp(-X)) \geq \exp(E(-X)) = \exp(-\tau) = T
$$

不偏透過率推定器には主に3つの種類があります。レギュラートラッキング[Amanatides and Woo 1987; Szirmay-Kalos et al. 2010]は、τを正確に計算するために高い計算コストがかかります。ヌル衝突法[Coleman 1968; Novák et al. 2014]は、均質化された媒質を確率的に走査し、元の媒質との差に基づいて透過率を推定します。不偏レイマーチング[Georgiev et al. 2019; Kettunen et al. 2021]は、確率的に選択された無制限の数の独立した光学的深さ推定値を組み合わせて、べき級数

$$
\exp(-\tau) = \sum_{j=0}^{\infty} \frac{(-\tau)^j}{j!}
$$

を推定します。正規トラッキングは予測可能ですがコストが高くなります。他の2つのアプローチは、実行される作業量がランダムであるため、GPU上での実行が発散しやすいという問題があります。さらに、ヌル衝突法は比較的高い分散を示します。

同時に、GPU上でのリアルタイムボリュームレンダリングが実現可能になりつつあり、効率的でGPUに適したアルゴリズムへの需要が高まっています[Hofmann et al. 2021, 2023; Schneider 2023]。このような状況では、多少のバイアスは許容範囲内です。しかし、exp(−𝑋)という推定値は最適とは言えないようです。バイアスが常に正であるという事実は、より良い推定値が存在する可能性を示唆しています。バイアスは𝑋の分散とともに増加します。もし何らかの方法でバイアスを正確に推定できれば、それを補正することでより良い透過率推定値を得ることができるでしょう。

これを実現するために、私たちは一つの重要な仮定に基づいています。それは、光学的深さの推定値が正規分布、すなわちガウス密度で分布しているという仮定です（3.1節）。この仮定の下では、exp(−𝜏) の一意な最小分散不偏推定量（UMVU推定量）が知られています。これは、いわゆる一般化ジャックナイフ法を用いて導出されたものです[Gray et al. 1973]（3.2節）。私たちは、光学的深さの2つの不偏で独立同分布（i.i.d.）な推定値X₀、X₁を用いてこの推定量を使用します。つまり、同じ方法で光学的深さτを2回推定しますが、それぞれ異なる乱数を使用します。この選択の結果、私たちの新しいジャックナイフ透過率推定値に対して、驚くほどシンプルな以下の式が得られます（3.3節）。

$$
T \approx \cosh\left(\frac{X_0 - X_1}{2}\right) \exp\left(-\frac{X_0 + X_1}{2}\right).
$$

光学的深さの推定値が正規分布に従うという仮定の下では、この推定値は不偏です。実際には、この仮定は満たされないことが予想され、その結果バイアスが生じます。しかし、一様ジッターサンプリングの代わりに層化ジッターサンプリングを用いることで、このバイアスを小さく抑えることができます[Pauly et al. 2000]（4.1節）。このようにすることで、光学的深さの推定値は独立な確率変数の和となり、中心極限定理[Knight 1999, p. 145]により、サンプル数が増加するにつれて正規分布に近づきます。疎なボリュームが存在する場合でもこの推論の信頼性を高めるために、分散を考慮した[Pantaleoni and Heitz 2017]重点サンプリング手法を提案し（4.2節）、サンプル数を一定に保ちます（4.3節）。

我々の推定の基礎となる理論[Gray et al. 1973]は、exp(−𝜏)の推定に限定されるものではありません。任意の解析関数𝑔 : C → Cに対して𝑔(𝜏)を推定することができます（5.1節）。この機能を利用して、距離サンプリング手法、例えば自由飛行距離サンプリングと等角サンプリングを組み合わせる[Kulla and Fajardo 2012]といった、複数の重点サンプリング（MIS）を実装します。この際によく知られている問題は、自由飛行サンプリングの確率密度が𝑇(𝑡)𝜇(𝑡)であり、正確には既知ではないことです[Miller et al. 2019; Novák et al. 2018]。この問題には2つの方法で対処します（5.2節）。1つは、MIS重みをバイアスのかかった方法で推定する方法です。この場合でも、最終的なMIS推定値は不偏になります[Veach and Guibas 1995]。もう1つは、MIS重み、逆密度、透過率の積を直接推定する方法です。これにより全体的な計算コストは​​削減されますが、わずかなバイアスが生じます。

私たちの透過率推定方法は、既存のバイアスのある代替手法 [Kettunen et al. 2021] と比較してバイアスが大幅に低く、トラック長推定法や比率トラッキング法 [Novák et al. 2014] よりも分散が小さく、バイアスのないレイマーチング法 [Kettunen et al. 2021] よりも計算コストが低くなっています（6.1節）。また、私たちのMIS重み推定方法は、同等のサンプル数において、比率トラッキングに基づく手法 [Miller et al. 2019] と比較して分散が大幅に低減されています（6.2節）。GPU実装のソースコードはプロジェクトのウェブページで公開しています。

# 3. 我々の透過率推定器
ここでは、我々が開発した新しい透過率推定器について説明します。計算コストを低く抑え、かつ予測可能なものにするために、光学的深さの推定値が正規分布に従うという仮定を置くことで、不偏性を犠牲にしています（3.1節）。この仮定の下で、exp(−τ) の最適な推定値は Gray ら [1973] によって導出されています（3.2節）。我々は彼らの結果を応用し、ジャックナイフ法を用いた透過率推定器を導出します（3.3節）。正規分布に近い分布を持つ適切な光学的深さの推定器は、4節で開発します。

## 3.1 私たちのアプローチ
2.2節では、既存の不偏透過率推定器の欠点について議論しました。通常のトラッキングは、決定論的なコストで実行できる唯一の方法ですが、このコストは最悪の場合のコストです。デルタトラッキング、比率トラッキング[Novák et al. 2014]、および不偏レイマーチング[Kettunen et al. 2021]では、μ(t)のサンプル数はランダムで上限がありません。このランダム性により、SIMDアーキテクチャ上での実行が非効率になり、全体的なコストが増加します。私たちは、これは消光プロファイルμ(t)に関する仮定の欠如と、指数関数が超越関数であるという事実の必然的な結果であると推測しています。この問題を無限次元領域における積分として解釈することで、この推測が裏付けられます[Georgiev et al. 2019]。

もしこれが真実であれば、コストが限定された手法に到達するためには、より強い仮定を置く必要があります。私たちは、追加の仮定なしに透過率推定値を効率的に計算できるほど十分にシンプルな、光学的深さ推定値の分布モデルを求めます。正規分布は、後述するようにこの要件を満たします。
同時に、私たちのモデルが真の分布に合理的に近いものであることも望んでいます。統計学において正規分布がこれほど重要である理由は、中心極限定理にあります。比較的弱い仮定の下で、独立な確率変数の和は、確率変数の数が増えるにつれて正規分布に近づきます[Knight 1999, p. 145]。4.1節では、各サンプルを独立にジッターさせる限り、これが光学的深さのレイマーチング推定にも当てはまることを示します。したがって、正規分布は実用的で自然な選択です。これは完璧なモデルではないため、私たちの推定値には依然としてバイアスが残りますが、単純な推定値exp(−𝑋)よりもはるかに低いバイアスを達成するには十分です。

## 3.2 一意な最小分散不偏推定

exp(−𝑋)よりも優れた透過率推定値を構築するには、光学的深さの推定値が複数必要になります。そこで、光学的深さの独立同分布（i.i.d.）推定値を複数計算します。つまり、異なる乱数を用いて同じ推定値を複数回評価します。本アプローチでは、各推定値は未知の平均τと未知の標準偏差を持つ正規分布に従うと仮定します。そして、透過率𝑇 = exp(−τ)の不偏推定値を求めます。さらに、この推定値は最小の分散を持つことが望ましいです。つまり、最小分散不偏推定量（UMVU推定量）を求めたいのです。UMVU推定量は統計学においてよく研究されており、様々な統計量や𝑋の異なる分布について知られています[Voinov and Nikulin 1993]。私たちは、この特定の場合の解を文献[Gray et al. 1973, Example 4]に見つけました。

定理1. 𝑚 ∈ N で𝑚 ≥ 2とし、𝑋0, . . . ,𝑋𝑚−1 を、平均𝜏 ∈ R と標準偏差𝜎 ≥ 0が未知の正規分布のi.i.d.標本とする。標本平均、バイアス付き標本分散、バイアス付き標本標準偏差は次のように定義される。

$$
\bar{X} := \frac{1}{m} \sum_{j=0}^{m-1} X_j, \quad S^2 := \frac{1}{m} \sum_{j=0}^{m-1} X_j^2 - \bar{X}^2, \quad S := \sqrt{S^2}.
$$


$$
K := \Gamma\left(\frac{m-1}{2}\right)\left(\frac{2}{S}\right)^{\frac{m-3}{2}} J_{\frac{m-3}{2}}(S) \exp(-\bar{X}),
$$

<!-- 
where Γ denotes the gamma function and 𝐽 𝑚−3
2
denotes the Bessel
function of order 𝑚−3
2
[Akhmedova and Akhmedov 2019, p. 44]. Then 𝐾
is the UMVU estimate of exp(−𝜏), i.e. E(𝐾) = exp(−𝜏) and if another
function 𝐿(𝑋0, . . . , 𝑋𝑚−1) satisfies E(𝐿(𝑋0, . . . , 𝑋𝑚−1)) = exp(−𝜏),
its variance cannot be lower:
V(𝐾) ≤ V(𝐿(𝑋0, . . . , 𝑋𝑚−1)).
 -->

ここで、$\Gamma$ はガンマ関数、$J_{\frac{m-3}{2}}$ は階数 $\frac{m-3}{2}$ のベッセル関数を表す [Akhmedova and Akhmedov 2019, p. 44]。このとき、$K$ は $\exp(-\tau)$ の UMVU 推定量であり、すなわち $E(K) = \exp(-\tau)$ であり、もし他の関数 $L(X_0, . . . , X_{m-1})$ が $E(L(X_0, . . . , X_{m-1})) = \exp(-\tau)$ を満たすならば、その分散は以下の不等式を満たす。

$$
V(K) \leq V(L(X_0, . . . , X_{m-1})).
$$

<!-- 
As is, Eq. 3 requires us to evaluate the Bessel function, but we will
eliminate the need for that shortly. Other than that, this looks like a
straight-forward solution to our problem with all the properties that
we wanted. The naive estimate exp(−𝑋¯) is a factor in the estimate
𝐾, but there are additional factors depending on the sample standard
deviation 𝑆, which compensate for the bias of this estimate.
 -->

そのままでは、式3はベッセル関数を評価する必要がありますが、すぐにその必要性を排除します。これ以外は、私たちの問題に対する非常に直接的な解決策のように見え、私たちが望んでいたすべての特性を備えています。素朴な推定値 $\exp(-\bar{X})$ は推定値 $K$ の一因ですが、サンプル標準偏差 $S$ に依存する追加の要因があり、この推定値のバイアスを補正しています。

<!-- 
To derive Eq. 3, Gray et al. [1973] apply the “generalized jackknife”
to a biased estimate, thus removing the bias, and then use so-called
Rao-Blackwellization to arrive at the UMVU estimate. The jackknife
in turn is called that because it is “a useful tool in a variety of
situations” [Gray et al. 1973]. Therefore, we refer to the estimate in
Eq. 3 as jackknife estimate.
 -->

式3を導出するために、Grayら[1973]は「一般化ジャックナイフ」をバイアスのある推定値に適用し、バイアスを除去し、その後、いわゆるRao-Blackwellizationを使用してUMVU推定値に到達します。ジャックナイフは「様々な状況で有用なツール」であるため[Gray et al. 1973]、式3の推定値をジャックナイフ推定値と呼びます。

<!-- 
If our assumption of normal-distributed estimates holds true,
the jackknife estimate is unbiased. Fig. 2 demonstrates that it still
achieves lower bias than exp(−𝑋¯) for a variety of other distributions.
In these experiments, the improvement is smallest for single-sided
distributions. The Laplace distribution is remarkable in that the
jackknife estimate slightly underestimates the transmittance. By
Jensen’s inequality, the naive estimate can never do so. In terms of
their standard deviation, both methods perform similarly.
 -->

もし私たちの正規分布に従う推定値の仮定が真であれば、ジャックナイフ推定値は不偏です。図2は、さまざまな他の分布に対しても、$\exp(-\bar{X})$よりも低いバイアスを達成することを示しています。これらの実験では、単一側分布に対する改善が最小です。ラプラス分布は、ジャックナイフ推定値が透過率をわずかに過小評価するという点で注目に値します。イェンセンの不等式によれば、素朴な推定値は決してそうすることはできません。標準偏差の観点からは、両方の方法は同様に機能します。

## 3.3 ジャックナイフ透過率推定

<!-- We now introduce our jackknife transmittance estimator. The main
design decision that we still have to make to complete this part of
our technique is what number of i.i.d. optical depth estimates 𝑚
we should use. We have to use at least 𝑚 = 2 estimates, because
otherwise it is not possible to estimate the sample variance 𝑆
2
. On
the other hand, we would like to use as few independent estimates
as possible, for two reasons: If we use fewer estimates, we can
afford more ray marching steps per estimate. Thus, the reasoning
that these estimates, as sum of many independent samples, are
normal-distributed is more sound and we expect less bias. Besides, 𝑋¯
converges at a rate of 1√
𝑚
, whereas stratified jittered sampling with
𝑁 ∈ N samples has a convergence rate closer to 1
𝑁
(Fig. 5). Therefore,
it is preferable to invest the available sample budget into more
ray marching steps. The data in Table 1 (and in the supplemental)
support this reasoning: The choice 𝑚 = 2 is optimal in terms of bias
and standard deviation. -->

ここで、私たちのジャックナイフ透過率推定器を紹介します。この技術のこの部分を完成させるためにまだ決定しなければならない主な設計上の決定は、i.i.d.光学的深さ推定値𝑚の数を何にするかです。サンプル分散𝑆²を推定することが不可能であるため、少なくとも𝑚 = 2の推定値を使用する必要があります。一方で、可能な限り少ない独立した推定値を使用したいと考えています。理由は2つあります。より少ない推定値を使用すれば、各推定値あたりのレイマーチングステップを増やすことができます。したがって、これらの推定値が多くの独立したサンプルの和として正規分布に従うという推論はより確かなものとなり、バイアスが少なくなると予想されます。さらに、𝑋¯は1/√𝑚の速度で収束しますが、𝑁 ∈ Nサンプルを用いた層化ジッターサンプリングは1/𝑁に近い収束速度を持ちます（図5）。したがって、利用可能なサンプル予算をより多くのレイマーチングステップに投資することが望ましいです。表1（および補足資料）のデータはこの推論を支持しています。バイアスと標準偏差の観点から、𝑚 = 2の選択が最適です。

<!-- This choice also has a third advantage in that it simplifies evaluation of Eq. 3. We know [Akhmedova and Akhmedov 2019, p. 49]
that -->

この選択には、式3の評価を簡素化するという3つ目の利点もあります。私たちは知っています[Akhmedova and Akhmedov 2019, p. 49]、

<!-- 𝐽−
1
2
(𝑆) =
√︂
2
𝜋𝑆
cos(𝑆), Γ

1
2

=
√
𝜋. -->

$$
J_{-\frac{1}{2}}(S) = \sqrt{\frac{2}{\pi S}} \cos(S), \quad \Gamma\left(\frac{1}{2}\right) = \sqrt{\pi}.
$$

<!-- Then for 𝑚 = 2 -->

そのとき、𝑚 = 2の場合

<!-- 𝐾 =
√
𝜋

2
𝑆
−
1
2
√︂
2
𝜋𝑆
cos(𝑆) exp(−𝑋¯) = cos(𝑆) exp(−𝑋¯). -->

$$
K = \sqrt{\pi} \left(\frac{2}{S}\right)^{-\frac{1}{2}} \sqrt{\frac{2}{\pi S}} \cos(S) \exp(-\bar{X}) = \cos(S) \exp(-\bar{X}).
$$

<!-- Furthermore -->

さらに、

<!-- 𝑆 =
√︄
𝑋
2
0
+ 𝑋
2
1
2
−

𝑋0 + 𝑋1
2
2
=
√︄
𝑋
2
0
− 2𝑋0𝑋1 + 𝑋
2
1
4
=
|𝑋0 − 𝑋1 |
2
. -->

$$
S = \sqrt{\frac{X_0^2 + X_1^2}{2} - \left(\frac{X_0 + X_1}{2}\right)^2} = \sqrt{\frac{X_0^2 - 2X_0X_1 + X_1^2}{4}} = \frac{|X_0 - X_1|}{2}.
$$

<!-- With that, we have derived the core of our method. Given two
unbiased i.i.d. estimates of optical depth 𝑋0, 𝑋1, our jackknife transmittance estimate is -->

これで、私たちの方法の核心を導き出しました。光学的深さの2つの不偏なi.i.d.推定値𝑋0、𝑋1が与えられた場合、私たちのジャックナイフ透過率推定値は次のようになります。

<!-- 𝐾 = cos(𝑆) exp(−𝑋¯), where 𝑋¯ =
𝑋0 + 𝑋1
2
, 𝑆 =
|𝑋0 − 𝑋1 |
2
. (4) -->

$$
K = \cos(S) \exp(-\bar{X}), \quad where \quad \bar{X} = \frac{X_0 + X_1}{2}, \quad S = \frac{|X_0 - X_1|}{2}. \quad (4)
$$

<!-- The computational cost is negligible. It is quite surprising that the
cosine shows up in this manner, but even without diving into the
proof of Thm. 1, we can convince ourselves that it is plausible. The
estimate exp(−𝑋¯) systematically overestimates the transmittance.
The factor cos(𝑆) ∈ [−1, 1] reduces this estimate to compensate for
this bias. For small sample standard deviation 𝑆, the correction factor
remains close to 1. For large standard deviations, our transmittance
estimate can become negative. That is an undesirable property, but
we found these negative values to be rare unless the transmittance
is close to zero (see the supplemental document) and clamping them
away causes increased bias. Sec. 5.1 generalizes Eq. 4 and provides
another explanation why cos(𝑆) arises in this formula. -->

計算コストは無視できるほど小さいです。このような形で余弦が現れるのは非常に驚くべきことですが、定理1の証明に深入りしなくても、それがもっともらしいことを納得させることができます。推定値 $\exp(-\bar{X})$ は透過率を体系的に過大評価します。因子 $\cos(S) \in [-1, 1]$ はこの推定値を減少させ、このバイアスを補正します。サンプル標準偏差 $S$ が小さい場合、補正因子は1に近いままです。標準偏差が大きい場合、私たちの透過率推定値は負になる可能性があります。それは望ましくない特性ですが、透過率がゼロに近い場合を除いて、これらの負の値はまれであることがわかりました（補足資料を参照）し、それらをクランプするとバイアスが増加します。5.1節では式4を一般化し、この式に $\cos(S)$ が現れるもう一つの説明を提供します。

<!-- In Appendix A.2, we derive the variance of our estimate for𝑚 = 2
and normal-distributed 𝑋0, 𝑋1 with standard deviation 𝜎 ≥ 0:
 -->

付録A.2では、𝑚 = 2および標準偏差𝜎 ≥ 0を持つ正規分布に従う𝑋0、𝑋1に対する私たちの推定値の分散を導出します。

<!-- V(𝐾) =
1
2
(exp(𝜎
2
) − 1) exp(−2𝜏). -->

$$
V(K) = \frac{1}{2} ( \exp(\sigma^2) - 1 ) \exp(-2\tau).
$$

<!-- The corresponding relative root mean square error (rRMSE) is -->

対応する相対二乗平均平方根誤差（rRMSE）は次のとおりです。

<!-- √︁
V(𝐾)
𝑇
=
√︂
exp(𝜎
2
) − 1
 -->
$$
\frac{\sqrt{V(K)}}{T} = \sqrt{\exp(\sigma^2) - 1}
$$

<!-- In Appendix A.1, we derive the rRMSE of exp(−𝑋¯) for the same two
samples: -->
付録A.1では、同じ2つのサンプルに対する $\exp(-\bar{X})$ の rRMSE を導出します。
<!-- exp(𝜎
2
) − 2 exp 
𝜎
2
4

+ 1 -->
  
$$
\sqrt{\exp(\sigma^2) - 2 \exp\left(\frac{\sigma^2}{4}\right) + 1}
$$

<!-- Fig. 3 shows plots of these two functions and their ratio. We observe
that the error of our method is never worse and lower by a factor of
nearly √
2 for large 𝜎. We note however, that this comparison is not
entirely fair, since the naive estimate could also use 𝑚 = 1 to benefit
from the faster convergence rate of stratified jittered sampling.
 -->
図3は、これら2つの関数とその比率のプロットを示しています。私たちの方法の誤差は決して悪くならず、大きな𝜎に対してはほぼ√2の要因で低くなることがわかります。ただし、この比較は完全に公平ではないことに注意してください。なぜなら、素朴な推定値も𝑚 = 1を使用して、層化ジッターサンプリングのより速い収束率の恩恵を受けることができるからです。

<!-- 4 Our Optical Depth Estimator -->
# 4 Our Optical Depth Estimator

<!-- We now have to design an optical depth estimator that fits the
requirements of our jackknife transmittance estimator well. First,
we elaborate on our reasoning that stratified jittered sampling results in nearly normal-distributed estimates (Sec. 4.1). Since this
reasoning can fail for sparse volumes, we use importance sampling
with a variance-minimizing strategy [Pantaleoni and Heitz 2017]
(Sec. 4.2). Finally, we describe non-trivial aspects of the implementation (Sec. 4.3).
 -->
私たちは今、ジャックナイフ透過率推定器の要件に適した光学的深さ推定器を設計しなければなりません。まず、層化ジッターサンプリングがほぼ正規分布に従う推定値をもたらすという私たちの推論について詳述します（4.1節）。この推論は疎なボリュームでは失敗する可能性があるため、分散最小化戦略を用いた重点サンプリングを使用します[Pantaleoni and Heitz 2017]（4.2節）。最後に、実装の非自明な側面について説明します（4.3節）。

## 4.1 Stratified Jittered versus Uniform Jittered Sampling

<!-- Ray marching with uniform jittered sampling [Pauly et al. 2000]
is a well-established way to attain unbiased estimates of optical
depth [Kettunen et al. 2021]. It depends on a single uniform random
number 𝜉 ∈ [0, 1) and uses it to jitter 𝑁 ∈ N equidistant samples.
Sample 𝑗 ∈ {0, . . . , 𝑁 −1} is placed at𝑡𝑗
:=
𝑗+𝜉
𝑁
𝑡max. Stratified jittered
sampling similarly ensures exactly one sample per stratum [
𝑗
𝑁
,
𝑗+1
𝑁
)
but places each sample independently. Sample 𝑗 is placed at 𝑡𝑗
:=
𝑗+𝜉 𝑗
𝑁
𝑡max where 𝜉0, . . . , 𝜉𝑁 −1 ∈ [0, 1) are uniform and independent.
In both cases, the Monte Carlo estimate of the optical depth is
𝑋 =
𝑡max
𝑁
Í𝑁 −1
𝑗=0
𝜇(𝑡𝑗). -->

均一ジッターサンプリング[Pauly et al. 2000]を用いたレイマーチングは、光学的深さの不偏推定値を得るための確立された方法です[Kettunen et al. 2021]。これは、単一の一様乱数𝜉 ∈ [0, 1)に依存し、これを用いて𝑁 ∈ N個の等間隔サンプルをジッターします。サンプル𝑗 ∈ {0, . . . , 𝑁 −1}は𝑡𝑗 := 𝑗+𝜉/𝑁 𝑡maxに配置されます。層化ジッターサンプリングも同様に各層[𝑗/𝑁, 𝑗+1/𝑁)に正確に1つのサンプルを確保しますが、各サンプルを独立に配置します。サンプル𝑗は𝑡𝑗 := 𝑗+𝜉_𝑗/𝑁 𝑡maxに配置され、ここで𝜉0, . . . , 𝜉_𝑁 −1 ∈ [0, 1)は一様かつ独立です。両方の場合において、光学的深さのモンテカルロ推定値は次のようになります。

<!-- 6 Results
We now evaluate our techniques in comparison to related work.
We start with transmittance estimation (Sec. 6.1), proceed to MIS
(Sec. 6.2) and finally discuss shared limitations (Sec. 6.3). To this end, 
we use four volumes: Bunny cloud (576 × 571 × 437), Intel cloud
(625 × 349 × 566), Disney cloud (993 × 675 × 1224) and explosion
(200×271×229). Our supplemental material provides an interactive
viewer with full sets of results for seven volumes.-->

# 6 結果

私たちは、関連する研究と比較して私たちの技術を評価します。まず透過率推定（6.1節）から始め、MIS（6.2節）に進み、最後に共有される制限事項について議論します（6.3節）。この目的のために、4つのボリュームを使用します：Bunny cloud（576 × 571 × 437）、Intel cloud（625 × 349 × 566）、Disney cloud（993 × 675 × 1224）、および explosion（200×271×229）。補足資料には、7つのボリュームの完全な結果セットを備えたインタラクティブビューアが提供されています。

<!-- Our implementation runs in a Vulkan fragment shader. All techniques use BC4-compressed super-voxel grids with one super-voxel
per 163 voxels. For unbiased ray marching [Kettunen et al. 2021], we
limit the maximal degree of the power series to eight, to avoid the
need for dynamic memory allocation within a shader, we do not use
endpoint matching and we use a variant of Alg. 1 for mean-based
importance sampling, including tight ray segments from Alg. 2. All
other parameters are chosen as proposed by Kettunen et al. [2021],
although we sometimes double the sample count for higher quality.
All transmittance estimators use stochastic texture filtering instead
of trilinear interpolation. The reported timings refer to frames of
resolution 1920 × 1080 rendered on an NVIDIA RTX 5070 Ti with
GPU and memory clocks locked to 2452 MHz and 13801 MHz, respectively -->
私たちの実装はVulkanフラグメントシェーダーで実行されます。すべての技術は、1つのスーパーボクセルあたり163ボクセルのBC4圧縮スーパーボクセルグリッドを使用します。不偏レイマーチング[Kettunen et al. 2021]では、パワーシリーズの最大次数を8に制限し、シェーダー内での動的メモリ割り当ての必要性を回避し、エンドポイントマッチングを使用せず、Alg. 2からのタイトなレイセグメントを含む平均ベースの重要サンプリングのためにAlg. 1のバリアントを使用します。他のすべてのパラメータはKettunenら[2021]によって提案されたものとして選択されますが、より高品質のためにサンプル数を2倍にすることもあります。すべての透過率推定器は、三線形補間の代わりに確率的テクスチャフィルタリングを使用します。報告されたタイミングは、NVIDIA RTX 5070 Tiでレンダリングされた解像度1920 × 1080のフレームを指し、GPUおよびメモリクロックはそれぞれ2452 MHzおよび13801 MHzにロックされています。

## 6.1 Transmittance Estimation

<!-- To evaluate transmittance estimators, we compute, display and analyze the transmittance for primary rays (with the exception of Figs. 1
and 12). That is not how they are typically used in a renderer (Sec. 2)
but makes it easier to assess the bias and variance.
 -->

透過率推定器を評価するために、主線（図1および12を除く）に対して透過率を計算、表示、分析します。これは通常レンダラーで使用される方法ではありません（2節）が、バイアスと分散を評価しやすくします。

<!-- Fig. 8. Results of biased transmittance estimators. We show 1 sample per pixel (spp) full-size images for our main technique (a) and magnified insets for other
techniques (b-d). Additional insets show the bias and standard deviation (computed from 2
18 spp). For each full-size image, we report the total frame time at
1 spp as well as the average bias and standard deviation across all pixels. Our main technique achieves remarkably low bias and a good standard deviation (e). -->
図8. バイアスのある透過率推定器の結果。私たちの主要な技術（a）と他の技術の拡大インセット（b-d）について、ピクセルあたり1サンプル（spp）のフルサイズ画像を示します。追加のインセットは、バイアスと標準偏差を示します（2^18 sppから計算）。各フルサイズ画像について、1 sppでの総フレーム時間とすべてのピクセルにわたる平均バイアスと標準偏差を報告します。私たちの主要な技術は、驚くほど低いバイアスと良好な標準偏差を達成します（e）。
