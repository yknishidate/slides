---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: ./images/representative_image.jpg
# some information about your slides (markdown enabled)
title: Jackknife Transmittance

# apply UnoCSS classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: fade-out
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
hideInToc: true
---

# Jackknife Transmittance and<br>MIS Weight Estimation

Christoph Peters, Delft University of Technology <br>
ACM Transactions on Graphics, 2025

<!--

-->

---

<Toc maxDepth="1"/>

---

# 前提：UnbiasedとConsistentの違い

///

### 不偏推定量(Unbiased Estimator)

$$
E[\hat{\theta}] - \theta = \epsilon
$$

> 推定量 $\hat{\theta}$ の期待値と真値 $\theta$ とのずれ $\epsilon$ を、推定量の偏り(bias)と呼ぶ。<br>
> $\epsilon$ がゼロの場合は、$\hat{\theta}$は不偏推定量(Unbiased Estimator)と呼ばれる。

<br>

### 一致推定量(Consistent Estimator)

$$
\lim_{N \to \infty} (\hat{\theta} - \theta) = 0
$$

> サンプル数を無限に増やしたときに、推定量が完全に真値に一致する場合、一致性(Consistency)を持つと言う。

---

### UnbiasedとConsistentの例

$$
\theta =\int_0^1 f(x) dx
$$

上の真値 $\theta$ を推定するために、$N$ 個のサンプル $x_i$ を用いて以下の推定量 $\hat{\theta}$ を考える。

||Unbiased|Biased|
|--|--|--|
|Consistent|$\hat{\theta}=\frac{1}{N} \Sigma_{i=1}^{N} f(x_i)$|$\hat{\theta}=\frac{1}{N} (\Sigma_{i=1}^{N} f(x_i) + A)$<br>$(\epsilon=\frac{A}{N})$|
|Inconsistent|$\hat{\theta}=\frac{1}{2}(f(x_1)+f(x_N))$|$\hat{\theta}=\frac{1}{N} \Sigma_{i=1}^{N} f(x_i)+ A$<br>$(\epsilon=A)$|

---

# ボリュームレンダリング

///

ボリュームレンダリングで重要な2つの操作

| 操作                           | 内容              |
| ---------------------------- | --------------- |
| **距離（自由行程）サンプリング**        | 散乱・吸収が起きる位置を決める |
| **透過率推定** | 光がどれだけ届くかを求める   |

![](https://rayspace.xyz/CG/contents/VLTE1/fig_volume_rendering_equation.svgz)

---

位置ごとの消散係数（extinction coefficient）を $\mu(t)$ とすると、透過率は以下の式で定義される。

$$
T = \exp(-\tau), \quad \text{where} \quad \tau := \int_0^{t_{max}} \mu(t) dt\\
$$

ここで、$\tau$ は光学的深度 (optical depth) と呼ばれる。


研究の貢献： 透過率推定のための効率的かつGPUに優しい手法

---

## 素朴な推定量と問題点

1. 光学的深度 $\tau$ の不偏推定量 $X$ を計算する<br>
   例：一様ジッターサンプリングを用いたレイマーチング

2. 透過率の推定量として $\exp(-X)$ を使用する


ただ、この戦略は Jensenの不等式 により、透過率を過大評価してしまう（バイアスがある）。

$$
E(\exp(-X)) \geq \exp(E(-X)) = \exp(-\tau) = T
$$

> 一般に、推定量 $\hat{\theta}$ が不偏であっても、非線形関数 $g()$ を通した結果 $g(\hat{\theta})$ は $g(\theta)$ の不偏推定量とは限らない。

---

## 既存の不偏透過率推定器の問題点

既存の不偏透過率推定器には主に3つの種類がある。

| 手法                                  | 概要                    | 問題点               |
| ----------------------------------- | --------------------- | ----------------- |
| **Regular tracking**                | 体積を正確にステップし、積分を厳密に求める | コストが高い            |
| **Null-collision / Delta tracking** | 均質媒質を仮定して確率的に通過距離を決定  | コストがランダム（GPUに不向き） |
| **Unbiased ray marching**           | 無限級数展開に基づく理論的に正確な推定   | 計算負荷が大きく、分岐もランダム  |

どれもGPU上でのボリュームレンダリングにはあまり適していない。

---

## 手法概要：ジャックナイフ透過率推定値

光学的深度の推定値がほぼ正規分布するという仮定の下では、$\exp(-\tau)$ の一意な最小分散不偏推定量 (UMVU推定量) が知られている。（一般化ジャックナイフ法を用いて導出される）

> 最小分散不偏推定量 (UMVU推定量) : 不偏であり、分散が最小となる推定量。つまり最強な推定量のこと。

特に、光学的深度の2つの不偏で独立同分布 (i.i.d.) な推定値 $X_0, X_1$ を用いることにすると、以下の透過率推定式が得られる。

$$
T \approx \cos \left(\frac{X_0 - X_1}{
2}\right) \exp\left(-\frac{X_0 + X_1}{2}\right)
$$

* 正規分布の仮定を満たす限りはUnbiased
* $\cos$項がバイアスの補正をしてくれる
* 2回の光学的深度推定（$X_0, X_1$）だけで済むのでGPU向き

---

## MIS重み推定への応用

提案した理論は $\exp(-\tau)$ の推定に限定されない

→ 任意の解析関数 $g$ に対して $g(\tau)$ を推定することができる

この機能を利用して距離サンプリングにおけるMISを実装したりもできるらしいが、おまけなので割愛

---
layout: image-right
image: ./images/graph_biased.png
backgroundSize: contain
---

# 実験：Jackknife法

///

通常のJackknife法について補足する。

正規分布 $N(0,1)$ に従う確率変数 $X$ と、非線形関数の一例として $g(x) = \exp(-x)$ を考える。

正規分布の平均を真値 $\theta = 0$ とし、$N$ 個のサンプル $x_i$ を平均すると不偏推定量 $\hat{\theta}$ が計算できる。

ここで、$\exp(-\theta) = \exp(0) = 1$であるが、$\exp(-\hat{\theta})$ を $N$ ごとにプロットすると、右のようになってしまう。

---
layout: image-right
image: ./images/jackknife2.png
backgroundSize: contain
---

<!-- 明らかに $N$ に依存するバイアスが存在する。ちなみに$N\to \infty$で真値に収束するので、一致性は満たす。 -->

Jackknife法では、Nを変えたときの推定値からバイアスの傾向を推定し、$N=\infty$ のときの推定値を予測する。

具体的には、$N$ 個のサンプルのうち1つを除いた $N-1$ 個の部分サンプルから不偏推定量 $\hat{\theta}_{N-1}$ を計算し、以下の式でJackknife推定量 $\hat{\theta}_{\infty}$ を求める。

$$
\hat{\theta}_{\infty} = N \hat{\theta}_{N} - (N-1) \hat{\theta}_{N-1}
$$

これを実装した結果が右のグラフで、バイアスがうまく打ち消されていることが分かる。


---

# ジャックナイフ透過率推定値

///

まず、$x$ が正規分布に従うとき、 $\exp(-x)$ の UMVU推定量が以下の式で知られている。(Gray et al. 1973)

> 定理1. <br>
> $m \in N$ で $m \geq 2$ とし、$X_0, . . . ,X_{m-1}$ を、平均 $\tau \in R$ と標準偏差 $\sigma \geq 0$ が未知の正規分布の i.i.d. 標本とする。標本平均、バイアス付き標本分散、バイアス付き標本標準偏差は次のように定義される。
> $$
> \bar{X} := \frac{1}{m} \sum_{j=
> 0}^{m-1} X_j, \quad S^2 := \frac{1}{m} \sum_{j=0}^{m-1} X_j^2 - \bar{X}^2, \quad S := \sqrt{S^2}.
> $$
> このとき、$\exp(-\tau)$ の UMVU推定量は以下の式で与えられる。
> $$
> K := \Gamma\left(\frac{m-1}{2}\right)\left(\frac{2}{S}\right)^{\frac{m-3}{2}} J_{\frac{m-3}{2}}(S) \exp(-\bar{X}),
> $$
> ここで、$\Gamma$ はガンマ関数、$J_{\nu}$ は第一種ベッセル関数。

---

定理1の導出は論文では省略されているが、考え方としては以下の2ステップ。

1. 素朴な推定量 $\exp(-\bar{X})$ のバイアスを一般化ジャックナイフ法で除去
2. Rao-Blackwellの定理を用いて分散を最小化

これによって、不偏かつ最小分散な推定量、つまりUMVU推定量が得られる。

---

## 補足：Rao-Blackwellの定理

<br>

> Rao-Blackwellの定理 : <br>
> 任意の推定量 $\hat{\theta}$ に対して、十分統計量 $T$ を用いて条件付き期待値 $E[\hat{\theta}|T]$ を計算すると、分散が小さくなる不偏推定量が得られる。

- 十分統計量 : 観測データ $X$ からパラメータ $\theta$ に関する全ての情報を保持する統計量 $S(X)$ のこと。つまり、$X$ を $S(X)$ に圧縮しても $\theta$ に関する情報は失われない。

- 条件付き期待値 $E[X|Y]$ : 確率変数 $Y$ の値が分かっているときの確率変数 $X$ の期待値。

---

コイン投げの例で説明する。

- コインを $n$ 回投げて、観測データ $X = (X_0, ..., X_{n-1})$ を得る。$X_i = 1$ は表、$X_i = 0$ は裏を表す。
- 表の確率を $\theta$ とし、これを推定したい。

自然な推定量は、観測データの平均 $\hat{\theta} = \frac{1}{n} \sum_{i=0}^{n-1} X_i$。

これはすでに UMVU 推定量なので、あえて変な推定量を考えてみる：

$$
T(X) = X_0
$$

上は「最初の1回だけ見る」推定量で、

- $E[T(X)] = \theta$ なので不偏
- ただ、分散が大きい

---

ここに、$\theta$ に関する十分統計量である、表の総数 $S(X) = s = \sum_{i=0}^{n-1} X_i$ を導入する。

> 観測データ全体 $X$ から、表の総数 $S(X)$ に圧縮しても、$\theta$ に関する情報は失われないので、$S(X)$ は十分統計量。

Rao-Blackwellの定理により、条件付き期待値

$$
E[T(X)∣S(X)] = E[X_0 ∣ s] = \frac{s}{n}
$$

これはちょうど標本平均 $\hat{\theta}$ に等しく、元の推定量 $T(X)$ よりも分散が小さい。

---
layout: image-right
image: ./images/fig2.png
backgroundSize: contain
---

## Grayらの式の性能評価

実際、Grayらによる式は $\exp(-\bar{X})$ よりもバイアスが小さいことが右の図から分かる。

とくに、元の分布が正規分布である場合は不偏になっている。

ちなみに、Laplace分布の場合は推定量の平均が真値よりもマイナスになっているが、これはナイーブな推定量では起こりえないので特徴的。

---

# m=2の場合の簡略式

///

光学的深度の推定値の数を $m=2$ とすると、Grayらの式

> $$
> K := \Gamma\left(\frac{m-1}{2}\right)\left(\frac{2}{S}\right)^{\frac{m-3}{2}} J_{\frac{m-3}{2}}(S) \exp(-\bar{X}),
> $$

は以下のように簡略化される。

$$
\Gamma\left(\frac{1}{2}\right) = \sqrt{\pi}, \quad
J_{-\frac{1}{2}}(S) = \sqrt{\frac{2}{\pi S}} \cos(S).
$$

$$
K = \sqrt{\pi} \left(\frac{2}{S}\right)^{-\frac{1}{2}} \sqrt{\frac{2}{\pi S}} \cos(S) \exp(-\bar{X}) = \cos(S) \exp(-\bar{X}).
$$

---

さらに、標準偏差 $S$ は、2つの推定値 $X_0, X_1$ を用いた以下であり、

$$
S = \sqrt{\frac{X_0^2 + X_1^2}{2} - \left(\frac{X_0 + X_1}{2}\right)^2} = \sqrt{\frac{X_0^2 - 2X_0X_1 + X_1^2}{4}} = \frac{|X_0 - X_1|}{2}.
$$

最終的に、以下の式が得られる。

$$
K = \cos(S) \exp(-\bar{X}), \quad where \quad \bar{X} = \frac{X_0 + X_1}{2}, \quad S = \frac{|X_0 - X_1|}{2}. \quad (4)
$$

計算コストは無視できるほど小さい！

---
layout: image-right
image: ./images/jitter.png
backgroundSize: contain
---

# 光学的深度の推定

///

均一ジッターサンプリングを用いたレイマーチングで、光学的深度 $\tau$ の不偏推定量を計算できるが、分布が正規分布に近いとは限らない。

層化ジッターサンプリングを用いると、独立な確率変数の和となり、中心極限定理により正規分布に近づく。

---
layout: image
image: ./images/fig8.png
backgroundSize: contain
---

---

# 制限

///

スパースすぎるボリュームに対しては、光学的深度の推定値が正規分布に従わない可能性がある。

