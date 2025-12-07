
# erfinv 関数の近似

Mike Giles∗

逆誤差関数 **erfinv** は数学ライブラリの標準的な構成要素であり、特に一様乱数を正規乱数へ変換する統計的応用で有用である。本章では、GPU 実行におけるワープ・ダイバージェンス（warp divergence）を大幅に低減することで、**erfinv** を従来よりも顕著に効率よく計算できる新しい近似法を提示する。

## 1 はじめに

$\cos x$、$\sin x$、$e^x$、$\log x$ と同様に、誤差関数

$$
\mathrm{erf}(x)=\frac{2}{\sqrt{\pi}}\int_{0}^{x} e^{-t^{2}} dt
$$

およびその逆関数 $\mathrm{erfinv}(x)$ は、Intel の MKL、AMD の ACML、NVIDIA の CUDA math library といったライブラリの標準的な一部である。

逆誤差関数は、計算ファイナンスにおけるモンテカルロ応用で特に有用である。というのも、誤差関数は正規分布の累積分布関数（CDF）と密接に関係しているからである：

$$
\Phi(x)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x} e^{-t^{2}/2} dt
= \frac{1}{2}+\frac{1}{2} \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)
$$

したがって

$$
\Phi^{-1}(x)=\sqrt{2} \mathrm{erfinv}(2x-1).
$$

もし $x$ が $(0,1)$ の範囲で一様分布する乱数であれば、$y=\Phi^{-1}(x)$ は平均 0・分散 1 の正規乱数となる。擬似乱数の一様分布を擬似正規乱数へ変換する方法としては polar 法や Marsaglia の Ziggurat 法などが通常用いられるが、Sobol 列や格子法から生成される準乱数（quasi-random）一様分布に対しては、低偏差列（low-discrepancy sequences）の有益な性質を保つため、$\Phi^{-1}(x)$ を用いる方法が好ましい [Gla04]。

三角関数と同様に、$\mathrm{erfinv}(x)$ は通常、多項式近似あるいは有理式近似によって実装される [BEJ76, Str68, Wic88]。しかし、これらの近似は従来型 CPU 上での計算コストが低くなるよう設計されている。[BEJ76] の単精度アルゴリズム（CUDA 3.0 の math library で用いられていたもの）は、表1の形式を取る。

### 表1：$y=\mathrm{erfinv}(x)$ を計算する疑似コード（$p_n(t)$ は $t$ の多項式関数）

```text
a = |x|
if a > 0.9375 then
    t = sqrt(log(a))
    y = p1(t) / p2(t)
else if a > 0.75 then
    y = p3(a) / p4(a)
else
    y = p5(a) / p6(a)
end if
if x < 0 then
    y = −y
end if
```

入力 (x) が $(-1,1)$ 上で一様分布しているとすると、このコードで第3分岐（else）が実行される確率は 0.75 であり、$\log(a)$ とその平方根の計算を必要とする高コストな第1分岐が実行される確率は 0.0625 に過ぎない。

しかし、NVIDIA GPU において 1 ワープが 32 スレッドであるとき、全スレッドが第3分岐を取る確率は $0.75^{32}\approx 0.0001$ であり、また第1分岐を取るスレッドが 1 つも存在しない確率は $0.9375^{32}\approx 0.13$ である。したがって大半のワープでは、3 つの分岐それぞれを取るスレッドが少なくとも 1 つは存在し、実行コストは概ね **3 分岐すべての実行コストの合計**に近くなる。

本稿の主目的は、単精度および倍精度の近似を構成し、ワープ・ダイバージェンスを大幅に低減することで $\mathrm{erfinv}(x)$ の実行速度を改善することにある。すなわち、多くのワープが条件分岐コードの主要分岐を 1 つだけ実行するようにする。この手法は他の特殊関数にも容易に適用できる。近似を生成する MATLAB コードは付随ウェブサイトで提供されており、erfinv 近似の CUDA コードおよび速度と精度を示すテストコードも併せて提供される。

新しい近似の効率は CUDA 3.0 の実装との比較で示される。本章の執筆中に CUDA 3.1 がリリースされた。CUDA 3.1 の単精度 erfinv 実装は本章のアイデアを取り入れているが、コードは表5に示すものとやや異なる。倍精度実装は CUDA 3.0 と同一だが、新しい版が開発中である。

これらの近似はもともと、GPU 向け数値計算ルーチンを提供する商用数学ライブラリ [NAG09, BTGTW10] の一部として開発された。同様の理由により、Shaw と Brickman [Sha09] によっても独立に同種の近似が開発されている。

CUDA ソースコード、近似生成に用いた MATLAB コード、評価用テストコードはすべて無償で入手可能である [Gil10]。

---

## 2 新しい erfinv 近似

### 2.1 単精度

新しい単精度近似 $\mathrm{erfinvSP}$ を次で定義する：

$$
\mathrm{erfinvSP}(x)=
\begin{cases}
x p_1(w), & w\le w_1 \quad \text{（中心領域）}\\
x p_2(s), & w_1 < w \quad \text{（尾部領域）}
\end{cases}
$$

ここで $w=-\log(1-x^2)$、$s=\sqrt{w}$ であり、$p_1(w)$ と $p_2(s)$ は 2 つの多項式関数である。この形を選ぶ動機（尾部領域は Strecok [Str68] が提案した形に類似）は次の通りである。

* $\mathrm{erfinv}$ は $x$ の奇関数であり、$x=0$ 近傍で $x$ の奇数冪のテイラー展開を持つ。これは $p_1(w)$ が $w=0$ で通常のテイラー展開を持つことに対応する。
* $x=\pm 1$ 近傍で $\mathrm{erfinv}$ はおおよそ $\pm \sqrt{w}$ に等しい。

$x=\sqrt{1-e^{-w}}$ を用いると、図1左は $0<w<16$（単精度で $|x|<1$ の全範囲に対応）における $\mathrm{erfinv}(x)/x$ を $w$ に対してプロットしたものである。図1右は $4<w<36$ における $\mathrm{erfinv}(x)/x$ を $s\equiv \sqrt{w}$ に対してプロットしたものである。$w\approx 36$ までの拡張は、後に倍精度入力を扱うために必要である。

次数 $n$ の多項式全体を $P_n$ とすると、中心領域に対する標準的な $L_\infty$ 近似は次で定義される：

$$
p_1=\arg\min_{p\in P_n}\max_{w\in (0,w_1)}
\left| p(w)-\frac{\mathrm{erfinv}(x)}{x}\right|.
$$

しかし実際に最小化したいのは相対誤差 $(\mathrm{erfinvSP}(x)-\mathrm{erfinv}(x))/\mathrm{erfinv}(x)$ なので、より望ましいのは次のように $p_1$ を定義することである：

$$
p_1=\arg\min_{p\in P_n}\max_{w\in (0,w_1)}
\left| \frac{x}{\mathrm{erfinv}(x)}
\left(p(w)-\frac{\mathrm{erfinv}(x)}{x}\right)\right|.
$$

この重み付き $L_\infty$ 最小化は MATLAB では実行できないため、代わりに重み付き最小二乗近似を行い、次を最小化する：

$$
\int_{0}^{w_1}\frac{1}{\sqrt{w(w_1-w)}}
\left(
\frac{x}{\mathrm{erfinv}(x)}
\left(p(w)-\frac{\mathrm{erfinv}(x)}{x}\right)
\right)^2 dw.
$$

図1 $\mathrm{erfinv}(x)/x$ を $w$ および $s\equiv\sqrt{w}$ に対してプロット。

#### 表2：単精度 (\mathrm{erfinv}) 近似の 3 つの選択肢

| $w_1$ | $p_1$ 次数 | $p_2$ 次数 | 尾部確率 |
| ----: | -------: | -------: | ---: |
|  5.00 |        8 |        8 | 0.3% |
|  6.25 |        9 |        7 | 0.1% |
| 16.00 |       14 |      n/a |   0% |

この重みは、チェビシェフ多項式が直交する標準的な重みであり、区間端点付近の誤差を制御するために導入される。$p_2$ に対しても同様の構成を用いる。

表2は、分割点 $w_1$ の選択に応じて、相対近似誤差を $10^{-7}$ 未満に抑えるために必要な $p_1$、$p_2$ の次数を示す。第4列は、((-1,1)) に一様分布する入力 (x) が尾部領域に入る確率の概算である。これに 32 を掛けると、32 スレッドからなる CUDA ワープ内で 1 つ以上の入力が尾部に入る確率（すなわち CUDA 実装がダイバージェントになる確率）のおおまかな見積もりになる。

$w_1=5$ を用いると、ダイバージェント・ワープの確率は約 10% だが、中心領域近似は $p_1$ が次数 8 で済むため最も低コストである。一方 $w_1=16$ を用いると尾部領域は存在せず中心領域が全区間を覆うが、$p_1$ は次数 14 となりコストが増える。第2の選択肢は $w_1=6.25$ を用い、このとき $p_1$ は次数 9 が必要であり、ダイバージェント・ワープの確率は 3% に過ぎない。

### 2.2 倍精度

倍精度近似 $\mathrm{erfinvDP}$ も同様に定義されるが、倍精度での (x) の全範囲を覆うため $w\approx 36$ まで拡張する必要があり、最大 2 つの尾部領域を用いる：

$$
\mathrm{erfinvDP}(x)=
\begin{cases}
x p_1(w), & w\le w_1 \quad \text{（中心領域）}\\
x p_2(s), & w_1<w\le w_2 \quad \text{（尾部領域1）}\\
x p_3(s), & w_2<w \quad \text{（尾部領域2）}
\end{cases}
$$

#### 表3：倍精度 $\mathrm{erfinv}$ 近似の 2 つの選択肢

| $w_1$ | $w_2$ | $p_1$ 次数 | $p_2$ 次数 | $p_3$ 次数 | 尾部確率 |
| ----: | ----: | -------: | -------: | -------: | ---: |
|  6.25 |  16.0 |       22 |       18 |       16 | 0.1% |
|  6.25 |  36.0 |       22 |       26 |      n/a | 0.1% |

表3は、分割点 $w_1,w_2$ の選択に応じて、相対近似誤差を $2\times 10^{-16}$ 未満に抑えるために必要な $p_1,p_2,p_3$ の次数を示す。最後の列は、一様分布する入力 (x) が中心領域に入らない確率の概算である。

MATLAB を用いて重み付き最小二乗近似を構成する際には、近似の精度を計算するために解析的誤差関数を倍精度よりも高い精度で評価する必要があり、そのため Symbolic Toolbox の可変精度演算（variable precision arithmetic：拡張精度機能）が必要となる。

### 2.3 浮動小数点誤差解析

本節では有限精度演算による誤差を考える。浮動小数点で $(1-x)(1+x)$ を評価すると、得られる値は

$$
(1-x)(1+x)(1+\varepsilon_1)
$$

に等しい。ここで $\varepsilon_1$ は最大 1 ulp（最下位桁単位）に相当し、単精度ではおよそ (10^{-7})、倍精度では $10^{-16}$ 程度である。これの対数を評価して $w$ を計算すると、近似的に

$$
w+\varepsilon_1+\varepsilon_2
$$

が得られる。ここで $\varepsilon_2$ は $\log$ 関数評価の誤差であり、CUDA の高速単精度関数 $\mathrm{logf}()$ を用いるとおおよそ $5\times 10^{-8}\max(1,3w)$、倍精度関数 $\log()$ を用いると $10^{-16}w$ 程度である。

次に $p_1(w)$ を計算すると、近似的に

$$
p_1(w)\left(
1+(\varepsilon_1+\varepsilon_2)\frac{p_1'(w)}{p_1(w)}+\varepsilon_3
\right)
$$

が得られる。ここで $\varepsilon_3$ は $p_1$ の評価における相対誤差である。最終積 $x p_1(w)$ の相対誤差は近似的に

$$
(\varepsilon_1+\varepsilon_2)\frac{p_1'(w)}{p_1(w)}+\varepsilon_3+\varepsilon_4
$$

となる。ここで $\varepsilon_4$ も $10^{-7}$ オーダーである。

$p_1(w)\approx 1+0.2w$ なので、$\varepsilon_1,\varepsilon_2$ による寄与の合計は単精度で約 $1.5$ ulp、倍精度で約 0.5 ulp である。$\varepsilon_3,\varepsilon_4$ による誤差はそれぞれさらに 0.5 ulp ずつ寄与する。これらに加えて、erfinv 近似自体による相対誤差が約 1 ulp 程度ある。したがって全体誤差は、単精度では最大 4 ulp、倍精度では最大 3 ulp 程度になりうる。

尾部領域では、追加ステップとして $s=\sqrt{w}$ を計算する必要がある。
$\sqrt{w}+\varepsilon \approx \sqrt{w}+\frac{\varepsilon}{2\sqrt{w}}$ なので、計算される $s$ は近似的に

$$
s+\frac{1}{2s}(\varepsilon_1+\varepsilon_2)+\varepsilon_5 s
$$

となる。ここで $\varepsilon_5$ は平方根評価の相対誤差であり、CUDA 実装では単精度で 3 ulp、倍精度で 0.5 ulp 未満である。また $p_2(s)\approx s$ であることに注意すると、最終結果の相対誤差は

$$
\frac{1}{2w}(\varepsilon_1+\varepsilon_2)+\varepsilon_5+\varepsilon_3+\varepsilon_4
$$

となる。単精度では $\varepsilon_2/w$ が約 3 ulp、$\varepsilon_5$ も 3 ulp なので、$p_2$ 近似誤差も含めた全体誤差は約 6 ulp となる。倍精度では $\varepsilon_2/w$ と $\varepsilon_5$ はそれぞれ約 1 ulp と 0.5 ulp であり、全体誤差は 2〜3 ulp である。

---

## 3 性能と精度

表4は、表2および表3にある新近似のうち 1 つ目の性能を、CUDA 3.0 における既存実装と比較して示す。表5は新しい単精度近似のコードを示す。計測時間は 100M 個の値を計算するのに要したミリ秒であり、28 ブロック×512 スレッドを用い、各スレッドが 7000 個の値を計算する。

テスト条件は “uniform” と “constant” の 2 つである。uniform の場合、各ワープに対する入力 (x) を、既存実装に対して 100% のワープ・ダイバージェンスが発生するように一様に散らしている。この最悪ケースは、はじめに述べた「ランダム入力で 87% のダイバージェンスになる」状況よりもわずかに悪い。constant は最良ケースであり、各ワープ内の全スレッドが同じ入力値を用いる。ただし計算中にその値は変化させ、全体としては各条件分岐が適切な割合で実行されるようにしている。

### 表4：CUDA 3.0 による C1060 / C2050 での計算時間（ms、100M 個、SP=単精度、DP=倍精度）

| 条件            | C1060 SP | C1060 DP | C2050 SP | C2050 DP |
| ------------- | -------: | -------: | -------: | -------: |
| uniform, old  |       24 |      479 |       31 |      114 |
| uniform, new  |        8 |      219 |       10 |       49 |
| constant, old |        8 |      123 |       11 |       30 |
| constant, new |        8 |      213 |        9 |       48 |

結果は、2 つのテストケースにおける既存実装の性能が 3 倍以上異なることを示している。これはワープ・ダイバージェンスのペナルティを反映しており、少なくとも 1 スレッドが取った分岐はすべて実行されるため、そのコストは取られた分岐のコスト合計に等しくなる。さらに主要分岐（最も頻繁に取られる分岐）は (\log) 計算を必要としないため、もともと大幅に低コストである点も影響する。

新しい近似のコストは、2 条件間でほとんど変化しない。なぜなら “uniform” の場合でさえ、ほぼすべてのワープが非ダイバージェントだからである。一方で、すべてのワープが (\log) 計算を実行しなければならないため、倍精度では constant 条件において新実装が既存より遅くなる。

精度については、新しい単精度近似の最大誤差（既存の倍精度版との比較）はおよそ $7\times 10^{-7}$ であり、既存単精度版より良い。また新旧倍精度実装の最大差はおよそ $2\times 10^{-15}$ である。

### 表5：単精度実装の CUDA コード

```cpp
__inline__ __device__ float MBG_erfinv(float x)
{
    float w, p;
    w = - __logf((1.0f-x)*(1.0f+x));
    if ( w < 5.000000f ) {
        w = w - 2.500000f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p*w;
        p = -3.5233877e-06f + p*w;
        p = -4.39150654e-06f + p*w;
        p = 0.00021858087f + p*w;
        p = -0.00125372503f + p*w;
        p = -0.00417768164f + p*w;
        p = 0.246640727f + p*w;
        p = 1.50140941f + p*w;
    }
    else {
        w = sqrtf(w) - 3.000000f;
        p = -0.000200214257f;
        p = 0.000100950558f + p*w;
        p = 0.00134934322f + p*w;
        p = -0.00367342844f + p*w;
        p = 0.00573950773f + p*w;
        p = -0.0076224613f + p*w;
        p = 0.00943887047f + p*w;
        p = 1.00167406f + p*w;
        p = 2.83297682f + p*w;
    }
    return p*x;
}
```

---

## 4 結論

本章は、ワープ・ダイバージェンスのコストと、それが従来型 CPU のために開発されたアルゴリズムや近似を再設計することで、場合によっては回避できることを示した。

また本章は、ライブラリ開発者が直面しうるジレンマも示している。新しい倍精度近似が既存のものより優れているかどうかは、利用形態に依存する。ランダム入力に対しては最大 3 倍速いが、各ワープ内の入力がすべて同一、あるいはほとんど変化しない場合には、逆に遅くなることもある。

---

## 参考文献

* [BEJ76] J.M. Blair, C.A. Edwards, and J.H. Johnson. *Rational Chebyshev approximations for the inverse of the error function.* Mathematics of Computation, 30(136):827–830, 1976.
* [BTGTW10] T. Bradley, J. du Toit, M. Giles, R. Tong, and P. Woodhams. *Parallelisation techniques for random number generators.* GPU Computing Gems, Volume 1, Morgan Kaufmann, 2010.
* [Gil10] M.B. Giles. *Approximating the erfinv function (source code).* [http://gpucomputing.net/?q=node/1828](http://gpucomputing.net/?q=node/1828), 2010
* [Gla04] P. Glasserman. *Monte Carlo Methods in Financial Engineering.* Springer, New York, 2004.
* [NAG09] Numerical Algorithms Group. *Numerical Routines for GPUs.* [http://www.nag.co.uk/numeric/GPUs/](http://www.nag.co.uk/numeric/GPUs/), 2009
* [Sha09] W.T. Shaw and N. Brickman. *Differential equations for Monte Carlo recycling and a GPU-optimized Normal quantile.* Working paper, available from arXiv:0901.0638v3, 2009.
* [Str68] A.J. Strecok. *On the calculation of the inverse of the error function.* Mathematics of Computation, 22(101):144–158, 1968.
* [Wic88] M.J. Wichura. *Algorithm AS 241: the percentage points of the Normal distribution.* Applied Statistics, 37(3):477–484, 1988.
