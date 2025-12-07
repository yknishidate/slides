<!--- gems_erfinv.pdf からの自動抽出コンテンツ（翻訳） --->
# erfinv 関数の近似
Mike Giles∗

逆誤差関数 erfinv は数学ライブラリの標準的なコンポーネントであり、特におよそ一様乱数を正規乱数に変換するための統計アプリケーションで有用です。本章では、ワープダイバージェンス（warp divergence）を大幅に低減することで、GPU 実行において著しく効率的な erfinv 関数の新しい近似法を提示します。

## 1 はじめに
$\cos x$、$\sin x$、$e^x$、$\log x$ と同様に、誤差関数

$$
\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt,
$$

とその逆関数 $\mathrm{erfinv}(x)$ は、Intel の MKL、AMD の ACML、NVIDIA の CUDA 数学ライブラリなどのライブラリの標準的な部分です。
逆誤差関数は、計算ファイナンスにおけるモンテカルロ・アプリケーションにとって特に有用な関数です。なぜなら、誤差関数は正規累積分布関数

$$
\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-t^2/2} dt = \frac{1}{2} + \frac{1}{2} \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)
$$

と密接に関連しており、したがって

$$
\Phi^{-1}(x) = \sqrt{2} \mathrm{erfinv}(2x-1).
$$

となるからです。もし $x$ が範囲 $(0, 1)$ の一様分布に従う乱数であれば、$y = \Phi^{-1}(x)$ は平均 0、分散 1 の正規確率変数となります。
Box-Muller 法（polar method）や Marsaglia の Ziggurat 法などの他の手法も、擬似乱数の一様分布を擬似乱数の正規分布に変換するために通常使用されますが、$\Phi^{-1}(x)$ は Sobol 列や格子法から生成される準乱数（quasi-random）の一様分布に対して推奨されるアプローチです。これは、これらの低食い違い列（low-discrepancy sequences）の有益な特性を保持するためです [Gla04]。

三角関数と同様に、$\mathrm{erfinv}(x)$ は通常、多項式近似または有理近似を使用して実装されます [BEJ76, Str68, Wic88]。しかし、これらの近似は従来の CPU 上で低い計算コストになるように設計されています。[BEJ76] による単精度アルゴリズムは、CUDA 3.0 数学ライブラリで使用されていましたが、Table 1 に示すような形式を持っています。

Table 1: $y = \mathrm{erfinv}(x)$ を計算するための擬似コード。ここで $p_n(t)$ は $t$ の多項式関数を表す。

```
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
    y = -y
end if
```

入力 $x$ が $(-1, 1)$ 上で一様に分布している場合、コード内の 3 番目の分岐を実行する確率は 0.75 であり、$\log(a)$ とその平方根の計算を必要とする高価な最初の分岐を実行する確率はわずか 0.0625 です。

しかし、ワープ内に 32 スレッドを持つ NVIDIA GPU 上では、それらすべてが 3 番目の分岐を通る確率は $0.75^{32} \approx 0.0001$ であり、一方で誰も最初の分岐を通らない確率は $0.9375^{32} \approx 0.13$ です。したがって、ほとんどのワープにおいて、3 つの分岐のそれぞれを通るスレッドが少なくとも 1 つは存在することになり、実行コストは 3 つすべての分岐の実行コストの合計にほぼ等しくなります。

本論文の主な目的は、ワープダイバージェンスを大幅に低減した、すなわち条件付きコードにおいてほとんどのワープが 1 つのメイン分岐のみを実行するような、単精度および倍精度の近似を構築することによって、erfinv(x) の実行速度を向上させることです。使用される手法は、他の特殊関数にも容易に適応できます。近似を生成する MATLAB コードは、erfinv 近似の CUDA コード、およびその速度と精度を実証するテストコードとともに、付属のウェブサイトで提供されています。

新しい近似の効率性は、CUDA 3.0 における実装との比較において実証されます。本章の執筆中に CUDA 3.1 がリリースされました。その単精度 erfinv 実装は本章のアイデアを取り入れていますが、コードは Table 5 に示されているものとはわずかに異なります。倍精度実装は CUDA 3.0 と同じままですが、新しいバージョンが開発中です。

これらの近似は元々、GPU 向けの数値ルーチンを提供する商用数学ライブラリの一部として開発されました [NAG09, BTGTW10]。同様の近似は、同じ理由から Shaw と Brickman によっても独立して開発されています [Sha09]。

CUDA ソースコード、近似を作成するために使用された MATLAB コード、および評価のためのテストコードはすべて無料で利用可能です [Gil10]。

## 2 新しい erfinv 近似
### 2.1 単精度
新しい単精度近似 $\mathrm{erfinvSP}$ は以下のように定義されます。

$$
\mathrm{erfinvSP}(x) = \begin{cases}
x p_1(w), & w \le w_1 \quad \text{central region（中央領域）} \\
x p_2(s), & w_1 < w \quad \text{tail region（裾領域）}
\end{cases}
$$

ここで $w = -\ln(1-x^2)$、$s = \sqrt{w}$ であり、$p_1(w)$ と $p_2(s)$ は 2 つの多項式関数です。この形式の近似（裾領域では Strecok [Str68] によって提案されたものと類似しています）の動機は以下の通りです。
- $\mathrm{erfinv}$ は $x$ の奇関数であり、$x=0$ 付近で $x$ の奇数べき乗のテイラー級数展開を持ちます。これは $p_1(w)$ が $w=0$ で標準的なテイラー級数を持つことに対応します。
- $\mathrm{erfinv}$ は $x=\pm 1$ 付近で $\pm \sqrt{w}$ に近似的に等しくなります。

$x = \sqrt{1 - e^{-w}}$ を用いて、Figure 1 の左側は $0 < w < 16$ に対する $\mathrm{erfinv}(x) / x$ 対 $w$ をプロットしています。これは絶対値が 1 未満の単精度浮動小数点数 $x$ の全範囲に対応します。Figure 1 の右側は $4 < w < 36$ に対する $\mathrm{erfinv}(x) / x$ 対 $s \equiv \sqrt{w}$ をプロットしています。$w \approx 36$ までの拡張は、後で倍精度入力のために必要となります。

次数 $n$ の多項式 $P_n$ を使用すると、中央領域に対する標準的な $L_\infty$ 近似は以下のように定義されます。

$$
p_1 = \arg \min_{p \in P_n} \max_{w \in (0, w_1)} \left| \frac{p(w) - \mathrm{erfinv}(x)}{x} \right|.
$$

しかし、我々が本当に最小化したいのは相対誤差 $(\mathrm{erfinvSP}(x) - \mathrm{erfinv}(x))/ \mathrm{erfinv}(x)$ なので、$p_1$ を以下のように定義する方が良いでしょう。

$$
p_1 = \arg \min_{p \in P_n} \max_{w \in (0, w_1)} \left| \frac{x}{\mathrm{erfinv}(x)} \left( \frac{p(w) - \mathrm{erfinv}(x)}{x} \right) \right|.
$$

この重み付き $L_\infty$ 最小化は MATLAB を使用して行うことができないため、$p_1$ は代わりに重み付き最小二乗最小化を実行することによって近似されます。

$$
\int_0^{w_1} \frac{1}{\sqrt{w(w_1 - w)}} \left( \frac{x}{\mathrm{erfinv}(x)} \left( \frac{p(w) - \mathrm{erfinv}(x)}{x} \right) \right)^2 dw.
$$

Figure 1: $\mathrm{erfinv}(x) / x$ versus $w$ and $s \equiv \sqrt{w}$ のプロット。

Table 2: $\mathrm{erfinv}$ の単精度近似のための 3 つの代替案

| $w_1$ | $p_1$ 次数 | $p_2$ 次数 | 裾確率 (tail prob.) |
| :--- | :--- | :--- | :--- |
| 5.00 | 8 | 8 | 0.3% |
| 6.25 | 9 | 7 | 0.1% |
| 16.00 | 14 | n/a | 0% |

重み付けはチェビシェフ多項式が直交する標準的なものであり、区間の両端付近での誤差を制御するために導入されています。$p_2$ にも同様の構成が使用されます。
Table 2 は、分割点 $w_1$ の選択に応じて、相対近似誤差を $10^{-7}$ 未満に減らすために必要な $p_1$ と $p_2$ の多項式の次数を示しています。4 列目は、$(-1, 1)$ 上で一様に分布するランダム入力 $x$ が裾領域（tail region）にあるおおよその確率を示しています。これに 32 を掛けると、32 スレッドの CUDA ワープ内の 1 つ以上の入力が裾領域にあるおおよその確率、したがって CUDA 実装が分岐ワープを持つおおよその確率が得られます。

$w_1 = 5$ を使用すると、分岐ワープの確率は約 10% ですが、$p_1$ に次数 8 の多項式しか必要としないため、中央領域の近似コストは最小になります。一方、$w_1 = 16$ を使用すると、裾領域はありません。中央領域が区間全体をカバーします。しかし、$p_1$ は次数 14 になるため、コストは増加します。2 番目のオプションは $w_1 = 6.25$ を使用しており、この場合 $p_1$ は次数 9 である必要があります。この場合、分岐ワープの確率はわずか 3% です。

### 2.2 倍精度
倍精度近似 $\mathrm{erfinvDP}$ も同様に定義されますが、$x$ の完全な倍精度範囲をカバーするために $w \approx 36$ まで拡張する必要があるため、最大 2 つの裾領域を持つように定義されます。

$$
\mathrm{erfinvDP}(x) = \begin{cases}
x p_1(w), & w \le w_1 \quad \text{central region（中央領域）} \\
x p_2(s), & w_1 < w \le w_2 \quad \text{tail region 1（裾領域 1）} \\
x p_3(s), & w_2 < w \quad \text{tail region 2（裾領域 2）}
\end{cases}
$$

Table 3 は、分割点 $w_1$ と $w_2$ の選択に応じて、相対近似誤差を $2 \times 10^{-16}$ 未満に減らすために必要な $p_1$、$p_2$、$p_3$ の多項式の次数を示しています。最後の列は再び、一様分布するランダム入力 $x$ が中央領域にないおおよその確率を示しています。

Table 3: $\mathrm{erfinv}$ の倍精度近似のための 2 つの代替案

| $w_1$ | $w_2$ | $p_1$ 次数 | $p_2$ 次数 | $p_3$ 次数 | 裾確率 (tail prob.) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 6.25 | 16.0 | 22 | 18 | 16 | 0.1% |
| 6.25 | 36.0 | 22 | 26 | n/a | 0.1% |

MATLAB を使用して重み付き最小二乗近似を構築する際、近似の精度を計算するために、解析的な誤差関数を倍精度よりも高い精度で評価する必要があり、可変精度演算（Symbolic Toolbox の拡張精度機能）が必要となります。

### 2.3 浮動小数点誤差解析
このセクションでは、有限精度演算による誤差について見ていきます。
$(1 - x)(1+x)$ の浮動小数点評価は、以下の値になります。

$$
(1- x)(1+x)(1 + \varepsilon_1)
$$

ここで $\varepsilon_1$ は最大 1 ulp（unit of least precision：最小精度の単位）に対応し、単精度ではおよそ $10^{-7}$、倍精度では $10^{-16}$ です。これの対数を評価して $w$ を計算すると、およそ以下のようになります。

$$
w + \varepsilon_1 + \varepsilon_2
$$

ここで $\varepsilon_2$ は対数関数の評価における誤差であり、CUDA の高速単精度関数 `logf()` を使用する場合はおよそ $5 \times 10^{-8} \max(1, 3w)$、倍精度関数 `log()` を使用する場合は $10^{-16}w$ です。
$p_1(w)$ を計算すると、およそ以下のようになります。

$$
p_1(w) \left( 1 + (\varepsilon_1 + \varepsilon_2) \frac{p'_1(w)}{p_1(w)} + \varepsilon_3 \right)
$$

ここで $\varepsilon_3$ は $p_1$ の評価における相対誤差です。最終的な積 $x p_1(w)$ の相対誤差は、およそ以下のようになります。

$$
(\varepsilon_1 + \varepsilon_2) \frac{p'_1(w)}{p_1(w)} + \varepsilon_3 + \varepsilon_4,
$$

ここで $\varepsilon_4$ もまた $10^{-7}$ のオーダーです。
$p_1(w) \approx 1 + 0.2w$ であるため、$\varepsilon_1$ と $\varepsilon_2$ による寄与の合計は、単精度で約 1.5 ulp、倍精度で約 0.5 ulp です。
$\varepsilon_3$ と $\varepsilon_4$ による誤差はそれぞれさらに 0.5 ulp 寄与します。これらは $\mathrm{erfinv}$ の近似によるおよそ 1 ulp の相対誤差に追加されます。したがって、全体的な誤差は単精度で最大 4 ulp、倍精度で最大 3 ulp になる可能性があります。

裾領域では、$s = \sqrt{w}$ を計算する追加のステップがあります。

$$
\sqrt{w + \varepsilon} \approx \sqrt{w} + \frac{\varepsilon}{2\sqrt{w}}
$$

であるため、$s$ に対して計算される値はおよそ

$$
s + \frac{1}{2s}(\varepsilon_1 + \varepsilon_2) + \varepsilon_5 s
$$

となります。ここで $\varepsilon_5$ は平方根自体の評価における相対誤差であり、CUDA 実装では単精度で 3 ulp、倍精度で 0.5 ulp 未満です。また $p_2(s) \approx s$ であることに注意すると、最終結果の相対誤差は

$$
\frac{1}{2w} (\varepsilon_1 + \varepsilon_2) + \varepsilon_5 + \varepsilon_3 + \varepsilon_4,
$$

となります。
単精度では、$\varepsilon_2/w$ は約 3 ulp、$\varepsilon_5$ も 3 ulp なので、全体的な誤差は $p_2$ 近似誤差を含めて約 6 ulp になります。倍精度では、$\varepsilon_2/w$ と $\varepsilon_5$ はそれぞれ約 1 ulp と 0.5 ulp なので、全体的な誤差は 2 〜 3 ulp です。

## 3 性能と精度
Table 4 は、Table 2 と 3 の新しい近似の最初のものの性能を、CUDA 3.0 の既存のものと比較して示しています。Table 5 は新しい単精度近似のコードを示しています。時間は 1 億個（100M）の値を計算するためのもので、単位はミリ秒です。28 ブロック、512 スレッドを使用し、各スレッドが 7000 個の値を計算しています。

テストには「uniform（一様）」と「constant（定数）」の 2 つの条件があります。
一様の場合、各ワープの入力 $x$ は、既存の実装で 100% のワープダイバージェンスが保証されるような方法で一様に分散されます。この最悪のシナリオは、はじめに述べたようにランダムな入力データから生じる 87% のダイバージェンスよりもわずかに悪いです。定数の場合は、各ワープ内のすべてのスレッドが同じ入力値を使用する最良のシナリオを表していますが、この値は計算中に変化するため、全体として各条件分岐は適切な割合で実行されます。

Table 4: C1060 および C2050 上の CUDA 3.0 を使用して 1 億個の単精度 (SP) および倍精度 (DP) 値を計算する時間（ミリ秒）

| | C1060 | | C2050 | |
| :--- | :--- | :--- | :--- | :--- |
| 時間 (ms) | SP | DP | SP | DP |
| uniform, old | 24 | 479 | 31 | 114 |
| uniform, new | 8 | 219 | 10 | 49 |
| constant, old | 8 | 123 | 11 | 30 |
| constant, new | 8 | 213 | 9 | 48 |

結果は、2 つのテストケースにおいて既存の実装の性能に 3 倍以上の差があることを示しています。これは、少なくとも 1 つのスレッドが通るすべての分岐の合計に等しいコストがかかるというワープダイバージェンスのペナルティと、メイン分岐（最も頻繁に通る分岐）は対数計算を必要としないため実行コストが大幅に低いという事実を反映しています。

新しい近似のコストは、2 つのケースでほとんど変わりません。なぜなら、「一様」の場合でもほとんどのワープは分岐しないからです。一方で、すべてのワープが対数計算を実行する必要があるため、倍精度では「定数」の場合に新しい実装が既存のものより遅くなります。

精度に関しては、新しい単精度近似の最大誤差は、既存の倍精度バージョンと比較して約 $7 \times 10^{-7}$ であり、これは既存の単精度バージョンよりも優れています。また、新しい倍精度実装と既存の倍精度実装の最大差は約 $2 \times 10^{-15}$ です。

Table 5: 単精度実装のための CUDA コード
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

## 4 結論
本章では、ワープダイバージェンスのコストと、従来の CPU 向けに開発されたアルゴリズムや近似を再設計することによって、いかにしてそれを回避できるかを示しました。

また、ライブラリ開発者が直面するジレンマも示しています。新しい倍精度近似が既存のものより優れていると見なされるかどうかは、それらがどのように使用される可能性が高いかに依存します。ランダムな入力に対しては最大 3 倍高速ですが、各ワープ内の入力がすべて同一であるか、ほとんど変化しない場合には遅くなる可能性もあります。

## 参考文献
[BEJ76] J.M. Blair, C.A. Edwards, and J.H. Johnson. Rational Chebyshev approximations for the inverse of the error function. Mathematics of Computation , 30(136):827–830, 1976.
[BTGTW10] T. Bradley, J. du Toit, M. Giles, R. Tong, and P. Woodhams. Parallelisation techniques for random number generators. GPU Computing Gems, Volume 1 , Morgan Kaufmann, 2010.
[Gil10] M.B. Giles. Approximating the erﬁnv function (source code). http://gpucomputing.net/?q=node/1828, 2010
[Gla04] P. Glasserman. Monte Carlo Methods in Financial Engineering. Springer, New York, 2004.
[NAG09] Numerical Algorithms Group. Numerical Routines for GPUs. http://www.nag.co.uk/numeric/GPUs/, 2009
[Sha09] W.T. Shaw and N. Brickman. Diﬀerential equations for Monte Carlo recycling and a GPU-optimized Normal quantile. Working paper, available from arXiv:0901.0638v3, 2009.
[Str68] A.J. Strecok. On the calculation of the inverse of the error function. Mathematics of Computation , 22(101):144–158, 1968.
[Wic88] M.J. Wichura. Algorithm AS 241: the percentage points of the Normal distribution. Applied Statistics , 37(3):477–484, 1988.
