# Theory of Error Diffusion Algorithm

## 概要

ED法ではNNにおけるニューロンを**興奮性細胞**(excitatory neuron)と**抑制性細胞**(inhibitory neuron)とに区別し、また重みについても正数であるべき(興奮的に働くべき)**興奮性シナプス**(excitatory synapse)と負数であるべき(抑制的に働くべき)**抑制性シナプス**(inhibitory synapse)とに区別する。

学習の際は、残差(教師信号-推定値)が正であれば興奮性細胞からのシナプス(興奮性シナプスとは限らない)を強化し、抑制性細胞からのシナプス(これも抑制性シナプスとは限らない)を弱化する。残差が負である場合はその逆を行う。

## 興奮性/抑制性 細胞/シナプス

生物学的には興奮性シナプスとはシナプス後細胞の活動電位発生を促進させるシナプスであり、抑制性シナプスとは逆に活動電位発生を抑制するシナプスである。更に、興奮性シナプスを構成するシナプス前細胞を興奮性細胞、逆に抑制性シナプスを構成するシナプス前細胞を抑制性細胞と呼ぶ。

理研によれば、大脳皮質の2割を占める抑制性細胞は、抑制性伝達物質(通称GABA)を放出することによって、残り8割の興奮性細胞の活動を制御しているらしい。

本モデルでは生物学的定義とは少し違い、興奮性シナプスは同種の細胞(つまり、興奮性-興奮性あるいは抑制性-抑制性)を繋ぐシナプス、抑制性シナプスは異種の細胞(興奮性-抑制性あるいは抑制性-興奮性)を繋ぐシナプスであると定義し、また興奮性細胞と抑制性細胞は同じ層の中に(ほぼ)同数存在しているとする。

繰り返すが、**興奮性シナプスは「興奮性細胞からのシナプス」という意味ではないし、抑制性シナプスは「抑制性細胞からのシナプス」という意味ではない**。接続先であるシナプス後細胞の興奮性/抑制性を強化するか、弱化するかという区別である。

## 入力層(第$1$層)

入力ベクトル$\boldsymbol{x} \in \mathbb{R}^{D^{\text{in}}}$の各値を興奮性細胞及び抑制性細胞の両方に割り当てる。この時、第$1$層の出力$\boldsymbol{o}^{(1)}$の次元数は$D^{(1)} = 2 D^{\text{in}}$となる。

例えば、$\boldsymbol{x} = (a, b)$であるとすると、入力層は$\boldsymbol{o}^{(1)} = (a, b, a, b)$となる。別に$(a, a, b, b)$としても構わないが、ここで重要なのは、出力$\boldsymbol{o}$は興奮性細胞の状態$\boldsymbol{o}_{\text{(ex)}}^{(1)} = (a, b)$と抑制性細胞の状態$\boldsymbol{o}_{\text{(in)}}^{(1)} = (a, b)$の両方を持つということである。そしてこの性質は後続の層の出力においても(次元数に差あれど)同様である。

## 中間層と出力層(第$l$層)

直前の層の出力が$\boldsymbol{o}^{(l-1)} \in \mathbb{R}^{D^{(l-1)}}$であったとする($l \geq 2$)。第$l$層では、以下の計算式に従って$\boldsymbol{o}^{(l)}$を計算する。

$$
\begin{align}
\boldsymbol{o}^{(l)}
 &= f \left( \boldsymbol{i}^{(l)} \right) \\
 &= f \left( {}^t\boldsymbol{W}^{(l)} \boldsymbol{o}^{(l-1)} + \boldsymbol{b}^{(l)} \right)
\end{align}
$$

ここで、$f (\cdot)$は活性化関数、$\boldsymbol{b}^{(l)} \in \mathbb{R}^{D^{(l+1)}}$はバイアス項である。ここで、後々のために活性化関数は広義単調増加であることにしよう。

## 興奮性/抑制性シナプスによる拘束

前述のように、興奮性シナプスとは同種の細胞を繋ぐシナプスであり、抑制性シナプスとは異種の細胞を繋ぐシナプスである。また、興奮性シナプスはシナプス後細胞の性質を強化する働きを、抑制性シナプスは後細胞の性質を弱化する働きをするべきである。

以上のような要件を満たすため、本モデルは接続行列$\boldsymbol{W}^{(l)}$の各要素に対し、それが興奮性シナプスを意味するのであれば正数に、抑制性シナプスを意味するのであれば負数になるよう強制している。簡易的な数式で表すなら、

$$
\begin{align}
w_{ij}^{(l)}
 & \left\{
    \begin{array}{cc}
        > 0 & \text{if neuron } i,j \text{ is homogenious} \\
        < 0 & \text{if neuron } i,j \text{ is heterogenious}
    \end{array}
 \right.
\end{align}
$$

と書ける。

では、バイアス項$\boldsymbol{b}^{(l)}$にはどのような拘束条件が課せられるべきであろうか。その答えを知るためには、バイアス項が「どちらの細胞から」「どちらのシナプスを通じて」加算される値なのかを明確にする必要がある。

本モデルにおけるバイアス項についての自然な解釈を得るため、直前の層、つまり第$l-1$層には、$D^{(l-1)}$個のニューロンの他に、値が$1$で固定である興奮性/抑制性細胞が$1$つずつ存在していたとしよう。その上で、この$2$つの細胞から第$l$層へのシナプスの重みの和こそが$\boldsymbol{b}^{(l)}$であると解釈する。

より詳細には、値が$1$の興奮性細胞からのシナプス$\boldsymbol{b}_1^{(l)}$と、同じく値が$1$の抑制性細胞からのシナプス$\boldsymbol{b}_2^{(l)}$があって、第$l-1$層から第$l$層へ伝播する際に、細胞の値が等しいことから、

$$
\boldsymbol{b}^{(l)} = \boldsymbol{b}_1^{(l)} + \boldsymbol{b}_2^{(l)}
$$

のように表現されていると解釈する。

以上の解釈に則れば、$\boldsymbol{b}^{(l)}$に課されるべき拘束条件を強いて言うのであれば、$\boldsymbol{b}^{(l)}$を興奮性シナプスと抑制性シナプス$\boldsymbol{b}_{\text{(ex)}}^{(l)}, \boldsymbol{b}_{\text{(in)}}^{(l)} \in \mathbb{R}^{D^{(l+1)}}$に分けた時、

$$
\begin{align}
b_{\text{(ex)}, j}^{(l)} &> 0 \\
b_{\text{(in)}, j}^{(l)} &> 0
\end{align}
$$

を満足することである。要するに$\boldsymbol{b}^{(l)}$そのものへの拘束条件は陽に与えられないのであるが、バイアス項をシナプスと捉えることでそれを更新することに対する妥当性が得られる。

- 実際にはこのような拘束条件は守らなくともそこそこの性能が出せている。

## 出力層の拘束

本モデルにおいて各層のニューロンは興奮性/抑制性細胞としての意味合いを持つことは前述の通りである。つまり、出力層上の細胞は特定事象に興奮する細胞と、それを抑制する細胞の2種しか持ち得ない。そのため、**出力次元数$D^{(L)}$は$1$である必要がある**。

多クラス分類を行いたい場合は、特定のクラスとその他を識別するモデルをクラス数分用意するOne-vs-Restや、2つの特定のクラスを識別するモデルを組み合わせ数分用意するOne-vs-Oneなどを行う必要がある。

- この主張は実験的には真であるとされているが、識別クラス数を$K$とした時、それぞれのクラスに対して興奮性/抑制性細胞を割り当てる、つまり$D^{(L)} = 2K$とした上で、それを縮約する行列(これはもはやシナプスではないが)$\boldsymbol{W}^{\text{out}} \in \mathbb{R}^{2K \times K}$とSoftmax関数を用いればどうにかなりそうな気がしないでもない。

## 更新則

入力層は入力を細胞に割り当てているだけであるから更新しない。更新対象は第$2$層以降のシナプス: $\boldsymbol{W}^{(l)}, \boldsymbol{b}^{(l)}$である。

### 誤差関数

教師信号を$y \in \mathbb{R}$、予測値を$o^{(L)} \in \mathbb{R}$として、

$$
\begin{align}
E
 &= \frac{1}{2} {r}^2 \\
 &= \frac{1}{2} {( y - o^{(L)} )}^2
\end{align}
$$

とする。ここで、$r$は残差である。

### 更新の基本方針

残差$r$が正である、つまり推定値$o^{(L)}$が教師信号$y$より小さい時、興奮性細胞からのシナプスを強化し、抑制性細胞からのシナプスを弱化する。ここで、強化とは「絶対値を大きくすること」を意味し、弱化とは逆に「絶対値を小さくすること」を意味する。また、残差が負である時はその逆を行う。なぜこれにより最適化が行えるのかは不明。

結局増加させるのか減少させるのかを明確にするために、シナプス前細胞と後細胞に着目してまとめる。興奮性シナプスは正、抑制性シナプスが負であることに注意すると、

- $o^{(L)} < y$の時の更新

| シナプス | 興奮性細胞から | 抑制性細胞から |
| ---- | ---- | ---- |
| 興奮性細胞へ | 増加(正の強化) | 増加(負の弱化) |
| 抑制性細胞へ | 減少(負の強化) | 減少(正の弱化) |

- $o^{(L)} > y$の時の更新

| シナプス | 興奮性細胞から | 抑制性細胞から |
| ---- | ---- | ---- |
| 興奮性細胞へ | 減少(正の弱化) | 減少(負の強化) |
| 抑制性細胞へ | 増加(負の弱化) | 増加(正の強化) |

のように表現できる。表より、接続先であるシナプス後細胞の極性によって増加/減少を決めれば良いことが分かる。以降の節では、接続行列$\boldsymbol{W}^{(l)}$及びバイアス$\boldsymbol{b}^{(l)}$の変量$\Delta \boldsymbol{W}^{(l)}, \Delta \boldsymbol{b}^{(l)}$について考える。

### 変量$\Delta \boldsymbol{W}^{(l)}$

端的に言えば勾配である。学習率を$\eta$とすると、各要素における変量の絶対値は、

$$
\begin{align}
\left| \Delta w_{ij}^{(l)} \right| = \eta \left| \frac{\partial E}{\partial w_{ij}^{(l)}} \right| = \eta \left| \frac{\partial E}{\partial o_j^{(l)}} \frac{\partial o_j^{(l)}}{\partial i_j^{(l)}} \frac{\partial i_j^{(l)}}{\partial w_{ij}^{(l)}} \right|
\end{align}
$$

と書ける。

一部の偏微分は容易に計算できる。

$$
\begin{align}
\frac{\partial o_j^{(l)}}{\partial i_j^{(l)}} &= f' \left( i_j^{(l)} \right)  \\
\frac{\partial i_j^{(l)}}{\partial w_{ij}^{(l)}} &= o_i^{(l-1)}
\end{align}
$$

しかし、$\partial E / \partial o_{j}^{(l)}$は層によっては容易に求まらない。もし$l=L$、つまり出力層であるならば、$D^{(L)} = 1$より添字$j$が必要ないことに注意すれば、

$$
\begin{align}
\frac{\partial E}{\partial o_j^{(L)}} = \frac{\partial E}{\partial o^{(L)}} = o^{(L)} - y = r
\end{align}
$$

のように書けるが、出力層ではない場合はそうではない。誤差逆伝播法では更に連鎖律を刻むが、本モデルは出力層の場合のそれに単純に減衰率$\varepsilon$をかけたもので代用する。つまり、$\partial E / \partial o_{j}^{(l)}$は添字$j$の値に関わらずに、

$$
\begin{align}
\frac{\partial E}{\partial o_j^{(l)}} \approx \varepsilon \frac{\partial E}{\partial o^{(L)}} = \varepsilon r
\end{align}
$$

と置く。活性化関数$f$は広義単調増加であり、その微分は非負であることに注意すれば、

$$
\begin{align}
\left| \Delta w_{ij}^{(l)} \right| = \eta \varepsilon \left| r \right| f' \left( i_j^{(l)} \right) \left| o_i^{(l-1)} \right|
\end{align}
$$

と書ける(前述のように出力層の場合は$\varepsilon$をかけないことに注意)。

さて、上の更新式はさらに以下のように簡略に書ける:

$$
\begin{align}
\Delta w_{ij}^{(l)} =
\left\{
    \begin{array}{cc}
    + \eta \varepsilon r \left| o_i^{(l-1)} \right| f' \left( i_j^{(l)} \right) & \text{if neuron } j \text{ is excitatory} \\
    - \eta \varepsilon r \left| o_i^{(l-1)} \right| f' \left( i_j^{(l)} \right) & \text{if neuron } j \text{ is inhibitory} \\
    \end{array}
\right.
\end{align}
$$

### 変量$\boldsymbol{b}^{(l)}$

同様に勾配である。

$$
\begin{align}
\left| \Delta b_{j}^{(l)} \right| = \eta \left|  \frac{\partial E}{\partial b_{j}^{(l)}} \right| = \eta \left| \frac{\partial E}{\partial o_j^{(l)}} \frac{\partial o_j^{(l)}}{\partial i_j^{(l)}} \frac{\partial i_j^{(l)}}{\partial b_{j}^{(l)}} \right|
\end{align}
$$

$\partial E / \partial o_j^{(l)}$と$\partial o_j^{(l)} / \partial i_j^{(l)}$は先程と同様である。また残る微分も

$$
\begin{align}
\frac{\partial i_j^{(l)}}{\partial b_{j}^{(l)}} = 1
\end{align}
$$

と書ける。よって

$$
\begin{align}
\left| \Delta b_{j}^{(l)} \right| &= \eta \varepsilon \left| r \right| f' \left( i_j^{(l)} \right) \\
\end{align}
$$

$$
\begin{align}
\therefore \Delta b_{j}^{(l)} =
\left\{
    \begin{array}{cc}
    + \eta \varepsilon r f' \left( i_j^{(l)} \right) & \text{if neuron } j \text{ is excitatory} \\
    - \eta \varepsilon r f' \left( i_j^{(l)} \right) & \text{if neuron } j \text{ is inhibitory} \\
    \end{array}
\right.
\end{align}
$$
