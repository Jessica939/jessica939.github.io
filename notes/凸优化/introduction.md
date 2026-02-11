# Introduction
## Optimization problems

$$
\begin{aligned}
& \underset{x}{\text{minimize}} & & f(x) \\
& \text{subject to} & & x \in \Omega
\end{aligned}
\quad \text{or} \quad \min_{x \in \Omega} f(x)
$$

* $f : \mathbb{R}^n \to \mathbb{R}$: **objective function**
* $x = (x_1, x_2, \dots, x_n) \in \mathbb{R}^n$: **optimization/decision variables**
* $\Omega \subset \mathbb{R}^n$: **feasible set** or **constraint set**
    * $x$ is called **feasible** if $x \in \Omega$ and **infeasible** if $x \notin \Omega$.

Maximizing $f$ is equivalent to minimizing $-f$; will focus on minimization.

The problem is **unconstrained** if $\Omega = \mathbb{R}^n$ and **constrained** if $\Omega \neq \mathbb{R}^n$.

$\Omega$ is often specified by **constraint functions**,

$$
\begin{aligned}
& \min_{x} & & f(x) \\
& \text{s.t.} & & g_i(x) \le 0, \quad i = 1, 2, \dots, m \\
& & & h_i(x) = 0, \quad i = 1, 2, \dots, k
\end{aligned}
$$
![](../../images/Pasted%20image%2020260116103003.png)

## Data fitting

Recall Hooke’s law in physics,
$$ F = -k(x - x_0) = -kx + b, \quad \text{where } b = kx_0 $$

* $F$: force
* $k$: spring constant
* $x$: length
* $x_0$: length at rest

Given $m$ measurements $(x_1, F_1), (x_2, F_2), \dots, (x_m, F_m)$,
$$ F_i = -kx_i + b + \epsilon_i $$
* $\epsilon_i$: measurement error

find $k, b$ by fitting a line through data.

**Least squares criterion**,
$$ \min_{k>0, b>0} \sum_{i=1}^m \epsilon_i^2 = \sum_{i=1}^m (F_i + kx_i - b)^2 $$
![|300](../../images/Pasted%20image%2020260116102921.png)
## Linear least squares regression线性最小二乘回归

A **linear model** predicts a response/target by a linear combination of predictors/features (plus an intercept截距/bias),
$$ \hat{y} = f(x) = b + \sum_{i=1}^n w_i x_i = x^T w + b $$

Given $m$ data points $(x_1, y_1), (x_2, y_2), \dots, (x_m, y_m)$, **linear (least squares) regression** finds $w$ and $b$ by minimizing the sum of squared errors,
$$ \min_{w \in \mathbb{R}^n, b \in \mathbb{R}} \sum_{i=1}^m (f(x_i) - y_i)^2 = \sum_{i=1}^m (x_i^T w + b - y_i)^2 $$

In a more compact form,
$$ \min_{w \in \mathbb{R}^n, b \in \mathbb{R}} \| Xw + b\mathbf{1} - y \|^2 $$

* $X = [x_1, \dots, x_m]^T \in \mathbb{R}^{m \times n}$, $y = (y_1, \dots, y_m) \in \mathbb{R}^m$
* $\mathbf{1} = (1, 1, \dots, 1) \in \mathbb{R}^m$
* $\|z\| = \sqrt{z^T z} = \sqrt{\sum_{i=1}^n z_i^2}$ for $z = (z_1, \dots, z_n) \in \mathbb{R}^n$
## Optimal transport problem

* need to ship products from $n$ warehouses to $m$ customers
* inventory at warehouse $i$ is $a_i, i = 1, 2, \dots, n$
* quantity ordered by customer $j$ is $b_j, j = 1, 2, \dots, m$
* unit shipping cost from warehouse $i$ to customer $j$ is $c_{ij}$

Let $x_{ij}$ be quantity shipped from warehouse $i$ to customer $j$
Minimize total cost by solving the following **linear program**

$$
\begin{aligned}
& \min_{(x_{ij})} & & \sum_{i=1}^n \sum_{j=1}^m c_{ij} x_{ij} \\
& \text{s.t.} & & \sum_{i=1}^n x_{ij} = b_j \quad \text{for } j = 1, 2, \dots, m \\
& & & \sum_{j=1}^m x_{ij} \le a_i \quad \text{for } i = 1, 2, \dots, n \\
& & & x_{ij} \ge 0 \quad \text{for } i = 1, 2, \dots, n; j = 1, 2, \dots, m
\end{aligned}
$$
## Power allocation

We want to transmit information over $n$ communication channels. The capacity of the $i$-th channel is
$$ C_i = W_i \log_2 (1 + \frac{P_i}{N_i}) \text{ bits/second} $$

* $W_i$: channel bandwidth in hertz
* $P_i$: signal power in watts
* $N_i$: noise power in watts

Given a total power constraint $P$,

$$
\begin{aligned}
& \max_{P_1, \dots, P_n} & & \sum_{i=1}^n W_i \log_2 (1 + \frac{P_i}{N_i}) \\
& \text{s.t.} & & \sum_{i=1}^n P_i \le P, \quad P_i \ge 0 \text{ for } i = 1, 2, \dots, n
\end{aligned}
$$
![|200](../../images/Pasted%20image%2020260116103142.png)
## Binary classification
![](../../images/Pasted%20image%2020260116103200.png)

Represent an image by a vector $x \in \mathbb{R}^n$, label $y \in \{+1, -1\}$

Given a set of images with labels $(x_1, y_1), (x_2, y_2), \dots, (x_m, y_m)$, want a function $f : \mathbb{R}^n \to \mathbb{R}$, called a **classifier**, such that

$$
\begin{cases}
f(x_i) > 0, & \text{iff } y_i = +1 \\
f(x_i) < 0, & \text{iff } y_i = -1
\end{cases}
\iff y_i f(x_i) > 0
$$

Once we find $f$, we can use $\hat{y} = \text{sign}[f(x)]$ to classify new images.

How to find $f$? Let’s consider **linear classifiers**, i.e. $f(x) = w^T x + b$, and two methods to determine $w, b$, i.e. **logistic regression** and **support vector machine**.
## Logistic regression

**Logistic regression** assumes the probability for an image $x$ to have label $y$ is modeled by
$$ p(y \mid x) = \sigma(yf(x)) = \sigma(y(w^T x + b)) $$
where $\sigma$ is the **sigmoid function**
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
y通常取1，-1；$w^T x+b$为打分，通常像1则为正，像-1则为负
![|300](../../images/Pasted%20image%2020260116103220.png)
To determine $w, b$, maximize the **likelihood**
$$ \max_{w, b} \quad L(w, b) = \prod_{i=1}^m p(y_i \mid x_i) $$

or equivalently, minimize the negative log likelihood
$$ \min_{w, b} \quad NLL(w, b) = -\log L(w, b) = \sum_{i=1}^m \log(1 + e^{-y_i(x_i^T w + b)}) $$
## Support vector machine支持向量机

Assume data is linearly separable, i.e. exists hyperplane $w^T x + b = 0$ s.t.
$$ y_i(w^T x_i + b) > 0, \quad \forall i $$

There may exist many such hyperplanes with different **margins**, i.e. minimum distance from data points to the hyperplane.
* A classifier with a larger margin is more robust鲁棒 against noise.

Given a hyperplane $P : w^T x + b = 0$,
* distance from $x_i$ to $P$,
$$ \text{dist}(x_i, P) = \frac{|w^T x_i + b|}{\|w\|} $$

* margin
$$ \min_{1 \le i \le m} \frac{|w^T x_i + b|}{\|w\|} $$

**Support vector machine (SVM)** maximizes the margin
$$
\begin{aligned}
& \max_{w, b} & & \min_{1 \le i \le m} \frac{|w^T x_i + b|}{\|w\|} \\
& \text{s.t.} & & y_i(w^T x_i + b) > 0, \quad i = 1, 2, \dots, m
\end{aligned}
$$

Not easy to solve in this form.

Problem reformulation
* Note $|w^T x_i + b| = y_i(w^T x_i + b)$, as $y_i = \text{sign}(w^T x_i + b)$.
* For $\alpha > 0$, $\tilde{w} = \alpha w$ and $\tilde{b} = \alpha b$ determine the same hyperplane $P$,
$$ x \in P \iff w^T x + b = 0 \iff \tilde{w}^T x + \tilde{b} = 0 $$
* Choosing $\alpha$ properly, we can assume $\min_{1 \le i \le m} y_i(\tilde{w}^T x_i + \tilde{b}) = 1$,

$$
\begin{aligned}
& \max_{\tilde{w}, \tilde{b}} & & \frac{1}{\|\tilde{w}\|} \\
& \text{s.t.} & & y_i(\tilde{w}^T x_i + \tilde{b}) \ge 1, \quad i = 1, 2, \dots, m
\end{aligned}
$$

* For $z > 0$, maximizing $\frac{1}{z}$ is equivalent to minimizing $\frac{1}{2}z^2$,

$$
\begin{aligned}
& \min_{\tilde{w}, \tilde{b}} & & \frac{1}{2} \|\tilde{w}\|^2 \\
& \text{s.t.} & & y_i(\tilde{w}^T x_i + \tilde{b}) \ge 1, \quad i = 1, 2, \dots, m
\end{aligned}
$$

We will see this reformulation is a convex problem and easy to solve.

## Appendix: Distance to hyperplane

* $w \perp \text{hyperplane } P : w^T x + b = 0$
* $x_i'$ is the **orthogonal projection** of $x_i$ onto $P$, i.e.
$$ x_i - x_i' \perp P $$
$$ w^T x_i' + b = 0 $$

* $x_i - x_i' = \gamma_i w$ for some $\gamma_i \in \mathbb{R}$,
$$ w^T(x_i - \gamma_i w) + b = 0 \implies \gamma_i = \frac{w^T x_i + b}{w^T w} $$

* distance from $x_i$ to $P$ is
$$ \min_{x \in P} \|x_i - x\| = \|x_i - x_i'\| = \|\gamma_i w\| = \frac{|w^T x_i + b|}{\|w\|} $$
## Soft margin SVM

Hard margin SVM requires linear separability
$$ \min_{\boldsymbol{w},b} \quad \frac{1}{2}\|\boldsymbol{w}\|^2 $$
$$ \text{s.t.} \quad y_i(\boldsymbol{w}^T \boldsymbol{x}_i + b) \geq 1, \quad \forall i $$

When not linearly separable,
*   relax constraints
*   penalize deviation

Soft margin SVM: introduce slack variables $\boldsymbol{\xi} = (\xi_1, \dots, \xi_n)$

$$ \min_{\boldsymbol{w},b,\boldsymbol{\xi}} \quad \frac{1}{2}\|\boldsymbol{w}\|_2^2 + C \sum_{i=1}^m \xi_i \quad (C > 0 \text{ is a hyperparameter}) $$
$$ \text{s.t.} \quad y_i(\boldsymbol{w}^T \boldsymbol{x}_i + b) \geq 1 - \xi_i, \quad i = 1, 2, \dots, m $$
$$ \boldsymbol{\xi} \geq \boldsymbol{0}, \quad (\text{i.e. } \xi_i \geq 0, \quad i = 1, 2, \dots, m) $$

## Global optima

$\boldsymbol{x}^* \in \Omega$ is a **global minimum**$^1$ of $f$ over $\Omega$ if
$$ f(\boldsymbol{x}^*) \leq f(\boldsymbol{x}), \quad \forall \boldsymbol{x} \in \Omega $$

It is an **optimal solution** of the minimization problem
$$ \min_{\boldsymbol{x} \in \Omega} f(\boldsymbol{x}) \quad \text{(P)} $$
and $f(\boldsymbol{x}^*)$ is the **optimal value** of (P).

**Global maximum** is defined by reversing the direction of the inequality.
Maximum and minimum are called **extremum**.

Note. Global extrema may not exist.
*   $f(x) = x, \Omega = \mathbb{R}, \inf_{x \in \Omega} f(x) = -\infty$ unbounded below
*   $f(x) = x, \Omega = (0, 1), \inf_{x \in \Omega} f(x) = 0$, but not achievable

$^1$Global minimum often also refers to the minimum value $f(\boldsymbol{x}^*)$.

## Inner product and norm

Euclidean inner product on $\mathbb{R}^n$: $\langle \boldsymbol{x}, \boldsymbol{y} \rangle = \boldsymbol{x}^T \boldsymbol{y} = \sum_{i=1}^n x_i y_i$

Euclidean norm (2-norm, $\ell_2$-norm): $\|\boldsymbol{x}\|_2 = \sqrt{\boldsymbol{x}^T \boldsymbol{x}} = \sqrt{\sum_{i=1}^n x_i^2}$

A **norm** on $\mathbb{R}^n$ is a function $\|\cdot\| : \mathbb{R}^n \to \mathbb{R}$ satisfying
1.  $\|\boldsymbol{x}\| \geq 0, \forall \boldsymbol{x} \in \mathbb{R}^n$
2.  $\|\boldsymbol{x}\| = 0 \text{ iff } \boldsymbol{x} = \boldsymbol{0}$
    } (positive definiteness)
3.  $\|a\boldsymbol{x}\| = |a|\|\boldsymbol{x}\|, \forall a \in \mathbb{R}, \boldsymbol{x} \in \mathbb{R}^n$ (positive homogeneity)
4.  $\|\boldsymbol{x} + \boldsymbol{y}\| \leq \|\boldsymbol{x}\| + \|\boldsymbol{y}\|, \forall \boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^n$ (triangle inequality)

Example.
*   $p$-norm ($\ell_p$-norm): $\|\boldsymbol{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}, p \geq 1$
*   $\infty$-norm ($\ell_\infty$-norm): $\|\boldsymbol{x}\|_\infty = \max_{1 \leq i \leq n} |x_i| = \lim_{p \to \infty} \|\boldsymbol{x}\|_p$

Property 4 of these norms is given by Minkowski’s inequality.
By default, $\|\boldsymbol{x}\|$ means $\|\boldsymbol{x}\|_2$.

##  Open and closed balls

Open ball of radius $r$ centered at $\boldsymbol{x}_0$
$$ B(\boldsymbol{x}_0, r) = \{\boldsymbol{x} : \|\boldsymbol{x} - \boldsymbol{x}_0\| < r\} $$

Closed ball of radius $r$ centered at $\boldsymbol{x}_0$
$$ \bar{B}(\boldsymbol{x}_0, r) = \{\boldsymbol{x} : \|\boldsymbol{x} - \boldsymbol{x}_0\| \leq r\} $$

![](../../images/Pasted%20image%2020260116104904.png)

Open ball of radius $r$ centered at $\boldsymbol{x}_0$
$$ B(\boldsymbol{x}_0, r) = \{\boldsymbol{x} : \|\boldsymbol{x} - \boldsymbol{x}_0\| < r\} $$

Closed ball of radius $r$ centered at $\boldsymbol{x}_0$
$$ \bar{B}(\boldsymbol{x}_0, r) = \{\boldsymbol{x} : \|\boldsymbol{x} - \boldsymbol{x}_0\| \leq r\} $$

![](../../images/Pasted%20image%2020260116104921.png)
## Open and closed sets

A set $S$ is **open** if for any $\boldsymbol{x} \in S$, there exists $\epsilon > 0$ s.t. $B(\boldsymbol{x}, \epsilon) \subset S$.
A set $S$ is **closed** if its complement $S^c$ is open.

Examples in $\mathbb{R}$.
*   $(0, 1)$ is open.
*   $[0, 1]$ is closed.
*   $(0, 1]$ is neither open nor closed.
*   $[1, \infty)$ is closed.

A sequence $\{\boldsymbol{x}_k\}$ converges to $\boldsymbol{x}$, denoted $\boldsymbol{x}_k \to \boldsymbol{x}$ or $\lim_{k \to \infty} \boldsymbol{x}_k = \boldsymbol{x}$ if
$$ \lim_{k \to \infty} \|\boldsymbol{x} - \boldsymbol{x}_k\| = 0 $$

Note. In $\mathbb{R}^n$, if $\boldsymbol{x}_k \to \boldsymbol{x}$ in one norm, it converges in any norm. The convergence in norm is equivalent to coordinate-wise convergence.

Theorem. $S$ is closed iff for any sequence $\{\boldsymbol{x}_k\} \subset S$,
$$ \boldsymbol{x}_k \to \boldsymbol{x} \implies \boldsymbol{x} \in S. $$

## Compactness

A set $S$ is **bounded** if there exists $M < \infty$ s.t. $\|\boldsymbol{x}\| \leq M, \forall \boldsymbol{x} \in S$.

Heine-Borel Theorem. $S \subset \mathbb{R}^n$ is **compact** iff it is closed and bounded.

Examples in $\mathbb{R}$.
*   $[0, 1]$ is compact
*   $(0, 1), (0, 1]$ are bounded but not closed, so not compact
*   $[1, \infty)$ is closed but not bounded, so not compact

Examples in $\mathbb{R}^n$.
*   For $0 < r < \infty$, the open ball $B(\boldsymbol{0}, r) = \{\boldsymbol{x} \in \mathbb{R}^n : \|\boldsymbol{x}\| < r\}$ is not compact.
*   For $r < \infty$, the closed ball $\bar{B}(\boldsymbol{0}, r) = \{\boldsymbol{x} \in \mathbb{R}^n : \|\boldsymbol{x}\| \leq r\}$ is compact.
*   $\{\boldsymbol{x} \in \mathbb{R}^n : \boldsymbol{x} \geq \boldsymbol{0}\}$ is closed but unbounded, so not compact.

## Continuity

A function $f : S \subset \mathbb{R}^n \to \mathbb{R}$ is **continuous at $\boldsymbol{x} \in S$** if for any $\epsilon > 0$, there exists $\delta > 0$ s.t.
$$ \boldsymbol{y} \in S \cap B(\boldsymbol{x}, \delta) \implies |f(\boldsymbol{y}) - f(\boldsymbol{x})| < \epsilon $$

Equivalently, $f$ is continuous at $\boldsymbol{x} \in S$ if
$$ \forall \{\boldsymbol{x}_k\} \subset S, \quad \boldsymbol{x}_k \to \boldsymbol{x} \implies f(\boldsymbol{x}_k) \to f(\boldsymbol{x}) $$

$f$ is **continuous** on $S$ if it is continuous at every $\boldsymbol{x} \in S$.

Theorem. If $f : \mathbb{R}^n \to \mathbb{R}$ is continuous on $\mathbb{R}^n$, then for any $c \in \mathbb{R}$,
1.  $\{\boldsymbol{x} : f(\boldsymbol{x}) < c\}$ is open;
2.  $\{\boldsymbol{x} : f(\boldsymbol{x}) \leq c\}$ is closed.

## Existence of Global Optima

Extreme Value Theorem. If $f$ is continuous on a nonempty compact set $K$, then $f$ attains its maximum and minimum on $K$, i.e. there exist $\boldsymbol{x}_1, \boldsymbol{x}_2 \in K$ (not necessarily unique) s.t.
$$ f(\boldsymbol{x}_1) \leq f(\boldsymbol{x}) \leq f(\boldsymbol{x}_2), \quad \forall \boldsymbol{x} \in K. $$

Example. $f(x) = x^2$ satisfies $f(0) \leq f(x) \leq f(2) = f(-2)$ on $[-2, 2]$.

The Extreme Value Theorem gives **sufficient** conditions for the existence of global optima, but they are **not necessary**.

Example. $f(x) = x^2$.
*   $\inf_{x \in (0,1)} f(x) = 0$, but $f(x) > 0$ for all $x \in (0, 1)$, no global min.
*   $\min_{x \in [0,1)} f(x) = f(0), x^* = 0$ is global min, but $[0, 1)$ is not closed.
*   $\min_{x \in \mathbb{R}} f(x) = f(0), x^* = 0$ is global min, but $\mathbb{R}$ unbounded.

Corollary. If $f$ is continuous on $\mathbb{R}^n$ and $f(\boldsymbol{x}) \to +\infty$ as $\|\boldsymbol{x}\| \to \infty$, then the global min exists, i.e. there exists $\boldsymbol{x}^*$ s.t. $f(\boldsymbol{x}^*) \leq f(\boldsymbol{x}), \forall \boldsymbol{x}$.

Proof.
*   Since $f(\boldsymbol{x}) \to +\infty$ as $\|\boldsymbol{x}\| \to \infty$, there exists $M > 0$ s.t. $f(\boldsymbol{x}) > f(\boldsymbol{0})$ when $\|\boldsymbol{x}\| > M$.
*   The closed ball $\bar{B}(\boldsymbol{0}, M) = \{\boldsymbol{x} : \|\boldsymbol{x}\| \leq M\}$ is compact.
*   By the Extreme Value Theorem, there exists $\boldsymbol{x}^* \in \bar{B}(\boldsymbol{0}, M)$ s.t.
    $$ f(\boldsymbol{x}^*) \leq f(\boldsymbol{x}), \quad \forall \boldsymbol{x} \in \bar{B}(\boldsymbol{0}, M) $$
*   For $\boldsymbol{x} \notin \bar{B}(\boldsymbol{0}, M), f(\boldsymbol{x}^*) \leq f(\boldsymbol{0}) < f(\boldsymbol{x})$, so $\boldsymbol{x}^*$ is a global min on $\mathbb{R}^n$.

A function $f$ is called **coercive** if $f(\boldsymbol{x}) \to +\infty$ as $\|\boldsymbol{x}\| \to \infty$.

Example. $f(\boldsymbol{x}) = \|\boldsymbol{x}\|^2$ coercive, $\boldsymbol{x}^* = \boldsymbol{0}$ is global minimum.
Example. $f(\boldsymbol{x}) = e^{-\|\boldsymbol{x}\|}$ not coercive, no global minimum.
Example. $f(x) = \sin x$ not coercive, $x^* = -\frac{\pi}{2}$ is global minimum.

## Local Minimum

$\boldsymbol{x}^* \in \Omega$ is a **local minimum** of $f$ on $\Omega$ if there exists $\epsilon > 0$ s.t.
$$ f(\boldsymbol{x}^*) \leq f(\boldsymbol{x}), \quad \forall \boldsymbol{x} \in \Omega \cap B(\boldsymbol{x}^*, \epsilon) $$

$\boldsymbol{x}^*$ is a **strict local minimum** if strict inequality holds for $\boldsymbol{x} \neq \boldsymbol{x}^*$.
**Local maximum** is defined by reversing direction of inequality.

*![](../../images/Pasted%20image%2020260116105150.png)*

Global minimum is always local minimum, but **not** vice versa.
*   We will see local min is global min for convex problems

$\boldsymbol{x}^* \in \Omega$ is a **local minimum** of $f$ on $\Omega$ if there exists $\epsilon > 0$ s.t.
$$ f(\boldsymbol{x}^*) \leq f(\boldsymbol{x}), \quad \forall \boldsymbol{x} \in \Omega \cap B(\boldsymbol{x}^*, \epsilon) $$

$\boldsymbol{x}^*$ is a **strict local minimum** if strict inequality holds for $\boldsymbol{x} \neq \boldsymbol{x}^*$.
**Local maximum** is defined by reversing direction of inequality.

![](../../images/Pasted%20image%2020260116105223.png)*

Global minimum is always local minimum, but **not** vice versa.
*   We will see local min is global min for convex problems

