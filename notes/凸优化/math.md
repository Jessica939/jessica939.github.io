# Math

## Derivative

$x$ is an **interior point** of $S \subset \mathbb{R}^n$ if there exists $\epsilon > 0$ s.t. $B(x, \epsilon) \subset S$.
The **interior** of $S$, denoted by $\text{int } S$, is the set of interior points of $S$.
A function $f : S \subset \mathbb{R}^n \to \mathbb{R}^m$ is **differentiable** at $x_0 \in \text{int } S$, if there exists a matrix $A \in \mathbb{R}^{m \times n}$ s.t.

$$
\lim_{\Delta x \to 0} \frac{f(x_0 + \Delta x) - f(x_0) - A\Delta x}{\|\Delta x\|} = \mathbf{0}
$$

i.e.

$$
\Delta f := f(x_0 + \Delta x) - f(x_0) = A\Delta x + o(\|\Delta x\|)
$$

The affine function $f(x_0) + A(x - x_0)$ is the first-order approximation of $f$ at $x_0$,

$$
f(x) = f(x_0) + A(x - x_0) + o(\|x - x_0\|)
$$


The matrix $A$ is called the derivative of $f$ at $x_0$, and we write

$$
f'(x_0) = Df(x_0) = A
$$

The derivative is given by the Jacobian matrix of $f = (f_1, \ldots, f_m)$

$$
f'(x_0) = \begin{bmatrix}
\frac{\partial f_1(x_0)}{\partial x_1} & \frac{\partial f_1(x_0)}{\partial x_2} & \cdots & \frac{\partial f_1(x_0)}{\partial x_n} \\
\frac{\partial f_2(x_0)}{\partial x_1} & \frac{\partial f_2(x_0)}{\partial x_2} & \cdots & \frac{\partial f_2(x_0)}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m(x_0)}{\partial x_1} & \frac{\partial f_m(x_0)}{\partial x_2} & \cdots & \frac{\partial f_m(x_0)}{\partial x_n}
\end{bmatrix}
$$

i.e.

$$
[f'(x_0)]_{ij} = \frac{\partial f_i(x_0)}{\partial x_j}, \quad i = 1, \ldots, m; j = 1, \ldots, n
$$

Note

$$
f_i(x_0 + \Delta x) = f_i(x_0) + \sum_{j=1}^n \frac{\partial f_i(x_0)}{\partial x_j} \Delta x_j + o(\|\Delta x\|), \quad i = 1, 2, \ldots, m
$$

Example. An affine function $f(x) = Ax + b$ from $\mathbb{R}^n$ to $\mathbb{R}^m$ has derivative $f'(x) = A$ at all $x$. In particular, when $m = 1$, $f(x) = a^T x + b$ has derivative $f'(x) = a^T$, which is a $1 \times n$ matrix, i.e. a row vector.

Proof. In component form,

$$
f_i(x) = \sum_{k=1}^n A_{ik}x_k + b_i = A_{i1}x_1 + A_{i2}x_2 + \cdots + A_{in}x_n + b_i
$$

so

$$
\frac{\partial f_i(x_0)}{\partial x_j} = A_{ij} \implies f'(x_0) = A
$$

Alternative proof.

$$
f(x_0 + \Delta x) - f(x_0) = A\Delta x \implies f'(x_0) = A
$$


Example. $f(x) = x^T A x = \sum_{i=1}^n \sum_{j=1}^n A_{ij}x_i x_j$ has derivative

$$
f'(x) = x^T(A + A^T)
$$

If $A$ is symmetric, then $f'(x) = 2x^T A$.

Proof.

$$
\frac{\partial f}{\partial x_k} = \sum_{i=1}^n \sum_{j=1}^n A_{ij} \left( x_j \frac{\partial x_i}{\partial x_k} + x_i \frac{\partial x_j}{\partial x_k} \right) = \sum_{j=1}^n A_{kj}x_j + \sum_{i=1}^n A_{ik}x_i
$$

Alternatively,

$$
f(x_0 + \Delta x) - f(x_0) = x_0^T (A + A^T)\Delta x + \underbrace{\Delta x^T A \Delta x}_{=o(\|\Delta x\|)}
$$

With $\tilde{A} = \frac{1}{2}(A + A^T)$, $\Delta x^T A \Delta x = o(\|\Delta x\|)$ follows from (cf. slide 32)

$$
\lambda_{\min}(\tilde{A})\|\Delta x\|^2 \le \Delta x^T A \Delta x = \Delta x^T \tilde{A} \Delta x \le \lambda_{\max}(\tilde{A})\|\Delta x\|^2
$$

## Gradient

For a real-valued function $f : \mathbb{R}^n \to \mathbb{R}$, the gradient of $f$ at $x$, denoted by $\nabla f(x)$, is the transpose of $f'(x)$,

$$
\nabla f(x) = [f'(x)]^T, \quad [\nabla f(x)]_i = \frac{\partial f(x)}{\partial x_i}, \quad i = 1, \ldots, n
$$

$\nabla f(x)$ is a column vector and satisfies

$$
f'(x)\Delta x = \langle \nabla f(x), \Delta x \rangle = \nabla f(x)^T \Delta x
$$

The first-order approximation of $f$ at $x_0$ is

$$
f(x_0) + \nabla f(x_0)^T (x - x_0)
$$

Example. For symmetric $A$, the gradient of $f(x) = x^T A x + b^T x + c$ is

$$
\nabla f(x) = 2Ax + b
$$


$\nabla f(x)$ is the steepest ascent direction of $f$ at $x$,

$$
f(x + d) - f(x) \approx \nabla f(x)^T d \le \|\nabla f(x)\| \cdot \|d\|
$$

where equality holds in the last step iff $d = \alpha \nabla f(x)$ for some $\alpha \ge 0$.

![](../../images/Pasted%20image%2020260116105520.png)*

## Chain rule

If $f : S \subset \mathbb{R}^n \to \mathbb{R}^m$ is differentiable at $x_0 \in S$, $g : Y \subset \mathbb{R}^m \to \mathbb{R}^p$ is differentiable at $y_0 = f(x_0)$, then the composition of $f$ and $g$ defined by $h(x) = g(f(x))$ is differentiable at $x_0$, and

$$
h'(x_0) = g'(y_0)f'(x_0) = g'(f(x_0))f'(x_0)
$$

Note. The order is important since $g'(y_0) \in \mathbb{R}^{p \times m}$ and $f'(x_0) \in \mathbb{R}^{m \times n}$ are matrices. In general $f'(x_0)g'(y_0)$ is undefined.

$$
\begin{align*}
\mathbb{R}^n &\xrightarrow{f} \mathbb{R}^m \xrightarrow{g} \mathbb{R}^p \\
x_0 &\mapsto y_0 = f(x_0) \mapsto z_0 = h(x_0) = g(y_0) \\
\Delta x &\xrightarrow{f'} \Delta y \approx f'(x_0)\Delta x \xrightarrow{g'} \Delta z \approx g'(y_0)\Delta y \approx g'(y_0)f'(x_0)\Delta x
\end{align*}
$$

In component form,

$$
[h'(x_0)]_{ij} = \frac{\partial h_i(x_0)}{\partial x_j} = \sum_{k=1}^m \frac{\partial g_i(y_0)}{\partial y_k} \cdot \frac{\partial f_k(x_0)}{\partial x_j} = \sum_{k=1}^m [g'(y_0)]_{ik} [f'(x_0)]_{kj}
$$


Example. $h(x) = f(Ax + b)$ has derivative $h'(x_0) = f'(Ax_0 + b)A$. If $f$ is real-valued,

$$
\nabla h(x_0) = A^T [f'(Ax_0 + b)]^T = A^T \nabla f(Ax_0 + b)
$$

Example. Given $f : \mathbb{R}^n \to \mathbb{R}$ and $x, d \in \mathbb{R}^n$, define

$$
g(t) = f(x + td)
$$

Then

$$
g'(t) = f'(x + td)d = \nabla f(x + td)^T d = d^T \nabla f(x + td)
$$

Note. $g$ is the restriction of $f$ to the straight line through $x$ with direction $d$. We can often get useful information about $f$ by looking at $g$, which is usually easier to deal with.

## First-order necessary condition

Consider unconstrained optimization problem, i.e. $\Omega = \mathbb{R}^n$.

Theorem. If $x^*$ is a local minimum of $f$ and $f$ is differentiable at $x^*$, then its gradient at $x^*$ vanishes, i.e.

$$
\nabla f(x^*) = \left[ \frac{\partial f(x^*)}{\partial x_1}, \ldots, \frac{\partial f(x^*)}{\partial x_n} \right]^T = \mathbf{0}.
$$

Proof. Let $d \in \mathbb{R}^n$. Define $g(t) = f(x^* + td)$.
*   Since $x^*$ is a local minimum, $g(t) \ge g(0)$
*   For $t > 0$,
    $$ \frac{g(t) - g(0)}{t} \ge 0 \implies g'(0) = \lim_{t \downarrow 0} \frac{g(t) - g(0)}{t} \ge 0 $$
*   By chain rule, $g'(0) = \sum_{i=1}^n d_i \frac{\partial f(x^*)}{\partial x_i} = d^T \nabla f(x^*) \ge 0$
*   Setting $d = -\nabla f(x^*) \implies -\|\nabla f(x^*)\|^2 \ge 0 \implies \|\nabla f(x^*)\|^2 \le 0 \implies \nabla f(x^*) = \mathbf{0}$
![](../../images/Pasted%20image%2020260116105551.png)*


A point $x^*$ with $\nabla f(x^*) = \mathbf{0}$ is called a stationary point of $f$.

![](../../images/Pasted%20image%2020260116105640.png)

Note. Will see stationarity is sufficient for convex optimization.


For constrained optimization problem, i.e. $\Omega \neq \mathbb{R}^n$,
*   if $x^*$ is in the interior of $\Omega$, i.e. $B(x^*, \epsilon) \subset \Omega$ for some $\epsilon > 0$, then the proof still works, so $\nabla f(x^*) = \mathbf{0}$
*   otherwise, the proof shows $d^T \nabla f(x^*) \ge 0$ for any feasible direction $d$ at $x^*$
    *   $d$ is a feasible direction at $x \in \Omega$ if $x + \alpha d \in \Omega$ for all sufficiently small $\alpha > 0$
*   will revisit later

Example. $\Omega = [a, b]$
*   $f'(x_1) = 0$
*   $d_1 f'(a) \ge 0 \implies f'(a) \ge 0$ (since $d_1 > 0$)
*   $d_2 f'(b) \ge 0 \implies f'(b) \le 0$ (since $d_2 < 0$)

![](../../images/Pasted%20image%2020260116105655.png)*



## Second derivative

The second-order partial derivatives of $f : S \subset \mathbb{R}^n \to \mathbb{R}$ at $x_0 \in \text{int } S$ are

$$
\frac{\partial^2 f(x_0)}{\partial x_i \partial x_j} \triangleq \left. \frac{\partial}{\partial x_i} \right|_{x=x_0} \left( \frac{\partial f(x)}{\partial x_j} \right), \quad i, j = 1, 2, \ldots, n
$$

The Hessian (matrix) of $f$ at $x_0$, denoted by $\nabla^2 f(x_0)$, is given by

$$
[\nabla^2 f(x_0)]_{ij} = \frac{\partial^2 f(x_0)}{\partial x_i \partial x_j}, \quad i, j = 1, 2, \ldots, n
$$

Note. $\nabla^2 f = [D(\nabla f)]^T$

If $\frac{\partial^2 f(x)}{\partial x_i \partial x_j}$ and $\frac{\partial^2 f(x)}{\partial x_j \partial x_i}$ exist in a neighborhood of $x_0$ and are continuous at $x_0$, then

$$
\frac{\partial^2 f(x_0)}{\partial x_i \partial x_j} = \frac{\partial^2 f(x_0)}{\partial x_j \partial x_i}
$$

so $\nabla^2 f(x_0)$ is symmetric and $\nabla^2 f = D(\nabla f)$
Will assume twice continuous differentiability when considering $\nabla^2 f$.

## Second derivative

Example. For an affine function $f(x) = b^T x + c$
$$ \nabla^2 f(x) = O $$

Example. For a quadratic function $f(x) = x^T A x$ with a symmetric $A$,
$$ \nabla^2 f(x) = 2A $$

Proof. Recall $\nabla f(x) = 2Ax$, so $\nabla^2 f(x) = D(2Ax) = 2A$. In components,

$$
\frac{\partial f(x)}{\partial x_j} = 2 \sum_{k=1}^n x_k A_{kj}
$$

so

$$
\frac{\partial^2 f(x)}{\partial x_i \partial x_j} = 2 \sum_{k=1}^n \frac{\partial x_k}{\partial x_i} A_{kj} = 2A_{ij}
$$

## Chain rule for second derivative

The composition with affine function $g(x) = f(Ax + b)$ has Hessian

$$
\nabla^2 g(x) = A^T \nabla^2 f(Ax + b) A
$$

Proof. Let $y = Ax + b$. Recall $\nabla g(x) = A^T \nabla f(y)$, so
$$ \nabla^2 g(x) = D_x(\nabla g(x)) = D_x(A^T \nabla f(y)) = A^T D_x(\nabla f(y)) $$
$$ = A^T D_y(\nabla f(y)) D_x y = A^T \nabla^2 f(y) A $$

In components,
$$ \frac{\partial g(x)}{\partial x_j} = \sum_{k} \frac{\partial f(y)}{\partial y_k} \frac{\partial y_k}{\partial x_j} = \sum_{k} \frac{\partial f(y)}{\partial y_k} A_{kj} $$
$$ \frac{\partial^2 g(x)}{\partial x_i \partial x_j} = \sum_{k} \frac{\partial}{\partial x_i} \frac{\partial f(y)}{\partial y_k} A_{kj} = \sum_{k} \sum_{\ell} \frac{\partial^2 f(y)}{\partial y_{\ell} \partial y_k} A_{\ell i} A_{kj} = [A^T \nabla^2 f(y) A]_{ij} $$

Special case. For $g(t) = f(x + td)$,
$$ g''(t) = d^T \nabla^2 f(x + td) d $$

Proof. Set $A \leftarrow d, x \leftarrow t, b \leftarrow x$ in the general formula above.

## Second-order Taylor expansion

The second-order Taylor expansion for $g : \mathbb{R} \to \mathbb{R}$ takes the form

$$ g(a + t) = g(a) + g'(a)t + \frac{1}{2}g''(a)t^2 + o(|t|^2) \quad \text{(T1)} $$

The second-order Taylor expansion for $f : \mathbb{R}^n \to \mathbb{R}$ takes the form

$$ f(x + d) = f(x) + \nabla f(x)^T d + \frac{1}{2}d^T \nabla^2 f(x) d + o(\|d\|^2) \quad \text{(T2)} $$

i.e.

$$ f(x + d) = f(x) + \sum_{i=1}^n \frac{\partial f(x)}{\partial x_i} d_i + \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \frac{\partial^2 f(x)}{\partial x_i \partial x_j} d_i d_j + o(\|d\|^2) $$

Note. (T2) can be obtained by applying (T1) to $g(t) = f(x + t\hat{d})$ at $a = 0$ and $t = \|d\|$, where $\hat{d}$ is the unit vector in the direction $d$, i.e. $d = \|d\|\hat{d}$,

$$ g(\|d\|) = g(0) + g'(0)\|d\| + \frac{1}{2}g''(0)\|d\|^2 + o(\|d\|^2) $$

By the chain rule, $g'(0) = \nabla f(x)^T \hat{d}, \quad g''(0) = \hat{d}^T \nabla^2 f(x) \hat{d}$


For a quadratic function $f(x) = x^T A x + b^T x + c$, the second-order Taylor expansion is exact with no $o(\|d\|^2)$ term, i.e.

$$ f(x + d) = f(x) + \nabla f(x)^T d + \frac{1}{2}d^T \nabla^2 f(x) d $$

Note. This can be used to find the expressions for $\nabla f$ and $\nabla^2 f$.

Assume $A$ is symmetric; otherwise, replace $A$ by $\tilde{A} = \frac{1}{2}(A + A^T)$.

$$
\begin{align*}
f(x + d) &= (x + d)^T A (x + d) + b^T (x + d) + c \\
&= x^T A x + d^T A x + x^T A d + d^T A d + b^T x + b^T d + c \\
&= f(x) + (2Ax + b)^T d + \frac{1}{2}d^T (2A) d
\end{align*}
$$

Comparison with the Taylor expansion shows that
$$ \nabla f(x) = 2Ax + b, \quad \nabla^2 f(x) = 2A. $$

## Postive definite matrices

A matrix $A \in \mathbb{R}^{n \times n}$ is positive semidefinite, denoted by $A \succeq O$, if
1. it is symmetric, i.e. $A = A^T$
2. $x^T A x \ge 0, \forall x \in \mathbb{R}^n$

It is positive definite, denoted by $A \succ O$, if condition 2 is replaced by
2'. $x^T A x > 0, \forall x \in \mathbb{R}^n$ and $x \neq \mathbf{0}$.

Note. For a quadratic form $x^T A x$, can always assume $A$ is symmetric, since

$$ x^T A x = x^T A^T x = x^T \left( \frac{A + A^T}{2} \right) x $$

$A$ is negative (semi)definite if $-A$ is positive (semi)definite.

$A$ is indefinite if it is neither positive semidefinite nor negative semidefinite, i.e. there exists $x_1, x_2 \in \mathbb{R}^n$ s.t.

$$ x_1^T A x_1 > 0 > x_2^T A x_2 $$



Example. $A = B^T B$ is positive semidefinite.

Proof. Obviously $A^T = A$. For any $x$,
$$ x^T A x = x^T B^T B x = (Bx)^T (Bx) = \|Bx\|^2 \ge 0 $$

Example. $A = B^T B$ is positive definite iff $B$ has full column rank.

Proof. Note
$$ x^T A x = 0 \iff \|Bx\|^2 = 0 \iff Bx = \mathbf{0} $$
so
$$
\begin{align*}
A \succ O &\iff (x^T A x = 0 \iff x = \mathbf{0}) \\
&\iff (Bx = \mathbf{0} \iff x = \mathbf{0}) \\
&\iff B \text{ has full column rank}
\end{align*}
$$

## Test for positive definiteness

A vector $x$ is an eigenvector of a matrix $A$ with associated eigenvalue $\lambda$ if
$$ Ax = \lambda x $$
We can find all eigenvalues by solving $\det(\lambda I - A) = 0$.

Theorem. Let $A$ be a symmetric matrix.
*   $A \succ O$ iff all its eigenvalues $\lambda > 0$.
*   $A \succeq O$ iff all its eigenvalues $\lambda \ge 0$.

Example. $A = \begin{bmatrix} 1 & 2 \\ 2 & 5 \end{bmatrix}$ is positive definite.

$$ \det(\lambda I - A) = (\lambda - 1)(\lambda - 5) - 4 = 0 \implies \lambda = 3 \pm 2\sqrt{2} > 0 $$

Example. $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ is positive semidefinite.

$$ \det(\lambda I - A) = (\lambda - 1)(\lambda - 4) - 4 = 0 \implies \lambda_1 = 0, \lambda_2 = 5 $$


Given matrix $A = (a_{ij}) \in \mathbb{R}^{n \times n}$, a $k \times k$ principal submatrix of $A$ consists of $k$ rows and $k$ columns with the same indices $I = \{i_1 < i_2 < \cdots < i_k\}$,

$$
A_I = \begin{bmatrix}
a_{i_1 i_1} & \cdots & a_{i_1 i_k} \\
\vdots & \ddots & \vdots \\
a_{i_k i_1} & \cdots & a_{i_k i_k}
\end{bmatrix}
$$

A principal minor of order $k$ of $A$ is $\det A_I$ for some $I$ with $|I| = k$.

If $I = \{1, 2, \ldots, k\}$, $D_k(A) \triangleq \det A_I$ is called the leading principal minor of order $k$.

Theorem (Sylvester). Let $A$ be a symmetric matrix.
*   $A \succ O$ iff $D_k(A) > 0$ for $k = 1, 2, \ldots, n$.
*   $A \succeq O$ iff $\det A_I \ge 0$ for all $I \subset \{1, 2, \ldots, n\}$

Note. For positive semidefiniteness, we need to check all principal minors, not just the leading principal minors.



Example. $A = \begin{bmatrix} 1 & 2 \\ 2 & 5 \end{bmatrix}$ is positive definite.

$$ D_1(A) = \det(1) = 1 > 0, \quad D_2(A) = \det A = 1 > 0 $$

Example. $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ is positive semidefinite.

$$ D_1(A) = \det(1) = 1, \quad \det A_{\{2\}} = \det(4) = 4, \quad D_2(A) = \det A = 0 $$

Note. It is not enough to check $D_k(A) \ge 0$ for all $k$!

Example. $A = \begin{bmatrix} 0 & 0 \\ 0 & -2 \end{bmatrix}$ is negative semidefinite,

$$ D_1(A) = \det(0) = 0, \quad D_2(A) = \det A = 0, $$

but
$$ \det A_{\{2\}} = \det(-2) = -2 < 0 $$



Example. $A = \begin{bmatrix} 1 & 2 \\ 2 & 5 \end{bmatrix}$ is positive definite.

*   Use definition,
    $$ x^T A x = x_1^2 + 4x_1 x_2 + 5x_2^2 = (x_1 + 2x_2)^2 + x_2^2 \ge 0, \quad \forall x \in \mathbb{R}^2 $$
    with equality $\iff \begin{cases} x_1 + 2x_2 = 0 \\ x_2 = 0 \end{cases} \iff x = \mathbf{0}$
*   Find eigenvalues by solving $\det(\lambda I - A) = 0$
    $$ \det \begin{bmatrix} \lambda - 1 & -2 \\ -2 & \lambda - 5 \end{bmatrix} = (\lambda - 1)(\lambda - 5) - 4 = 0 \implies \lambda = 3 \pm 2\sqrt{2} > 0 $$
*   Check leading principal minors
    $$ D_1(A) = \det(1) = 1 > 0, \quad D_2(A) = \det A = 1 > 0 $$



Example. $A = \begin{bmatrix} 1 & 2 & 1 \\ 2 & 5 & 8 \\ 1 & 8 & 1 \end{bmatrix}$ is not positive definite.

Check leading principal minors

$$ D_1(A) = \det(1) = 1 > 0, \quad D_2(A) = \det \begin{bmatrix} 1 & 2 \\ 2 & 5 \end{bmatrix} = 1 > 0 $$

$$ D_3(A) = \det A = 1 \times \begin{vmatrix} 5 & 8 \\ 8 & 1 \end{vmatrix} - 2 \times \begin{vmatrix} 2 & 8 \\ 1 & 1 \end{vmatrix} + 1 \times \begin{vmatrix} 2 & 5 \\ 1 & 8 \end{vmatrix} = -36 < 0 $$

Can also check eigenvalues, e.g. using `numpy.linalg.eig`,
$$ \lambda_1 = 11.69585173, \quad \lambda_2 = 0.58307572, \quad \lambda_3 = -5.27892745 $$

## Eigendecomposition

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ has the following eigendecomposition

$$ A = Q\Lambda Q^T = \sum_{i=1}^n \lambda_i v_i v_i^T $$

where $\Lambda = \text{diag}\{\lambda_1, \ldots, \lambda_n\}$, $Q = [v_1, \ldots, v_n]$ is an orthogonal matrix, i.e. $Q^T Q = QQ^T = I$, and $Av_i = \lambda_i v_i$.

Example. $A = \frac{1}{4} \begin{bmatrix} 3 & -1 \\ -1 & 3 \end{bmatrix}$ has eigenvalues $\lambda_1 = \frac{1}{2}$ and $\lambda_2 = 1$, with corresponding eigenvectors $v_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ and $v_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}$. The eigendecomposition is

$$
A = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} \frac{1}{2} & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} = \frac{1}{2} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}^T + \begin{bmatrix} \frac{-1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} \frac{-1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}^T
$$



The linear transformation $x \mapsto y = Ax = Q\Lambda Q^T x$ can be decomposed into three steps

$$ x \mapsto \tilde{x} = Q^T x \mapsto \tilde{y} = \Lambda \tilde{x} \mapsto y = Q\tilde{y} $$

Recall $v_1, \ldots, v_n$ form an orthonormal basis of $\mathbb{R}^n$,
1. Find the coordinate $\tilde{x}$ of $x$ in the basis $v_1, \ldots, v_n$,
   $$ x = Q\tilde{x} = \sum_{i=1}^n \tilde{x}_i v_i, \quad \tilde{x}_i = v_i^T x $$
2. Scale the components of $\tilde{x}$ by the corresponding eigenvalues,
   $$ \tilde{y} = \Lambda \tilde{x}, \quad \tilde{y}_i = \lambda_i \tilde{x}_i $$
3. Find $y$ from its coordinate $\tilde{y}$ in the basis $v_1, \ldots, v_n$,
   $$ y = Q\tilde{y} = \sum_{i=1}^n \tilde{y}_i v_i $$
![](../../images/Pasted%20image%2020260116105717.png)*

## Geometry of quadratic forms


![](../../images/Pasted%20image%2020260116105732.png)

![](../../images/Pasted%20image%2020260116105753.png)*

##  Bounds on quadratic forms

Proposition. For a symmetric matrix $A \in \mathbb{R}^{n \times n}$,

$$ \lambda_{\min}\|x\|_2^2 \le x^T A x \le \lambda_{\max}\|x\|_2^2, \quad \forall x \in \mathbb{R}^n $$

where $\lambda_{\max}$ and $\lambda_{\min}$ are the largest and the smallest eigenvalues of $A$, respectively.

Proof. Recall that $A$ can be orthogonally diagonalized, i.e. $A = Q\Lambda Q^T$, where $\Lambda = \text{diag}\{\lambda_1, \ldots, \lambda_n\}$ and $Q^T Q = I$. Let $x = Q\tilde{x}$.

$$ x^T A x = \tilde{x}^T (Q^T A Q) \tilde{x} = \tilde{x}^T \Lambda \tilde{x} = \sum_{i=1}^n \lambda_i \tilde{x}_i^2 \le \sum_{i=1}^n \lambda_{\max} \tilde{x}_i^2 = \lambda_{\max}\|\tilde{x}\|_2^2 $$

Then use the fact that orthogonal transformations preserve 2-norm, i.e.

$$ \|x\|_2^2 = x^T x = (Q\tilde{x})^T (Q\tilde{x}) = \tilde{x}^T (Q^T Q) \tilde{x} = \tilde{x}^T \tilde{x} = \|\tilde{x}\|_2^2. $$

Similarly for $x^T A x \ge \lambda_{\min}\|x\|_2^2$.

## Second-order necessary condition

**Theorem. If $f : \mathbb{R}^n \to \mathbb{R}$ is twice continuously differentiable and $x^*$ is a local minimum of $f$, then its Hessian matrix $\nabla^2 f(x^*)$ is positive semidefinite, i.e.**

$$ d^T \nabla^2 f(x^*) d \ge 0, \quad \forall d \in \mathbb{R}^n $$

Proof. Fix $d \in \mathbb{R}^n$. By the first-order necessary condition, $\nabla f(x^*) = \mathbf{0}$.
By the second-order Taylor expansion, for any $t > 0$,

$$ f(x^* + td) = f(x^*) + \frac{t^2}{2} d^T \nabla^2 f(x) d + o(t^2\|d\|^2) \ge f(x^*) $$

So
$$ \frac{1}{2} d^T \nabla^2 f(x) d + o(\|d\|^2) \ge 0, \quad \text{as } t \to 0 $$
Taking the limit $t \to 0$ yields $d^T \nabla f(x^*) d^T \ge 0$. (Note: typo in slide, likely meant $d^T \nabla^2 f(x^*) d \ge 0$)

Note. Can apply the same argument to $g(t) = f(x^* + td)$ with local minimum $t^* = 0$ and use chain rule to obtain $g''(0) = d^T \nabla^2 f(x^*) d \ge 0$.

## Second-order sufficient condition

**Theorem. Suppose $f$ is twice continuously differentiable. If**
1. $\nabla f(x^*) = 0$
2. $\nabla^2 f(x^*)$ **is positive definite, i.e.**
   $$ d^T \nabla^2 f(x^*) d > 0, \quad \forall d \neq \mathbf{0} $$

**then $x^*$ is a local minimum.**

Proof. Use second-order Tayler expansion.

Note. In condition 2, positive definiteness cannot be replaced by positive semidefiniteness.

![](../../images/Pasted%20image%2020260116105816.png)
![](../../images/Pasted%20image%2020260116105827.png)