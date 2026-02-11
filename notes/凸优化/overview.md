# Overview

本笔记根据上海交通大学CS 2601 Linear and Convex Optimization，江波老师的讲义整理而成
## Convex sets 

*   **convex set**
    $$ \boldsymbol{x}, \boldsymbol{y} \in C, \theta \in [0, 1] \implies \theta \boldsymbol{x} + \bar{\theta}\boldsymbol{y} \in C $$

*   **convex combination**
    $$ \sum_{i=1}^k \theta_i \boldsymbol{x}_i, \quad \text{where } \theta_i \ge 0, \quad \sum_{i=1}^k \theta_i = 1 $$

*   **convex hull of $S$**: smallest convex set containing $S$, set of all convex combinations of points in $S$,
    $$ \text{conv } S = \left\{ \sum_{i=1}^m \theta_i \boldsymbol{x}_i : m \in \mathbb{N}; \boldsymbol{x}_i \in S, \theta_i \ge 0, i = 1, \dots, m; \sum_{i=1}^m \theta_i = 1 \right\} $$

*   **examples**: lines, rays, line segments, hyperplanes, half space, affine space, polyhedron, norm ball, ellipsoid, simplex, positive semidefinite cone

*   **convexity-preserving operations**
    *   intersection of convex sets
    *   image/preimage of convex set under affine transformation

*   **cone**
    $$ \boldsymbol{x} \in C, \theta \ge 0 \implies \theta \boldsymbol{x} \in C $$

*   **convex cone**
    $$ \boldsymbol{x}_1, \boldsymbol{x}_2 \in C, \theta_1, \theta_2 \ge 0 \implies \theta_1 \boldsymbol{x}_1 + \theta_2 \boldsymbol{x}_2 \in C. $$

*   **conic combination**
    $$ \sum_{i=1}^m \theta_i \boldsymbol{x}_i, \quad \text{where } \theta_i \ge 0 \ \forall i. $$

*   **conic hull of $S$**: smallest convex cone containing $S$, set of all conic combinations of points in $S$,
    $$ \text{cone}(S) = \left\{ \sum_{i=1}^m \theta_i \boldsymbol{x}_i : m \in \mathbb{N}; \boldsymbol{x}_i \in S, \theta_i \ge 0, i = 1, \dots, m \right\} $$

*   **projection onto closed convex set**
    $$ \mathcal{P}_C(\boldsymbol{x}) = \underset{\boldsymbol{z} \in C}{\text{argmin}} \|\boldsymbol{x} - \boldsymbol{z}\|_2 = \underset{\boldsymbol{z} \in C}{\text{argmin}} \frac{1}{2} \|\boldsymbol{x} - \boldsymbol{z}\|_2^2 $$

*   **supporting hyperplane theorem**
*   **separating hyperplane theorem**
*   $*$ **Farkas' Lemma**

*  **methods for proving convexity:**
	*   definition
	*   convexity-preserving operations
	*   sublevel/superlevel set of convex/concave functions
	*   epigraph/hypograph of convex/concave functions

## Convex functions 

*   **definition**: $f$ is convex if it has convex domain $\text{dom} f$, and
    $$ \boldsymbol{x}, \boldsymbol{y} \in \text{dom} f, \theta \in (0, 1) \implies f(\theta \boldsymbol{x} + \bar{\theta}\boldsymbol{y}) \le \theta f(\boldsymbol{x}) + \bar{\theta}f(\boldsymbol{y}) $$
    $f$ is concave if $-f$ is convex.

*   **affine functions** are the only functions that are both convex and concave.

*   **strict convexity**
    $$ \boldsymbol{x} \ne \boldsymbol{y} \in \text{dom} f, \theta \in (0, 1) \implies f(\theta \boldsymbol{x} + \bar{\theta}\boldsymbol{y}) < \theta f(\boldsymbol{x}) + \bar{\theta}f(\boldsymbol{y}) $$

*   **strong convexity**: $f$ is $m$-strongly convex if $f(\boldsymbol{x}) - \frac{m}{2}\|\boldsymbol{x}\|_2^2$ is convex.

*   **examples**: norm, negative entropy, log-sum-exp function, quadratic function with PSD quadratic term,...

*   **epigraph**
    $$ \text{epi} f = \{(\boldsymbol{x}, y) : \boldsymbol{x} \in \text{dom} f, y \ge f(\boldsymbol{x})\} $$
    $f$ is a convex function iff $\text{epi} f$ is a convex set.

*   **sublevel sets of convex functions are convex**
    $$ C_\alpha(f) = \{ \boldsymbol{x} \in \text{dom} f : f(\boldsymbol{x}) \le \alpha \} $$

*   **zeroth order condition**
    *   restriction to any line is (strictly/strongly) convex

*   **first-order conditions**
    *   **convexity**
        $$ f(\boldsymbol{y}) \ge f(\boldsymbol{x}) + \nabla f(\boldsymbol{x})^T (\boldsymbol{y} - \boldsymbol{x}), \quad \forall \boldsymbol{x}, \boldsymbol{y} \in \text{dom} f $$
    *   **strict convexity**
        $$ f(\boldsymbol{y}) > f(\boldsymbol{x}) + \nabla f(\boldsymbol{x})^T (\boldsymbol{y} - \boldsymbol{x}), \quad \forall \boldsymbol{x} \ne \boldsymbol{y} \in \text{dom} f $$
    *   **$m$-strong convexity**
        $$ f(\boldsymbol{y}) \ge f(\boldsymbol{x}) + \nabla f(\boldsymbol{x})^T (\boldsymbol{y} - \boldsymbol{x}) + \frac{m}{2}\|\boldsymbol{x} - \boldsymbol{y}\|_2^2, \quad \forall \boldsymbol{x}, \boldsymbol{y} \in \text{dom} f $$

*   **second-order conditions**
    *   **convexity**
        $$ \nabla^2 f(\boldsymbol{x}) \succeq \boldsymbol{O}, \quad \forall \boldsymbol{x} \in \text{dom} f $$
    *   **strict convexity**
        $$ \nabla^2 f(\boldsymbol{x}) \succ \boldsymbol{O}, \quad \forall \boldsymbol{x} \in \text{dom} f $$
    *   **$m$-strong convexity**
        $$ \nabla^2 f(\boldsymbol{x}) \succeq m\boldsymbol{I}, \quad \forall \boldsymbol{x} \in \text{dom} f $$

*   **convexity preserving operations**
    *   **nonnegative combinations**
        $$ f(\boldsymbol{x}) = \sum_{i=1}^m c_i f_i(\boldsymbol{x}) $$
    *   **composition with affine functions**
        $$ f(\boldsymbol{x}) = g(\boldsymbol{A}\boldsymbol{x} + \boldsymbol{b}) $$
    *   **certain composition of monotonic convex/concave functions**
        $$ f(\boldsymbol{x}) = h(g_1(\boldsymbol{x}), \dots, g_m(\boldsymbol{x})) $$
    *   **pointwise maximum/supremum**
        $$ f(\boldsymbol{x}) = \sup_{i \in I} f_i(\boldsymbol{x}) $$
    *   **partial minimization**: for convex $g$ and convex $C$,
        $$ f(\boldsymbol{x}) = \inf_{\boldsymbol{y} \in C} g(\boldsymbol{x}, \boldsymbol{y}) $$
## Optimization problems

$$
\begin{aligned}
\min_{\boldsymbol{x}} \quad & f(\boldsymbol{x}) \\
\text{s.t.} \quad & \boldsymbol{g}(\boldsymbol{x}) \le \boldsymbol{0} \\
& \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0}
\end{aligned}
$$

*   **domain** $D = \text{dom} f \cap (\bigcap_i \text{dom} g_i) \cap (\bigcap_j \text{dom} h_j)$

*   **feasible set**
    $$ \Omega = \{ \boldsymbol{x} \in D : \boldsymbol{g}(\boldsymbol{x}) \le \boldsymbol{0}, \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0} \} $$
    $\boldsymbol{x}$ is feasible if $\boldsymbol{x} \in \Omega$

*   $f^* = \inf_{\boldsymbol{x} \in \Omega} f(\boldsymbol{x})$ is the **optimal value**

*   $\boldsymbol{x}^* \in \Omega$ is a **global minimum** if $f^* = f(\boldsymbol{x}^*)$, or equivalently
    $$ f(\boldsymbol{x}^*) \le f(\boldsymbol{x}), \quad \forall \boldsymbol{x} \in \Omega $$

*   $\boldsymbol{x}^* \in \Omega$ is a **local minimum** if for some $\delta > 0$,
    $$ f(\boldsymbol{x}^*) \le f(\boldsymbol{x}), \quad \forall \boldsymbol{x} \in \Omega \cap B(\boldsymbol{x}^*, \delta) $$
## Convex optimization problems

$$
\begin{aligned}
\min_{\boldsymbol{x}} \quad & f(\boldsymbol{x}) \\
\text{s.t.} \quad & \boldsymbol{g}(\boldsymbol{x}) \le \boldsymbol{0} \\
& \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0}
\end{aligned}
$$

*   **$f, \boldsymbol{g}$ are convex, $\boldsymbol{h} = \boldsymbol{Ax} - \boldsymbol{b}$ is affine.**

*   **key property: local minima are global minima.**
    *   no assertion about existence; $*$ some conditions for existence
    *   no assertion about uniqueness; if $f$ is strictly convex, solution is unique if exists.

*   **examples**: LP, QP, QCQP, GP

*   **equivalent problems**: informally, solution of one problem readily yields solution to the other
    *   some simple transformation: changing variables, eliminating equality constraints, introducing slack variables, transforming objective/constraints,...

*   **be able to formulate simple convex optimization problems**

## Optimality conditions

*   **unconstrained problem**
    $$ \nabla f(\boldsymbol{x}^*) = \boldsymbol{0} $$

*   **equality constrained problem**: Lagrange condition
    $$
    \begin{cases}
    \nabla f(\boldsymbol{x}^*) + \boldsymbol{A}^T \boldsymbol{\lambda}^* = \boldsymbol{0} \\
    \boldsymbol{Ax}^* = \boldsymbol{b}
    \end{cases}
    \quad \text{or} \quad \nabla \mathcal{L}(\boldsymbol{x}^*, \boldsymbol{\lambda}^*) = \boldsymbol{0}
    $$

*   **inequality constrained problem**: KKT conditions
    *   primal feasibility: $\boldsymbol{h}(\boldsymbol{x}^*) = \boldsymbol{0}, \boldsymbol{g}(\boldsymbol{x}^*) \le \boldsymbol{0}$
    *   dual feasibility: $\boldsymbol{\mu}^* \ge \boldsymbol{0}$
    *   stationarity: $\nabla_{\boldsymbol{x}} \mathcal{L}(\boldsymbol{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\mu}^*) = \boldsymbol{0}$
    *   complementary slackness: $\mu_j^* g_j(\boldsymbol{x}^*) = 0, \forall j$

*   **convex problems**
    $$ \nabla f(\boldsymbol{x}^*)^T (\boldsymbol{x} - \boldsymbol{x}^*) \ge 0, \quad \forall \boldsymbol{x} \in \Omega $$
## Lagrange duality

*   **general primal problem**,
    $$
    \begin{aligned}
    \min_{\boldsymbol{x}} \quad & f(\boldsymbol{x}) \\
    \text{s.t.} \quad & \boldsymbol{g}(\boldsymbol{x}) \le \boldsymbol{0} \\
    & \boldsymbol{h}(\boldsymbol{x}) = \boldsymbol{0}
    \end{aligned}
    $$

*   **Lagrangian**
    $$ \mathcal{L}(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\boldsymbol{x}) + \sum_{i=1}^k \lambda_i h_i(\boldsymbol{x}) + \sum_{j=1}^m \mu_j g_j(\boldsymbol{x}) $$
    *   $\mathcal{L}(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) \le f(\boldsymbol{x})$ for feasible $\boldsymbol{x}$ and $\boldsymbol{\mu} \ge \boldsymbol{0}$

*   **dual function**
    $$ \phi(\boldsymbol{\lambda}, \boldsymbol{\mu}) = \inf_{\boldsymbol{x} \in D} \mathcal{L}(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) $$
    *   always concave
    *   domain: $\{(\boldsymbol{\lambda}, \boldsymbol{\mu}) : \phi(\boldsymbol{\lambda}, \boldsymbol{\mu}) > -\infty\}$
    *   lower bound property $\phi(\boldsymbol{\lambda}, \boldsymbol{\mu}) \le \phi^* \le f^* \le f(\boldsymbol{x})$ for $\boldsymbol{\mu} \ge \boldsymbol{0}, \boldsymbol{x} \in \Omega$

*   **dual problem**
    $$
    \begin{aligned}
    \max_{\boldsymbol{\lambda}, \boldsymbol{\mu}} \quad & \phi(\boldsymbol{\lambda}, \boldsymbol{\mu}) \\
    \text{s.t.} \quad & \boldsymbol{\mu} \ge \boldsymbol{0}
    \end{aligned}
    $$
    *   always a convex optimization problem
    *   dual LP that makes constraints explicit

*   **weak duality**: $\phi^* \le f^*$
    *   optimal duality gap $f^* - \phi^*$

*   **strong duality**: $\phi^* = f^*$
    *   (refined) Slater's condition for convex problems
    *   strong duality almost always holds for LP

*   **KKT conditions and strong duality for convex problems**

    KKT $\Longleftrightarrow$ strong duality + primal optimality + dual optimality

## Algorithms

#### LP
*   **simplex method**
    *   Fundamental Theorem of LP
    *   tableau method:
        $$
        \begin{aligned}
        \min \quad & \boldsymbol{c}^T \boldsymbol{x} \\
        \text{s.t.} \quad & \boldsymbol{Ax} = \boldsymbol{b} \\
        & \boldsymbol{x} \ge \boldsymbol{0}
        \end{aligned}
        \implies
        \begin{bmatrix}
        -\boldsymbol{c}^T & 0 \\
        \boldsymbol{A} & \boldsymbol{b}
        \end{bmatrix}
        $$

        $$
        \begin{bmatrix}
        -\boldsymbol{c}_1^T & -\boldsymbol{c}_2^T & 0 \\
        \boldsymbol{B} & \boldsymbol{D} & \boldsymbol{b}
        \end{bmatrix}
        \implies
        \begin{bmatrix}
        \boldsymbol{0} & -\boldsymbol{c}_2^T + \boldsymbol{c}_1^T \boldsymbol{B}^{-1}\boldsymbol{D} & \boldsymbol{c}_1^T \boldsymbol{B}^{-1}\boldsymbol{b} \\
        \boldsymbol{I} & \boldsymbol{B}^{-1}\boldsymbol{D} & \boldsymbol{B}^{-1}\boldsymbol{b}
        \end{bmatrix}
        $$

    *   basic solution $(\boldsymbol{B}^{-1}\boldsymbol{b}, \boldsymbol{0})$
    *   value $\boldsymbol{c}_1^T \boldsymbol{B}^{-1}\boldsymbol{b}$
    *   negative reduced cost $-\boldsymbol{c}_2^T + \boldsymbol{c}_1^T \boldsymbol{B}^{-1}\boldsymbol{D}$
    *   two-phase method
        $$
        \begin{aligned}
        \min \quad & \boldsymbol{1}^T \boldsymbol{y} \\
        \text{s.t.} \quad & \boldsymbol{Ax} + \boldsymbol{y} = \boldsymbol{b} \\
        & \boldsymbol{x}, \boldsymbol{y} \ge \boldsymbol{0}
        \end{aligned}
        $$

*   $*$ **barrier method**

#### **unconstrained problems**
*   **smooth $f$**
    *   descent method: $\boldsymbol{x}_{k+1} = \boldsymbol{x}_k + t_k \boldsymbol{d}_k$
    *   descent direction
        *   negative gradient: $\boldsymbol{d}_k = -\nabla f(\boldsymbol{x}_k)$
        *   Newton direction: $\boldsymbol{d}_k = -[\nabla^2 f(\boldsymbol{x}_k)]^{-1} \nabla f(\boldsymbol{x}_k)$
    *   step size
        *   constant
        *   exact line search
        *   backtracking line search (Armijo's rule)
    *   condition number
    *   $*$ convergence analysis

*   $*$ **smooth $f$ + nonsmooth $h$**
    *   proximal gradient descent
        $$ \boldsymbol{x}_{k+1} = \text{prox}_{t_k h}(\boldsymbol{x}_k - t_k \nabla f(\boldsymbol{x}_k)) $$
        $$ \text{prox}_h(\boldsymbol{x}) = \underset{\boldsymbol{z}}{\text{argmin}} \left\{ \frac{1}{2}\|\boldsymbol{z} - \boldsymbol{x}\|_2^2 + h(\boldsymbol{z}) \right\} $$

#### **constrained problems**
*   **equality constraints**
    *   constraint elimination
    *   Newton's method with feasible start
        $$ \boldsymbol{x}_{k+1} = \boldsymbol{x}_k + t_k \boldsymbol{v}_k \quad \text{where} \quad \begin{bmatrix} \nabla^2 f(\boldsymbol{x}_k) & \boldsymbol{A}^T \\ \boldsymbol{A} & \boldsymbol{O} \end{bmatrix} \begin{bmatrix} \boldsymbol{v}_k \\ \boldsymbol{\lambda}_k \end{bmatrix} = \begin{bmatrix} -\nabla f(\boldsymbol{x}_k) \\ \boldsymbol{0} \end{bmatrix} $$

    *   $*$ Newton's method with infeasible start
        $$
        \begin{bmatrix} \boldsymbol{x}_{k+1} \\ \boldsymbol{\lambda}_{k+1} \end{bmatrix} =
        \begin{bmatrix} \boldsymbol{x}_{k} \\ \boldsymbol{\lambda}_{k} \end{bmatrix} + t_k
        \begin{bmatrix} \boldsymbol{v}_{k} \\ \boldsymbol{w}_{k} \end{bmatrix}
        \quad \text{where} \quad
        \begin{bmatrix} \nabla^2 f(\boldsymbol{x}_k) & \boldsymbol{A}^T \\ \boldsymbol{A} & \boldsymbol{O} \end{bmatrix} \begin{bmatrix} \boldsymbol{v}_k \\ \boldsymbol{w}_k \end{bmatrix} = -
        \begin{bmatrix} \nabla f(\boldsymbol{x}_k) + \boldsymbol{A}^T \boldsymbol{\lambda}_k \\ \boldsymbol{A}\boldsymbol{x}_k - \boldsymbol{b} \end{bmatrix}
        $$

*   $*$ **equality+inequality constraints**
    *   projected gradient descent
        $$ \boldsymbol{x}_{k+1} = \mathcal{P}_\Omega (\boldsymbol{x}_k - t_k \nabla f(\boldsymbol{x}_k)) $$
    *   barrier method
