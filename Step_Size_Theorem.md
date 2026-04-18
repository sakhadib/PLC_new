## Formal Proof: The Structural Contiguity Constraint (ΔL ≤ 1)

1. Definitions & Notation
Let $P_n$ be a set of exactly $n$ distinct points in a 2D geometric space.
Let $P_{n+1} = P_n \cup \{p_{n+1}\}$, where $p_{n+1}$ is a newly introduced, distinct coordinate point.
Let $C(P)$ represent a valid line cover for a set of points $P$, meaning every point in $P$ is incident to at least one line in $C(P)$.
Let $L(n)$ denote the cardinality of the optimal (minimum) line cover for $P_n$. Mathematically, $L(n) = \min |C(P_n)|$.
Let $L_{opt}(n)$ be the specific set of lines that achieves this minimum cover for $P_n$, such that $|L_{opt}(n)| = L(n)$.

2. Lemma 1: The Non-Decreasing Property ($L(n) \le L(n+1)$)
Proof by Contradiction:
Assume that adding a point to the universe could strictly decrease the minimum number of lines required to cover the universe. That is, assume $L(n+1) < L(n)$.
Let $L_{opt}(n+1)$ be the optimal line cover for $P_{n+1}$. By definition, this set of lines covers every point in $P_{n+1}$.
Since $P_n \subset P_{n+1}$, the line cover $L_{opt}(n+1)$ inherently covers every point in $P_n$ as well.
This means $L_{opt}(n+1)$ is a valid line cover for $P_n$.
If $L(n+1) < L(n)$, then we have found a valid cover for $P_n$ that is strictly smaller than $L(n)$.
This directly contradicts the foundational definition of $L(n)$ as the absolute minimum cover for $P_n$.
Therefore, the assumption is false, and $L(n) \le L(n+1)$ must hold true.

3. Lemma 2: The Upper Bound Property ($L(n+1) \le L(n) + 1$)
Proof by Constructive Existence:
Consider the set of points $P_{n+1}$ and the known optimal line cover for the previous step, $L_{opt}(n)$.
By definition, $L_{opt}(n)$ successfully covers all points in $P_n$.
The only point in $P_{n+1}$ that is not guaranteed to be covered by $L_{opt}(n)$ is the newly added point, $p_{n+1}$.
We can construct a new, valid line cover for $P_{n+1}$ by taking $L_{opt}(n)$ and adding exactly one arbitrary mathematical line $l'$ that passes through $p_{n+1}$ (e.g., a line connecting $p_{n+1}$ to any $p_i \in P_n$).
Let this newly constructed cover be $C_{construct} = L_{opt}(n) \cup \{l'\}$.
The cardinality of this constructed cover is $|C_{construct}| = |L_{opt}(n)| + 1 = L(n) + 1$.
Because $L(n+1)$ is defined as the absolute minimum possible cover for $P_{n+1}$, it must be less than or equal to the size of any known valid cover we can construct.
Therefore, $L(n+1) \le |C_{construct}|$.
Substituting the cardinality, we derive $L(n+1) \le L(n) + 1$.

4. Theorem Conclusion
By combining Lemma 1 and Lemma 2, we establish the absolute bounds for the phase transition:
$L(n) \le L(n+1) \le L(n) + 1$

Subtracting $L(n)$ from all terms yields:
$0 \le L(n+1) - L(n) \le 1$

Letting $\Delta L = L(n+1) - L(n)$, we conclude that:
$\Delta L \in \{0, 1\}$

Q.E.D.

5. Strategic Implication for the RL Environment
This proof establishes that the geometry of the Prime Line Cover problem grows continuously. A jump of $\Delta L > 1$ represents a mathematical impossibility under optimal play. If the agent's proposed sequence yields $\Delta L > 1$, it constitutes a failure of extendability (a greedy collapse), justifying the catastrophic physics penalty $\Phi$ applied to the agent's reward function.
