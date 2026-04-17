## Formal State Representation: Dynamic Bipartite Graph for Prime Line Cover

1. Core Architecture
The environment state at any given time step $t$ is formulated as a dynamically pruned bipartite graph, denoted as $G_t = (U_t, V_t, E_t)$. This structure maps the geometric relationship between uncovered coordinate points and the candidate lines that can cover them, explicitly embedding collinearity into the graph topology.

2. Node Definitions
The graph consists of two disjoint sets of nodes:
* Point Nodes ($U_t$): Represents the set of all prime coordinate points that have not yet been covered at time step $t$. 
  - Feature Vector: Each point node $u \in U_t$ is encoded with a feature vector $[x, y, d_u]$, where $x$ and $y$ are the geometric coordinates, and $d_u$ is the node's current degree (the number of active candidate lines passing through it).
* Line Nodes ($V_t$): Represents the set of mathematically valid candidate lines that can be drawn through the currently uncovered points.
  - Feature Vector: Each line node $v \in V_t$ is encoded with a feature vector $[a, b, d_v]$, where $a$ is the rational slope, $b$ is the y-intercept, and $d_v$ is the node's current degree (the exact number of uncovered points lying on this line).

3. Edge Definition ($E_t$)
An undirected edge $e_{ij}$ exists between a point node $u_i$ and a line node $v_j$ if and only if the coordinate point lies exactly on the line. Mathematically, the incidence is validated by the strict linear equation $y_i = a \cdot x_i + b$. Edges carry no specific weights; the topological presence of the edge itself is the geometric proof of collinearity.

4. The Dynamic Pruning Mechanism (Memory Optimization)
To circumvent the combinatorial explosion inherent in the integer linear programming (ILP) formulation (which requires allocating memory for up to $O(n^2)$ lines simultaneously), the graph $G_t$ is dynamically generated and surgically pruned at every step.
* A line node $v$ is only instantiated and included in $V_t$ if its current active degree $d_v \ge 2$. 
* Any line that covers only one active point is mathematically trivial and strategically irrelevant for deep geometric scaffolding. Therefore, these $O(n)$ redundant lines are excluded from the state representation, drastically reducing the memory footprint and allowing the Graph Neural Network (GNN) to process massive scales (e.g., $n=5000$) within strict VRAM limits.

5. Strategic Advantages of the Bipartite Formulation
* Permutation Invariance: The GNN processes the graph irrespective of the arbitrary ordering of primes, focusing purely on structural geometry.
* Algorithmic Offloading: The neural network is not forced to learn Euclidean geometry or algebraic collinearity. The physical rules of the system are hardcoded into the edge connections, allowing the Reinforcement Learning agent to dedicate 100% of its computational capacity to combinatorial strategy and phase-transition optimization.


---

## Formal Action Space: Single Line Node Selection via Masked Softmax

1. Core Architecture
At any given time step $t$, the action space $A_t$ is defined as a discrete, dynamically sized set corresponding exactly to the available Line Nodes in the current bipartite graph $G_t$. 
Mathematically, $A_t = V_t$. The Reinforcement Learning (RL) agent must select exactly one action $a_t \in A_t$, which represents choosing one specific line to add to the permanent global line cover.

2. The Selection Mechanism
The Graph Neural Network (GNN) acts as the policy network $\pi(a_t | s_t)$. It processes the graph $G_t$ and outputs a scalar logit $z_v$ for every line node $v \in V_t$. 
* Masked Softmax: Because the size of $V_t$ changes at every step, a dynamic mask is applied to ensure probabilities are only calculated for currently valid lines. 
* Probability Distribution: The logits are passed through a softmax function to generate a probability distribution over the available lines: $P(v) = \exp(z_v) / \sum \exp(z_i)$.
* Sampling: During training, the agent samples from this distribution to encourage exploration. During inference/evaluation, the agent selects the line with the highest probability (argmax).

3. State Transition Dynamics (The Environment Step)
Once the agent selects a specific line $v^*$, the environment executes the following deterministic transition to generate the next state $G_{t+1}$:
* Registration: The line $v^*$ is appended to the global list of chosen lines.
* Point Elimination: Every Point Node $u \in U_t$ that shares an edge with $v^*$ is considered "covered" and is permanently deleted from the graph.
* Graph Pruning: The degrees of all remaining Line Nodes are recalculated. Any line node whose active degree drops below 2 (because its constituent points were just covered by $v^*$) is permanently deleted from the action space.
* Time Step Increment: $t \to t+1$.

4. Strategic Justification (Defeating the Greedy Trap)
By restricting the action space to a single sequential line choice—rather than a simultaneous multi-line prediction—the agent is forced to witness the geometric consequences of its actions. Guided by the RL reward function (Bellman Equation), the network learns to evaluate the expected future state rather than immediate payout. It learns to sacrifice immediate maximum-point-coverage (the greedy trap) in favor of selecting structurally superior lines that minimize the total sequence length $L(n)$ over the entire episode.


---

## Formal Reward Function: Step-Penalty with Bounded Baseline Jackpot and Physics Enforcement

1. Core Architecture
The Reward Function $R$ is formulated to guide the Reinforcement Learning (RL) agent via a combination of constant negative pressure (dense step rewards) and a mathematically rigorous terminal evaluation (sparse episodic rewards). The objective is to maximize the expected cumulative return $G = \sum_{t=0}^{T_{end}} \gamma^t R_t$, where $T_{end}$ is the total number of lines used to clear the board, and $\gamma=1$ (undiscounted) since the episode length is finite and exactly equals the line count.

2. Dense Reward: The Efficiency Driver
At every time step $t$ where an action $a_t$ (selecting a line) is taken, the environment issues a strictly negative reward:
$R_t = -1$
* Strategic Justification: This represents the "cost" of using a line. By penalizing every single move, the environment natively forces the agent to seek the shortest possible path to a cleared board. It mathematically eliminates the "Greedy Clone" trap, as there is zero positive reinforcement for simply covering large numbers of points inefficiently.

3. Sparse Terminal Reward: The Baseline Jackpot
When the episode terminates (i.e., $|U_t| = 0$, all points covered), the total lines used $T_{end}$ is compared against the established greedy baseline $L_{greedy}(n)$. A terminal reward is applied:
$R_{terminal} = \lambda \cdot (L_{greedy}(n) - T_{end})$
* $\lambda$ (Jackpot Multiplier): A massive positive scaling factor (e.g., $\lambda = 100$).
* Strategic Justification: If the agent beats the greedy algorithm ($T_{end} < L_{greedy}(n)$), it receives a massive positive payout. If it matches greedy, $R_{terminal} = 0$. If it performs worse than greedy, it suffers a scaled penalty. This asymmetric framing heavily incentivizes the discovery of novel structural phase transitions over mere baseline matching.

4. The Physics Enforcement Constraint ($\Delta L \le 1$)
To embed the fundamental laws of prime geometry into the RL environment, the agent must respect the contiguous structural growth bound. Mathematically, covering $n$ points should never require more than one additional line compared to the optimal cover for $n-1$ points. Let $L_{best}(n-1)$ be the absolute best known line cover for the previous integer step.
Upon termination, the physics constraint is evaluated:
If $T_{end} > L_{best}(n-1) + 1$:
    $R_{physics} = -\Phi$
* $\Phi$ (Physics Violation Penalty): An overwhelmingly large negative scalar (e.g., $\Phi = 1000$).
* Strategic Justification: This forces the agent to learn "scaffolding." If the AI discovers a cheap trick that clears $n$ points but violently breaks the geometric infrastructure established at $n-1$, it is hit with a catastrophic penalty. This guarantees that the AI's solutions are structurally continuous and scientifically valid for proving theoretical upper bounds.

5. Mitigation of High-Reward Traps
* Trap A: The Infinite Loop (Reward Farming). Because $R_t$ is strictly negative and the environment deterministically removes covered points, there are no positive feedback loops to exploit mid-game. The episode must monotonically approach termination.
* Trap B: Deliberate Episode Extension. An agent might theoretically try to prolong an episode if a massive terminal reward was guaranteed. However, because $R_{terminal}$ decreases linearly as $T_{end}$ increases, and $R_t$ constantly subtracts, the mathematical gradients strictly point toward the absolute minimum $T_{end}$.
* Trap C: Subverting the Baseline. The baseline $L_{greedy}(n)$ and $L_{best}(n-1)$ are injected into the environment as immutable external constants, pre-calculated before the episode begins. The agent cannot mathematically manipulate the threshold required to trigger the jackpot or physics constraint.
