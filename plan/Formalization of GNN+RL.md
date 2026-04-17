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
