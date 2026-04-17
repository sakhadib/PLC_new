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


---


## Formal Network Architecture: Deep Bipartite Graph Attention Network (GATv2)

1. Core Architecture
The policy network $\pi_\theta(s_t)$ is implemented as a deep Bipartite Graph Attention Network v2 (GATv2). It is specifically engineered to ingest the dynamically pruned bipartite graph $G_t = (U_t, V_t, E_t)$ and process the asymmetrical geometric relationship between discrete coordinate points and continuous mathematical lines.

2. Feature Initialization & Dimensionality
* Input Mapping: Point Node features $[x, y, d_u]$ and Line Node features $[a, b, d_v]$ are initially passed through separate, dedicated Linear projection layers to map them into a shared hidden state space.
* Hidden Dimension ($D$): Set to 128 dimensions. This provides massive representational capacity to memorize prime distribution sequences without choking the Kaggle P100's 16 GB VRAM.

3. Message Passing (The Receptive Field)
To ensure the network captures broad Phase 2 resonance structures rather than just localized greedy overlaps, the architecture utilizes exactly 3 Message Passing Layers.
* Layer Mechanics: Information flows strictly across the bipartite edges ($E_t$). 
    - Pass 1 (Lines $\to$ Points): Points update their state based on intersecting lines.
    - Pass 2 (Points $\to$ Lines): Lines update their state based on their constituent points.
* Receptive Scope: At 3 layers deep, a Line Node assesses its direct points, the intersecting lines crossing those points, and the extended point structures those secondary lines cover. This grants the agent a "regional awareness" of the geometric scaffolding.

4. Multi-Head Attention (GATv2)
Each Message Passing layer is equipped with 8 Attention Heads.
* Strategic Justification: GATv2 computes dynamic attention scores, allowing the network to heavily weight mathematically critical edges (e.g., dense structural hubs or $a=8$ slopes) while ignoring trivial connections. 
* Dimension Splitting: The 128-dimensional hidden state is split across the 8 heads (16 dimensions per head), allowing each head to independently specialize in recognizing different arithmetic or topological phenomena. The heads are concatenated after each layer.

5. The Readout Layer (Action Logits)
After the 3rd message passing layer, the final 128-dimensional embedding of every Line Node $v \in V_t$ is extracted.
* Projection: These embeddings are passed through a final Multi-Layer Perceptron (MLP) structured as $128 \to 64 \to 1$.
* Output: This produces a single, unnormalized scalar score (logit) $z_v$ for every available line.
* Masking: A boolean mask is applied to filter out any lines that were dynamically pruned, and a Softmax is applied to convert the valid logits into the final probability distribution $P(v)$ from which the RL agent samples its action $a_t$.

6. Constraint Alignment (Kaggle P100 Limits)
* VRAM Limits: Because the 12 million global lines are heavily pruned at the State Representation level, the node count stays strictly in the thousands. A 3-layer, 128-dim GATv2 on a graph of this size consumes less than 3 GB of VRAM, leaving massive overhead for RL rollout buffers.
* Storage Limits: The `.pth` weights for this architecture will be approximately 5-10 MB, easily allowing thousands of safe checkpoint saves within the Kaggle 19.5 GB working directory limit over the 12-hour session.


---


## Formal Training & Optimization Strategy: Masked PPO, Curriculum Learning, and API Chaining

1. Core Optimization Algorithm: Proximal Policy Optimization (PPO)
The agent is trained using PPO, an Actor-Critic policy gradient method, chosen for its stability in complex combinatorial spaces and its sample efficiency.
* The Actor Network: The main Bipartite GATv2 outputs a probability distribution over the available lines.
* The Critic Network: A parallel linear head attached to the GAT's readout layer that predicts the Value Function $V(s_t)$—the expected total number of lines to finish the game from the current state.
* The Objective: PPO updates the network weights by comparing the Actor's actual reward against the Critic's prediction (the Advantage). To prevent catastrophic forgetting during training, PPO clips the policy update so the AI's behavior does not drastically shift from a single lucky (or unlucky) episode.

2. Action Masking Mechanics
Because the bipartite graph dynamically prunes Line Nodes, the action space $A_t$ shrinks at every step. Standard RL algorithms fail here.
* Implementation: Before the Actor network applies the softmax function to its output logits, an Action Mask is applied. The logits of mathematically invalid or pruned lines are set to $-\infty$. This guarantees the softmax probability for invalid lines is exactly $0.0$, forcing the PPO optimizer to only explore mathematically sound geometry.

3. Curriculum Learning Protocol (The Training Escalation)
To prevent early-stage gradient collapse, the environment strictly controls the scale of the point coordinates ($n$) presented to the agent.
* Tier 1 (The Sandbox, $n=10$ to $50$): Rapid episodes where the agent learns basic point-coverage, the step-penalty survival mechanic, and avoiding the $\Delta L > 1$ physics violation.
* Tier 2 (The Benchmark, $n=161$): The AI must achieve $\ge 95\%$ match with the known ILP ground truth. This serves as mathematical proof of architectural competence.
* Tier 3 (The Wall, $n=859$): The agent maps the upper bound of Neil Sloane's exact ILP calculations, learning deep Phase 2 structural resonances.
* Tier 4 (The Frontier, $n=1000$ to $5000$): The environment unlocks massive scales where the AI hunts for novel upper bounds that beat the greedy baseline.

4. The Kaggle Survival Pipeline (Cross-Account Chaining)
To bypass the 12-hour session timeout and the 19.5 GB working directory limit across 5 rotating Kaggle P100 accounts, a rolling API chain is deployed.
* Rolling Checkpoints: The training script strictly maintains only three files: `model_latest.pth`, `model_best.pth`, and `optimizer.pth`. These are overwritten every 100 episodes or 30 minutes. Total disk usage remains strictly under 50 MB, eliminating OOM storage crashes.
* The Baton Pass (Automated Hand-off): At hour 11.5 of the Kaggle session, a time-hook triggers. The script safely pauses training and pushes the `latest.pth` and `optimizer.pth` files to a centralized cloud repository (e.g., Kaggle Datasets API or Weights & Biases Artifacts).
* Seamless Resumption: When the next co-author account spins up, the initialization script automatically fetches the artifacts from the cloud, loads the exact model weights and PPO optimizer momentum, and resumes training at the exact episode it left off. This achieves a continuous 150-hour weekly compute block without manual data transfer.
