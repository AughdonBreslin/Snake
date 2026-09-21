# AlphaZero Snake: Design

Date: 2026-08-24
Status: Approved for planning

## 1. Purpose

Build a single-player AlphaZero-style snake agent: a convolutional network emitting a policy and a value, guided by Monte Carlo tree search, trained on its own self-play.

The work is a standalone research project with its own entry point.
It is not wired into the home screen during development.
The eventual endpoint is that a trained checkpoint replaces the Hamiltonian cycle currently animating the main menu, which makes rule fidelity between the training environment and the shipped game a hard requirement rather than a nicety.

## 2. Context and prior art in this repo

`app.py` contains a working pygame snake with a main menu, settings, leaderboard, and gameplay.
`graph.py` contains a Hamiltonian cycle generator that produces perfect play and currently drives the menu background.
`god.py` contains an unrun sketch of a CNN, an MCTS, and a trainer, added in commit 564cb24 as a base for this work.

`god.py` has never executed end to end.
The following are defects rather than design choices, recorded here so the spec is not read as a critique of decisions that were never made:

- `SnakeGame.step` is non-functional.
  `Direction.__add__` returns a plain `int`, so `move()` compares `int == Direction.RIGHT`, every branch fails, the head never advances, and the head-into-body check fires.
  Every agent step reports game over immediately.
- `get_valid_inputs()` returns pygame keycodes such as `K_RIGHT` (1073741903), which are then used to index a length-4 policy array.
- Three methods are called that do not exist: `make_move`, `get_next_state`, and `to_tensor`.
- `SnakeGame` extends `Background`, holds a `pygame.Surface` and font objects, and re-renders text inside `move()`, so it cannot be cloned cheaply or stepped without a display.
- The MCTS selection loop has no terminal check, and the expansion block reads `action` after the `for` loop has ended, so a fully expanded node re-descends into the last action's child indefinitely.
- Unvisited children score `float('inf')`, which discards the prior at exactly the point where it is most informative.
- The policy head applies `relu` before `softmax`, flooring every negative logit to zero.
- Training applies `F.cross_entropy` to an already-softmaxed tensor, softmaxing twice.
- `grid_size=10` sizes the fully connected layers for a 10x10 board; the playable board is 20x20.
- BatchNorm is never switched to eval mode for single-sample search inference.
- `get_state` indexes `state[1, x, y]` against an array shaped `(4, GRID_HEIGHT, GRID_WIDTH)`, transposing body and food relative to the wall channel.

`god.py` is superseded by the `snake/` package described below and is deleted at the end of Phase 1.
Git history preserves it.

## 3. Decisions taken

These were settled during design and are not revisited by the implementation plan.

| Decision | Choice |
|---|---|
| Menu role | Standalone project now; trained checkpoint replaces the Hamiltonian background at the end |
| Board sizes | Curriculum from 6x6 up to 20x20, with a fully convolutional size-agnostic network |
| Stochastic food | Open-loop determinized MCTS: nodes keyed by action path, fresh RNG per simulation |
| Value target | Fraction of the whole board still to be filled from this state |
| Scaffolding | Config dataclasses, explicit seeding, TensorBoard, checkpoints with resume |
| Rule ownership | A single headless `SnakeEnv`; `app.py` keeps rendering and delegates all rules |

## 4. Module layout

```
snake/
  __init__.py
  env.py         SnakeEnv: rules, step, clone, legal mask. No pygame, no torch.
  encoding.py    observation planes and D4 symmetry transforms
  model.py       SnakeNet: residual tower, head-centric policy head, pooled value head
  evaluator.py   Evaluator interface, single-leaf and batched implementations
  mcts.py        open-loop determinized PUCT
  selfplay.py    game generation, target computation, replay buffer
  train.py       optimizer loop, losses, checkpointing, resume
  arena.py       evaluation harness, metrics, baseline comparison
  baselines.py   random, greedy, BFS-with-tail-safety, hamiltonian agents
  config.py      run configuration dataclasses
  cli.py         entry points: train, eval, play
app.py           SnakeGame delegates rules to SnakeEnv, keeps rendering and input
graph.py         unchanged; HamiltonianCycle is reused by baselines.py
```

The boundary that carries the most weight is `evaluator.py`.
MCTS never calls the network directly; it calls an `Evaluator`.
Replacing single-leaf inference with inference batched across concurrent games, or later with multi-process workers, is a new `Evaluator` implementation and requires no change to search.
This is the deliberate scaling extension point rather than a later rewrite.

## 5. Environment

### 5.1 Ownership

`SnakeEnv` is the single definition of what snake is.
`SnakeGame` in `app.py` retains its surface, fonts, score text, highscore persistence, and keyboard handling, and delegates every rule decision to an env instance.

This preserves the existing quirk on `app.py:375`, where the tail is popped before the collision check so the head may legally move into the cell the tail is vacating.
That behavior is canonical and is pinned by tests before the refactor begins.

### 5.2 Interface

```python
class SnakeEnv:
    def __init__(self, width: int, height: int, rng: np.random.Generator,
                 starvation_limit: int | None = None) -> None: ...
    def reset(self) -> None: ...
    def clone(self) -> "SnakeEnv": ...
    def step(self, action: int) -> bool: ...                 # done   # (reward, done)
    def legal_actions(self) -> np.ndarray: ...               # bool[4]
    def observation(self) -> np.ndarray: ...                 # float32[11, H, W]

    @property
    def length(self) -> int: ...
    @property
    def total_cells(self) -> int: ...                        # playable cells, the C in section 8
    @property
    def death_cause(self) -> str | None: ...                 # "wall" | "self" | "starvation"
```

`clone()` copies the body deque and the scalars explicitly.
`copy.deepcopy` is too slow for a search that steps the environment tens of thousands of times per move.

The env owns a `numpy.random.Generator` rather than using the global RNG, so open-loop search can reseed per simulation and so runs are reproducible.

`step` returns only whether the episode ended.
There is deliberately no reward signal in the environment, because the value target in section 8 is computed from the length trajectory of a finished episode rather than accumulated per step.
This removes reward shaping as a tuning surface entirely.

### 5.3 Action space

Four absolute actions: RIGHT=0, DOWN=1, LEFT=2, UP=3, matching the existing `Direction` enum.
The reverse of the current heading is masked out of `legal_actions()`.

Masking the reverse is not a rule change.
For a length-3 snake moving right, turning left pops the tail and then moves the head into `snake[1]`, which is still in the body list, so the move is always fatal under the existing rules.
Masking removes a move that is always a loss.

Absolute actions are chosen over relative turn-left/straight/turn-right because rotating the board permutes the four absolute directions by a simple index map, which makes dihedral augmentation clean.
Relative actions do not survive reflection as neatly.

### 5.4 Starvation cap

`starvation_limit` truncates an episode after that many steps without eating.
The default is `2 * total_playable_cells`, so 72 on a 6x6 board and 800 on a 20x20 board.

This exists only for training.
A weak agent will loop indefinitely and consume GPU time on a single game.
`app.py` constructs its env with `starvation_limit=None`, so the shipped game is unaffected.

Truncation is recorded as `death_cause == "starvation"` and is distinguished from wall and self collisions in evaluation metrics.

## 6. Observation encoding

Observations are `float32[11, H, W]`, indexed `[channel, y, x]`.
Note that this fixes the transposition in `app.py:get_state`, which indexes `state[1, x, y]` against a `(4, H, W)` array.

| Plane | Contents |
|---|---|
| 0 | Walls, the existing one-cell border |
| 1 | Body excluding the head |
| 2 | Head |
| 3 | Food |
| 4 | Tail cell |
| 5 | Body age: each body cell holds steps-until-vacated, divided by current length |
| 6 | Heading is RIGHT, constant plane |
| 7 | Heading is DOWN, constant plane |
| 8 | Heading is LEFT, constant plane |
| 9 | Heading is UP, constant plane |
| 10 | Fill fraction, constant plane holding length divided by total playable cells |

Every plane is board-size invariant in meaning, which is what permits one set of weights to serve every curriculum stage.

Plane 5 carries the most information per channel.
Snake's central difficulty is deciding whether a corridor is a trap or will have opened by the time the head arrives, which is precisely the question of how many steps remain until a cell vacates.
Without that plane the network must infer body ordering from an unordered occupancy mask, which convolutions do poorly.
With it, the answer is available locally.

The wall border stays in the grid rather than being represented by convolution padding, so boundaries are visible as content and one set of weights behaves consistently across board sizes.

## 7. Network

`SnakeNet` is a fully convolutional residual tower.

Stem: 3x3 convolution from 11 planes to `channels`, GroupNorm, ReLU.
Body: `blocks` residual blocks, each two 3x3 convolutions with GroupNorm and a skip connection.

Policy head: gather the feature vector at the head cell, concatenate the global average pooled feature vector, then a small MLP to 4 logits.
The head-cell gather is size-agnostic and semantically correct, since the decision is taken at the head; the pooled term supplies board-level context.
The head emits logits, not probabilities, so the loss can use `log_softmax` directly.

Value head: 1x1 convolution, global average pool, MLP to one scalar, sigmoid.
Sigmoid rather than tanh because the value target defined in section 8 lives in [0, 1].
PUCT operates correctly with Q in [0, 1]; that is the range AlphaGo's win probability occupied.

Normalization is GroupNorm throughout, never BatchNorm.
BatchNorm in AlphaZero self-play is a recurring failure mode: search evaluates batches of size one, running statistics are polluted, and a forgotten `model.eval()` degrades a run silently instead of raising.
GroupNorm is batch-size independent and removes the failure mode rather than documenting it.

No layer may be sized from the board dimensions.
A test asserts that one set of weights runs on 6x6 and 20x20 inputs and produces correctly shaped outputs.

## 8. Value and policy targets

### 8.1 Value

For a state `s` in an episode that ended with the snake at `final_length`, with `C` the total playable cells:

```
z(s) = (final_length - length(s)) / C
```

The quantity is "what fraction of the whole board will I still go on to fill from here."
It lies in [0, 1], requires no scale tuning, and is board-size invariant, so it transfers directly across curriculum stages.
It is a return-to-go, so credit assignment is local rather than smeared uniformly across a two-thousand-step episode.

Critically, `z` is the value function of a fixed-reward MDP: undiscounted return with a reward of `1/C` for each food eaten and zero otherwise.
That property is what makes the search backup in section 9 well defined.
An earlier draft normalized by remaining capacity rather than by `C`, which put parent and child values in different units and left PUCT comparing incommensurable numbers across siblings.
Normalizing by a constant removes that problem rather than requiring the search to correct for it.

The known cost is resolution late in a game: at length 380 on a 400-cell board, targets occupy [0, 0.05] rather than the full range, so late-game states contribute smaller gradients and may underfit.
Plane 10 carries the fill fraction so the network can condition on where it sits on that scale.
If TensorBoard shows value mean absolute error concentrated on high-fill states in Phase 3, the response is target reweighting, decided then with data rather than pre-emptively.

Edge cases:

- A state from which the snake never eats again yields `z(s) = 0.0`, whether the episode ended in a collision or in starvation truncation.
- A solved board has nothing left to fill and therefore also yields `z(s) = 0.0`.

### 8.2 Policy

The target is the root visit distribution over the four actions, normalized to sum to one, with masked actions held at zero.

## 9. Search

Open-loop determinized PUCT.

Nodes are keyed by action path rather than by state.
Each simulation resets a scratch environment to the root state with a fresh RNG seed, descends by PUCT while applying actions to that scratch environment, expands the first unvisited node, evaluates the leaf through the `Evaluator`, and backs the value up the path.

Backup carries no sign alternation, since there is no opponent.
It does carry the per-edge reward from section 8.1: a value arriving at a parent from a child is `v_child + 1/C` when that edge ate food, and `v_child` otherwise.
Every value in the tree is therefore in the same units, and comparing siblings at a node is well defined even when one was reached by eating and another was not.

Repeated visits to a node therefore encounter different food futures, and Q converges to an average over them rather than committing to one sampled world.
Environment steps are O(1) and the descent is O(depth) regardless, so replaying from the root costs essentially nothing.

Selection score:

```
Q(a)     = reward(a) + W(a) / N(a)      where reward(a) is 1/C if action a eats, else 0
score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))
```

Whether an action eats is a deterministic function of the current scratch environment, since the food position is known at the parent; only where the replacement food spawns is random.

Unvisited children are initialized to the parent's Q rather than to infinity, so priors influence the first visits instead of being overridden by tie-breaking order.

Terminal nodes back up a value of 0, representing no further capacity filled, and are never expanded.

Illegal actions receive zero prior and are never selected.

During self-play, Dirichlet noise is mixed into the root prior.
During evaluation, no noise is added.

Move selection samples from visit counts at temperature `tau`, which is 1.0 for the first `tau_threshold` moves of an episode and `tau_final` thereafter.

## 10. Training loop

Self-play emits `(observation, visit_distribution, value_target)` triples into a fixed-window replay buffer over recent positions.

Loss is the sum of three terms:

- Policy: cross-entropy between `log_softmax(logits)` and the visit distribution.
- Value: mean squared error against `z`.
- Regularization: weight decay via the optimizer.

### 10.1 Dihedral augmentation

Every board in the curriculum is square, so all eight D4 transforms are valid, giving eight times the data at negligible cost.
Each transform permutes the four action logits by a known index map, and also permutes planes 6 through 9, which encode heading.

The permutation map is derived in code and verified by test, not hardcoded from a table in this document.
Getting the permutation backwards produces a subtly worse agent rather than an error, so a test asserting that transforming then acting equals acting then transforming is mandatory.

### 10.2 Throughput

Phase 1 runs K games concurrently within one process, batching their MCTS leaf evaluations into single GPU calls.
This saturates the available GPU without introducing multiprocessing.

If concurrency within one process stops keeping the GPU busy, multi-process self-play workers are added as an additional `Evaluator` implementation, with no change to `mcts.py`.

## 11. Curriculum

The config declares stages as a list of `(board_size, promotion_threshold, min_games)`.

Because the tower is fully convolutional and the value target is normalized by board area, promotion is a configuration change: the same weights train on a larger board.

Promotion is gated on evaluation performance, defaulting to mean fill fraction at or above 0.9 over 100 deterministic games.

After the first stage, board size is sampled per game from the set of unlocked sizes rather than training exclusively on the newest size, so training on 20x20 does not erase 6x6 competence.

## 12. Evaluation and baselines

`arena.py` runs N deterministic games on fixed seeds with no Dirichlet noise and `tau = 0`, and reports:

- Mean, median, and maximum score
- Mean fill fraction
- Mean episode length
- Death cause histogram across wall, self, and starvation

Score is the count of food eaten, matching `app.py`, so `length = 3 + score`.
Fill fraction is final length divided by total playable cells, so a solved board scores 1.0.

The death cause histogram is the primary diagnostic for identifying which failure mode an agent is stuck on.

The agent is compared against four baselines on identical seeds:

| Baseline | Role |
|---|---|
| Random legal move | Floor |
| Greedy toward food | Trivial heuristic |
| BFS to food gated on the head still reaching its tail afterward | Strong classical agent |
| Hamiltonian cycle, reusing `graph.py` | Ceiling, 100% by construction |

The middle two carry the interpretive weight.
BFS-with-tail-safety is a genuinely strong snake agent, and if the learned agent cannot beat it, the honest conclusion is that AlphaZero has not learned anything the heuristic did not already contain.
Without these baselines, a mean score of 30 on a 20x20 board is uninterpretable.

Baseline results are computed once in Phase 0 and stored as JSON, so every later run is measured against numbers that already exist.

## 13. Scaffolding

Run configuration is a set of dataclasses in `config.py`, serialized to JSON alongside each checkpoint.

Seeding is explicit and covers the environment generator, torch, numpy, and python `random`.

TensorBoard scalars: policy loss, value loss, total loss, policy entropy, value mean absolute error, mean score, mean fill fraction, games per second, simulations per second, replay buffer size, and learning rate.
Policy entropy is the early warning signal for policy collapse.

Checkpoints are written every N optimizer steps, retaining the last K plus the best by evaluation score.
Resume restores model weights, optimizer state, replay buffer, and step count.

`torch` and `tensorboard` are declared in a separate `requirements-train.txt`.
`requirements.txt` keeps only numpy and pygame, so `app.py` continues to run without the training stack installed.

## 14. Default hyperparameters

These are Phase 1 starting points for 6x6, all configurable.

| Parameter | Default |
|---|---|
| channels | 64 |
| residual blocks | 6 |
| simulations per move | 100 |
| c_puct | 1.5 |
| Dirichlet alpha | 0.8 |
| Dirichlet epsilon | 0.25 |
| tau_threshold | 40 moves |
| tau_final | 0.2 |
| optimizer | Adam |
| learning rate | 2e-3 |
| weight decay | 1e-4 |
| batch size | 512 |
| replay window | 200,000 positions |
| games per iteration | 100 |
| concurrent self-play games | 32 |
| evaluation games | 100 |
| promotion threshold | mean fill fraction 0.9 |

## 15. Menu integration, deferred to Phase 4

`Home` takes a background strategy object.
`HamiltonianBackground` wraps the behavior that exists today.
`LearnedBackground` loads a checkpoint and steps an env once per render tick, at a low simulation budget or on the raw policy with no search.

If torch or the checkpoint is missing, `Home` falls back to `HamiltonianBackground`, so the game never acquires a hard dependency on the training stack.

If live inference cannot meet the menu frame budget, the fallback is to generate a game offline and replay the recorded move list.

## 16. Testing strategy

### Phase 0

- Golden rule tests: wall death, self collision, legality of moving into the vacating tail cell, growth on eating, food never respawning on the body.
- Equivalence test: drive `app.py`'s `SnakeGame` and a bare `SnakeEnv` through the same action sequence under `SDL_VIDEODRIVER=dummy`, asserting identical body, food, score, and game-over state at every step, across many seeded episodes.
- Clone isolation: cloning then stepping the clone leaves the original unchanged.
- Legal mask: the reverse action is masked, and no agent ever returns a masked action.
- Starvation cap truncates at the configured limit and reports `death_cause == "starvation"`.

### Phase 1

- Encoding: plane shapes and per-plane semantics on hand-built positions.
- D4 round trip: applying a transform and its inverse recovers the original observation and policy.
- D4 action permutation: acting in the transformed frame equals transforming the action taken in the original frame.
- Size agnosticism: one set of weights forwards on 6x6 and 20x20 with correct output shapes.
- MCTS against a stub evaluator returning a uniform policy and a constant value: visit counts follow PUCT expectations, and illegal actions receive zero visits.
- MCTS terminal handling: a terminal leaf backs up 0 and is never expanded.
- Value target: a hand-built episode produces the expected `z` for every state, including a solved board and an episode ending in starvation.
- Backup units: on a hand-built tree where one child eats and a sibling does not, the two siblings' Q values are expressed in the same units, and a full backup reproduces the analytic `z` of the root.
- Overfit test: repeated optimizer steps on one fixed batch drive the loss down.
- Checkpoint resume: save then load reproduces identical model outputs and optimizer state.

## 17. Phasing and acceptance criteria

This spec covers the full arc, but it is too large for a single implementation plan.
The first plan covers Phase 0 and Phase 1 only, which together produce a headless environment, a measuring stick, and a working learning loop on 6x6.
Phases 2 through 4 get their own plans, written once Phase 1 has produced a result worth scaling.

### Phase 0: environment and measuring stick, no machine learning

Pin `app.py`'s current behavior with tests, extract `SnakeEnv`, wire `app.py` to it, then build the baselines and the evaluation harness.

Accepted when the equivalence test passes across many seeded episodes, `app.py` plays identically by hand, and baseline results for 6x6 and 20x20 are stored as JSON.

### Phase 1: learning loop on 6x6

Network, open-loop MCTS, single-process batched self-play, training loop, augmentation, and the config, TensorBoard, and checkpoint scaffolding.
`god.py` is deleted at the end of this phase.

Accepted when a trained 6x6 agent exceeds the greedy baseline's mean score, training resumes correctly from a checkpoint, and TensorBoard shows value mean absolute error decreasing without policy entropy collapsing to zero.

### Phase 2: curriculum and throughput

Transfer to 10x10, mixed-size sampling, and self-play throughput scaling.

Accepted when 10x10 weights transferred from 6x6 beat the greedy baseline without training from scratch, and 6x6 performance does not regress below its Phase 1 result.

### Phase 3: scale and ablations

Training on 20x20, plus the ablations the scaffolding makes possible, including whether open-loop determinization beats a single determinization at the chosen simulation budget.

Accepted when 20x20 results are reported against all four baselines and the determinization ablation has a recorded result either way.

### Phase 4: menu integration

`Home` gains a background strategy, and the learned agent replaces the Hamiltonian cycle.

Accepted when the menu renders the learned agent at full frame rate and falls back to the Hamiltonian background when torch or the checkpoint is absent.

## 18. Risks

Twenty by twenty may never solve.
The curriculum is a mitigation, not a guarantee.
The baselines exist so that this outcome is legible rather than ambiguous.

Open-loop determinization has higher variance at low simulation counts than explicit chance nodes would.
The configuration treats it as an ablation rather than an assumption, and Phase 3 records the comparison.

Long episodes make late-stage self-play expensive in ways that early stages do not predict.
Throughput measured on 6x6 should not be extrapolated to 20x20.

Refactoring `app.py` risks regressing a working game.
This is mitigated by pinning current behavior with tests before the refactor and by the step-for-step equivalence test afterward.
