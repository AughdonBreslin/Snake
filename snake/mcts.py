"""Open loop determinized PUCT for a single player game.

Nodes are keyed by action path rather than by state. Every simulation replays
from the root with a fresh generator, so repeated visits to a node average over
different food futures instead of committing to one sampled world. Environment
steps are constant time and the descent is linear in depth regardless, so
replaying costs essentially nothing.

Values are the fraction of the whole board still to be filled, which is the
value function of an undiscounted MDP with a reward of 1/C per food. Backup
therefore adds that reward on eating edges, and every value in the tree is in
the same units, so comparing siblings is well defined even when one was reached
by eating and another was not.
"""

from __future__ import annotations

import math

import numpy as np

_SEED_MAX = 2**63 - 1


class MinMaxStats:
    """Tracks the range of Q values seen in one tree, so selection can compare
    them on a unit scale.

    PUCT adds an exploration term to Q and weights it by c_puct. That only
    balances when the two are the same order of magnitude. Here Q values are
    fractions of the board still to be filled, so siblings typically differ by
    one food, 1/C, which is 0.028 on a 6x6 board. Against a conventional c_puct
    the exploration term is roughly fifty times the signal it is meant to
    modulate, and visits spread almost evenly no matter what the value head
    says. Normalizing Q against the range actually observed in this tree
    restores the balance and makes c_puct scale free, so it does not need
    retuning per board size. This is the modification MuZero introduced for
    domains whose rewards have no natural scale.
    """

    __slots__ = ("minimum", "maximum")

    def __init__(self):
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def update(self, value):
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    def normalize(self, value):
        # Until two distinct values have been seen there is nothing to
        # normalize against. Pass the value through rather than inventing a
        # constant, which would erase the only signal an early search has.
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children", "terminal")

    def __init__(self, prior):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = None
        self.terminal = False

    @property
    def expanded(self):
        return self.children is not None

    def expand(self, priors, legal):
        masked = np.where(legal, priors, 0.0)
        total = masked.sum()
        if total > 0:
            masked = masked / total
        else:
            masked = legal.astype(np.float64) / legal.sum()
        self.children = [
            Node(float(masked[a])) if legal[a] else None for a in range(4)
        ]


class Search:
    def __init__(self, env, cfg, rng):
        self.env = env
        self.cfg = cfg
        self.rng = rng
        self.root = Node(1.0)
        # Per tree, never shared across moves: normalizing against a range from
        # a position that no longer exists would be meaningless.
        self.stats = MinMaxStats()
        self._reward = 1.0 / env.total_cells
        self._pending = None

    # Lockstep interface

    def descend(self):
        """Walk to a leaf. Returns the observation needing evaluation, or None
        when this determinization died on the way, in which case backup has
        already run.

        Terminality is re-decided every simulation rather than cached. Death is
        absorbing and survival is not, so a cached flag would record "has ever
        died in any sampled world", a union over samples rather than an average
        over them, and the resulting pessimism would grow with the simulation
        budget instead of shrinking."""
        if self._pending is not None:
            raise RuntimeError(
                "descend() called while a simulation is still pending; "
                "apply() must complete the previous one first"
            )
        scratch = self.env.clone(np.random.default_rng(int(self.rng.integers(_SEED_MAX))))
        node = self.root
        path = [node]
        rewards = []
        done = False

        while node.expanded:
            action = self._select(node, scratch)
            ate = scratch.would_eat(action)
            done = scratch.step(action)
            rewards.append(self._reward if ate else 0.0)
            node = node.children[action]
            path.append(node)
            if done:
                # Recorded so callers can see that this node has died at least
                # once, but never consulted on the way down. Whether a node is
                # terminal is decided by the current determinization alone.
                node.terminal = True
                break

        if done:
            self._backup(path, rewards, 0.0)
            return None

        self._pending = (path, rewards, scratch)
        return scratch.observation()

    def apply(self, priors, value):
        """Complete the pending simulation with the evaluator's answer."""
        path, rewards, scratch = self._pending
        self._pending = None
        leaf = path[-1]
        leaf.expand(priors, scratch.legal_actions())
        if leaf is self.root and self.cfg.dirichlet_epsilon > 0:
            self._add_root_noise()
        self._backup(path, rewards, float(value))

    def visit_counts(self):
        return np.array(
            [0 if child is None else child.visits for child in self.root.children],
            dtype=np.float64,
        )

    # Internals

    def _select(self, node, scratch):
        sqrt_parent = math.sqrt(max(1, node.visits))
        parent_value = node.value_sum / node.visits if node.visits else 0.0
        best_action, best_score = -1, -math.inf
        for action in range(4):
            child = node.children[action]
            if child is None:
                continue
            # An unvisited child inherits the parent's estimate rather than
            # infinity, so the prior still shapes the first few visits.
            child_value = (
                child.value_sum / child.visits if child.visits else parent_value
            )
            reward = self._reward if scratch.would_eat(action) else 0.0
            # Normalized only for the comparison. The value itself is never
            # rescaled, so backup keeps its fixed units and siblings stay
            # commensurable exactly as before.
            q = self.stats.normalize(reward + child_value)
            score = q + self.cfg.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            if score > best_score:
                best_score, best_action = score, action
        return best_action

    def _backup(self, path, rewards, value):
        # value is V(leaf) in the shared units. Walking up, V(parent) is the
        # edge reward plus V(child), with no sign alternation because there is
        # no opponent.
        current = value
        for index in range(len(path) - 1, -1, -1):
            path[index].visits += 1
            path[index].value_sum += current
            self.stats.update(path[index].value_sum / path[index].visits)
            if index > 0:
                current = current + rewards[index - 1]

    def _add_root_noise(self):
        legal = [a for a in range(4) if self.root.children[a] is not None]
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(legal))
        epsilon = self.cfg.dirichlet_epsilon
        for action, sample in zip(legal, noise, strict=True):
            child = self.root.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * float(sample)


def run_search(searches, evaluator, simulations):
    """Advance every search in lockstep, batching the leaf evaluations at each
    simulation index into a single call. This is what keeps a GPU busy without
    introducing threads or processes."""
    for _ in range(simulations):
        pending = []
        for search in searches:
            obs = search.descend()
            if obs is not None:
                pending.append((search, obs))
        if not pending:
            continue
        batch = np.stack([obs for _, obs in pending])
        priors, values = evaluator.evaluate_batch(batch)
        for (search, _), prior, value in zip(pending, priors, values, strict=True):
            search.apply(prior, value)
