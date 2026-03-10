"""
Outer Ensemble — Multi-Modal Genetic Algorithm with Niche Sharing.

Finds the optimal (and simplest) combination of base classifiers by
maximizing G-mean of the ensemble on the original (imbalanced) training set.

Algorithm follows Paper Algorithm 2 and Section 6 parameters exactly.
Base classifier predictions are precomputed once per run for efficiency.
"""

import numpy as np
from .metrics import g_mean_score


class MultiModalGA:
    """
    Multi-modal GA with fitness sharing for ensemble structure selection.

    Parameters (from Paper Section 6)
    ----------------------------------
    pop_size      : 50
    max_iter      : 100
    pc            : 0.6   single-point crossover probability
    pm            : 0.08  single-point bit-flip mutation probability
    niche_radius  : 2.0   sharing radius L (inferred; paper uses symbol only)
    archive_size  : 50
    random_state  : 42
    """

    def __init__(
        self,
        pop_size: int = 50,
        max_iter: int = 100,
        pc: float = 0.6,
        pm: float = 0.08,
        niche_radius: float = 2.0,
        archive_size: int = 50,
        random_state: int = 42,
    ):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.pc = pc
        self.pm = pm
        self.niche_radius = niche_radius
        self.archive_size = archive_size
        self.random_state = random_state

        # Populated after run()
        self.archive_: list = []
        self.best_fitness_history_: list = []

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, classifiers: list, X_train, y_train) -> np.ndarray:
        """
        Search for the optimal ensemble structure.

        Parameters
        ----------
        classifiers : list of n fitted estimators (n = 30)
        X_train, y_train : original imbalanced training data

        Returns
        -------
        selected_indices : ndarray of int — indices into `classifiers`
        """
        rng = np.random.RandomState(self.random_state)
        n = len(classifiers)  # individual length = n_sub_datasets

        # Precompute base-classifier predictions on X_train (major speedup)
        all_probas = np.stack(
            [clf.predict_proba(X_train) for clf in classifiers]
        )  # shape (n, N, 2)

        # Initialise binary population; each row = one individual
        pop = rng.randint(0, 2, size=(self.pop_size, n)).astype(np.float64)
        # Ensure at least one active gene per individual
        for i in range(self.pop_size):
            if pop[i].sum() == 0:
                pop[i, rng.randint(0, n)] = 1.0

        self.archive_ = []
        self.best_fitness_history_ = []

        for _ in range(self.max_iter):
            fitness = self._evaluate(pop, all_probas, y_train)
            self.best_fitness_history_.append(float(fitness.max()))

            # Update external archive with current best individual
            self._update_archive(pop, fitness)

            # Reproduce: selection → crossover → mutation
            offspring = self._roulette_select(pop, fitness, rng)
            offspring = self._crossover(offspring, rng)
            offspring = self._mutation(offspring, n, rng)

            # Combine parent population + offspring
            combined = np.vstack([pop, offspring])
            combined_fit = self._evaluate(combined, all_probas, y_train)

            # Niche-based survivor selection → next generation
            pop = self._niche_select(combined, combined_fit)

        return self._select_simplest()

    # ── Evaluation ───────────────────────────────────────────────────────────

    @staticmethod
    def _evaluate(population: np.ndarray, all_probas: np.ndarray,
                  y: np.ndarray) -> np.ndarray:
        """
        Compute G-mean for each individual using precomputed probabilities.

        all_probas : shape (n_classifiers, N_train, 2)
        """
        fitness = np.zeros(len(population))
        for i, individual in enumerate(population):
            active = np.where(individual > 0.5)[0]
            if len(active) == 0:
                continue
            # Equal-weight average of selected classifiers' probabilities
            avg_proba = all_probas[active].mean(axis=0)   # (N_train, 2)
            pred = np.argmax(avg_proba, axis=1)
            fitness[i] = g_mean_score(y, pred)
        return fitness

    # ── GA operators ─────────────────────────────────────────────────────────

    @staticmethod
    def _roulette_select(pop: np.ndarray, fitness: np.ndarray,
                         rng: np.random.RandomState) -> np.ndarray:
        """Fitness-proportionate (roulette wheel) selection."""
        f = fitness - fitness.min() + 1e-10
        probs = f / f.sum()
        idx = rng.choice(len(pop), size=len(pop), p=probs, replace=True)
        return pop[idx].copy()

    def _crossover(self, pool: np.ndarray,
                   rng: np.random.RandomState) -> np.ndarray:
        """Single-point crossover applied with probability pc."""
        offspring = pool.copy()
        n_ind, n_genes = pool.shape
        for i in range(0, n_ind - 1, 2):
            if rng.random() < self.pc:
                point = rng.randint(1, n_genes)
                offspring[i, point:] = pool[i + 1, point:].copy()
                offspring[i + 1, point:] = pool[i, point:].copy()
        return offspring

    def _mutation(self, pool: np.ndarray, n_genes: int,
                  rng: np.random.RandomState) -> np.ndarray:
        """
        Single-point bit-flip mutation.
        With probability pm, flip one randomly chosen gene per individual.
        """
        offspring = pool.copy()
        for i in range(len(offspring)):
            if rng.random() < self.pm:
                gene = rng.randint(0, n_genes)
                offspring[i, gene] = 1.0 - offspring[i, gene]
        # Ensure each individual stays valid (≥1 active gene)
        for i in range(len(offspring)):
            if offspring[i].sum() == 0:
                offspring[i, rng.randint(0, n_genes)] = 1.0
        return offspring

    def _niche_select(self, combined: np.ndarray,
                      fitness: np.ndarray) -> np.ndarray:
        """
        Fitness sharing: divide each individual's fitness by its niche count,
        then keep the top pop_size individuals.

        Niche count for individual i = Σ_j sh(d_ij)
        where sh(d) = 1 - d/L  if d < L,  else 0
        """
        n = len(combined)
        # Vectorised pairwise Euclidean distances
        diff = combined[:, np.newaxis, :] - combined[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))            # (n, n)

        sh = np.where(dist < self.niche_radius,
                      1.0 - dist / self.niche_radius, 0.0)  # (n, n)
        niche_counts = sh.sum(axis=1)                       # (n,)

        shared = fitness / np.maximum(niche_counts, 1e-10)
        idx = np.argsort(shared)[::-1][: self.pop_size]
        return combined[idx]

    # ── Archive ───────────────────────────────────────────────────────────────

    def _update_archive(self, population: np.ndarray,
                        fitness: np.ndarray) -> None:
        """
        Add the best individual of the current generation to the archive.
        Evict the worst (lowest fitness) entry when the archive is full.
        Duplicate individuals (identical bit-strings) are not added.
        """
        best_i = int(np.argmax(fitness))
        candidate_ind = population[best_i].copy()
        candidate_fit = float(fitness[best_i])

        # Skip duplicates
        for ind, _ in self.archive_:
            if np.array_equal(ind, candidate_ind):
                return

        self.archive_.append((candidate_ind, candidate_fit))

        if len(self.archive_) > self.archive_size:
            # Remove the entry with the lowest fitness
            self.archive_.sort(key=lambda x: x[1])
            self.archive_.pop(0)

    def _select_simplest(self) -> np.ndarray:
        """
        From the archive, select the simplest ensemble structure:
          xcout = argmin_{xci ∈ Pbest} Σ xcij
        where Pbest = archive entries within 1 % of the best fitness.
        Ties broken by highest fitness.

        Returns indices of active (selected) base classifiers.
        """
        if not self.archive_:
            return np.array([0])

        max_fit = max(f for _, f in self.archive_)
        threshold = max_fit * 0.99
        candidates = [
            (ind, f) for ind, f in self.archive_ if f >= threshold
        ]

        # (min active genes, then max fitness as tiebreak)
        simplest_ind, _ = min(candidates, key=lambda x: (x[0].sum(), -x[1]))
        return np.where(simplest_ind > 0.5)[0]
