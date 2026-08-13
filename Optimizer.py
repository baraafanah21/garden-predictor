import random
import math


def euclideanDist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


class SimulatedAnnealingOptimizer:
    def __init__(self, plants, start_temp=100.0, cooling_rate=0.95,
                 max_iter=500, miss_penalty=50, extra_penalty=50,
                 convergence_patience=50):

        self.plants = plants
        self.needs_water = [p for p in plants if p.get("predicted", 0) == 1]
        self.start_temp = start_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.miss_penalty = miss_penalty
        self.extra_penalty = extra_penalty
        self.convergence_patience = convergence_patience
        self.history = []

    def total_distance(self, sequence):
        if len(sequence) < 2:
            return 0.0
        distance = 0.0
        for i in range(len(sequence) - 1):
            p1 = (sequence[i]["x"], sequence[i]["y"])
            p2 = (sequence[i + 1]["x"], sequence[i + 1]["y"])
            distance += euclideanDist(p1, p2)
        return distance

    def costFun(self, sequence):
        selectedIds = {p["id"] for p in sequence}
        missed = sum(1 for p in self.needs_water if p["id"] not in selectedIds)
        distance = self.total_distance(sequence)
        extra = sum(1 for p in sequence if p.get("predicted", 0) == 0)
        return distance + (missed * self.miss_penalty) + (extra * self.extra_penalty)

    def get_initial_solution(self, selectedCount):
        """Randomly pick selectedCount plants from ALL plants (needy or not)."""
        sol = random.sample(self.plants, selectedCount)
        random.shuffle(sol)
        return sol, self.costFun(sol)

    def replace_with_closest(self, sol):
        """Swap each non-needy plant in sol with the closest needy plant not in sol."""
        newSol = sol[:]
        selected_ids = {p["id"] for p in newSol}
        needy_outside = [p for p in self.needs_water if p["id"] not in selected_ids]

        for i, plant in enumerate(newSol):
            if plant.get("predicted", 0) == 0 and needy_outside:
                px, py = plant["x"], plant["y"]
                closest = min(needy_outside,
                              key=lambda q: euclideanDist((px, py), (q["x"], q["y"])))
                needy_outside.remove(closest)
                selected_ids.discard(plant["id"])
                selected_ids.add(closest["id"])
                newSol[i] = closest

        return newSol

    def replace_non_needy(self, sol):
        """Replace non-needy plants in sol with random needy plants not in sol (used by neighbor)."""
        newSol = sol[:]
        selected_ids = {p["id"] for p in newSol}
        needy_outside = [p for p in self.needs_water if p["id"] not in selected_ids]

        for i, plant in enumerate(newSol):
            if plant.get("predicted", 0) == 0 and needy_outside:
                replacement = random.choice(needy_outside)
                needy_outside.remove(replacement)
                selected_ids.discard(plant["id"])
                selected_ids.add(replacement["id"])
                newSol[i] = replacement

        return newSol

    def neighbor(self, sol):
        newSol = sol[:]
        if len(newSol) == 0:
            return newSol

        if random.random() < 0.7 and len(newSol) >= 2:
            i, j = random.sample(range(len(newSol)), 2)
            newSol[i], newSol[j] = newSol[j], newSol[i]
        else:
            current_ids = {p["id"] for p in newSol}
            outside = [p for p in self.needs_water if p["id"] not in current_ids]
            if outside:
                i = random.randrange(len(newSol))
                newSol[i] = random.choice(outside)
            elif len(newSol) >= 2:
                i, j = random.sample(range(len(newSol)), 2)
                newSol[i], newSol[j] = newSol[j], newSol[i]
        return self.replace_non_needy(newSol)

    def run_sa(self, initial_sol):
        """Run SA starting from initial_sol. Returns (bestSol, bestCost)."""
        self.history = []
        currentSol = initial_sol[:]
        currentCost = self.costFun(currentSol)
        bestSol = currentSol[:]
        bestCost = currentCost
        temperature = self.start_temp
        stagnant_steps = 0

        for step in range(self.max_iter):
            newSol = self.neighbor(currentSol)
            newCost = self.costFun(newSol)
            delta = newCost - currentCost

            if delta < 0:
                currentSol = newSol
                currentCost = newCost
            else:
                probability = math.exp(-delta / temperature) if temperature > 0 else 0
                if random.random() < probability:
                    currentSol = newSol
                    currentCost = newCost

            if currentCost < bestCost:
                bestSol = currentSol[:]
                bestCost = currentCost
                stagnant_steps = 0
            else:
                stagnant_steps += 1

            self.history.append({
                "step": step,
                "temperature": temperature,
                "currentCost": currentCost,
                "bestCost": bestCost
            })

            temperature *= self.cooling_rate

            if stagnant_steps >= self.convergence_patience:
                break
            if temperature < 0.001:
                break

        return bestSol, bestCost

    def optimize(self, selectedCount):
        if selectedCount <= 0:
            raise ValueError("selectedCount must be greater than 0.")
        if not self.needs_water:
            raise ValueError("No plants are predicted to need water.")
        if selectedCount > len(self.plants):
            raise ValueError(
                f"selectedCount cannot be greater than {len(self.plants)} total plants."
            )

        initial_sol, _ = self.get_initial_solution(selectedCount)
        return self.run_sa(initial_sol)
