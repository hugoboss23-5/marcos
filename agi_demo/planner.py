class Planner:
    """Goal-directed action planning"""

    def __init__(self, world_model, memory):
        self.world_model = world_model
        self.memory = memory
        self.actions = ["push_right", "push_left", "heat", "cool", "wait"]

    def plan(self, state, goal, max_depth=10):
        # Stored skill check
        if hasattr(goal, "get") and "task_id" in goal:
            skill = self.memory.get_skill(goal["task_id"])
            if skill:
                final_state, conf = self.simulate(state, skill)
                if self._goal_achieved(final_state, goal) and conf > 0.7:
                    return skill

        best_plan = None
        best_score = float("inf")

        frontier = [(state.copy(), [], 0, set())]  # (state, plan, cost, visited_in_path)

        for _ in range(2000):
            if not frontier:
                break

            curr_state, plan, cost, path_visited = frontier.pop(0)

            coarsened = self._coarsen_state(curr_state)
            if coarsened in path_visited:
                cost += 10
            else:
                path_visited = path_visited.copy()
                path_visited.add(coarsened)

            if self._goal_achieved(curr_state, goal):
                if cost < best_score:
                    best_score = cost
                    best_plan = plan
                continue

            if len(plan) >= max_depth:
                continue

            for action in self.actions:
                next_state, conf = self.world_model.predict(curr_state, action)
                new_cost = cost + 1 + (1.0 - conf) * 3
                frontier.append((next_state, plan + [action], new_cost, path_visited))

            frontier.sort(key=lambda x: x[2])
            frontier = frontier[:100]

        return best_plan or []

    def simulate(self, state, action_sequence):
        curr_state = state.copy()
        total_confidence = 0.0

        for action in action_sequence:
            next_state, conf = self.world_model.predict(curr_state, action)
            curr_state = next_state
            total_confidence += conf

        avg_conf = total_confidence / len(action_sequence) if action_sequence else 0.0
        return curr_state, avg_conf

    def _goal_achieved(self, state, goal):
        tol = 0.50
        if hasattr(goal, "get"):
            tol = float(goal.get("_tol", 0.50))

        for key, target in goal.items():
            if key in ("task_id", "get", "_tol"):
                continue
            if key == "state":
                if state["state"] != target:
                    return False
            else:
                if abs(state[key] - target) > tol:
                    return False
        return True

    def _coarsen_state(self, state):
        x = round(state["x"])
        y = round(state["y"])
        vx = round(state["vx"] * 2) / 2
        vy = round(state["vy"] * 2) / 2
        temp = round(state["temp"] / 10) * 10
        st = state["state"]
        return f"{x},{y},{vx},{vy},{temp},{st}"
