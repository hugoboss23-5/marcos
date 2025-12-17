class PhysicsWorld:
    """Deterministic toy world with hidden rules"""

    def __init__(self):
        self.hidden_rules = {
            "gravity": 0.3,
            "friction": 0.95,
            "temp_threshold": 50.0,
        }
        # Default per-task tolerance used by the *spec* (planner should respect this)
        self.task_tol = {1: 0.55, 2: 0.50}

    def step(self, state, action):
        s = state.copy()

        # Apply action
        if action == "push_right":
            s["vx"] += 1.0
        elif action == "push_left":
            s["vx"] -= 1.0
        elif action == "heat":
            s["temp"] += 15.0
        elif action == "cool":
            s["temp"] -= 15.0
        elif action == "wait":
            pass
        else:
            raise ValueError(f"Unknown action: {action}")

        # Hidden physics
        s["vy"] -= self.hidden_rules["gravity"]
        s["vx"] *= self.hidden_rules["friction"]
        s["vy"] *= self.hidden_rules["friction"]

        s["x"] += s["vx"]
        s["y"] += s["vy"]

        # Hidden state transitions
        if s["temp"] > self.hidden_rules["temp_threshold"]:
            if s["state"] == 0:
                s["state"] = 1
            elif s["state"] == 1:
                s["state"] = 2
        elif s["temp"] < 20.0:
            if s["state"] == 2:
                s["state"] = 0

        # Bounds
        if s["x"] < -10:
            s["x"] = -10
        if s["x"] > 10:
            s["x"] = 10
        if s["y"] < 0:
            s["y"] = 0
        if s["y"] > 20:
            s["y"] = 20

        return s

    def get_tasks(self):
        return {
            1: {
                "goal_state": {"x": 8.0, "y": 5.0, "state": 0},
                "initial_state": {"x": 0.0, "y": 10.0, "vx": 0.0, "vy": 0.0, "temp": 25.0, "state": 0},
            },
            2: {
                "goal_state": {"state": 2},
                "initial_state": {"x": 5.0, "y": 5.0, "vx": 0.0, "vy": 0.0, "temp": 25.0, "state": 0},
            },
        }

    def reset(self, task_id):
        return self.get_tasks()[task_id]["initial_state"].copy()

    def get_goal(self, task_id):
        # IMPORTANT: embed the spec tolerance into the goal so planners can’t “solve the wrong problem”.
        g = self.get_tasks()[task_id]["goal_state"].copy()
        g["task_id"] = task_id
        g["_tol"] = self.task_tol.get(task_id, 0.50)
        return g

    def goal_achieved(self, state, task_id, tol=None):
        goal = self.get_goal(task_id)
        use_tol = goal.get("_tol", 0.50) if tol is None else tol

        for key, target in goal.items():
            if key in ("task_id", "_tol"):
                continue
            if key == "state":
                if state["state"] != target:
                    return False
            else:
                if abs(state[key] - target) > use_tol:
                    return False
        return True

    def reachability_check(self, task_id, max_steps=30, tol=None, max_expansions=250_000):
        # Note: this is your existing validator interface; keep behavior consistent.
        from collections import deque

        initial = self.reset(task_id)
        goal = self.get_goal(task_id)
        use_tol = goal.get("_tol", 0.50) if tol is None else tol

        actions = ["push_right", "push_left", "heat", "cool", "wait"]

        def discretize(st):
            # coarse hash to prevent infinite explosion
            return (
                round(st["x"] * 2) / 2,
                round(st["y"] * 2) / 2,
                round(st["vx"] * 2) / 2,
                round(st["vy"] * 2) / 2,
                round(st["temp"] / 5) * 5,
                st["state"],
            )

        def matches(st):
            for k, t in goal.items():
                if k in ("task_id", "_tol"):
                    continue
                if k == "state":
                    if st["state"] != t:
                        return False
                else:
                    if abs(st[k] - t) > use_tol:
                        return False
            return True

        q = deque()
        q.append((initial.copy(), []))
        visited = {discretize(initial)}
        expanded = 0

        while q and expanded < max_expansions:
            st, path = q.popleft()
            expanded += 1

            if matches(st):
                return True, path, expanded

            if len(path) >= max_steps:
                continue

            for a in actions:
                ns = self.step(st.copy(), a)
                d = discretize(ns)
                if d not in visited:
                    visited.add(d)
                    q.append((ns, path + [a]))

        return False, [], expanded
