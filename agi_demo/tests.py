#!/usr/bin/env python3
import unittest

from world import PhysicsWorld
from agent import Agent


class TestAGIDemo(unittest.TestCase):
    def test_world_deterministic(self):
        w = PhysicsWorld()
        s = w.reset(1)
        a = "wait"
        s1 = w.step(s.copy(), a)
        s2 = w.step(s.copy(), a)
        self.assertEqual(s1, s2)

    def test_world_gravity_effect(self):
        w = PhysicsWorld()
        s = w.reset(1)
        s["vy"] = 0.0
        s2 = w.step(s.copy(), "wait")
        self.assertLess(s2["y"], s["y"])

    def test_agent_learns_some_rule(self):
        w = PhysicsWorld()
        agent = Agent()

        # Learn a bit
        for _ in range(10):
            s = w.reset(1)
            a = "wait"
            s2 = w.step(s.copy(), a)
            agent.learn_from_experience(s, a, s2, w)

        stats = agent.get_stats()
        self.assertGreater(stats["memory_episodes"], 0)

    def test_transfer_task2_possible(self):
        w = PhysicsWorld()
        agent = Agent()
        ok, steps = agent.transfer_skill(2, w)
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(steps, int)

    # -----------------------------
    # New: Reachability validator tests
    # -----------------------------
    def test_task1_unsat_at_strict_tol(self):
        """
        Empirically certified: at tol=0.50, no plan found within 30 steps
        under the discretized BFS budget. This prevents 'optimize an impossible spec'.
        """
        w = PhysicsWorld()
        reachable, path, expanded = w.reachability_check(
            1, max_steps=30, tol=0.50, max_expansions=250_000
        )
        self.assertFalse(reachable, f"Expected UNSAT at tol=0.50 but found path={path} expanded={expanded}")
        self.assertEqual(path, [])

    def test_task1_sat_at_relaxed_tol(self):
        """
        Minimal spec change: relax tolerance from 0.50 -> 0.55.
        At tol=0.55, reachability finds a 6-step witness quickly.
        """
        w = PhysicsWorld()
        reachable, path, expanded = w.reachability_check(
            1, max_steps=30, tol=0.55, max_expansions=250_000
        )
        self.assertTrue(reachable, f"Expected SAT at tol=0.55 but none found (expanded={expanded})")
        self.assertGreater(len(path), 0)
        self.assertLessEqual(len(path), 30)

        # Verify the witness actually reaches the tol=0.55 goal
        s = w.reset(1)
        for a in path:
            s = w.step(s.copy(), a)
        self.assertTrue(w.goal_achieved(s, 1, tol=0.55), f"Witness failed. Final state: {s}, path={path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
