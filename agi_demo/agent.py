from memory import Memory
from world_model import WorldModel
from causal_library import CausalLibrary
from planner import Planner
from self_audit import SelfAudit
from compute_controller import ComputeController

class Agent:
    # Main AGI-like agent

    def __init__(self):
        self.memory = Memory()
        self.world_model = WorldModel(self.memory)
        self.causal_library = CausalLibrary(self.memory)
        self.planner = Planner(self.world_model, self.memory)
        self.self_audit = SelfAudit(self.causal_library)
        self.compute_controller = ComputeController()

        self.episode_count = 0
        self.learned_rules = 0

    def act(self, state, goal, world):
        # Adjust thinking based on confidence
        _, confidence = self.world_model.predict(state, 'wait')
        thinking_level = self.compute_controller.adjust_thinking(confidence)

        # Plan
        max_depth = self.compute_controller.get_planning_depth()
        plan = self.planner.plan(state, goal, max_depth)

        if plan:
            action = plan[0]
            predicted_state, pred_conf = self.world_model.predict(state, action)

            # Suppress computation if high confidence
            if pred_conf > 0.9 and thinking_level == 1:
                pass

            return action, predicted_state

        # No plan found, use heuristic
        return self._heuristic_action(state, goal), None

    def learn_from_experience(self, state, action, next_state, world):
        # Update world model
        surprise = self.world_model.update_from_experience(state, action, next_state)

        # Store episode
        self.memory.store_episode(state, action, next_state, surprise < 0.5)

        # Learn new rules if surprised
        if self.compute_controller.should_generate_hypotheses(surprise):
            hypotheses = self.causal_library.generate_hypotheses(state, action, surprise)

            best_hyp = None
            best_gain = 0.0

            for hyp in hypotheses:
                err = self.causal_library.test_hypothesis(hyp, state, action, world)
                gain = surprise - err
                if gain > best_gain:
                    best_gain = gain
                    best_hyp = hyp

            # Accept if it reduces error by at least 20%
            if best_hyp is not None and best_gain > surprise * 0.2:
                if self.causal_library.add_rule(best_hyp):
                    self.learned_rules += 1
                    print(f"  Learned rule: {best_hyp['type']}")

        # Run self-audit if failure was large
        if self.self_audit.should_activate(surprise):
            predicted_state, _ = self.world_model.predict(state, action)
            diagnosis, new_rule = self.self_audit.analyze_failure(
                state, {}, next_state, predicted_state, action
            )
            if new_rule:
                # ONLY print if rule actually got added (prevents spam)
                if self.causal_library.add_rule(new_rule):
                    print(f"  Self-audit added rule: {diagnosis}")

    def transfer_skill(self, task_id, world):
        state = world.reset(task_id)
        goal = world.get_goal(task_id)

        steps = 0
        max_steps = 30

        while steps < max_steps:
            action, _ = self.act(state, goal, world)
            next_state = world.step(state.copy(), action)

            self.learn_from_experience(state, action, next_state, world)

            if world.goal_achieved(next_state, task_id):
                self.memory.store_skill(task_id, [action])
                return True, steps + 1

            state = next_state
            steps += 1

        return False, steps

    def _heuristic_action(self, state, goal):
        if 'x' in goal and abs(state['x'] - goal['x']) > 0.5:
            return 'push_right' if goal['x'] > state['x'] else 'push_left'
        if 'state' in goal and state['state'] != goal['state']:
            return 'heat' if goal['state'] > state['state'] else 'cool'
        return 'wait'

    def get_stats(self):
        return {
            'episodes': self.episode_count,
            'rules_learned': self.learned_rules,
            'memory_episodes': len(self.memory.episodes),
            'thinking_level': self.compute_controller.thinking_level,
        }
