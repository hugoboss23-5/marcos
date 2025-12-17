class CausalLibrary:
    """Generates and tests causal hypotheses"""

    def __init__(self, memory):
        self.memory = memory

        # Expanded hypothesis templates with better coverage
        self.templates = [
            # Gravity values around true value (0.3)
            {'type': 'gravity', 'value': 0.1, 'test_range': [0.1, 0.2, 0.3, 0.4, 0.5]},
            {'type': 'gravity', 'value': 0.2, 'test_range': [0.1, 0.2, 0.3, 0.4, 0.5]},
            {'type': 'gravity', 'value': 0.3, 'test_range': [0.1, 0.2, 0.3, 0.4, 0.5]},
            {'type': 'gravity', 'value': 0.4, 'test_range': [0.1, 0.2, 0.3, 0.4, 0.5]},
            {'type': 'gravity', 'value': 0.5, 'test_range': [0.1, 0.2, 0.3, 0.4, 0.5]},

            # Friction values around true value (0.95)
            {'type': 'friction', 'value': 0.8,  'test_range': [0.8, 0.85, 0.9, 0.95, 0.99]},
            {'type': 'friction', 'value': 0.9,  'test_range': [0.8, 0.85, 0.9, 0.95, 0.99]},
            {'type': 'friction', 'value': 0.95, 'test_range': [0.8, 0.85, 0.9, 0.95, 0.99]},
            {'type': 'friction', 'value': 0.97, 'test_range': [0.8, 0.85, 0.9, 0.95, 0.99]},
            {'type': 'friction', 'value': 0.99, 'test_range': [0.8, 0.85, 0.9, 0.95, 0.99]},

            # State transition thresholds
            {'type': 'state_transition', 'threshold': 30.0, 'new_state': 1, 'test_range': [30, 40, 50, 60]},
            {'type': 'state_transition', 'threshold': 40.0, 'new_state': 1, 'test_range': [30, 40, 50, 60]},
            {'type': 'state_transition', 'threshold': 50.0, 'new_state': 1, 'test_range': [30, 40, 50, 60]},
            {'type': 'state_transition', 'threshold': 50.0, 'new_state': 2, 'test_range': [30, 40, 50, 60]},
            {'type': 'state_transition', 'threshold': 60.0, 'new_state': 2, 'test_range': [30, 40, 50, 60]},
        ]

    def generate_hypotheses(self, state, action, surprise):
        """Generate causal hypotheses when surprised"""
        hypotheses = []

        for template in self.templates:
            if self._rule_exists(template):
                continue

            for val in template.get('test_range', [template.get('value')]):
                hyp = template.copy()
                if 'value' in hyp:
                    hyp['value'] = float(val)
                if 'threshold' in hyp:
                    hyp['threshold'] = float(val)
                hypotheses.append(hyp)

        return hypotheses[:15]

    def test_hypothesis(self, hypothesis, state, action, world):
        """Test hypothesis in mental simulation"""
        test_state = state.copy()

        # Apply action
        if action == 'push_right':
            test_state['vx'] += 1.0
        elif action == 'push_left':
            test_state['vx'] -= 1.0
        elif action == 'heat':
            test_state['temp'] += 15.0
        elif action == 'cool':
            test_state['temp'] -= 15.0
        elif action == 'wait':
            pass

        # Apply hypothesis
        test_state = self._apply_rule(test_state, hypothesis)

        # Update position
        test_state['x'] += test_state['vx']
        test_state['y'] += test_state['vy']

        actual_state = world.step(state.copy(), action)
        return self._calculate_error(test_state, actual_state)

    def add_rule(self, rule):
        """Add validated rule to memory"""
        return self.memory.store_rule(rule)

    def apply_rules(self, state):
        result = state.copy()
        for rule in self.memory.get_rules():
            result = self._apply_rule(result, rule)
        return result

    def _apply_rule(self, state, rule):
        s = state.copy()
        t = rule.get('type')

        if t == 'gravity':
            s['vy'] -= rule['value']
        elif t == 'friction':
            s['vx'] *= rule['value']
            s['vy'] *= rule['value']
        elif t == 'state_transition':
            if s['temp'] > rule['threshold']:
                s['state'] = rule['new_state']

        return s

    def _calculate_error(self, pred, actual):
        error = 0.0
        for key in ['x', 'y', 'vx', 'vy', 'temp']:
            error += abs(pred[key] - actual[key])
        error += abs(pred['state'] - actual['state']) * 2.0
        return error

    def _rule_exists(self, rule):
        existing = self.memory.get_rules()

        for r in existing:
            if r.get('type') != rule.get('type'):
                continue

            t = r.get('type')
            if t == 'gravity':
                if abs(r['value'] - rule.get('value', r['value'])) < 0.05:
                    return True
            elif t == 'friction':
                if abs(r['value'] - rule.get('value', r['value'])) < 0.01:
                    return True
            elif t == 'state_transition':
                if (abs(r['threshold'] - rule.get('threshold', r['threshold'])) < 5.0 and
                    r.get('new_state') == rule.get('new_state')):
                    return True

        return False
