class WorldModel:
    """Predictive model of world dynamics"""

    def __init__(self, memory):
        self.memory = memory
        self.prediction_cache = {}
        self.recent_errors = []
        self.learned_dynamics = {}

    def predict(self, state, action):
        """Predict next state given action"""
        cache_key = (
            f"{state['x']:.1f},{state['y']:.1f},{state['vx']:.1f},{state['vy']:.1f},"
            f"{state['temp']:.1f},{state['state']},{action}"
        )

        if cache_key in self.prediction_cache:
            pred, conf = self.prediction_cache[cache_key]
            return pred.copy(), conf

        pred = state.copy()

        # Apply known action effects
        if action == 'push_right':
            pred['vx'] += 1.0
        elif action == 'push_left':
            pred['vx'] -= 1.0
        elif action == 'heat':
            pred['temp'] += 15.0
        elif action == 'cool':
            pred['temp'] -= 15.0
        elif action == 'wait':
            pass

        # Apply learned rules
        for rule in self.memory.get_rules():
            pred = self._apply_rule(pred, rule)

        # Update position
        pred['x'] += pred['vx']
        pred['y'] += pred['vy']

        confidence = self._calculate_confidence()

        self.prediction_cache[cache_key] = (pred.copy(), confidence)
        return pred, confidence

    def update_from_experience(self, state, action, next_state):
        """Learn from discrepancy"""
        pred, _ = self.predict(state, action)

        # Clear cache since model might be updated soon
        self.prediction_cache = {}

        error = self._calculate_error(pred, next_state)

        self.recent_errors.append(error)
        if len(self.recent_errors) > 50:
            self.recent_errors = self.recent_errors[-50:]

        return error

    def get_prediction_error(self):
        """Average error of recent predictions"""
        if not self.recent_errors:
            return 1.0
        return sum(self.recent_errors) / len(self.recent_errors)

    def _calculate_confidence(self):
        """Real confidence based on recent prediction errors"""
        if not self.recent_errors:
            return 0.1

        avg_error = self.get_prediction_error()
        # map error -> confidence (bounded)
        conf = 1.0 / (1.0 + avg_error)
        conf = max(0.05, min(0.95, conf))
        return conf

    def _apply_rule(self, state, rule):
        """Apply a causal rule to state"""
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
        """Calculate total prediction error"""
        error = 0.0
        for key in ['x', 'y', 'vx', 'vy', 'temp']:
            error += abs(pred[key] - actual[key])
        error += abs(pred['state'] - actual['state']) * 2.0
        return error
