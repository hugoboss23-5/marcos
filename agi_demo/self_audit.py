class SelfAudit:
    # Analyzes failures and generates corrections

    def __init__(self, causal_library):
        self.causal_library = causal_library
        self.failure_history = []

    def analyze_failure(self, state, goal, actual_outcome, prediction, action):
        # Diagnose failure and propose fix
        # IMPORTANT: friction before gravity

        diagnosis = "Unknown failure mode"
        new_rule = None

        # Friction diagnosis: vx/vy mismatch (slowing) is usually friction
        if abs(prediction.get('vx', 0.0) - actual_outcome.get('vx', 0.0)) > 0.1 or \
           abs(prediction.get('vy', 0.0) - actual_outcome.get('vy', 0.0)) > 0.1:
            diagnosis = "Velocity mismatch (friction missing?)"
            new_rule = {'type': 'friction', 'value': 0.95, 'source': 'audit'}

        # Gravity diagnosis: y drift unexplained
        elif abs(prediction.get('y', 0.0) - actual_outcome.get('y', 0.0)) > 0.5:
            diagnosis = "Vertical prediction error (gravity missing?)"
            new_rule = {'type': 'gravity', 'value': 0.3, 'source': 'audit'}

        # State transition diagnosis
        elif prediction.get('state') != actual_outcome.get('state'):
            diagnosis = "State transition incorrect (threshold rule missing?)"
            new_rule = {
                'type': 'state_transition',
                'threshold': 50.0,
                'new_state': actual_outcome.get('state'),
                'source': 'audit'
            }

        self.failure_history.append({
            'diagnosis': diagnosis,
            'state': state,
            'goal': goal,
            'action': action
        })

        return diagnosis, new_rule

    def should_activate(self, prediction_error):
        return prediction_error > 0.5
