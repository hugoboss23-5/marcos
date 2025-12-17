class ComputeController:
    """Controls computational effort based on confidence"""

    def __init__(self):
        self.thinking_level = 1  # 1=minimal, 3=deep
        self.confidence_history = []

    def adjust_thinking(self, confidence):
        self.confidence_history.append(float(confidence))
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)

        avg = sum(self.confidence_history) / len(self.confidence_history)

        if avg > 0.8:
            self.thinking_level = 1
        elif avg > 0.5:
            self.thinking_level = 2
        else:
            self.thinking_level = 3

        return self.thinking_level

    def get_planning_depth(self):
        return {1: 5, 2: 10, 3: 15}[self.thinking_level]

    def should_generate_hypotheses(self, surprise):
        return float(surprise) > 0.7 or self.thinking_level == 3
