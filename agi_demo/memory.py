import json
import hashlib

class Memory:
    """Episodic memory + rule storage (with semantic canonicalization + tolerance de-dupe)."""

    def __init__(self):
        self.episodes = []
        self.rules = []
        self.skills = {}  # task_id -> successful action sequence

    # -------------------------
    # Episodes / skills
    # -------------------------
    def store_episode(self, state, action, next_state, success):
        episode = {
            'state': state.copy(),
            'action': action,
            'next_state': next_state.copy(),
            'success': bool(success),
            'hash': self._state_hash(state),
        }
        self.episodes.append(episode)
        if len(self.episodes) > 1000:
            self.episodes = self.episodes[-1000:]

    def get_similar_episodes(self, state, k=5):
        if not self.episodes:
            return []
        target_hash = self._state_hash(state)
        scored = []
        for ep in self.episodes[-100:]:
            sim = 1.0 if ep['hash'] == target_hash else 0.0
            scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def store_skill(self, task_id, action_sequence):
        self.skills[task_id] = list(action_sequence)

    def get_skill(self, task_id):
        return self.skills.get(task_id, None)

    # -------------------------
    # Rules (canonical + dedupe)
    # -------------------------
    def store_rule(self, rule):
        """
        Store a rule if it's not a semantic duplicate of an existing rule.
        We intentionally ignore non-semantic metadata like 'source' so audits/tests
        don't create duplicates.
        """
        canonical = self._canonicalize_rule(rule)

        for existing in self.rules:
            if self._rules_equal(canonical, existing):
                return False

        self.rules.append(canonical)
        return True

    def get_rules(self):
        return [r.copy() for r in self.rules]

    # -------------------------
    # Internals
    # -------------------------
    def _state_hash(self, state):
        state_str = json.dumps(state, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()[:8]

    def _canonicalize_rule(self, rule):
        """
        Keep only semantic fields + apply physics-aware rounding.
        This prevents duplicate rules that differ only by metadata (e.g. 'source').
        """
        if not isinstance(rule, dict) or 'type' not in rule:
            return {'type': 'unknown'}

        t = rule.get('type')

        # Only keep semantic keys per rule type
        if t == 'gravity':
            val = float(rule.get('value', 0.0))
            return {'type': 'gravity', 'value': round(val, 2)}  # 0.01 granularity
        if t == 'friction':
            val = float(rule.get('value', 1.0))
            return {'type': 'friction', 'value': round(val, 3)}  # 0.001 granularity
        if t == 'state_transition':
            thr = float(rule.get('threshold', 0.0))
            ns = int(rule.get('new_state', 0))
            return {'type': 'state_transition', 'threshold': round(thr, 1), 'new_state': ns}

        # Fallback: strip obvious metadata keys but preserve remaining fields
        stripped = {}
        for k, v in rule.items():
            if k in ('source', 'timestamp', 'meta', 'debug'):
                continue
            stripped[k] = v
        # ensure stable ordering via json roundtrip
        return json.loads(json.dumps(stripped, sort_keys=True))

    def _rules_equal(self, a, b):
        """
        Physics-aware tolerance equality.
        Assumes both rules already canonicalized (no 'source', rounded).
        """
        if a.get('type') != b.get('type'):
            return False

        t = a.get('type')
        if t == 'gravity':
            return abs(float(a.get('value', 0.0)) - float(b.get('value', 0.0))) < 0.05
        if t == 'friction':
            return abs(float(a.get('value', 1.0)) - float(b.get('value', 1.0))) < 0.01
        if t == 'state_transition':
            return (
                abs(float(a.get('threshold', 0.0)) - float(b.get('threshold', 0.0))) < 5.0
                and int(a.get('new_state', 0)) == int(b.get('new_state', 0))
            )

        # Generic dict equality for unknown rule types after canonicalization
        return a == b
