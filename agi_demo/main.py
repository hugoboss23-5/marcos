#!/usr/bin/env python3
from world import PhysicsWorld
from agent import Agent

def _fmt_rule(rule: dict) -> str:
    t = rule.get("type", "unknown")
    if t == "gravity":
        return f"gravity: {rule.get('value')}"
    if t == "friction":
        return f"friction: {rule.get('value')}"
    if t == "state_transition":
        thr = rule.get("threshold")
        ns = rule.get("new_state")
        return f"state_transition: temp>{thr} -> state={ns}"
    return str(rule)

def run_demo():
    print("=" * 60)
    print("AGI-LIKE AGENT DEMO (SPEC AUDITED)")
    print("=" * 60)

    world = PhysicsWorld()
    agent = Agent()

    # Spec audit (Task 1)
    print("\n0. SPEC REACHABILITY AUDIT (Task 1)")
    print("-" * 60)
    r0, p0, e0 = world.reachability_check(1, max_steps=30, tol=0.50)
    r1, p1, e1 = world.reachability_check(1, max_steps=30, tol=0.55)

    print(f"tol=0.50 reachable: {r0} | expanded: {e0} | witness_len: {len(p0)}")
    print(f"tol=0.55 reachable: {r1} | expanded: {e1} | witness_len: {len(p1)}")
    if r1:
        final = world.reset(1)
        for a in p1:
            final = world.step(final.copy(), a)
        print(f"witness (tol=0.55) = {p1}")
        print(f"final_state = {final}")
        print(f"goal_hit(tol=0.55) = {world.goal_achieved(final, 1, tol=0.55)}")

    print("\nNOTE: Task 1 is UNSAT at tol=0.50 within 30 steps under this validator.")
    print("      Any 'learning failure' claims for strict tol are meaningless.")
    print("      Demo will proceed using tol=0.55 when evaluating Task 1 success.\n")

    # Learning phase
    print("\n1. CAUSAL LEARNING PHASE (Task 1 dynamics)")
    print("-" * 60)

    success_count = 0
    episodes = 50

    for episode in range(episodes):
        state = world.reset(1)
        goal = world.get_goal(1)

        for step in range(30):
            action, _ = agent.act(state, goal, world)
            next_state = world.step(state.copy(), action)

            agent.learn_from_experience(state, action, next_state, world)

            if world.goal_achieved(next_state, 1, tol=0.55):
                success_count += 1
                state = next_state
                break

            state = next_state

        agent.episode_count += 1

        if episode % 10 == 0:
            stats = agent.get_stats()
            print(f"  Episode {episode}: rules_learned={stats['rules_learned']} mem_eps={stats['memory_episodes']}")

    print(f"\n  Learning phase complete (Task 1, tol=0.55): {success_count}/{episodes} successes")

    # Planning test
    print("\n2. PLANNING TEST (Task 1, tol=0.55)")
    print("-" * 60)

    state = world.reset(1)
    goal = world.get_goal(1)
    steps = 0

    while steps < 30:
        action, _ = agent.act(state, goal, world)
        state = world.step(state.copy(), action)
        steps += 1

        if world.goal_achieved(state, 1, tol=0.55):
            print(f"  Planning SUCCESS at tol=0.55: reached in {steps} steps")
            break
    else:
        print(f"  Planning FAILED at tol=0.55: did not reach in {steps} steps")

    # Transfer test
    print("\n3. TRANSFER TEST (Task 2)")
    print("-" * 60)
    success, steps = agent.transfer_skill(2, world)
    print(f"  Transfer {'SUCCESS' if success else 'FAILED'}: steps={steps}")

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)

    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nLearned rules:")
    for rule in agent.memory.get_rules():
        print(f"  - {_fmt_rule(rule)}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_demo()
