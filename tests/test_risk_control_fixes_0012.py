"""Task 0012: conformal risk control — realized false-action rate <= epsilon,
abstain routes to escalation (never a silent guess)."""
import random

from pydantic import BaseModel

from aura_state import AuraEngine, Node, CompiledTransition, RiskController, learn_then_test


def _labeled_set(n, seed):
    """Confidence scores in [0,1]; a decision is correct with prob = score
    (higher confidence -> more likely right). Realistic monotone relationship."""
    rng = random.Random(seed)
    scores, correct = [], []
    for _ in range(n):
        s = rng.random()
        scores.append(s)
        correct.append(rng.random() < s)
    return scores, correct


def test_crc_risk_bound_fixes_0012():
    epsilon = 0.1
    # Calibrate on one draw, measure realized false-action rate on a fresh draw.
    cal_s, cal_c = _labeled_set(500, seed=1)
    ctrl = RiskController(epsilon=epsilon).calibrate(cal_s, cal_c)
    assert ctrl.calibrated and ctrl.can_act

    test_s, test_c = _labeled_set(2000, seed=2)
    acted = [(s, c) for s, c in zip(test_s, test_c) if ctrl.should_act(s)]
    assert acted, "controller should act on at least some high-confidence points"
    false_action_rate = sum(1 for s, c in acted if not c) / len(test_s)
    # CRC bounds E[false action]; allow a small finite-sample tolerance.
    assert false_action_rate <= epsilon + 0.03


def test_abstention_rate_moves_with_epsilon_fixes_0012():
    cal_s, cal_c = _labeled_set(500, seed=3)
    strict = RiskController(epsilon=0.02).calibrate(cal_s, cal_c)
    loose = RiskController(epsilon=0.2).calibrate(cal_s, cal_c)
    # Tighter epsilon -> higher acting threshold -> abstains more.
    assert strict.threshold >= loose.threshold


def test_abstention_routes_to_escalation_fixes_0012():
    # A node calibrated to always abstain must route to escalation, not guess.
    class Low(BaseModel):
        v: int = 1

    always_abstain = RiskController(epsilon=0.5)
    always_abstain.calibrated = True
    always_abstain.can_act = True
    always_abstain.threshold = 2.0  # score 0.0 < 2.0 -> always abstain

    class Decide(Node):
        system_prompt = "decide"
        risk_controller = always_abstain
        escalation_node = "Human"

        def risk_score(self, extracted_data=None, conformal=None, memory=None):
            return 0.0  # never confident enough

        def handle(self, user_text, extracted_data=None, memory=None):
            return "AutoAct", {}   # what it WOULD have done

    class AutoAct(Node):
        system_prompt = "auto"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    class Human(Node):
        system_prompt = "human in the loop"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    e = AuraEngine()
    e.register(Decide, AutoAct, Human)
    e.connect([CompiledTransition(from_node=Decide, to_node=AutoAct)])

    next_state, payload = e.process("Decide", "something risky")
    assert next_state == "Human"                 # escalated, not AutoAct
    assert payload["abstained"] is True
    assert e.verification_reports()[-1]["abstained"] is True


def test_learn_then_test_selects_controlled_configs_fixes_0012():
    # Three configs; only the low-risk ones with enough evidence pass at level eps.
    risks = [0.01, 0.05, 0.30]
    controlled = learn_then_test(risks, n=5000, epsilon=0.1, delta=0.05)
    assert 0 in controlled          # 0.01 clearly controlled
    assert 2 not in controlled      # 0.30 > epsilon, never controlled
