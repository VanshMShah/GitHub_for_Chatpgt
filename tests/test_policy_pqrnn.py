from src.ats.policy import DMAPolicy


def test_pqrnn_registration_is_bounded_and_prior_labeled():
    p = DMAPolicy()
    p.register_model("pqrnn", prior=0.10)
    assert round(sum(p.weights.values()), 10) == 1.0
    assert abs(p.weights["pqrnn"] - 0.10) < 1e-9
    assert p.state()["weight_source"] == "prior"


def test_policy_becomes_learned_after_observation():
    from src.ats.models import Forecast
    p = DMAPolicy()
    p.register_model("pqrnn", prior=0.10)
    f = Forecast(0.001, 0.6, -0.01, -0.003, 0.001, 0.004, 0.01, 0.006, {})
    p.observe(0.002, {"pqrnn": f})
    assert p.state()["weight_source"] == "learned"
    assert p.sample_count["pqrnn"] == 1
