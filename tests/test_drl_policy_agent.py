import pytest
import torch
import numpy as np
from src.drl_policy_agent import (
    ActorCriticPolicy,
    DRLTradingEnvironment,
    evaluate_drl_policy_action,
    train_drl_policy,
)


def test_actor_critic_policy_forward():
    model = ActorCriticPolicy(state_dim=6, hidden_dim=16)
    dummy_state = torch.randn(2, 6)
    action_mean, action_std, state_val = model(dummy_state)

    assert action_mean.shape == (2, 1)
    assert action_std.shape == (2, 1)
    assert state_val.shape == (2, 1)
    assert (action_mean >= 0.0).all() and (action_mean <= 1.5).all()


def test_drl_trading_environment_step():
    returns = np.array([0.01, -0.02, 0.015, 0.005], dtype=np.float32)
    vols = np.array([0.01, 0.02, 0.015, 0.01], dtype=np.float32)
    sents = np.array([0.5, -0.3, 0.2, 0.8], dtype=np.float32)

    env = DRLTradingEnvironment(returns, vols, sents)
    state = env.reset()
    assert len(state) == 6

    next_state, reward, done, info = env.step(1.0)
    assert len(next_state) == 6
    assert isinstance(reward, float)
    assert "capital" in info


def test_evaluate_drl_policy_action():
    res = evaluate_drl_policy_action("NVDA")
    assert "recommended_leverage" in res
    assert 0.0 <= res["recommended_leverage"] <= 1.5
    assert "action_label" in res
    assert "state_value_score" in res


def test_train_drl_policy_fast():
    # 3 quick episodes to test training pipeline
    res = train_drl_policy("NVDA", episodes=3, learning_rate=0.01)
    assert res["status"] == "TRAINED_SUCCESS"
    assert len(res["learning_curve_rewards"]) == 3
