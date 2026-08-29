import numpy as np
from experimental.rl_allocator import (
    TradingEnvironment,
    PPOPolicyAgent,
    optimize_rl_position_allocation,
)


def test_trading_environment_step_and_reset():
    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
    vols = np.array([0.02, 0.025, 0.02, 0.018, 0.022])
    sents = np.array([0.6, 0.4, 0.7, 0.65, 0.5])

    env = TradingEnvironment(returns, vols, sents)
    init_state = env.reset()

    assert init_state.shape == (5,)
    next_state, reward, done, info = env.step(action_leverage=1.2)
    assert next_state.shape == (5,)
    assert isinstance(reward, float)
    assert "capital" in info
    assert "drawdown" in info


def test_ppo_agent_forward_pass():
    agent = PPOPolicyAgent(state_dim=5, action_dim=1)
    state = np.array([0.01, 0.02, 0.7, 1.0, 0.0])

    leverage = agent.forward_actor(state)
    assert 0.0 <= leverage <= 2.0

    value = agent.forward_critic(state)
    assert isinstance(value, float)


def test_optimize_rl_position_allocation():
    res = optimize_rl_position_allocation(
        ticker="NVDA",
        recent_returns=[0.02, 0.01, -0.005, 0.015, 0.03, -0.01, 0.02],
        volatility=0.025,
        sentiment_score=0.82,
        ai_confidence=0.78,
    )

    assert res["ticker"] == "NVDA"
    assert "recommended_leverage" in res
    assert 0.0 <= res["recommended_leverage"] <= 2.0
    assert "cash_buffer_pct" in res
    assert "policy_action" in res
