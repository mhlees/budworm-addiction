from models.budworm_model import simulate_budworm

def test_simulation_runs():
    t, N = simulate_budworm()
    assert len(t) == len(N)
    assert all(N >= 0), 'Population should be non-negative'
