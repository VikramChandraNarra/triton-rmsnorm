# Presence of a conftest.py at the repo root makes pytest add the repo root to
# sys.path, so `from kernels.rmsnorm import ...` resolves when running `pytest`
# from the project directory. Intentionally empty otherwise.
