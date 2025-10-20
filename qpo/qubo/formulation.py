"""QUBO formulation for portfolio optimization."""

import numpy as np
import pandas as pd
from typing import List
import dimod


class QUBOFormulator:
    """Formulate portfolio optimization as QUBO."""

    def __init__(self,
                 n_bits: int = 10,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 lambda_budget: float = 10.0):
        """
        Initialize QUBO formulator.

        Args:
            n_bits: Binary discretization bits (default 10)
            alpha: Weight on expected return (higher = more aggressive)
            beta: Weight on risk (higher = more conservative)
            lambda_budget: Penalty for budget violations (default 10.0)

        Note:
            With n_bits=10, weights are discretized to 1024 levels (0 to 1023/1023).
        """
        self.n_bits = n_bits
        self.K = 2**n_bits - 1  # Normalization constant (1023 for n_bits=10)
        self.alpha = alpha
        self.beta = beta
        self.lambda_budget = lambda_budget

    def formulate_cluster(self,
                         tickers: List[str],
                         mu: pd.Series,
                         Sigma: pd.DataFrame,
                         budget: float = 1.0) -> dimod.BinaryQuadraticModel:
        """
        Create QUBO for a single cluster.

        Binary encoding: w_n = (1/K) * Σ 2^q * x_{n,q}

        Objective: H = H_risk - H_return + H_budget
            H_risk = β * w^T Σ w  (minimize variance)
            H_return = α * μ^T w  (maximize return, so negative)
            H_budget = λ * (Σw - B)²  (enforce budget)

        Args:
            tickers: Asset tickers in this cluster
            mu: Expected returns for these assets (annualized)
            Sigma: Covariance matrix for these assets (annualized)
            budget: Budget constraint (sum of weights, default 1.0)

        Returns:
            dimod.BinaryQuadraticModel
        """
        N = len(tickers)

        # Initialize QUBO matrices
        Q = {}  # Quadratic terms: {(var_i, var_j): coefficient}
        h = {}  # Linear terms: {var_i: coefficient}

        # Create variable names: "TICKER_BIT" (e.g., "AAPL_0", "AAPL_1", ..., "AAPL_9")
        var_names = [[f"{ticker}_{q}" for q in range(self.n_bits)]
                     for ticker in tickers]

        # 1. RETURN TERM (linear, negative to maximize)
        # H_return = -α * Σ μ_n * w_n
        #          = -α * Σ μ_n * (1/K) * Σ 2^q * x_{n,q}
        for n, ticker in enumerate(tickers):
            for q in range(self.n_bits):
                var = var_names[n][q]
                coeff = -self.alpha * mu[ticker] * (2**q) / self.K
                h[var] = h.get(var, 0) + coeff

        # 2. RISK TERM (quadratic)
        # H_risk = β * Σ_{i,j} w_i * Σ_{i,j} * w_j
        #        = β * Σ_{i,j} Σ_{i,j} * (1/K²) * Σ_{q,p} 2^{q+p} * x_{i,q} * x_{j,p}
        for i in range(N):
            for j in range(N):
                sigma_ij = Sigma.iloc[i, j]

                for q in range(self.n_bits):
                    for p in range(self.n_bits):
                        var_i = var_names[i][q]
                        var_j = var_names[j][p]

                        coeff = self.beta * sigma_ij * (2**(q+p)) / (self.K**2)

                        if var_i == var_j:
                            # Diagonal term (self-interaction)
                            h[var_i] = h.get(var_i, 0) + coeff
                        else:
                            # Off-diagonal term
                            key = tuple(sorted([var_i, var_j]))
                            Q[key] = Q.get(key, 0) + coeff

        # 3. BUDGET CONSTRAINT (penalty)
        # H_budget = λ * (Σw - B)²
        #          = λ * [Σw² + ΣΣ w_i*w_j - 2B*Σw + B²]

        # Linear term: -2B * Σw
        for i in range(N):
            for q in range(self.n_bits):
                var = var_names[i][q]
                coeff = -2 * self.lambda_budget * budget * (2**q) / self.K
                h[var] = h.get(var, 0) + coeff

        # Quadratic terms: w_i² (self-terms)
        for i in range(N):
            for q in range(self.n_bits):
                for p in range(self.n_bits):
                    var_q = var_names[i][q]
                    var_p = var_names[i][p]

                    coeff = self.lambda_budget * (2**(q+p)) / (self.K**2)

                    if var_q == var_p:
                        h[var_q] = h.get(var_q, 0) + coeff
                    else:
                        key = tuple(sorted([var_q, var_p]))
                        Q[key] = Q.get(key, 0) + coeff

        # Cross terms: w_i * w_j for i ≠ j
        for i in range(N):
            for j in range(i+1, N):
                for q in range(self.n_bits):
                    for p in range(self.n_bits):
                        var_i = var_names[i][q]
                        var_j = var_names[j][p]

                        coeff = 2 * self.lambda_budget * (2**(q+p)) / (self.K**2)
                        key = tuple(sorted([var_i, var_j]))
                        Q[key] = Q.get(key, 0) + coeff

        # Constant offset: B²
        offset = self.lambda_budget * budget**2

        # Build Binary Quadratic Model
        bqm = dimod.BinaryQuadraticModel(h, Q, offset, dimod.BINARY)

        return bqm

    def get_num_variables(self, n_assets: int) -> int:
        """
        Get total number of binary variables for n_assets.

        Args:
            n_assets: Number of assets in cluster

        Returns:
            Total binary variables (n_assets * n_bits)
        """
        return n_assets * self.n_bits

    def get_discretization_levels(self) -> int:
        """
        Get number of discretization levels.

        Returns:
            2^n_bits (e.g., 1024 for n_bits=10)
        """
        return 2**self.n_bits
