#!/bin/bash
# Auto-generated submission script for backtest jobs

cd /Users/carback1/Code/portfolio-optimization/test_budget_constraint

echo 'Submitting train.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @train.json
echo ''

echo 'Submitting period-1.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-1.json
echo ''

echo 'Submitting period-2.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-2.json
echo ''

echo 'Submitting period-3.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-3.json
echo ''

echo 'Submitting period-4.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-4.json
echo ''

echo 'Submitting period-5.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-5.json
echo ''

echo 'Submitting period-6.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-6.json
echo ''

echo 'Submitting period-7.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-7.json
echo ''

echo 'Submitting period-8.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-8.json
echo ''

echo 'Submitting period-9.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-9.json
echo ''

echo 'Submitting period-10.json...'
curl -X POST https://qpu-1.nodes.quip.network:20049/solve -H "Content-Type: application/json" -d @period-10.json
echo ''

