# In-context SARSA and Actor-Critic

This repository provides experiment code for our paper **Transformers Provably Implement In-Context Reinforcement Learning with Policy Improvement**.

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run all cells in `experiments.ipynb` to train and evaluate the SARSA and Actor-Critic transformers and generate the paper's figures.

Checkpoints, results, and figures are saved in `runs/`. Saved results are reused; change `OUT` for a new run.

## Runtime

Training and evaluation were conducted on a MacBook Pro with the following specifications:

- Apple M5 chip
- 16 GB Unified memory

Total training time: Approximately 15 minutes.

Total evaluation time: Approximately 7 minutes.
