## Optimal Charge Security Camera

An intelligent security camera system that dynamically selects YOLO models based on battery level and clean energy availability. Balances accuracy, efficiency, and sustainability through three controller approaches: Oracle (optimal), Custom (learned via imitation), and Naive (baseline).

📖 **Full documentation, results, and technical details:** https://therealsamyak.github.io/ECM202A_2025Fall_Project_2/

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/therealsamyak/ECM202A_2025Fall_Project_2.git
cd ECM202A_2025Fall_Project_2
python3 -m venv .venv
source .venv/bin/activate
pip install .

# Run simulation
python3 simulation/run.py
```

## Key Components

- **Model Profiler**: Benchmarks YOLOv10 power/latency/accuracy
- **Oracle Controller**: Optimal MILP-based decision making
- **ML Controller**: Real-time policy via imitation learning
- **Simulation Framework**: Seasonal evaluation across energy data

## Data Sources

Carbon intensity data from [Electricity Maps](https://www.electricitymaps.com/) for US regions (2024, 5-minute granularity).

---

Visit our [documentation site](https://therealsamyak.github.io/ECM202A_2025Fall_Project_2/) for comprehensive technical details, results, and implementation guide.
