#!/usr/bin/env bash
# Setup venv, install deps, fetch data, train model
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.fetch_games
python -m src.build_dataset
python -m src.train_model

echo "Done. Start the API with: source .venv/bin/activate && python app.py"
