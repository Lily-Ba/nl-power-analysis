# Dutch Power Market Analysis

Analysis of Dutch (NL) day-ahead electricity prices from the ENTSO-E Transparency Platform, 2019–2025.

## Questions
1. How do prices vary by season, weekday and hour, and how volatile are they?
2. When and why do negative prices occur, and what role does solar play?
3. What would an industrial consumer pay under different procurement strategies (spot, fixed, layered hedge, solar PPA, load shifting)?

## Stack
Python (pandas, entsoe-py, plotly) · Power BI

## Project structure
- `data/raw/`: downloaded ENTSO-E data (not committed)
- `data/processed/`: cleaned tables for analysis and Power BI
- `notebooks/`: exploratory analysis
- `src/`: download and processing scripts
- `powerbi/`: dashboard file

## Setup
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    # create .env containing ENTSOE_API_KEY=<your token>

## Status
🚧 In progress
