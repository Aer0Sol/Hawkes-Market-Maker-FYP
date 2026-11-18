# Hawkes Market Maker — reproducible pipeline.
#
#   make all          engine + diagnostics + analysis (+ features/hawkes when raw data is present)
#   make features     step 1 (needs data/databento/*.csv, see scripts/download_script.py)
#   make hawkes       step 2 (needs raw trades)
#   make build        compile bin/step3_mm
#   make engine       run the engine once on the default config
#   make diagnostics  Hawkes calibration diagnostics + fill-model calibration
#   make analysis     step 4 multi-seed sweep + plots
#   make causal       step 2b: walk-forward Hawkes features, corrected labels, trade file for replay fills
#   make fixed        step 4b: Hawkes vs baseline with replay fills and standardised OFI (the corrected comparison)
#   make alpha        step 5 walk-forward alpha-signal ablation (Hawkes vs EWMA, no engine needed)
#   make validate-binance  step 6 on 7 BTCUSDT days (downloads ~550 MB once; not in `make all`)
#   make live         live BTCUSDT monitor + dashboard on http://127.0.0.1:8765 (needs `validate-binance` once)
#   make live-replay  the same dashboard replaying a historical day (no network needed)
#   make taker        paper P&L of trading the 1 s forecasts on 6 out-of-sample BTCUSDT days (option 1)
#   make ensemble     step 7: the Hawkes ensemble vs every single-family model (BTCUSDT cross-day + AAPL)
#   make maker-calibration  fit the paper market maker's A-S fill decay k on BTCUSDT
#   make execution    step 8: does the order-flow forecast make a forced trade cheaper? (6 BTCUSDT test days)
#   make reconcile DATE=YYYY-MM-DD  live paper maker fills vs the engine's per-second replay of that session
#   make train-pairs  download a week of 8 more Binance pairs and train + cross-day-validate their models
#   make validate-strategies  replay every test day through the live code: output/validation/
#   make hub          start every pair's live monitor (never the original BTCUSDT one) and the hub on :8800
#   make test         lint + pytest
#   make bench        engine throughput/latency benchmark

PY       ?= uv run --frozen python
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

DATA_DIR  := data/databento
RAW_MBP   := $(DATA_DIR)/aapl_mbp-10_2024-06-03.csv
RAW_TRD   := $(DATA_DIR)/aapl_trades_2024-06-03.csv
STEP1_OUT := output/step1_features
STEP2_OUT := output/step2_hawkes
FEATURES  := $(STEP2_OUT)/feature_table_step2_1s.csv

BIN := bin/step3_mm
SRC := engine/event_driven_mm_engine.cpp

HAVE_RAW := $(and $(wildcard $(RAW_MBP)),$(wildcard $(RAW_TRD)))

.PHONY: all features hawkes build engine diagnostics analysis causal fixed alpha download-binance validate-binance live live-replay taker ensemble maker-calibration reconcile execution train-pairs validate-strategies hub hub-stop plots test lint pytest bench clean check-raw

ifeq ($(HAVE_RAW),)
all: engine diagnostics analysis fixed alpha
	@echo "NOTE: raw Databento data not found; steps 1-2 used the committed outputs in output/. See scripts/download_script.py."
else
all: features hawkes engine diagnostics analysis fixed alpha
endif

check-raw:
	@test -n "$(HAVE_RAW)" || { echo "Raw data missing: $(RAW_MBP), $(RAW_TRD). See scripts/download_script.py."; exit 1; }

features: check-raw
	$(PY) scripts/step1_build_feature_table.py --mbp $(RAW_MBP) --trades $(RAW_TRD) --outdir $(STEP1_OUT)

hawkes: check-raw
	$(PY) scripts/step2_trade_label_and_hawkes.py --features $(STEP1_OUT)/feature_table_1s.parquet \
		--trades $(RAW_TRD) --outdir $(STEP2_OUT)

$(BIN): $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $<

build: $(BIN)

engine: $(BIN)
	./$(BIN) --features $(FEATURES) --outdir output/step3_engine_cpp --use-hawkes

diagnostics:
	$(PY) scripts/hawkes_diagnostics.py
	$(PY) scripts/calibrate_fill_model.py

analysis: $(BIN)
	$(PY) scripts/step4_analysis.py
	$(PY) scripts/plot_graphs.py

causal:
	$(PY) scripts/step2b_causal_features.py

fixed: $(BIN) causal
	$(PY) scripts/step4b_fixed_comparison.py

alpha:
	$(PY) scripts/step5_alpha_signal.py

BINANCE_DATES := 2024-06-03 2024-06-04 2024-06-05 2024-06-06 2024-06-07 2024-06-08 2024-06-09

download-binance:
	@for d in $(BINANCE_DATES); do $(PY) scripts/download_binance.py --date $$d || exit 1; done

validate-binance: download-binance
	$(PY) scripts/step6_binance_validation.py --dates $(BINANCE_DATES)

live:
	uv run --frozen --group live python scripts/live_monitor.py

PAIRS      ?= ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT NEARUSDT
PAIR_DATES ?= 2026-11-10 2026-11-11 2026-11-12 2026-11-13 2026-11-14 2026-11-15 2026-11-16

train-pairs:
	for s in $(PAIRS); do for d in $(PAIR_DATES); do $(PY) scripts/download_binance.py --symbol $$s --date $$d; done; done
	printf "%s\n" $(PAIRS) | xargs -P 4 -I{} $(PY) scripts/train_pair.py --symbol {} --dates $(PAIR_DATES)

validate-strategies:
	uv run --frozen --group live python scripts/validate_live_strategies.py --symbol BTCUSDT --dates $(BINANCE_DATES)
	printf "%s\\n" $(PAIRS) | xargs -P 4 -I{} uv run --frozen --group live python scripts/validate_live_strategies.py --symbol {} --dates $(PAIR_DATES)
	$(PY) scripts/summarise_validation.py

hub:
	uv run --frozen --group live python scripts/live_hub.py start
	uv run --frozen --group live python scripts/live_hub.py serve

hub-stop:
	uv run --frozen --group live python scripts/live_hub.py stop

execution: download-binance
	$(PY) scripts/step8_execution.py

maker-calibration: download-binance
	$(PY) scripts/calibrate_btc_maker.py

DATE ?= $(shell date -u +%Y-%m-%d)
reconcile:
	$(PY) scripts/reconcile_live_maker.py --date $(DATE)

ensemble: download-binance
	$(PY) scripts/step7_hawkes_ensemble.py

taker: download-binance
	$(PY) scripts/taker_backtest.py

live-replay:
	uv run --frozen --group live python scripts/live_monitor.py --no-record \
		--replay data/binance/BTCUSDT-aggTrades-2024-06-09.csv --speed 20

plots:
	$(PY) scripts/plot_graphs.py

lint:
	uv run --frozen ruff check scripts tests

pytest: build
	uv run --frozen pytest -q

test: lint pytest

bench: $(BIN)
	$(PY) scripts/benchmark_engine.py

clean:
	rm -rf bin
