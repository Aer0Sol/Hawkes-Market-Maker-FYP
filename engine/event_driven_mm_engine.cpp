#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct MMParams {
    double gamma = 0.10;
    double k = 6.00;
    double A = 0.08;
    int inventory_limit = 500;
    int max_order_size = 25;
    double tick_size = 0.01;
    int min_spread_ticks = 2;
    bool use_hawkes = false;
    double ofi_skew_coef = 0.03;
    double microprice_skew_coef = 0.10;
    double latency_budget_us = 500.0;
    uint64_t seed = 42;

    double maker_fee_bps = 0.20;
    double queue_haircut_base = 0.25;
    double toxicity_haircut_strength = 0.60;
    double toxic_ofi_threshold = 0.20;
    double toxic_side_widen_ticks = 3.0;
    bool allow_one_sided_pull = true;

    int markout_horizon_steps = 1;

    // Inventory-risk horizon in seconds for the A-S terms (sigma^2 * tau, with sigma in
    // price units per second). tau = min(risk_horizon_s, seconds left in the session).
    // <= 0 restores the original session-fraction tau in [0, 1].
    double risk_horizon_s = 60.0;
    // rv_60 fallback (log-return per second) for warm-up rows or non-positive values.
    double sigma_fallback = 1e-4;
    // |ofi| above which the threatened side's quote is pulled (was hard-coded 0.45).
    double toxic_pull_threshold = 0.45;
};

struct EngineState {
    int inventory = 0;
    double cash = 0.0;
    int total_bid_fills = 0;
    int total_ask_fills = 0;
    int total_bid_volume = 0;
    int total_ask_volume = 0;
    int risk_rejects = 0;
    int one_sided_bid_pulls = 0;
    int one_sided_ask_pulls = 0;
};

struct Row {
    std::string ts;
    double bid_px_1 = std::numeric_limits<double>::quiet_NaN();
    double ask_px_1 = std::numeric_limits<double>::quiet_NaN();
    double bid_sz_1 = std::numeric_limits<double>::quiet_NaN();
    double ask_sz_1 = std::numeric_limits<double>::quiet_NaN();
    double depth_bid_10 = std::numeric_limits<double>::quiet_NaN();
    double depth_ask_10 = std::numeric_limits<double>::quiet_NaN();
    double midprice = std::numeric_limits<double>::quiet_NaN();
    double spread = std::numeric_limits<double>::quiet_NaN();
    double microprice = std::numeric_limits<double>::quiet_NaN();
    double rv_60 = std::numeric_limits<double>::quiet_NaN();
    double imbalance_l1 = std::numeric_limits<double>::quiet_NaN();
    double lambda_buy_hawkes = std::numeric_limits<double>::quiet_NaN();
    double lambda_sell_hawkes = std::numeric_limits<double>::quiet_NaN();
    double hawkes_ofi = std::numeric_limits<double>::quiet_NaN();
    double lambda_buy_ewm = std::numeric_limits<double>::quiet_NaN();
    double lambda_sell_ewm = std::numeric_limits<double>::quiet_NaN();
    double order_flow_imbalance = std::numeric_limits<double>::quiet_NaN();
};

struct ResultRow {
    std::string ts;
    double midprice = 0.0;
    double next_midprice = 0.0;
    double best_bid = 0.0;
    double best_ask = 0.0;
    double reservation_price = 0.0;
    double quoted_bid = 0.0;
    double quoted_ask = 0.0;
    double half_spread = 0.0;
    int inventory = 0;
    double cash = 0.0;
    double mtm_pnl = 0.0;
    double sigma = 0.0;
    double bid_distance = 0.0;
    double ask_distance = 0.0;
    double buy_pressure = 0.0;
    double sell_pressure = 0.0;
    double queue_haircut_bid = 0.0;
    double queue_haircut_ask = 0.0;
    double toxicity_haircut_bid = 0.0;
    double toxicity_haircut_ask = 0.0;
    double p_bid_fill = 0.0;
    double p_ask_fill = 0.0;
    int bid_fill = 0;
    int ask_fill = 0;
    int bid_fill_qty = 0;
    int ask_fill_qty = 0;
    int can_bid = 0;
    int can_ask = 0;
    int quote_bid_active = 0;
    int quote_ask_active = 0;
    double fee_paid = 0.0;
    double bid_markout = 0.0;
    double ask_markout = 0.0;
    double adverse_selection_cost = 0.0;
    double realized_spread_component = 0.0;
    double latency_us = 0.0;
    int latency_budget_exceeded = 0;
};

static inline bool is_nan(double x) { return std::isnan(x); }

static inline double parse_double(const std::string& s) {
    if (s.empty()) return std::numeric_limits<double>::quiet_NaN();
    try { return std::stod(s); }
    catch (...) { return std::numeric_limits<double>::quiet_NaN(); }
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> result;
    std::string cur;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) {
            result.push_back(cur);
            cur.clear();
        } else cur.push_back(c);
    }
    result.push_back(cur);
    return result;
}

static std::unordered_map<std::string, size_t> build_header_index(const std::vector<std::string>& header) {
    std::unordered_map<std::string, size_t> idx;
    for (size_t i = 0; i < header.size(); ++i) idx[header[i]] = i;
    return idx;
}

static std::string get_field(const std::vector<std::string>& fields,
                             const std::unordered_map<std::string, size_t>& idx,
                             const std::string& key) {
    auto it = idx.find(key);
    if (it == idx.end()) return "";
    size_t pos = it->second;
    if (pos >= fields.size()) return "";
    return fields[pos];
}

static std::vector<Row> load_features_csv(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) throw std::runtime_error("Could not open input CSV: " + path);

    std::string line;
    if (!std::getline(fin, line)) throw std::runtime_error("Empty CSV: " + path);

    auto header = split_csv_line(line);
    auto idx = build_header_index(header);

    std::vector<std::string> required = {
        "ts", "bid_px_1", "ask_px_1", "midprice", "spread", "microprice", "rv_60", "imbalance_l1"
    };
    for (const auto& col : required) {
        if (idx.find(col) == idx.end()) throw std::runtime_error("Missing required column: " + col);
    }

    std::vector<Row> rows;
    rows.reserve(50000);

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto fields = split_csv_line(line);

        Row r;
        r.ts = get_field(fields, idx, "ts");
        r.bid_px_1 = parse_double(get_field(fields, idx, "bid_px_1"));
        r.ask_px_1 = parse_double(get_field(fields, idx, "ask_px_1"));
        r.bid_sz_1 = parse_double(get_field(fields, idx, "bid_sz_1"));
        r.ask_sz_1 = parse_double(get_field(fields, idx, "ask_sz_1"));
        r.depth_bid_10 = parse_double(get_field(fields, idx, "depth_bid_10"));
        r.depth_ask_10 = parse_double(get_field(fields, idx, "depth_ask_10"));
        r.midprice = parse_double(get_field(fields, idx, "midprice"));
        r.spread = parse_double(get_field(fields, idx, "spread"));
        r.microprice = parse_double(get_field(fields, idx, "microprice"));
        r.rv_60 = parse_double(get_field(fields, idx, "rv_60"));
        r.imbalance_l1 = parse_double(get_field(fields, idx, "imbalance_l1"));
        r.lambda_buy_hawkes = parse_double(get_field(fields, idx, "lambda_buy_hawkes"));
        r.lambda_sell_hawkes = parse_double(get_field(fields, idx, "lambda_sell_hawkes"));
        r.hawkes_ofi = parse_double(get_field(fields, idx, "hawkes_ofi"));
        r.lambda_buy_ewm = parse_double(get_field(fields, idx, "lambda_buy_ewm"));
        r.lambda_sell_ewm = parse_double(get_field(fields, idx, "lambda_sell_ewm"));
        r.order_flow_imbalance = parse_double(get_field(fields, idx, "order_flow_imbalance"));

        if (is_nan(r.bid_px_1) || is_nan(r.ask_px_1) || is_nan(r.midprice)) continue;
        rows.push_back(r);
    }

    return rows;
}

static inline double round_to_tick(double px, double tick) {
    return std::round(px / tick) * tick;
}

// rv_60 is the std of 1-second log returns; A-S needs price units, so scale by mid.
static inline double compute_sigma(const Row& row, const MMParams& params) {
    double rv = (is_nan(row.rv_60) || row.rv_60 <= 0.0) ? params.sigma_fallback : row.rv_60;
    return rv * row.midprice;
}

static inline double compute_ofi_signal(const Row& row, bool use_hawkes) {
    if (use_hawkes && !is_nan(row.hawkes_ofi)) return row.hawkes_ofi;
    if (!is_nan(row.order_flow_imbalance)) return row.order_flow_imbalance;
    return 0.0;
}

struct Quote {
    double mid = 0.0;
    double best_bid = 0.0;
    double best_ask = 0.0;
    double sigma = 0.0;
    double tau = 0.0;
    double reservation = 0.0;
    double half_spread = 0.0;
    double bid_px = 0.0;
    double ask_px = 0.0;
    bool bid_active = true;
    bool ask_active = true;
};

static Quote avellaneda_stoikov_quotes(const Row& row,
                                       const EngineState& state,
                                       const MMParams& params,
                                       double t_frac,
                                       double seconds_left) {
    Quote q;
    q.mid = row.midprice;
    q.best_bid = row.bid_px_1;
    q.best_ask = row.ask_px_1;
    q.sigma = compute_sigma(row, params);

    double tau = (params.risk_horizon_s > 0.0)
        ? std::max(1e-6, std::min(params.risk_horizon_s, seconds_left))
        : std::max(1e-6, 1.0 - t_frac);
    q.tau = tau;
    double ofi = compute_ofi_signal(row, params.use_hawkes);

    q.reservation = q.mid - state.inventory * params.gamma * q.sigma * q.sigma * tau;
    q.reservation += params.ofi_skew_coef * ofi;
    q.reservation += params.microprice_skew_coef * (row.microprice - q.mid);

    q.half_spread = (1.0 / params.gamma) * std::log(1.0 + params.gamma / params.k);
    q.half_spread += 0.5 * params.gamma * q.sigma * q.sigma * tau;

    double min_half_spread = params.min_spread_ticks * params.tick_size / 2.0;
    q.half_spread = std::max(q.half_spread, min_half_spread);

    double raw_bid = q.reservation - q.half_spread;
    double raw_ask = q.reservation + q.half_spread;

    // Informed buy flow (ofi > 0) lifts our ask; informed sell flow hits our bid.
    // Widen (and optionally pull) the side that flow is about to trade against.
    if (ofi > params.toxic_ofi_threshold) {
        raw_ask += params.toxic_side_widen_ticks * params.tick_size;
        if (params.allow_one_sided_pull && ofi > params.toxic_pull_threshold) q.ask_active = false;
    } else if (ofi < -params.toxic_ofi_threshold) {
        raw_bid -= params.toxic_side_widen_ticks * params.tick_size;
        if (params.allow_one_sided_pull && ofi < -params.toxic_pull_threshold) q.bid_active = false;
    }

    q.bid_px = std::min(round_to_tick(raw_bid, params.tick_size), q.best_bid);
    q.ask_px = std::max(round_to_tick(raw_ask, params.tick_size), q.best_ask);

    if (q.ask_px <= q.bid_px) q.ask_px = q.bid_px + params.tick_size;
    return q;
}

struct RiskDecision {
    bool can_bid = true;
    bool can_ask = true;
};

static RiskDecision risk_check(const EngineState& state, const MMParams& params) {
    RiskDecision r;
    r.can_bid = (state.inventory + params.max_order_size <= params.inventory_limit);
    r.can_ask = (state.inventory - params.max_order_size >= -params.inventory_limit);
    return r;
}

static double clamp01(double x) {
    return std::max(0.0, std::min(1.0, x));
}

static double queue_haircut_bid(const Row& row, int order_size, double base) {
    double size_ahead = !is_nan(row.bid_sz_1) ? row.bid_sz_1 : 100.0;
    double ratio = static_cast<double>(order_size) / std::max(size_ahead + order_size, 1.0);
    double depth_term = !is_nan(row.depth_bid_10) ? std::min(1.0, 100.0 / std::max(row.depth_bid_10, 1.0)) : 0.2;
    return clamp01(base + 0.45 * ratio + 0.30 * depth_term);
}

static double queue_haircut_ask(const Row& row, int order_size, double base) {
    double size_ahead = !is_nan(row.ask_sz_1) ? row.ask_sz_1 : 100.0;
    double ratio = static_cast<double>(order_size) / std::max(size_ahead + order_size, 1.0);
    double depth_term = !is_nan(row.depth_ask_10) ? std::min(1.0, 100.0 / std::max(row.depth_ask_10, 1.0)) : 0.2;
    return clamp01(base + 0.45 * ratio + 0.30 * depth_term);
}

// Haircut on the threatened side, consistent with the widening above:
// sell flow (ofi < 0) threatens the bid, buy flow (ofi > 0) threatens the ask.
static double toxicity_haircut_for_bid(const Row& row, const MMParams& params) {
    double ofi = compute_ofi_signal(row, params.use_hawkes);
    if (ofi < 0.0) return clamp01(1.0 - params.toxicity_haircut_strength * std::abs(ofi));
    return 1.0;
}

static double toxicity_haircut_for_ask(const Row& row, const MMParams& params) {
    double ofi = compute_ofi_signal(row, params.use_hawkes);
    if (ofi > 0.0) return clamp01(1.0 - params.toxicity_haircut_strength * std::abs(ofi));
    return 1.0;
}

static double fill_probability(double distance,
                               double base_intensity,
                               double A,
                               double k,
                               double dt_seconds,
                               double queue_haircut,
                               double toxicity_haircut) {
    distance = std::max(0.0, distance);
    double lam = std::max(1e-8, A * std::max(base_intensity, 1e-6) * std::exp(-k * distance));
    double p = 1.0 - std::exp(-lam * dt_seconds);
    p *= queue_haircut;
    p *= toxicity_haircut;
    return clamp01(p);
}

static double maker_fee(double fill_price, int qty, double maker_fee_bps) {
    return fill_price * static_cast<double>(qty) * maker_fee_bps / 10000.0;
}

// Markouts are per share and positive when the fill was profitable at the horizon.
static double bid_markout(double fill_price, double future_mid) {
    return future_mid - fill_price;
}

static double ask_markout(double fill_price, double future_mid) {
    return fill_price - future_mid;
}

static std::vector<ResultRow> run_engine(const std::vector<Row>& rows, const MMParams& params,
                                         EngineState& state) {
    std::vector<ResultRow> out;
    out.reserve(rows.size());

    state = EngineState{};
    std::mt19937_64 rng(params.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    const size_t n = rows.size();
    for (size_t i = 0; i < n; ++i) {
        const Row& row = rows[i];
        double t_frac = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
        double seconds_left = static_cast<double>(n - 1 - i);  // rows are 1 s apart

        size_t future_idx = std::min(i + static_cast<size_t>(std::max(1, params.markout_horizon_steps)), n - 1);
        double future_mid = rows[future_idx].midprice;

        auto start = std::chrono::high_resolution_clock::now();

        Quote q = avellaneda_stoikov_quotes(row, state, params, t_frac, seconds_left);
        RiskDecision risk = risk_check(state, params);

        if (!risk.can_bid || !risk.can_ask) state.risk_rejects++;
        if (!q.bid_active) state.one_sided_bid_pulls++;
        if (!q.ask_active) state.one_sided_ask_pulls++;

        double bid_distance = std::max(q.mid - q.bid_px, 0.0);
        double ask_distance = std::max(q.ask_px - q.mid, 0.0);

        double buy_pressure = 1.0;
        double sell_pressure = 1.0;
        if (params.use_hawkes && !is_nan(row.lambda_buy_hawkes) && !is_nan(row.lambda_sell_hawkes)) {
            buy_pressure = row.lambda_buy_hawkes;
            sell_pressure = row.lambda_sell_hawkes;
        } else {
            if (!is_nan(row.lambda_buy_ewm)) buy_pressure = row.lambda_buy_ewm;
            if (!is_nan(row.lambda_sell_ewm)) sell_pressure = row.lambda_sell_ewm;
        }

        double qh_bid = queue_haircut_bid(row, params.max_order_size, params.queue_haircut_base);
        double qh_ask = queue_haircut_ask(row, params.max_order_size, params.queue_haircut_base);
        double th_bid = toxicity_haircut_for_bid(row, params);
        double th_ask = toxicity_haircut_for_ask(row, params);

        double dt_seconds = 1.0;
        double p_bid_fill = (risk.can_bid && q.bid_active)
            ? fill_probability(bid_distance, sell_pressure, params.A, params.k, dt_seconds, qh_bid, th_bid)
            : 0.0;
        double p_ask_fill = (risk.can_ask && q.ask_active)
            ? fill_probability(ask_distance, buy_pressure, params.A, params.k, dt_seconds, qh_ask, th_ask)
            : 0.0;

        int bid_fill = 0, ask_fill = 0;
        int bid_fill_qty = 0, ask_fill_qty = 0;
        double fee_paid = 0.0;
        double bid_mo = 0.0, ask_mo = 0.0;
        double adverse_cost = 0.0;
        double realized_spread_component = 0.0;

        if (uni(rng) < p_bid_fill) {
            bid_fill = 1;
            bid_fill_qty = params.max_order_size;
            double fee = maker_fee(q.bid_px, bid_fill_qty, params.maker_fee_bps);
            fee_paid += fee;
            state.inventory += bid_fill_qty;
            state.cash -= static_cast<double>(bid_fill_qty) * q.bid_px;
            state.cash -= fee;
            state.total_bid_fills++;
            state.total_bid_volume += bid_fill_qty;

            bid_mo = bid_markout(q.bid_px, future_mid);
            adverse_cost += std::max(0.0, q.mid - future_mid) * static_cast<double>(bid_fill_qty);
            realized_spread_component += (q.mid - q.bid_px) * static_cast<double>(bid_fill_qty);
        }

        if (uni(rng) < p_ask_fill) {
            ask_fill = 1;
            ask_fill_qty = params.max_order_size;
            double fee = maker_fee(q.ask_px, ask_fill_qty, params.maker_fee_bps);
            fee_paid += fee;
            state.inventory -= ask_fill_qty;
            state.cash += static_cast<double>(ask_fill_qty) * q.ask_px;
            state.cash -= fee;
            state.total_ask_fills++;
            state.total_ask_volume += ask_fill_qty;

            ask_mo = ask_markout(q.ask_px, future_mid);
            adverse_cost += std::max(0.0, future_mid - q.mid) * static_cast<double>(ask_fill_qty);
            realized_spread_component += (q.ask_px - q.mid) * static_cast<double>(ask_fill_qty);
        }

        double mtm_pnl = state.cash + static_cast<double>(state.inventory) * q.mid;

        auto end = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;

        ResultRow rr;
        rr.ts = row.ts;
        rr.midprice = q.mid;
        rr.next_midprice = future_mid;
        rr.best_bid = q.best_bid;
        rr.best_ask = q.best_ask;
        rr.reservation_price = q.reservation;
        rr.quoted_bid = q.bid_px;
        rr.quoted_ask = q.ask_px;
        rr.half_spread = q.half_spread;
        rr.inventory = state.inventory;
        rr.cash = state.cash;
        rr.mtm_pnl = mtm_pnl;
        rr.sigma = q.sigma;
        rr.bid_distance = bid_distance;
        rr.ask_distance = ask_distance;
        rr.buy_pressure = buy_pressure;
        rr.sell_pressure = sell_pressure;
        rr.queue_haircut_bid = qh_bid;
        rr.queue_haircut_ask = qh_ask;
        rr.toxicity_haircut_bid = th_bid;
        rr.toxicity_haircut_ask = th_ask;
        rr.p_bid_fill = p_bid_fill;
        rr.p_ask_fill = p_ask_fill;
        rr.bid_fill = bid_fill;
        rr.ask_fill = ask_fill;
        rr.bid_fill_qty = bid_fill_qty;
        rr.ask_fill_qty = ask_fill_qty;
        rr.can_bid = risk.can_bid ? 1 : 0;
        rr.can_ask = risk.can_ask ? 1 : 0;
        rr.quote_bid_active = q.bid_active ? 1 : 0;
        rr.quote_ask_active = q.ask_active ? 1 : 0;
        rr.fee_paid = fee_paid;
        rr.bid_markout = bid_mo;
        rr.ask_markout = ask_mo;
        rr.adverse_selection_cost = adverse_cost;
        rr.realized_spread_component = realized_spread_component;
        rr.latency_us = latency_us;
        rr.latency_budget_exceeded = (latency_us > params.latency_budget_us) ? 1 : 0;

        out.push_back(rr);
    }

    return out;
}

static void write_results_csv(const std::string& path, const std::vector<ResultRow>& rows) {
    std::ofstream fout(path);
    if (!fout.is_open()) throw std::runtime_error("Could not open output CSV: " + path);

    fout << "ts,midprice,next_midprice,best_bid,best_ask,reservation_price,quoted_bid,quoted_ask,half_spread,"
         << "inventory,cash,mtm_pnl,sigma,bid_distance,ask_distance,buy_pressure,sell_pressure,"
         << "queue_haircut_bid,queue_haircut_ask,toxicity_haircut_bid,toxicity_haircut_ask,"
         << "p_bid_fill,p_ask_fill,bid_fill,ask_fill,bid_fill_qty,ask_fill_qty,can_bid,can_ask,"
         << "quote_bid_active,quote_ask_active,fee_paid,bid_markout,ask_markout,adverse_selection_cost,"
         << "realized_spread_component,latency_us,latency_budget_exceeded\n";

    fout << std::fixed << std::setprecision(8);
    for (const auto& r : rows) {
        fout << r.ts << ","
             << r.midprice << "," << r.next_midprice << "," << r.best_bid << "," << r.best_ask << ","
             << r.reservation_price << "," << r.quoted_bid << "," << r.quoted_ask << "," << r.half_spread << ","
             << r.inventory << "," << r.cash << "," << r.mtm_pnl << "," << r.sigma << ","
             << r.bid_distance << "," << r.ask_distance << "," << r.buy_pressure << "," << r.sell_pressure << ","
             << r.queue_haircut_bid << "," << r.queue_haircut_ask << ","
             << r.toxicity_haircut_bid << "," << r.toxicity_haircut_ask << ","
             << r.p_bid_fill << "," << r.p_ask_fill << "," << r.bid_fill << "," << r.ask_fill << ","
             << r.bid_fill_qty << "," << r.ask_fill_qty << "," << r.can_bid << "," << r.can_ask << ","
             << r.quote_bid_active << "," << r.quote_ask_active << "," << r.fee_paid << ","
             << r.bid_markout << "," << r.ask_markout << "," << r.adverse_selection_cost << ","
             << r.realized_spread_component << "," << r.latency_us << "," << r.latency_budget_exceeded << "\n";
    }
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = (p / 100.0) * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    double frac = idx - static_cast<double>(lo);
    if (lo == hi) return v[lo];
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static void write_summary_json(const std::string& path,
                               const std::vector<ResultRow>& rows,
                               const MMParams& params,
                               const EngineState& state) {
    if (rows.empty()) throw std::runtime_error("No result rows to summarize.");

    std::vector<double> latencies;
    latencies.reserve(rows.size());

    double pnl_sum = 0.0, pnl_sq_sum = 0.0;
    double fee_sum = 0.0, adverse_sum = 0.0, realized_spread_sum = 0.0;
    double bid_markout_sum = 0.0, ask_markout_sum = 0.0;
    int bid_markout_n = 0, ask_markout_n = 0;
    int bid_fills = 0, ask_fills = 0, latency_exceeded = 0, max_abs_inventory = 0;
    int bid_pulls = 0, ask_pulls = 0;
    double avg_bid_fill_prob = 0.0, avg_ask_fill_prob = 0.0;

    for (const auto& r : rows) {
        latencies.push_back(r.latency_us);
        pnl_sum += r.mtm_pnl;
        pnl_sq_sum += r.mtm_pnl * r.mtm_pnl;
        fee_sum += r.fee_paid;
        adverse_sum += r.adverse_selection_cost;
        realized_spread_sum += r.realized_spread_component;
        bid_fills += r.bid_fill;
        ask_fills += r.ask_fill;
        latency_exceeded += r.latency_budget_exceeded;
        max_abs_inventory = std::max(max_abs_inventory, std::abs(r.inventory));
        avg_bid_fill_prob += r.p_bid_fill;
        avg_ask_fill_prob += r.p_ask_fill;
        if (r.bid_fill) { bid_markout_sum += r.bid_markout; bid_markout_n++; }
        if (r.ask_fill) { ask_markout_sum += r.ask_markout; ask_markout_n++; }
        if (!r.quote_bid_active) bid_pulls++;
        if (!r.quote_ask_active) ask_pulls++;
    }

    size_t n = rows.size();
    double mean_pnl = pnl_sum / static_cast<double>(n);
    double var_pnl = std::max(0.0, pnl_sq_sum / static_cast<double>(n) - mean_pnl * mean_pnl);
    double std_pnl = std::sqrt(var_pnl);
    avg_bid_fill_prob /= static_cast<double>(n);
    avg_ask_fill_prob /= static_cast<double>(n);
    double mean_lat = std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(latencies.size());

    std::ofstream fout(path);
    if (!fout.is_open()) throw std::runtime_error("Could not open summary JSON: " + path);

    fout << std::fixed << std::setprecision(6);
    fout << "{\n";
    fout << "  \"params\": {\n";
    fout << "    \"gamma\": " << params.gamma << ",\n";
    fout << "    \"k\": " << params.k << ",\n";
    fout << "    \"A\": " << params.A << ",\n";
    fout << "    \"inventory_limit\": " << params.inventory_limit << ",\n";
    fout << "    \"max_order_size\": " << params.max_order_size << ",\n";
    fout << "    \"tick_size\": " << params.tick_size << ",\n";
    fout << "    \"min_spread_ticks\": " << params.min_spread_ticks << ",\n";
    fout << "    \"use_hawkes\": " << (params.use_hawkes ? "true" : "false") << ",\n";
    fout << "    \"ofi_skew_coef\": " << params.ofi_skew_coef << ",\n";
    fout << "    \"microprice_skew_coef\": " << params.microprice_skew_coef << ",\n";
    fout << "    \"latency_budget_us\": " << params.latency_budget_us << ",\n";
    fout << "    \"maker_fee_bps\": " << params.maker_fee_bps << ",\n";
    fout << "    \"queue_haircut_base\": " << params.queue_haircut_base << ",\n";
    fout << "    \"toxicity_haircut_strength\": " << params.toxicity_haircut_strength << ",\n";
    fout << "    \"toxic_ofi_threshold\": " << params.toxic_ofi_threshold << ",\n";
    fout << "    \"toxic_side_widen_ticks\": " << params.toxic_side_widen_ticks << ",\n";
    fout << "    \"markout_horizon_steps\": " << params.markout_horizon_steps << ",\n";
    fout << "    \"risk_horizon_s\": " << params.risk_horizon_s << ",\n";
    fout << "    \"sigma_fallback\": " << params.sigma_fallback << ",\n";
    fout << "    \"toxic_pull_threshold\": " << params.toxic_pull_threshold << ",\n";
    fout << "    \"allow_one_sided_pull\": " << (params.allow_one_sided_pull ? "true" : "false") << ",\n";
    fout << "    \"seed\": " << params.seed << "\n";
    fout << "  },\n";
    fout << "  \"n_steps\": " << n << ",\n";
    fout << "  \"final_inventory\": " << rows.back().inventory << ",\n";
    fout << "  \"final_cash\": " << rows.back().cash << ",\n";
    fout << "  \"final_mtm_pnl\": " << rows.back().mtm_pnl << ",\n";
    fout << "  \"mean_mtm_pnl\": " << mean_pnl << ",\n";
    fout << "  \"std_mtm_pnl\": " << std_pnl << ",\n";
    fout << "  \"max_abs_inventory\": " << max_abs_inventory << ",\n";
    fout << "  \"bid_fill_count\": " << bid_fills << ",\n";
    fout << "  \"ask_fill_count\": " << ask_fills << ",\n";
    fout << "  \"fill_count_total\": " << (bid_fills + ask_fills) << ",\n";
    fout << "  \"avg_bid_fill_prob\": " << avg_bid_fill_prob << ",\n";
    fout << "  \"avg_ask_fill_prob\": " << avg_ask_fill_prob << ",\n";
    fout << "  \"total_fees_paid\": " << fee_sum << ",\n";
    fout << "  \"total_adverse_selection_cost\": " << adverse_sum << ",\n";
    fout << "  \"total_realized_spread_component\": " << realized_spread_sum << ",\n";
    fout << "  \"avg_bid_markout_per_fill\": " << (bid_markout_n ? bid_markout_sum / bid_markout_n : 0.0) << ",\n";
    fout << "  \"avg_ask_markout_per_fill\": " << (ask_markout_n ? ask_markout_sum / ask_markout_n : 0.0) << ",\n";
    fout << "  \"bid_quote_pulls\": " << bid_pulls << ",\n";
    fout << "  \"ask_quote_pulls\": " << ask_pulls << ",\n";
    fout << "  \"risk_reject_steps\": " << state.risk_rejects << ",\n";
    fout << "  \"mean_latency_us\": " << mean_lat << ",\n";
    fout << "  \"p50_latency_us\": " << percentile(latencies, 50.0) << ",\n";
    fout << "  \"p95_latency_us\": " << percentile(latencies, 95.0) << ",\n";
    fout << "  \"p99_latency_us\": " << percentile(latencies, 99.0) << ",\n";
    fout << "  \"max_latency_us\": " << *std::max_element(latencies.begin(), latencies.end()) << ",\n";
    fout << "  \"latency_budget_exceeded_count\": " << latency_exceeded << "\n";
    fout << "}\n";
}

static MMParams parse_args(int argc, char** argv, std::string& input_csv, std::string& outdir) {
    MMParams params;
    input_csv.clear();
    outdir = "output/step3_engine_cpp";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_d = [&](double& x) { if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg); x = std::stod(argv[++i]); };
        auto next_i = [&](int& x) { if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg); x = std::stoi(argv[++i]); };
        auto next_u = [&](uint64_t& x) { if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg); x = static_cast<uint64_t>(std::stoull(argv[++i])); };
        auto next_s = [&](std::string& x) { if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg); x = argv[++i]; };

        if (arg == "--features") next_s(input_csv);
        else if (arg == "--outdir") next_s(outdir);
        else if (arg == "--gamma") next_d(params.gamma);
        else if (arg == "--k") next_d(params.k);
        else if (arg == "--A") next_d(params.A);
        else if (arg == "--inventory-limit") next_i(params.inventory_limit);
        else if (arg == "--max-order-size") next_i(params.max_order_size);
        else if (arg == "--tick-size") next_d(params.tick_size);
        else if (arg == "--min-spread-ticks") next_i(params.min_spread_ticks);
        else if (arg == "--use-hawkes") params.use_hawkes = true;
        else if (arg == "--ofi-skew-coef") next_d(params.ofi_skew_coef);
        else if (arg == "--microprice-skew-coef") next_d(params.microprice_skew_coef);
        else if (arg == "--latency-budget-us") next_d(params.latency_budget_us);
        else if (arg == "--seed") next_u(params.seed);
        else if (arg == "--maker-fee-bps") next_d(params.maker_fee_bps);
        else if (arg == "--queue-haircut-base") next_d(params.queue_haircut_base);
        else if (arg == "--toxicity-haircut-strength") next_d(params.toxicity_haircut_strength);
        else if (arg == "--toxic-ofi-threshold") next_d(params.toxic_ofi_threshold);
        else if (arg == "--toxic-side-widen-ticks") next_d(params.toxic_side_widen_ticks);
        else if (arg == "--disable-one-sided-pull") params.allow_one_sided_pull = false;
        else if (arg == "--markout-horizon-steps") next_i(params.markout_horizon_steps);
        else if (arg == "--risk-horizon-s") next_d(params.risk_horizon_s);
        else if (arg == "--sigma-fallback") next_d(params.sigma_fallback);
        else if (arg == "--toxic-pull-threshold") next_d(params.toxic_pull_threshold);
        else throw std::runtime_error("Unknown argument: " + arg);
    }

    if (input_csv.empty()) throw std::runtime_error("Missing required --features <csv>");
    return params;
}

int main(int argc, char** argv) {
    try {
        std::string input_csv, outdir;
        MMParams params = parse_args(argc, argv, input_csv, outdir);

        std::cout << "Loading Step 2 CSV...\n";
        auto rows = load_features_csv(input_csv);
        std::cout << "Rows loaded: " << rows.size() << "\n";

        std::cout << "Running calibrated C++ event-driven market-making engine...\n";
        EngineState state;
        auto results = run_engine(rows, params, state);

        std::filesystem::create_directories(outdir);

        std::string results_csv = outdir + "/engine_results.csv";
        std::string summary_json = outdir + "/step3_summary.json";

        write_results_csv(results_csv, results);
        write_summary_json(summary_json, results, params, state);

        std::cout << "Saved files:\n";
        std::cout << " - " << results_csv << "\n";
        std::cout << " - " << summary_json << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}