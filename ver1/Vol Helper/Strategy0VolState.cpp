#include "Strategy0VolState.h"

Strategy0VolState::Strategy0VolState() 
    : sampling_interval_sec_(1.0), rv_window_size_(60), ewma_lambda_(0.94),
      shock_lambda_(0.94), z_low_(1.0), z_high_(3.0), alpha_min_(0.0),
      last_sample_time_ns_(0), last_midprice_(0.0), rv_sum_(0.0), 
      vol_state_(0.0), prev_vol_state_(0.0), shock_var_(0.0), exposure_multiplier_(1.0) {}

bool Strategy0VolState::ShouldSample(uint64_t now_ns) {
    if (last_sample_time_ns_ == 0) return true;
    double elapsed_sec = (now_ns - last_sample_time_ns_) * 1e-9;
    return elapsed_sec >= sampling_interval_sec_;
}

void Strategy0VolState::OnQuote(const RCM::StrategyStudio::QuoteEventMsg& msg) {
    double bid = msg.best_bid_price();
    double ask = msg.best_ask_price();
    if (bid <= 0.0 |

| ask <= 0.0) return;

    double mid = 0.5 * (bid + ask);
    uint64_t now_ns = msg.source_time().time_since_epoch().count();

    if (last_midprice_ == 0.0) {
        last_midprice_ = mid;
        last_sample_time_ns_ = now_ns;
        return;
    }

    if (!ShouldSample(now_ns)) return;

    // Calculate variance
    double ret = std::log(mid / last_midprice_);
    double sq_ret = ret * ret;

    squared_returns_.push_back(sq_ret);
    rv_sum_ += sq_ret;

    if (squared_returns_.size() > static_cast<size_t>(rv_window_size_)) {
        rv_sum_ -= squared_returns_.front();
        squared_returns_.pop_front();
    }

    UpdateVolState(rv_sum_);
    UpdateShockAndExposure();

    last_midprice_ = mid;
    last_sample_time_ns_ = now_ns;
}

void Strategy0VolState::UpdateVolState(double realized_var) {
    prev_vol_state_ = vol_state_;
    if (vol_state_ == 0.0) {
        vol_state_ = realized_var;
    } else {
        vol_state_ = ewma_lambda_ * vol_state_ + (1.0 - ewma_lambda_) * realized_var;
    }
}

void Strategy0VolState::UpdateShockAndExposure() {
    double shock = vol_state_ - prev_vol_state_;
    if (shock_var_ == 0.0) {
        shock_var_ = shock * shock;
    } else {
        shock_var_ = shock_lambda_ * shock_var_ + (1.0 - shock_lambda_) * (shock * shock);
    }

    double abs_z = 0.0;
    if (shock_var_ > 1e-12) {
        abs_z = std::abs(shock / std::sqrt(shock_var_));
    }

    // Exposure gate
    if (abs_z <= z_low_) {
        exposure_multiplier_ = 1.0;
    } else if (abs_z >= z_high_) {
        exposure_multiplier_ = alpha_min_;
    } else {
        double t = (abs_z - z_low_) / (z_high_ - z_low_);
        exposure_multiplier_ = 1.0 - t * (1.0 - alpha_min_);
    }
}