#pragma once
#include <deque>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <StrategyStudioApi.h>

class Strategy0VolState {
public:
    Strategy0VolState();
    void OnQuote(const RCM::StrategyStudio::QuoteEventMsg& msg);
    
    double GetVolLevel() const { return vol_state_; }
    double GetVolShock() const { return vol_state_ - prev_vol_state_; }
    double GetExposureMultiplier() const { return exposure_multiplier_; }

private:
    bool ShouldSample(uint64_t now_ns);
    void UpdateVolState(double realized_var);
    void UpdateShockAndExposure();

    double sampling_interval_sec_;
    int rv_window_size_;
    double ewma_lambda_;
    double shock_lambda_;
    double z_low_;
    double z_high_;
    double alpha_min_;

    std::deque<double> squared_returns_;
    uint64_t last_sample_time_ns_;
    double last_midprice_;
    double rv_sum_;
    
    double vol_state_;
    double prev_vol_state_;
    double shock_var_;
    double exposure_multiplier_;
};