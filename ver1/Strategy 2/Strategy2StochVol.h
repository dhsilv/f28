#pragma once
#include <deque>
#include <cmath>
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <Strategy.h>
#include "Strategy0VolState.h"

class Strategy2StochVol : public RCM::StrategyStudio::Strategy {
public:
    Strategy2StochVol(StrategyID strategyID, const std::string& strategyName, const std::string& groupName);
    virtual ~Strategy2StochVol() {}

    virtual void OnQuote(const RCM::StrategyStudio::QuoteEventMsg& msg) override;
    virtual void OnResetStrategyState() override;
    virtual void OnParamChanged(RCM::StrategyStudio::StrategyParam& param) override;

private:
    virtual void DefineStrategyParams() override;

    void UpdateKalmanFilter(double observed_rv, double dt_years);
    double CalculateModelForwardVariance(double T1, double T2);
    double CalculateMarketForwardVariance(double T1, double T2);
    void EvaluateVolatilityMispricing(const RCM::StrategyStudio::QuoteEventMsg& msg);
    void AdjustPortfolio(int direction);
    void UpdateFarVariance(double mid, uint64_t now_ns);

    const RCM::StrategyStudio::Instrument* near_inst_;
    const RCM::StrategyStudio::Instrument* far_inst_;

    // Far Instrument Variance Tracker
    std::deque<double> far_squared_returns_;
    uint64_t last_far_sample_time_ns_;
    double last_far_midprice_;
    double far_rv_sum_;
    
    // Extended Kalman Filter (EKF) State 
    double v_est_;  // A posteriori state estimate (latent spot variance)
    double p_est_;  // A posteriori estimate error covariance
    uint64_t last_ekf_update_ns_;

    // CIR Process Parameters 
    double kappa_;  // Mean reversion speed
    double theta_;  // Long-term variance mean
    double xi_;     // Volatility of volatility
    double r_var_;  // Measurement noise variance

    //  Strategy Parameters 
    int base_spread_size_;
    int rv_window_size_;
    double vol_z_entry_;
    double vol_z_exit_;
    
    int spread_position_;
    Strategy0VolState vol_state_;
};