#include "Strategy2StochVol.h"
#include <iostream>

using namespace RCM::StrategyStudio;

Strategy2StochVol::Strategy2StochVol(StrategyID strategyID, const std::string& strategyName, const std::string& groupName)
    : Strategy(strategyID, strategyName, groupName),
      near_inst_(nullptr), far_inst_(nullptr),
      last_far_sample_time_ns_(0), last_far_midprice_(0.0), far_rv_sum_(0.0),
      v_est_(0.02), p_est_(1.0), last_ekf_update_ns_(0),
      kappa_(3.0), theta_(0.04), xi_(0.2), r_var_(1e-4), // CIR Params
      base_spread_size_(1), rv_window_size_(60), vol_z_entry_(2.0), vol_z_exit_(0.5),
      spread_position_(0) {}

void Strategy2StochVol::DefineStrategyParams() {
    params().CreateParam(CreateStrategyParamArgs("kappa", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, kappa_));
    params().CreateParam(CreateStrategyParamArgs("theta", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, theta_));
    params().CreateParam(CreateStrategyParamArgs("xi", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, xi_));
    params().CreateParam(CreateStrategyParamArgs("meas_noise_r", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, r_var_));
    params().CreateParam(CreateStrategyParamArgs("vol_z_entry", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, vol_z_entry_));
    params().CreateParam(CreateStrategyParamArgs("vol_z_exit", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, vol_z_exit_));
    params().CreateParam(CreateStrategyParamArgs("base_spread_size", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_INT, base_spread_size_));
}

void Strategy2StochVol::OnParamChanged(StrategyParam& param) {
    if (param.param_name() == "kappa") param.Get(&kappa_);
    else if (param.param_name() == "theta") param.Get(&theta_);
    else if (param.param_name() == "xi") param.Get(&xi_);
    else if (param.param_name() == "meas_noise_r") param.Get(&r_var_);
    else if (param.param_name() == "vol_z_entry") param.Get(&vol_z_entry_);
    else if (param.param_name() == "vol_z_exit") param.Get(&vol_z_exit_);
}

void Strategy2StochVol::OnResetStrategyState() {
    v_est_ = theta_;
    p_est_ = 1.0;
    last_ekf_update_ns_ = 0;
    spread_position_ = 0;
    far_squared_returns_.clear();
    far_rv_sum_ = 0.0;
    last_far_sample_time_ns_ = 0;
    last_far_midprice_ = 0.0;
}

void Strategy2StochVol::OnQuote(const QuoteEventMsg& msg) {
    const Instrument* inst = &msg.instrument();
    double bid = msg.best_bid_price();
    double ask = msg.best_ask_price();
    
    if (bid <= 0.0 || ask <= 0.0) return;
    double mid = 0.5 * (bid + ask);
    uint64_t now_ns = msg.source_time().time_since_epoch().count();

    if (!near_inst_) near_inst_ = inst;
    else if (!far_inst_ && inst!= near_inst_) far_inst_ = inst;

    if (inst == near_inst_) {
        vol_state_.OnQuote(msg); 
    } else if (inst == far_inst_) {
        UpdateFarVariance(mid, now_ns);
    }

    if (near_inst_ && far_inst_) {
        EvaluateVolatilityMispricing(msg);
    }
}

void Strategy2StochVol::UpdateFarVariance(double mid, uint64_t now_ns) {
    if (last_far_midprice_ == 0.0) {
        last_far_midprice_ = mid;
        last_far_sample_time_ns_ = now_ns;
        return;
    }

    double elapsed_sec = (now_ns - last_far_sample_time_ns_) * 1e-9;
    if (elapsed_sec < 1.0) return;

    double ret = std::log(mid / last_far_midprice_);
    double sq_ret = ret * ret;

    far_squared_returns_.push_back(sq_ret);
    far_rv_sum_ += sq_ret;

    if (far_squared_returns_.size() > static_cast<size_t>(rv_window_size_)) {
        far_rv_sum_ -= far_squared_returns_.front();
        far_squared_returns_.pop_front();
    }

    last_far_midprice_ = mid;
    last_far_sample_time_ns_ = now_ns;
}

void Strategy2StochVol::UpdateKalmanFilter(double observed_rv, double dt_years) {
    if (dt_years <= 0.0) return;

    // 1. EKF Prediction Step
    double v_pred = v_est_ + kappa_ * (theta_ - v_est_) * dt_years;
    v_pred = std::max(v_pred, 1e-8); 

    double F = 1.0 - kappa_ * dt_years;
    double Q = (xi_ * xi_) * v_est_ * dt_years; 
    double p_pred = (F * p_est_ * F) + Q;

    // 2. EKF Update Step 
    double S = p_pred + r_var_; 
    double K = p_pred / S;

    double residual = observed_rv - v_pred;
    v_est_ = v_pred + K * residual;
    v_est_ = std::max(v_est_, 1e-8); 

    p_est_ = (1.0 - K) * p_pred;
}

double Strategy2StochVol::CalculateModelForwardVariance(double T1, double T2) {
    // Exact mathematical integral of the CIR process expectation between T1 and T2
    double exp_term = (std::exp(-kappa_ * T1) - std::exp(-kappa_ * T2)) / kappa_;
    double forward_var = theta_ + ((v_est_ - theta_) / (T2 - T1)) * exp_term;
    return std::max(forward_var, 1e-8);
}

double Strategy2StochVol::CalculateMarketForwardVariance(double T1, double T2) {
    // Annualize the 1-second sampled rolling variance (assuming 252 trading days, can adjust later) @TODO
    double ann_factor = 252.0 * 24.0 * 3600.0; 
    double avg_rv_near = (vol_state_.GetRawRealizedVar() / rv_window_size_) * ann_factor;
    double avg_rv_far = (far_rv_sum_ / rv_window_size_) * ann_factor;

    // Extract forward variance from the futures term structure curve
    double forward_var = (avg_rv_far * T2 - avg_rv_near * T1) / (T2 - T1);
    return std::max(forward_var, 1e-8); // Protect against inverted curve anomalies
}

void Strategy2StochVol::EvaluateVolatilityMispricing(const QuoteEventMsg& msg) {
    uint64_t current_time_ns = msg.source_time().time_since_epoch().count();
    
    // Time Sync for EKF
    uint64_t strategy_0_time = vol_state_.GetLastSampleTimeNs();
    if (strategy_0_time > last_ekf_update_ns_) {
        double dt_years = (strategy_0_time - last_ekf_update_ns_) / (365.0 * 24.0 * 3600.0 * 1e9);
        if (last_ekf_update_ns_ == 0) dt_years = 1.0 / (365.0 * 24.0 * 3600.0 * 1e9); 
        
        // Annualize raw measurement for the filter
        double ann_factor = 252.0 * 24.0 * 3600.0;
        double ann_rv = (vol_state_.GetRawRealizedVar() / rv_window_size_) * ann_factor;
        
        UpdateKalmanFilter(ann_rv, dt_years);
        last_ekf_update_ns_ = strategy_0_time;
    }

    if (far_squared_returns_.size() < static_cast<size_t>(rv_window_size_)) return;

    // Calculate Time to Maturities (T1, T2) in Years
    uint64_t near_exp_ns = near_inst_->expiration_date().time_since_epoch().count();
    uint64_t far_exp_ns = far_inst_->expiration_date().time_since_epoch().count();
    
    if (near_exp_ns <= current_time_ns || far_exp_ns <= near_exp_ns) return;

    double T1 = (near_exp_ns - current_time_ns) / (365.0 * 24.0 * 3600.0 * 1e9);
    double T2 = (far_exp_ns - current_time_ns) / (365.0 * 24.0 * 3600.0 * 1e9);

    double model_fwd_var = CalculateModelForwardVariance(T1, T2);
    double market_fwd_var = CalculateMarketForwardVariance(T1, T2);

    // Z-score normalized by EKF estimation uncertainty
    double mispricing_z = (model_fwd_var - market_fwd_var) / std::sqrt(p_est_ + 1e-8);
    
    double alpha = vol_state_.GetExposureMultiplier();
    if (alpha <= 0.0 && spread_position_!= 0) {
        AdjustPortfolio(-spread_position_); 
        return;
    }

    if (spread_position_ == 0 && alpha > 0.0) {
        if (mispricing_z > vol_z_entry_) {
            // Model expects higher forward variance than the futures curve implies
            AdjustPortfolio(1); 
        } else if (mispricing_z < -vol_z_entry_) {
            // Futures term structure is overpricing volatility (High VRP)
            AdjustPortfolio(-1); 
        }
    } else if (spread_position_!= 0) {
        if (std::abs(mispricing_z) < vol_z_exit_) {
            AdjustPortfolio(-spread_position_); // Reversion hit
        }
    }
}

void Strategy2StochVol::AdjustPortfolio(int direction) {
    if (direction == 0) return;

    double alpha = vol_state_.GetExposureMultiplier();
    int desired_size = std::max(1, static_cast<int>(base_spread_size_ * alpha));

    OrderParams near_order;
    OrderParams far_order;
    near_order.inst = near_inst_;
    far_order.inst = far_inst_;
    near_order.size = desired_size;
    far_order.size = desired_size;
    near_order.type = ORDER_TYPE_MARKET; 
    far_order.type = ORDER_TYPE_MARKET;
    near_order.time_in_force = ORDER_TIF_DAY;
    far_order.time_in_force = ORDER_TIF_DAY;

    if (direction == 1) { 
        near_order.action = ORDER_ACTION_BUY;
        far_order.action = ORDER_ACTION_SELL;
    } else { 
        near_order.action = ORDER_ACTION_SELL;
        far_order.action = ORDER_ACTION_BUY;
    }

    trade_actions()->SendNewOrder(near_order);
    trade_actions()->SendNewOrder(far_order);

    spread_position_ += direction;
}