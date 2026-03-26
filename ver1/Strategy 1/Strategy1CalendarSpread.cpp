#include "Strategy1CalendarSpread.h"
#include <iostream>

using namespace RCM::StrategyStudio;

Strategy1CalendarSpread::Strategy1CalendarSpread(StrategyID strategyID, const std::string& strategyName, const std::string& groupName)
    : Strategy(strategyID, strategyName, groupName),
      base_spread_size_(1),
      mean_window_(300),
      z_entry_(2.0),
      z_exit_(0.0),
      rollover_days_(20),
      volume_switch_ratio_(1.0),
      samuelson_lambda_(2.0),
      spread_position_(0),
      near_inst_(nullptr),
      far_inst_(nullptr) {}

void Strategy1CalendarSpread::DefineStrategyParams() {
    params().CreateParam(CreateStrategyParamArgs("base_spread_size", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_INT, base_spread_size_));
    params().CreateParam(CreateStrategyParamArgs("mean_window", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_INT, mean_window_));
    params().CreateParam(CreateStrategyParamArgs("z_entry", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, z_entry_));
    params().CreateParam(CreateStrategyParamArgs("z_exit", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, z_exit_));
    params().CreateParam(CreateStrategyParamArgs("rollover_days", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_INT, rollover_days_));
    params().CreateParam(CreateStrategyParamArgs("volume_switch_ratio", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, volume_switch_ratio_));
    params().CreateParam(CreateStrategyParamArgs("samuelson_lambda", STRATEGY_PARAM_TYPE_RUNTIME, VALUE_TYPE_DOUBLE, samuelson_lambda_));
}

void Strategy1CalendarSpread::RegisterForStrategyEvents(StrategyEventRegister* eventRegister, DateType currDate) {
}

void Strategy1CalendarSpread::OnParamChanged(StrategyParam& param) {
    if (param.param_name() == "base_spread_size") param.Get(&base_spread_size_);
    else if (param.param_name() == "mean_window") param.Get(&mean_window_);
    else if (param.param_name() == "z_entry") param.Get(&z_entry_);
    else if (param.param_name() == "z_exit") param.Get(&z_exit_);
    else if (param.param_name() == "rollover_days") param.Get(&rollover_days_);
    else if (param.param_name() == "volume_switch_ratio") param.Get(&volume_switch_ratio_);
    else if (param.param_name() == "samuelson_lambda") param.Get(&samuelson_lambda_);
}

void Strategy1CalendarSpread::OnResetStrategyState() {
    spread_history_.clear();
    time_history_.clear();
    sum_x_ = sum_y_ = sum_xx_ = sum_yy_ = sum_xy_ = 0.0;
    kappa_ = mu_ = sigma_ = 0.0;
    spread_position_ = 0;
    quotes_.clear();
    volumes_.clear();
    near_inst_ = nullptr;
    far_inst_ = nullptr;
}

void Strategy1CalendarSpread::OnTrade(const TradeEventMsg& msg) {
    volumes_[&msg.instrument()].trade_volume += msg.trade().size();
}

void Strategy1CalendarSpread::OnQuote(const QuoteEventMsg& msg) {
    vol_state_.OnQuote(msg);
    
    const Instrument* inst = &msg.instrument();
    quotes_[inst].bid = msg.best_bid_price();
    quotes_[inst].ask = msg.best_ask_price();
    quotes_[inst].bid_size = msg.best_bid_size();
    quotes_[inst].ask_size = msg.best_ask_size();

    // Assign near and far dynamically
    if (!near_inst_) {
        near_inst_ = inst;
    } else if (!far_inst_ && inst!= near_inst_) {
        far_inst_ = inst;
    }

    if (near_inst_ && far_inst_) {
        EvaluateSpread(msg);
    }
}

void Strategy1CalendarSpread::UpdateSpreadStats(double spread, uint64_t timestamp_ns) {
    if (spread_history_.empty()) {
        spread_history_.push_back(spread);
        time_history_.push_back(timestamp_ns);
        return;
    }

    double prev_spread = spread_history_.back();
    
    // Prevent division by zero if updates arrive in the exact same nanosecond
    if (timestamp_ns <= time_history_.back()) return; 

    spread_history_.push_back(spread);
    time_history_.push_back(timestamp_ns);

    // Update OLS sum matrices
    sum_x_ += prev_spread;
    sum_y_ += spread;
    sum_xx_ += (prev_spread * prev_spread);
    sum_yy_ += (spread * spread);
    sum_xy_ += (prev_spread * spread);

    int n = spread_history_.size() - 1; 

    if (n > mean_window_) {
        double old_prev = spread_history_;
        double old_curr = spread_history_[1];
        
        sum_x_ -= old_prev;
        sum_y_ -= old_curr;
        sum_xx_ -= (old_prev * old_prev);
        sum_yy_ -= (old_curr * old_curr);
        sum_xy_ -= (old_prev * old_curr);
        
        spread_history_.pop_front();
        time_history_.pop_front();
        n--;
    }

    // Exact OLS mapping to continuous OU process
    if (n >= 2) {
        double denominator = (n * sum_xx_) - (sum_x_ * sum_x_);
        if (denominator == 0.0) return;

        double beta = ((n * sum_xy_) - (sum_x_ * sum_y_)) / denominator;
        double alpha = (sum_y_ - beta * sum_x_) / n;

        double sse = sum_yy_ - (alpha * sum_y_) - (beta * sum_xy_);
        double standard_error = std::sqrt(std::max(sse / (n - 2), 1e-12));

        // Exact timestamp drift tracking
        double window_duration_ns = static_cast<double>(time_history_.back() - time_history_.front());
        double avg_dt_years = (window_duration_ns / static_cast<double>(n)) / (365.0 * 24.0 * 3600.0 * 1e9);

        // Inverse Transformation
        if (beta > 0.0 && beta < 1.0 && avg_dt_years > 0.0) {
            kappa_ = -std::log(beta) / avg_dt_years;
            mu_ = alpha / (1.0 - beta);
            sigma_ = standard_error * std::sqrt((-2.0 * std::log(beta)) / (avg_dt_years * (1.0 - (beta * beta))));
        } else {
            kappa_ = -1.0; 
            mu_ = spread;
            sigma_ = standard_error; 
        }
    }
}

void Strategy1CalendarSpread::EvaluateSpread(const QuoteEventMsg& msg) {
    if (!quotes_[near_inst_].valid() ||!quotes_[far_inst_].valid()) return;

    uint64_t current_time_ns = msg.source_time().time_since_epoch().count();
    uint64_t expiration_ns = near_inst_->expiration_date().time_since_epoch().count();
    
    double tau_years = 0.0;
    double tau_days = 0.0;
    if (expiration_ns > current_time_ns) {
        double diff_ns = static_cast<double>(expiration_ns - current_time_ns);
        tau_years = diff_ns / (365.0 * 24.0 * 3600.0 * 1e9);
        tau_days = diff_ns / (24.0 * 3600.0 * 1e9);
    }

    // Check dual-trigger temporal rollover boundary
    bool warning_phase = (tau_days <= static_cast<double>(rollover_days_));
    
    uint64_t near_vol = volumes_.count(near_inst_)? volumes_.at(near_inst_).trade_volume : 0;
    uint64_t far_vol = volumes_.count(far_inst_)? volumes_.at(far_inst_).trade_volume : 0;
    bool hard_termination = (near_vol > 0 && far_vol >= static_cast<uint64_t>(near_vol * volume_switch_ratio_));

    if (hard_termination) {
        if (spread_position_!= 0) {
            AdjustPortfolio(-spread_position_); 
        }
        return; 
    }

    double near_mid = 0.5 * (quotes_[near_inst_].bid + quotes_[near_inst_].ask);
    double far_mid = 0.5 * (quotes_[far_inst_].bid + quotes_[far_inst_].ask);
    double spread = near_mid - far_mid;

    UpdateSpreadStats(spread, current_time_ns);

    double alpha = vol_state_.GetExposureMultiplier();
    if (alpha <= 0.0 && spread_position_!= 0) {
        AdjustPortfolio(-spread_position_);
        return;
    }

    if (spread_history_.size() < 3) return;

    // Apply Samuelson volatility expansion
    double damped_sigma = sigma_ * std::exp(-samuelson_lambda_ * tau_years);
    double z_score = (spread - mu_) / std::max(damped_sigma, 1e-6);

    // Structural constraints execution
    if (spread_position_ == 0 &&!warning_phase && alpha > 0.0) {
        // Validate autoregression structure
        if (kappa_ > 0.0) {
            if (z_score > z_entry_) {
                AdjustPortfolio(-1); 
            } else if (z_score < -z_entry_) {
                AdjustPortfolio(1);  
            }
        }
    } else if (spread_position_!= 0) {
        if (std::abs(z_score) < z_exit_) {
            AdjustPortfolio(-spread_position_); 
        }
    }
}

void Strategy1CalendarSpread::AdjustPortfolio(int direction) {
    if (direction == 0) return;

    double alpha = vol_state_.GetExposureMultiplier();
    int desired_size = std::max(1, static_cast<int>(base_spread_size_ * alpha));

    // Validating crossing liquidity depth
    int min_near_size = std::min(quotes_[near_inst_].bid_size, quotes_[near_inst_].ask_size);
    int min_far_size = std::min(quotes_[far_inst_].bid_size, quotes_[far_inst_].ask_size);
    int max_size = std::min(min_near_size, min_far_size);
    
    int trade_size = std::min(desired_size, max_size);
    if (trade_size <= 0) return;

    OrderParams near_order;
    OrderParams far_order;
    
    near_order.inst = near_inst_;
    far_order.inst = far_inst_;
    
    near_order.size = trade_size;
    far_order.size = trade_size;
    
    near_order.type = ORDER_TYPE_LIMIT;
    far_order.type = ORDER_TYPE_LIMIT;
    
    near_order.time_in_force = ORDER_TIF_DAY;
    far_order.time_in_force = ORDER_TIF_DAY;

    if (direction == 1) { 
        near_order.action = ORDER_ACTION_BUY;
        near_order.price = quotes_[near_inst_].ask;
        far_order.action = ORDER_ACTION_SELL;
        far_order.price = quotes_[far_inst_].bid;
    } else { 
        near_order.action = ORDER_ACTION_SELL;
        near_order.price = quotes_[near_inst_].bid;
        far_order.action = ORDER_ACTION_BUY;
        far_order.price = quotes_[far_inst_].ask;
    }

    trade_actions()->SendNewOrder(near_order);
    trade_actions()->SendNewOrder(far_order);

    spread_position_ += direction;
}