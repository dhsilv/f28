#pragma once

#ifndef _STRATEGY_STUDIO_CALENDAR_SPREAD_STRATEGY_H_
#define _STRATEGY_STUDIO_CALENDAR_SPREAD_STRATEGY_H_

#ifdef _WIN32
    #define _STRATEGY_EXPORTS __declspec(dllexport)
#else
    #ifndef _STRATEGY_EXPORTS
        #define _STRATEGY_EXPORTS
    #endif
#endif


#include <deque>
#include <cmath>
#include <algorithm>
#include <boost/unordered_map.hpp>
#include <Strategy.h>
#include "Strategy0VolState.h"

struct ContractQuote {
    double bid = 0.0;
    double ask = 0.0;
    int bid_size = 0;
    int ask_size = 0;
    bool valid() const { return bid > 0.0 && ask > 0.0 && bid_size > 0 && ask_size > 0; }
};

struct VolumeState {
    uint64_t trade_volume = 0;
};

class Strategy1CalendarSpread : public RCM::StrategyStudio::Strategy {
public:
    Strategy1CalendarSpread(StrategyID strategyID, const std::string& strategyName, const std::string& groupName);
    virtual ~Strategy1CalendarSpread() {}

    virtual void OnQuote(const RCM::StrategyStudio::QuoteEventMsg& msg) override;
    virtual void OnTrade(const RCM::StrategyStudio::TradeEventMsg& msg) override;
    virtual void OnOrderUpdate(const RCM::StrategyStudio::OrderUpdateEventMsg& msg) override {}
    virtual void OnResetStrategyState() override;
    virtual void OnParamChanged(RCM::StrategyStudio::StrategyParam& param) override;

private:
    virtual void RegisterForStrategyEvents(RCM::StrategyStudio::StrategyEventRegister* eventRegister, RCM::StrategyStudio::DateType currDate) override;
    virtual void DefineStrategyParams() override;

    void EvaluateSpread(const RCM::StrategyStudio::QuoteEventMsg& msg);
    void AdjustPortfolio(int direction);
    void UpdateSpreadStats(double spread, uint64_t timestamp_ns);

    boost::unordered_map<const RCM::StrategyStudio::Instrument*, ContractQuote> quotes_;
    boost::unordered_map<const RCM::StrategyStudio::Instrument*, VolumeState> volumes_;
    const RCM::StrategyStudio::Instrument* near_inst_;
    const RCM::StrategyStudio::Instrument* far_inst_;

    // OLS AR(1) Running Accumulators for Discrete OU Process
    std::deque<double> spread_history_;
    std::deque<uint64_t> time_history_;
    double sum_x_ = 0.0;
    double sum_y_ = 0.0;
    double sum_xx_ = 0.0;
    double sum_yy_ = 0.0;
    double sum_xy_ = 0.0;

    // Calibrated Continuous OU Parameters
    double kappa_ = 0.0; 
    double mu_ = 0.0;    
    double sigma_ = 0.0; 

    // Runtime Parameters
    int base_spread_size_;
    int mean_window_;
    double z_entry_;
    double z_exit_;
    int rollover_days_;
    double volume_switch_ratio_;
    double samuelson_lambda_; 

    int spread_position_;
    Strategy0VolState vol_state_;
};

extern "C" {

    _STRATEGY_EXPORTS const char* GetType() {
        return "VenueArb";
    }

    _STRATEGY_EXPORTS IStrategy* CreateStrategy(const char* strategyType,
                                                unsigned strategyID,
                                                const char* strategyName,
                                                const char* groupName) {
        if (strcmp(strategyType, GetType()) == 0)
            return *(new VenueArb(strategyID, strategyName, groupName));
        return nullptr;
    }

    _STRATEGY_EXPORTS const char* GetAuthor() {
        return "Danny Silverstein";
    }

    _STRATEGY_EXPORTS const char* GetAuthorGroup() {
        return "UIUC";
    }

    _STRATEGY_EXPORTS const char* GetReleaseVersion() {
        return Strategy::release_version();
    }
}

#endif