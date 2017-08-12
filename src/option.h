#ifndef __OPTION_H__
#define __OPTION_H__

#include "option_enum.h"

struct Option {
  OptionStyle ostyle;
  OptionType otype;
  double startPrice, strikePrice, timeToExpiry, vol, riskFree;
  Option(OptionStyle ostyle,
  		 OptionType otype,
  		 double startPrice,
  		 double strikePrice,
  		 double timeToExpiry,
  		 double vol,
  		 double riskFree) : ostyle{ostyle}, otype{otype}, startPrice{startPrice}, strikePrice{strikePrice},
  							timeToExpiry{timeToExpiry}, vol{vol},riskFree{riskFree} {}
};

#endif