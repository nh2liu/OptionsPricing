#ifndef __ABSTRACT_VALUATION__
#define __ABSTRACT_VALUATION__

#include "option.h"

class AbstractValuation {
public:
	virtual double calcPrice(Option & opt) = 0;
	AbstractValuation() {};
	virtual ~AbstractValuation() {};
};


#endif