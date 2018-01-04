#ifndef __ABSTRACT_VALUATION__
#define __ABSTRACT_VALUATION__

#include <string>
#include "option.h"

class AbstractValuation {
public:
	AbstractValuation() {};
	AbstractValuation(std::string nameOfValuation) : nameOfValuation{nameOfValuation} {};
	virtual double calcPrice(Option & opt) = 0;
	virtual ~AbstractValuation() {};
	std::string getName() {return nameOfValuation;}
private:
	std::string nameOfValuation;
};


#endif