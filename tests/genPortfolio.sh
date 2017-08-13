#! /bin/bash

function rng() {
	echo $(bc <<< "scale=2; $RANDOM*${1}/32767 + ${2}")
}

function rndArg() {
	ARGS=("$@")
	randomArg=$(($RANDOM % $#))
	echo ${ARGS[${randomArg}]}
}
numberOfOptions=${1}

counter=0
while [ $counter -lt $numberOfOptions ]; do
	# generates price from 10 to 200
	curPrice=$(rng 190 10)
	curPriceFloor=$(bc <<< "scale=0; $curPrice/1")

	if [ $curPriceFloor -lt 50 ]; then
		strikeGap=2.5
	elif [ $curPriceFloor -lt 100 ]; then
		strikeGap=5
	else
		strikeGap=10
	fi

	strikeAway=$(( ( RANDOM % 9 )  - 4 ))
	strike=$(bc <<< "scale=2; $(bc <<< "scale=0; $curPrice/$strikeGap + $strikeAway") * $strikeGap")
	
	vol=$(bc <<< "scale=2; $(rng 25 5) / 100")
	r=$(bc <<< "scale=3; $(rng 7 0) / 100")
	echo $curPrice $strike $vol $r $(rndArg a e) $(rndArg c p)

	let counter=counter+1
done