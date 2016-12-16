import math

def calcPrecision():
	u = 1
	v = 1
	E = calcE(u,v)
	n = 0.1
	iterations = 0
	EVals = []

	while(iterations < 10):

		newU = u - (n * calcPartialU(u,v))
		newV = v - (n * calcPartialV(u,v))
		E = calcE(newU, newV)
		EVals.append(E)	
		u = newU
		v = newV
		iterations = iterations + 1

	return (EVals, u, v)


def calcPrecisionCD():
	u = 1
	v = 1
	E = calcE(u,v)
	n = 0.1
	iterations = 0

	while(iterations < 15):

		newU = u - (n * calcPartialU(u,v))
		newV = v - (n * calcPartialV(newU,v))
		E = calcE(newU, newV)
		u = newU
		v = newV
		iterations = iterations + 1

	return E

def calcE(u,v):
	result = math.pow(((u * math.exp(v)) - (2 * v * math.exp(-u))), 2)
	return result

def calcPartialU(u,v):
	result = 2 * ((u * math.exp(v)) - (2 * v * math.exp(-u))) * (math.exp(v) + (2 * v * math.exp(-u)))
	return result

def calcPartialV(u,v):
	result = 2 * ((u * math.exp(v)) - (2 * v * math.exp(-u))) * ((u * math.exp(v)) - (2 * math.exp(-u)))
	return result

print calcPrecision()
print calcPrecisionCD()
