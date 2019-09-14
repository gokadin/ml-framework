package layer

import "math"

func Identity(x float64) float64 {
	return x
}

func IdentityDerivative(x float64) float64 {
	return 1
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func SigmoidDerivative(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func Relu(x float64) float64 {
	return math.Max(0.0, x)
}

func ReluDerivative(x float64) float64 {
    if x >= 0.0 {
    	return 1.0
	}

    return 0.0
}

func LeakyRelu(x float64) float64 {
    if x >= 0 {
    	return x
	}

    return 0.01 * x
}

func LeakyReluDerivative(x float64) float64 {
	if x >= 0.0 {
		return 1.0
	}

	return 0.01
}

func Softmax(x float64, others []float64) float64 {
	sumOfOthers := 0.0
	for _, other := range others {
        sumOfOthers += math.Pow(math.E, other)
	}

    return math.Pow(math.E, x) / sumOfOthers
}

func SoftmaxDerivative(y float64) float64 {
	// y is Softmax of x
    return y * (1 - y)
}
