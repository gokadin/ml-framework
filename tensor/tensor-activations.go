package tensor

import ("github.com/gokadin/ml-framework/mat"
	"math"
)

func (t *Tensor) Sigmoid() *Tensor {
	result := NewTensor(mat.DivScalarBy(mat.AddScalar(mat.Exp(mat.MulScalar(t.mat, -1)), 1), 1))
	result.operation = newOperation(operationActivationSigmoid, result, []*operation{t.operation})
	return result
}

func (t *Tensor) Relu() *Tensor {
	mat := make([][]float64, len(t.mat))
	for i := range t.mat {
		mat[i] = make([]float64, len(t.mat[i]))
		for j := range t.mat[i] {
			mat[i][j] = math.Max(0, t.mat[i][j])
		}
	}
	result := NewTensor(mat)
	result.operation = newOperation(operationActivationRelu, result, []*operation{t.operation})
	return result
}

func (t *Tensor) Softmax() *Tensor {
	mat := make([][]float64, len(t.mat))
	for i := range t.mat {
		mat[i] = make([]float64, len(t.mat[i]))
		sum := 0.0
		for j := range t.mat[i] {
			sum += math.Pow(math.E, t.mat[i][j])
		}
		for j := range t.mat[i] {
			mat[i][j] = math.Pow(math.E, t.mat[i][j]) / sum
		}
	}
	result := NewTensor(mat)
	result.operation = newOperation(operationActivationSoftmax, result, []*operation{t.operation})
	return result
}