package tensor

const (
	ActivationFunctionIdentity = "activationFunctionIdentity"
	ActivationFunctionSigmoid = "activationFunctionSigmoid"
)

func (t *Tensor) Sigmoid() *Tensor {
	result := NewTensor(divideScalarBy(addScalar(exp(mulScalar(t.mat, -1)), 1), 1))
	result.operation = newOperation(operationActivationSigmoid, result, []*operation{t.operation})
	return result
}
