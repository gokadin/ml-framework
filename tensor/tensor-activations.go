package tensor

const (
	ActivationFunctionIdentity = "activationFunctionIdentity"
	ActivationFunctionSigmoid = "activationFunctionSigmoid"
	ActivationFunctionTanh = "activationFunctionTanh"
)

func (t *Tensor) Sigmoid() *Tensor {
	result := NewTensor(divideScalarBy(addScalar(exp(mulScalar(t.mat, -1)), 1), 1))
	result.addOperation(newOperationSigmoid(result, []*Tensor{t}))
	return result
}
