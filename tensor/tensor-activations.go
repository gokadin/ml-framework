package tensor

func (t *Tensor) Sigmoid() *Tensor {
	result := NewTensor(divideScalarBy(addScalar(exp(mulScalar(t.mat, -1)), 1), 1))
	result.addCreator(ActivationFunctionSigmoid, t)
	return result
}
