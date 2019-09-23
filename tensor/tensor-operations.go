package tensor

func Add(a, b *Tensor) *Tensor {
	t := NewTensor(add(a.mat, b.mat))
	t.operation = newOperation(operationAdd, t, []*operation{a.operation, b.operation})
	return t
}

func Sub(a, b *Tensor) *Tensor {
	t := NewTensor(sub(a.mat, b.mat))
	t.operation = newOperation(operationSub, t, []*operation{a.operation, b.operation})
	return t
}

func (t *Tensor) Sub(b *Tensor) *Tensor {
	t2 := NewTensor(sub(t.mat, b.mat))
	return t2
}

func Mul(a, b *Tensor) *Tensor {
	t := NewTensor(mul(a.mat, b.mat))
	return t
}

func MulScalar(a *Tensor, scalar float64) *Tensor {
	t := NewTensor(mulScalar(a.mat, scalar))
	return t
}

func DivScalar(a *Tensor, scalar float64) *Tensor {
	t := NewTensor(divScalar(a.mat, scalar))
	t.operation = newOperation(operationDivScalar, t, []*operation{a.operation}, scalar)
	return t
}

func Div(a, b *Tensor) *Tensor {
	t := NewTensor(div(a.mat, b.mat))
	return t
}

func Pow(a *Tensor, power float64) *Tensor {
	t := NewTensor(pow(a.mat, power))
	t.operation = newOperation(operationPow, t, []*operation{a.operation}, power)
	return t
}

func (t *Tensor) Pow(power float64) *Tensor {
	//t.mat = pow(t.mat, power)
	//t.addOperation(newOperationPowSelf(t, power))
	//return t

	t2 := NewTensor(pow(t.mat, power))
	return t2
}

func Dot(a, b *Tensor) *Tensor {
	t := NewTensor(dot(a.mat, b.mat))
	t.operation = newOperation(operationDot, t, []*operation{a.operation, b.operation})
	return t
}

func Sum(t *Tensor, axis int) *Tensor {
	result := NewTensor(sum(t.mat, axis))
	result.operation = newOperation(operationSum, result, []*operation{t.operation}, float64(axis))
	return result
}

func Expand(t *Tensor, axis, copies int) *Tensor {
	result := NewTensor(expand(t.mat, axis, copies))
	return result
}
