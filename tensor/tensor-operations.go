package tensor

func Add(a, b *Tensor) *Tensor {
	t := NewTensor(add(a.mat, b.mat))
	t.addOperation(newOperationAdd(t, []*Tensor{a, b}))
	return t
}

func Sub(a, b *Tensor) *Tensor {
	t := NewTensor(sub(a.mat, b.mat))
	t.addOperation(newOperationSub(t, []*Tensor{a, b}))
	return t
}

func (t *Tensor) Sub(b *Tensor) *Tensor {
	t2 := NewTensor(sub(t.mat, b.mat))
	t2.addOperation(newOperationSub(t2, []*Tensor{t, b}))
	return t2
}

func Mul(a, b *Tensor) *Tensor {
	t := NewTensor(mul(a.mat, b.mat))
	t.addOperation(newOperationMul(t, []*Tensor{a, b}))
	return t
}

func MulScalar(a *Tensor, scalar float64) *Tensor {
	t := NewTensor(mulScalar(a.mat, scalar))
	t.addOperation(newOperationMulScalar(t, []*Tensor{a}, scalar))
	return t
}

func DivScalar(a *Tensor, scalar float64) *Tensor {
	t := NewTensor(divScalar(a.mat, scalar))
    t.addOperation(newOperationDivScalar(t, []*Tensor{a}, scalar))
	return t
}

func Div(a, b *Tensor) *Tensor {
	t := NewTensor(div(a.mat, b.mat))
	t.addOperation(newOperationDiv(t, []*Tensor{a, b}))
	return t
}

func Pow(a *Tensor, power float64) *Tensor {
	t := NewTensor(pow(a.mat, power))
	t.addOperation(newOperationPow(t, []*Tensor{a}, power))
	return t
}

func (t *Tensor) Pow(power float64) *Tensor {
	//t.mat = pow(t.mat, power)
	//t.addOperation(newOperationPowSelf(t, power))
	//return t

	t2 := NewTensor(pow(t.mat, power))
	t2.addOperation(newOperationPow(t2, []*Tensor{t}, power))
	return t2
}

func Dot(a, b *Tensor) *Tensor {
	t := NewTensor(dot(a.mat, b.mat))
    t.addOperation(newOperationDot(t, []*Tensor{a, b}))
	return t
}

func Sum(t *Tensor, axis int) *Tensor {
	result := NewTensor(sum(t.mat, axis))
	result.addOperation(newOperationSum(result, []*Tensor{t}, float64(axis)))
	return result
}

func Expand(t *Tensor, axis, copies int) *Tensor {
	result := NewTensor(expand(t.mat, axis, copies))
	result.addOperation(newOperationExpand(result, []*Tensor{t}, float64(axis)))
	return result
}
