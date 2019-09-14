package tensor

func Add(a, b *Tensor) *Tensor {
	t := NewTensor(add(a.mat, b.mat))
	t.addCreator(creationOperatorAdd, a, b)
	return t
}

func (t *Tensor) Neg() *Tensor {
	result := NewTensor(mulScalar(t.mat, -1))
	result.addCreator(creationOperatorNeg, t)
	return result
}

func Sub(a, b *Tensor) *Tensor {
	t := NewTensor(sub(a.mat, b.mat))
	t.addCreator(creationOperatorSub, a, b)
	return t
}

func Mul(a, b *Tensor) *Tensor {
	t := NewTensor(mul(a.mat, b.mat))
	t.addCreator(creationOperatorMul, a, b)
	return t
}

func MulScalar(a *Tensor, scalar float64) *Tensor {
	t := NewTensor(mulScalar(a.mat, scalar))
	t.addCreator(creationOperatorMulScalar, a)
	t.creationMetadataFloat64 = scalar
	return t
}

func DivScalar(a *Tensor, scalar float64) *Tensor {
	t := NewTensor(divScalar(a.mat, scalar))
	t.addCreator(creationOperatorDivScalar, a)
	t.creationMetadataFloat64 = scalar
	return t
}

func Pow(a *Tensor, power float64) *Tensor {
	t := NewTensor(pow(a.mat, power))
	t.addCreator(creationOperatorPow, a)
	t.creationMetadataFloat64 = power
	return t
}

func Dot(a, b *Tensor) *Tensor {
	t := NewTensor(dot(a.mat, b.mat))
	t.addCreator(creationOperatorDot, a, b)
	return t
}

func (t *Tensor) Transpose() *Tensor {
	result := NewTensor(transpose(t.mat))
	result.addCreator(creationOperatorTranspose, t)
	return t
}

func SumX(t *Tensor) *Tensor {
	result := NewTensor(sum(t.mat, 0))
	result.addCreator(creationOperatorSumX, t)
	return result
}

func SumY(t *Tensor) *Tensor {
	result := NewTensor(sum(t.mat, 1))
	result.addCreator(creationOperatorSumY, t)
	return result
}

func ExpandX(t *Tensor, copies int) *Tensor {
	result := NewTensor(expand(t.mat, 0, copies))
	result.addCreator(creationOperatorExpandX, t)
	return result
}

func ExpandY(t *Tensor, copies int) *Tensor {
	result := NewTensor(expand(t.mat, 1, copies))
	result.addCreator(creationOperatorExpandY, t)
	return result
}
