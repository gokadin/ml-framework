package tensor

import "github.com/gokadin/ml-framework/mat"

func Add(a, b *Tensor) *Tensor {
	t := NewTensor(mat.Add(a.mat, b.mat))
	t.operation = newOperation(operationAdd, t, []*operation{a.operation, b.operation})
	t.operation2 = &operationAdd2{}
	return t
}

func Sub(a, b *Tensor) *Tensor {
	t := NewTensor(mat.Sub(a.mat, b.mat))
	if a.isGradientEnabled || b.isGradientEnabled {
		t.operation = newOperation(operationSub, t, []*operation{a.operation, b.operation})
	}
	return t
}

func (t *Tensor) Sub(b *Tensor) *Tensor {
	t2 := NewTensor(mat.Sub(t.mat, b.mat))
	return t2
}

func Mul(a, b *Tensor) *Tensor {
	t := NewTensor(mat.Mul(a.mat, b.mat))
	return t
}

func MulScalar(a *Tensor, scalar float64) *Tensor {
	t := NewTensor(mat.MulScalar(a.mat, scalar))
	return t
}

func DivScalar(a *Tensor, scalar float64) *Tensor {
	t := NewTensor(mat.DivScalar(a.mat, scalar))
	t.operation = newOperation(operationDivScalar, t, []*operation{a.operation}, scalar)
	return t
}

func Div(a, b *Tensor) *Tensor {
	t := NewTensor(mat.Div(a.mat, b.mat))
	return t
}

func Pow(a *Tensor, power float64) *Tensor {
	t := NewTensor(mat.Pow(a.mat, power))
	t.operation = newOperation(operationPow, t, []*operation{a.operation}, power)
	return t
}

func (t *Tensor) Pow(power float64) *Tensor {
	//t.mat = pow(t.mat, power)
	//t.addOperation(newOperationPowSelf(t, power))
	//return t

	t2 := NewTensor(mat.Pow(t.mat, power))
	return t2
}

func Dot(a, b *Tensor) *Tensor {
	t := NewTensor(mat.Dot(a.mat, b.mat))
	t.operation = newOperation(operationDot, t, []*operation{a.operation, b.operation})
	return t
}

func Sum(t *Tensor, axis int) *Tensor {
	result := NewTensor(mat.Sum(t.mat, axis))
	result.operation = newOperation(operationSum, result, []*operation{t.operation}, float64(axis))
	return result
}

func Expand(t *Tensor, axis, copies int) *Tensor {
	result := NewTensor(mat.Expand(t.mat, axis, copies))
	result.operation = newOperation(operationExpand, result, []*operation{t.operation}, float64(axis))
	return result
}

func Log(t *Tensor) *Tensor {
	result := NewTensor(mat.Log(t.mat))
	result.operation = newOperation(operationLog, result, []*operation{t.operation})
	return result
}