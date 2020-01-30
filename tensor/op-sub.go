package tensor

import "github.com/gokadin/ml-framework/mat"

const operationSub = "opSub"

type opSub struct {
	a, b *Tensor
}

func (os *opSub) name() string {
	return operationSub
}

func (os *opSub) dependencies() []*Tensor {
	return []*Tensor{os.a, os.b}
}

func (os *opSub) forward(tensor *Tensor) {
	for i := range tensor.mat {
		for j := range tensor.mat[i] {
			tensor.mat[i][j] = os.a.mat[i][j] - os.b.mat[i][j]
		}
	}
}

func (os *opSub) backward(tensor *Tensor) {
	if os.a.isGradientEnabled {
		os.a.grad = tensor.grad
		//os.a.grad = mat.MulScalar(tensor.grad, 1)
	}

	if os.b.isGradientEnabled {
		os.b.grad = mat.Neg(tensor.grad)
	}
}

func Sub(a, b *Tensor) *Tensor {
	result := Variable(len(a.mat), len(a.mat[0]))
	result.op = &opSub{a, b}
	return result
}
