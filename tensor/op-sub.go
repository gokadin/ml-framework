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
	tensor.mat = mat.Sub(os.a.mat, os.b.mat)
}

func (os *opSub) backward(tensor *Tensor) {
	if os.a.isGradientEnabled {
		os.a.grad = tensor.grad
	}

	if os.b.isGradientEnabled {
		os.b.grad = mat.Neg(tensor.grad)
	}
}

func Sub(a, b *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opSub{a, b}
	return result
}
