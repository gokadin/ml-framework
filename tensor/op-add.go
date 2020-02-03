package tensor

import "github.com/gokadin/ml-framework/mat"

const operationAdd = "opAdd"

type opAdd struct {
	a, b *Tensor
}

func (oa *opAdd) name() string {
	return operationAdd
}

func (oa *opAdd) dependencies() []*Tensor {
	return []*Tensor{oa.a, oa.b}
}

func (oa *opAdd) forward(tensor *Tensor) {
	tensor.mat = mat.Add(oa.a.mat, oa.b.mat)
}


func (oa *opAdd) backward(tensor *Tensor) {
	if oa.a.isGradientEnabled {
		oa.a.grad = tensor.grad
	}

	if oa.b.isGradientEnabled {
		oa.b.grad = tensor.grad
	}
}

func Add(a, b *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opAdd{a, b}
	return result
}
