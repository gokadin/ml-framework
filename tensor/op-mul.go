package tensor

import "github.com/gokadin/ml-framework/mat"

const operationMul = "opMul"

type opMul struct {
	a, b *Tensor
}

func (om *opMul) name() string {
	return operationMul
}

func (om *opMul) dependencies() []*Tensor {
	return []*Tensor{om.a, om.b}
}

func (om *opMul) forward(tensor *Tensor) {
	tensor.SetData(mat.Mul(om.a.mat, om.b.mat).Data())
}


func (om *opMul) backward(tensor *Tensor) {
	if om.a.isGradientEnabled {
		om.a.grad = mat.Mul(tensor.grad, om.b.mat)
	}

	if om.b.isGradientEnabled {
		om.b.grad = mat.Mul(tensor.grad, om.a.mat)
	}
}

func Mul(a, b *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opMul{a, b}
	return result
}
