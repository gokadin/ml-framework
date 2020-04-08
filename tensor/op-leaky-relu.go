package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationLeakyRelu = "opLeakyRelu"

type opLeakyRelu struct {
	a *Tensor
}

func (opw *opLeakyRelu) name() string {
	return operationLeakyRelu
}

func (opw *opLeakyRelu) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opLeakyRelu) forward(tensor *Tensor) {
	tensor.mat = mat.Apply(opw.a.mat, func(value float32) float32 {
		if value > 0 {
			return value
		}
		return 0.01 * value
	})
}

func (opw *opLeakyRelu) backward(tensor *Tensor) {
	d := mat.Apply(tensor.mat, func(value float32) float32 {
		if value > 0 {
			return 1
		}
		return 0.01
	})
	opw.a.grad = mat.Mul(tensor.grad, d)
}

func LeakyRelu(a *Tensor) *Tensor {
	result := Variable(a.mat.Shape())
	result.op = &opLeakyRelu{a}
	return result
}
