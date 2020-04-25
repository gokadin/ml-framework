package tensor

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationSoftmax = "opSoftmax"

type opSoftmax struct {
	a *Tensor
}

func (opw *opSoftmax) name() string {
	return operationSoftmax
}

func (opw *opSoftmax) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opSoftmax) forward(tensor *Tensor) {
	tensor.adjustShape(opw.a.shape)
	tensor.SetData(mat.Softmax(opw.a.ToMat32f()).Data())
}

func (opw *opSoftmax) backward(tensor *Tensor) {
	opw.a.SetGradient(tensor.GradientToFloat32())
}

func Softmax(a *Tensor) *Tensor {
	result := Variable(a.shape.X, a.shape.Y)
	result.op = &opSoftmax{a}
	return result
}
