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
	tensor.SetData(mat.Apply(opw.a.ToMat32f(), func(value float32) float32 {
		if value > 0 {
			return value
		}
		return 0.01 * value
	}).Data())
}

func (opw *opLeakyRelu) backward(tensor *Tensor) {
	d := mat.Apply(tensor.ToMat32f(), func(value float32) float32 {
		if value > 0 {
			return 1
		}
		return 0.01
	})
	opw.a.SetGradient(mat.Mul(tensor.GradientToMat32(), d).Data())
}

func LeakyRelu(a *Tensor) *Tensor {
	result := Variable(a.Shape().X, a.Shape().Y)
	result.op = &opLeakyRelu{a}
	return result
}
