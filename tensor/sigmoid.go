package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"math"
)

const operationSigmoid = "opSigmoid"

type opSigmoid struct {
	a *Tensor
}

func (opw *opSigmoid) name() string {
	return operationSigmoid
}

func (opw *opSigmoid) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opSigmoid) forward(tensor *Tensor) {
	tensor.SetData(mat.Apply(opw.a.ToMat32f(), func(value float32) float32 {
		return float32(1 / (math.Exp(-float64(value)) + 1))
	}).Data())
}

func (opw *opSigmoid) backward(tensor *Tensor) {
	opw.a.SetGradient(mat.Mul(tensor.GradientToMat32(), mat.Mul(tensor.ToMat32f(), mat.SubFromScalar(tensor.ToMat32f(), 1))).Data())
}

func Sigmoid(a *Tensor) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opSigmoid{a}
	return result
}
