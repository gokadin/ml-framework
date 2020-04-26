package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"math"
)

const operationTanh = "opTanh"

type opTanh struct {
	a *Tensor
}

func (opw *opTanh) name() string {
	return operationTanh
}

func (opw *opTanh) dependencies() []*Tensor {
	return []*Tensor{opw.a}
}

func (opw *opTanh) forward(tensor *Tensor) {
	tensor.adjustShape(opw.a.shape)
	tensor.SetData(mat.Apply(opw.a.ToMat32f(), func(value float32) float32 {
		z := float64(value)
		expZ := math.Exp(z)
		negExpZ := math.Exp(-z)
		return float32((expZ - negExpZ) / (expZ + negExpZ))
	}).Data())
}

func (opw *opTanh) backward(tensor *Tensor) {
	opw.a.SetGradient(mat.Mul(tensor.GradientToMat32(), mat.SubFromScalar(mat.Pow(tensor.ToMat32f(), 2), 1)).Data())
}

func Tanh(a *Tensor) *Tensor {
	result := Variable(a.shape.X, a.shape.Y)
	result.op = &opTanh{a}
	return result
}
