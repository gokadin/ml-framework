package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"math"
)

const operationTanh = "opTanh"

type opTanh struct {
	a *Tensor
}

func (o *opTanh) name() string {
	return operationTanh
}

func (o *opTanh) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opTanh) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opTanh) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape}
}

func (o *opTanh) forward(tensor *Tensor) {
	tensor.SetData(mat.Apply(o.a.ToMat32f(), func(value float32) float32 {
		z := float64(value)
		expZ := math.Exp(z)
		negExpZ := math.Exp(-z)
		return float32((expZ - negExpZ) / (expZ + negExpZ))
	}).Data())
}

func (o *opTanh) backward(tensor *Tensor) {
	o.a.SetGradient(mat.Mul(tensor.GradientToMat32(), mat.SubFromScalar(mat.Pow(tensor.ToMat32f(), 2), 1)).Data())
}

func Tanh(a *Tensor) *Tensor {
	o := &opTanh{a}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
