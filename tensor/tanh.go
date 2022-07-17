package tensor

import (
	"math"
	"ml-framework/mat"
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

func (o *opTanh) forwardShape() mat.Shape {
	return o.a.Shape()
}

func (o *opTanh) backwardShapes(tensorShape mat.Shape) []mat.Shape {
	return []mat.Shape{tensorShape}
}

func (o *opTanh) forward(tensor *Tensor) {
	tensor.SetData(mat.Apply(o.a.ToMat32f(), func(value float32) float32 {
		return float32(math.Tanh(float64(value)))
	}).Data())
}

func (o *opTanh) backward(tensor *Tensor) {
	o.a.SetGradient(mat.Mul(tensor.GradientToMat32(), mat.Apply(tensor.ToMat32f(), func(value float32) float32 {
		return float32(1 - math.Pow(math.Tanh(float64(value)), 2))
	})).Data())
}

func Tanh(a *Tensor) *Tensor {
	o := &opTanh{a}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
