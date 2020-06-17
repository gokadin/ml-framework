package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"math"
)

const operationSigmoid = "opSigmoid"

type opSigmoid struct {
	a *Tensor
}

func (o *opSigmoid) name() string {
	return operationSigmoid
}

func (o *opSigmoid) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opSigmoid) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opSigmoid) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape}
}

func (o *opSigmoid) forward(tensor *Tensor) {
	tensor.SetData(mat.Apply(o.a.ToMat32f(), func(value float32) float32 {
		return float32(1 / (math.Exp(-float64(value)) + 1))
	}).Data())
}

func (o *opSigmoid) backward(tensor *Tensor) {
	o.a.SetGradient(mat.Mul(tensor.GradientToMat32(), mat.Mul(tensor.ToMat32f(), mat.SubFromScalar(tensor.ToMat32f(), 1))).Data())
}

func Sigmoid(a *Tensor) *Tensor {
	o := &opSigmoid{a}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
