package tensor

import (
	"ml-framework/mat"
)

const operationLeakyRelu = "opLeakyRelu"

type opLeakyRelu struct {
	a *Tensor
}

func (o *opLeakyRelu) name() string {
	return operationLeakyRelu
}

func (o *opLeakyRelu) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opLeakyRelu) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opLeakyRelu) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape}
}

func (o *opLeakyRelu) forward(tensor *Tensor) {
	tensor.SetData(mat.Apply(o.a.ToMat32f(), func(value float32) float32 {
		if value > 0 {
			return value
		}
		return 0.01 * value
	}).Data())
}

func (o *opLeakyRelu) backward(tensor *Tensor) {
	d := mat.Apply(tensor.ToMat32f(), func(value float32) float32 {
		if value > 0 {
			return 1
		}
		return 0.01
	})
	o.a.SetGradient(mat.Mul(tensor.GradientToMat32(), d).Data())
}

func LeakyRelu(a *Tensor) *Tensor {
	o := &opLeakyRelu{a}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
