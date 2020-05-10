package tensor

//#include <sum.h>
import "C"

import (
	"github.com/gokadin/ml-framework/mat"
	"log"
)

const operationSum = "opSum"

type opSum struct {
	a *Tensor
	axis int
	originalShape Shape
}

func (o *opSum) name() string {
	return operationSum
}

func (o *opSum) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opSum) forwardShape() Shape {
	if o.axis == 0 {
		return Shape{1, o.a.Shape().Y}
	}

	return Shape{o.a.Shape().X, 1}
}

func (o *opSum) backwardShapes(shape Shape) []Shape {
	if o.axis == 0 {
		return []Shape{{o.originalShape.X, shape.Y}}
	}

	return []Shape{{shape.X, shape.Y * o.originalShape.Y}}
}

func (o *opSum) forward(tensor *Tensor) {
	C.gpu_sum_forward(o.a._tensor, C.int(o.axis), tensor._tensor)
}

func (o *opSum) backward(tensor *Tensor) {
	if o.axis == 0 {
		o.a.SetGradient(mat.Expand(tensor.GradientToMat32(), 0, o.originalShape.X).Data())
	} else if o.axis == 1 {
		o.a.SetGradient(mat.Expand(tensor.GradientToMat32(), 1, o.originalShape.Y).Data())
	}
}

func Sum(a *Tensor, axis int) *Tensor {
	var result *Tensor
	if axis == 0 {
		result = OfShape(1, a.Shape().Y)
	} else if axis == 1 {
		result = OfShape(a.Shape().X, 1)
	} else {
		log.Fatal("axis unknown: " + string(axis))
	}
	result.op = &opSum{a, axis, a.Shape()}
	return result
}
