package tensor

//#include <sum.h>
import "C"

import (
	"fmt"
	"ml-framework/mat"
)

const operationSum = "opSum"

type opSum struct {
	a             *Tensor
	axis          int
	originalShape mat.Shape
}

func (o *opSum) name() string {
	return operationSum
}

func (o *opSum) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opSum) forwardShape() mat.Shape {
	if o.axis == 0 {
		return mat.Dim(1, o.a.Shape().D[1])
	}

	return mat.Dim(o.a.Shape().D[0], 1)
}

func (o *opSum) backwardShapes(shape mat.Shape) []mat.Shape {
	if o.axis == 0 {
		return []mat.Shape{mat.Dim(o.originalShape.D[0], shape.D[1])}
	}

	return []mat.Shape{mat.Dim(shape.D[0], shape.D[1]*o.originalShape.D[1])}
}

func (o *opSum) forward(tensor *Tensor) {
	C.gpu_sum_forward(o.a._tensor, C.int(o.axis), tensor._tensor)
}

func (o *opSum) backward(tensor *Tensor) {
	if o.axis == 0 {
		o.a.SetGradient(mat.Expand(tensor.GradientToMat32(), 0, o.originalShape.D[0]).Data())
	} else if o.axis == 1 {
		o.a.SetGradient(mat.Expand(tensor.GradientToMat32(), 1, o.originalShape.D[1]).Data())
	}
}

func Sum(a *Tensor, axis int) *Tensor {
	if axis != 0 && axis != 1 {
		panic(fmt.Sprintf("invalid axis provided for sum operation: %d. Valid values are 0 and 1", axis))
	}

	o := &opSum{a, axis, a.Shape()}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
