package tensor

//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -lexpand
//#include <expand.h>
import "C"

import (
	"fmt"
	"github.com/gokadin/ml-framework/mat"
)

const operationExpand = "opExpand"

type opExpand struct {
	a *Tensor
	axis int
	copies int
}

func (o *opExpand) name() string {
	return operationExpand
}

func (o *opExpand) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opExpand) forwardShape() Shape {
	if o.axis == 0 && o.a.Shape().X != 1 {
		handleIncompatibleShapes("expand 0", o.a.Shape())
	}

	if o.axis == 1 && o.a.Shape().Y != 1 {
		handleIncompatibleShapes("expand 1", o.a.Shape())
	}

	if o.axis == 0 {
		return Shape{o.copies, o.a.Shape().Y}
	}

	return Shape{o.a.Shape().X, o.copies}
}

func (o *opExpand) backwardShapes(shape Shape) []Shape {
	if o.axis == 0 {
		return []Shape{{1, shape.Y}}
	}

	return []Shape{{shape.X, 1}}
}

func (o *opExpand) forward(tensor *Tensor) {
	C.expand_forward(tensor._tensor, o.a._tensor, C.int(o.axis), C.int(o.copies))
}

func (o *opExpand) backward(tensor *Tensor) {
	if o.axis == 0 {
		o.a.SetGradient(mat.Sum(tensor.GradientToMat32(), 0).Data())
	} else {
		o.a.SetGradient(mat.Sum(tensor.GradientToMat32(), 1).Data())
	}
}

func Expand(a *Tensor, axis, copies int) *Tensor {
	if axis != 0 && axis != 1 {
		panic(fmt.Sprintf("invalid axis provided for expand operation: %d. Valid values are 0 and 1", axis))
	}

	o := &opExpand{a, axis, copies}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
