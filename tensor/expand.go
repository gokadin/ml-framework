package tensor

//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -lexpand
//#include <expand.h>
import "C"

import (
	"fmt"
	"ml-framework/mat"
)

const operationExpand = "opExpand"

type opExpand struct {
	a      *Tensor
	axis   int
	copies int
}

func (o *opExpand) name() string {
	return operationExpand
}

func (o *opExpand) dependencies() []*Tensor {
	return []*Tensor{o.a}
}

func (o *opExpand) forwardShape() mat.Shape {
	if o.axis == 0 && o.a.Shape().D[0] != 1 {
		handleIncompatibleShapes("expand 0", o.a.Shape())
	}

	if o.axis == 1 && o.a.Shape().D[1] != 1 {
		handleIncompatibleShapes("expand 1", o.a.Shape())
	}

	if o.axis == 0 {
		return mat.Dim(o.copies, o.a.Shape().D[1])
	}

	return mat.Dim(o.a.Shape().D[0], o.copies)
}

func (o *opExpand) backwardShapes(shape mat.Shape) []mat.Shape {
	if o.axis == 0 {
		return []mat.Shape{mat.Dim(1, shape.D[1])}
	}

	return []mat.Shape{mat.Dim(shape.D[0], 1)}
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
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
