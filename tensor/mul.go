package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lmul
//#include <mul.h>
import "C"

import (
	"github.com/gokadin/ml-framework/mat"
)

const operationMul = "opMul"

type opMul struct {
	a, b *Tensor
}

func (o *opMul) name() string {
	return operationMul
}

func (o *opMul) dependencies() []*Tensor {
	return []*Tensor{o.a, o.b}
}

func (o *opMul) forwardShape() Shape {
	return o.a.Shape()
}

func (o *opMul) backwardShapes(tensorShape Shape) []Shape {
	return []Shape{tensorShape, tensorShape}
}

func (o *opMul) forward(tensor *Tensor) {
	C.mul(o.a._tensor, o.b._tensor, tensor._tensor)
}

func (o *opMul) backward(tensor *Tensor) {
	if o.a.isGradientEnabled {
		o.a.SetGradient(mat.Mul(tensor.GradientToMat32(), o.b.ToMat32f()).Data())
	}

	if o.b.isGradientEnabled {
		o.b.SetGradient(mat.Mul(tensor.GradientToMat32(), o.a.ToMat32f()).Data())
	}
}

func Mul(a, b *Tensor) *Tensor {
	result := OfShape(a.Shape().X, a.Shape().Y)
	result.op = &opMul{a, b}
	return result
}
