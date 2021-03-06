package tensor

//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lmul
//#include <mul.h>
import "C"

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
	C.mul_forward(tensor._tensor, o.a._tensor, o.b._tensor)
}

func (o *opMul) backward(tensor *Tensor) {
	C.mul_backward(tensor._tensor, o.a._tensor, o.b._tensor)
}

func Mul(a, b *Tensor) *Tensor {
	o := &opMul{a, b}
	result := OfShape(o.forwardShape().ToArray()...)
	result.op = o
	return result
}
