package tensor

//#include <matmul.h>
import "C"
import "ml-framework/mat"

const operationMatmul = "opMatmul"

type opMatmul struct {
	a, b *Tensor
}

func (o *opMatmul) name() string {
	return operationMatmul
}

func (o *opMatmul) dependencies() []*Tensor {
	return []*Tensor{o.a, o.b}
}

func (o *opMatmul) forwardShape() mat.Shape {
	return mat.Dim(o.a.Shape().D[0], o.b.Shape().D[1])
}

func (o *opMatmul) backwardShapes(shape mat.Shape) []mat.Shape {
	return []mat.Shape{
		mat.Dim(shape.D[0], o.a.Shape().D[1]),
		mat.Dim(o.b.Shape().D[0], shape.D[1]),
	}
}

func (o *opMatmul) forward(tensor *Tensor) {
	if o.a.Shape().D[1] != o.b.Shape().D[0] {
		handleIncompatibleShapes("linear", o.a.Shape(), o.b.Shape())
	}
	C.matmul_forward(tensor._tensor, o.a._tensor, o.b._tensor)
}

func (o *opMatmul) backward(tensor *Tensor) {
	C.matmul_backward(tensor._tensor, o.a._tensor, o.b._tensor)
}

func Matmul(a, b *Tensor) *Tensor {
	o := &opMatmul{a, b}
	result := OfShape(o.forwardShape().D...)
	result.op = o
	return result
}
