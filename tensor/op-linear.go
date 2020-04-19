package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR} -llinear
//#include <linear.h>
//#include <matmul.h>
import "C"
import "github.com/gokadin/ml-framework/mat"

const operationLinear = "opLinear"

type opLinear struct {
	a, x, b *Tensor
}

func (oa *opLinear) name() string {
	return operationLinear
}

func (oa *opLinear) dependencies() []*Tensor {
	return []*Tensor{oa.a, oa.x, oa.b}
}

func (oa *opLinear) forward(tensor *Tensor) {
	tensor.adjustShape(Shape{oa.a.shape.X, oa.b.shape.Y})
	C.linear(oa.a._tensor, oa.x._tensor, oa.b._tensor, tensor._tensor)
}

func (oa *opLinear) backward(tensor *Tensor) {
	oa.b.SetGradient(mat.Sum(mat.NewMat32f(mat.WithShape(tensor.shape.X, tensor.shape.Y), tensor.GradientToFloat32()), 0).Data())
	C.matmul_backward(tensor._tensor, oa.a._tensor, oa.x._tensor)
}

func Linear(a, x, b *Tensor) *Tensor {
	result := Variable(a.shape.X, b.shape.Y)
	result.op = &opLinear{a, x, b}
	return result
}
