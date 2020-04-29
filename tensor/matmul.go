package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}  -lmatmul
//#include <matmul.h>
import "C"
import "github.com/gokadin/ml-framework/mat"

const operationMatmul = "opMatmul"

type opMatmul struct {
	a, b *Tensor
}

func (omm *opMatmul) name() string {
	return operationMatmul
}

func (omm *opMatmul) dependencies() []*Tensor {
	return []*Tensor{omm.a, omm.b}
}

func (omm *opMatmul) forward(tensor *Tensor) {
	tensor.adjustShape(Shape{omm.a.shape.X, omm.b.shape.Y})
	aMat := mat.NewMat32f(mat.WithShape(omm.a.shape.X, omm.a.shape.Y), omm.a.ToFloat32())
	bMat := mat.NewMat32f(mat.WithShape(omm.b.shape.X, omm.b.shape.Y), omm.b.ToFloat32())
	tensor.SetData(mat.MatMulParallel(aMat, bMat).Data())
}

func (omm *opMatmul) backward(tensor *Tensor) {
	C.matmul_backward(tensor._tensor, omm.a._tensor, omm.b._tensor)
}

func Matmul(a, b *Tensor) *Tensor {
	result := Variable(a.shape.X, b.shape.Y)
	result.op = &opMatmul{a, b}
	return result
}
