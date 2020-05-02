package tensor

//#include <matmul.h>
import "C"

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
	//aMat := mat.NewMat32f(mat.WithShape(omm.a.Shape().X, omm.a.Shape().Y), omm.a.ToFloat32())
	//bMat := mat.NewMat32f(mat.WithShape(omm.b.Shape().X, omm.b.Shape().Y), omm.b.ToFloat32())
	//tensor.SetData(mat.MatMulParallel(aMat, bMat).Data())
}

func (omm *opMatmul) backward(tensor *Tensor) {
	//C.gpu_matmul_backward(tensor._tensor, omm.a._tensor, omm.b._tensor)
}

func Matmul(a, b *Tensor) *Tensor {
	result := OfShape(a.Shape().X, b.Shape().Y)
	result.op = &opMatmul{a, b}
	result._tensor.op = C.alloc_matmul(a._tensor, b._tensor)
	return result
}
