package tensor2

import "github.com/google/uuid"

type Tensor struct {
	id string
	name string
	op op
	mat [][]float64
	grad [][]float64
	isGradientEnabled bool
}

func Constant(mat [][]float64) *Tensor {
	return &Tensor{
		id: uuid.New().String(),
		mat: mat,
	}
}

func Variable(shapeX, shapeY int) *Tensor {
	return &Tensor {
		id: uuid.New().String(),
		mat: buildMat(shapeX, shapeY),
	}
}

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) SetName(name string) *Tensor {
	t.name = name
	return t
}

func (t *Tensor) forward() {
	t.op.forward(t.mat)
}

func (t *Tensor) backward() {
	t.op.backward(t.grad)
}

func buildMat(shapeX, shapeY int) [][]float64 {
	mat := make([][]float64, shapeX)
	for i := range mat {
		col := make([]float64, shapeY)
		for j := range col {
			col[j] = 0
		}
		mat[i] = col
	}
	return mat
}
