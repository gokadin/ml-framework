package tensor2

import "github.com/google/uuid"

type Tensor struct {
	id string
	op op
	mat [][]float64
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

func (t *Tensor) forward() {
	t.op.forward(t.mat)
}

func (t *Tensor) backward() {
	t.op.backward()
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
