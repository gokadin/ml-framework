package tensor

import (
	"github.com/google/uuid"
)

type Tensor struct {
	id string
	name string
	shapeX int
	shapeY int
	op op
	mat [][]float64
	grad [][]float64
	isGradientEnabled bool
}

func Constant(mat [][]float64) *Tensor {
	return &Tensor{
		id: uuid.New().String(),
		shapeX: len(mat),
		shapeY: len(mat[0]),
		mat: mat,
	}
}

func Variable(shapeX, shapeY int) *Tensor {
	return &Tensor {
		id: uuid.New().String(),
		shapeX: shapeX,
		shapeY: shapeY,
		mat: buildMat(shapeX, shapeY),
	}
}

func (t *Tensor) Id() string {
	return t.id
}

func (t *Tensor) SetData(data [][]float64) {
	t.mat = data
}

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) SetName(name string) *Tensor {
	t.name = name
	return t
}

func (t *Tensor) Data() [][]float64 {
	return t.mat
}

func (t *Tensor) Gradient() [][]float64 {
	return t.grad
}

func (t *Tensor) Reduce(grad [][]float64) {
	for i := range t.mat {
		for j := range t.mat[i] {
			t.mat[i][j] -= grad[i][j]
		}
	}
}

func (t *Tensor) Shape() (x, y int) {
	return t.shapeX, t.shapeY
}

func (t *Tensor) forward() {
	t.op.forward(t)
}

func (t *Tensor) backward() {
	t.op.backward(t)
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
