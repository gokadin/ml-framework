package tensor

import (
	"github.com/google/uuid"
	"math/rand"
)

type Tensor struct {
	id uuid.UUID
	mat [][]float64
	grad [][]float64
	creators []*Tensor
	creationOperator string
    creationMetadataFloat64 float64
}

func NewTensor(mat [][]float64) *Tensor {
	return &Tensor{
		id: uuid.New(),
        mat: mat,
        creators: make([]*Tensor, 0),
	}
}

func NewRandomTensor(x, y int) *Tensor {
    mat := make([][]float64, x)
    for i := range mat {
    	mat[i] = make([]float64, y)
    	for j := range mat[i] {
    		mat[i][j] = rand.Float64()
		}
	}

    return NewTensor(mat)
}

func (t *Tensor) addCreator(creationOperation string, creators ...*Tensor) {
	t.creationOperator = creationOperation
	for _, creator := range creators {
		t.creators = append(t.creators, creator)
	}
}

func (t *Tensor) Data() [][]float64 {
	return t.mat
}

func (t *Tensor) Equals(other *Tensor) bool {
	return equals(t.mat, other.mat)
}

func (t *Tensor) ResetGradient() {
	t.grad = mulScalar(t.grad, 0)
}

func (t *Tensor) ReduceFromGradient(coefficient float64, count int) {
	t.mat = sub(t.mat, divScalar(mulScalar(t.grad, coefficient), float64(count)))
}