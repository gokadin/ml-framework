package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_crossEntropy_forward(t *testing.T) {
	a := OfShape(3, 4).SetData([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	b := OfShape(3, 4).SetData([]float32{2, 3, 4, 2, 11, 6, 15, 1, 4, 2, 10, 8})
	c := CrossEntropy(a, b)

	c.forward()

	expected := mat.DivScalar(mat.Sum(mat.Neg(mat.Log(mat.Sum(mat.Mul(a.ToMat32f(), b.ToMat32f()), 1))), 0), float32(a.ToMat32f().Shape().X)).Data()
	assert.Equal(t, expected, c.ToFloat32())
}
