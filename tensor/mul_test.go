package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_mul_simple(t *testing.T) {
	a := OfShape(3, 4).SetData([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	b := OfShape(3, 4).SetData([]float32{2, 3, 4, 2, 11, 6, 15, 1, 4, 2, 10, 8})
	c := Mul(a, b)

	c.forward()

	expected := mat.Mul(a.ToMat32f(), b.ToMat32f())
	assert.Equal(t, expected.Data(), c.ToFloat32())
}

func Test_mul_big(t *testing.T) {
	width := 500
	height := 1000
	aMat := make([]float32, width * height)
	bMat := make([]float32, width * height)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 2
		bMat[i] = 3
	}
	a := OfShape(width, height).SetData(aMat)
	b := OfShape(width, height).SetData(bMat)
	c := Mul(a, b)

	c.forward()

	expected := mat.Mul(a.ToMat32f(), b.ToMat32f())
	assert.Equal(t, expected.Data(), c.ToFloat32())
}
