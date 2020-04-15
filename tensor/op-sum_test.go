package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_Sum1_small(t *testing.T) {
	a := Variable(mat.WithShape(3, 4)).SetData([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	c := Sum(a, 1)

	c.forward()

	expected := mat.Sum(a.mat, 1).Data()

	assert.Equal(t, expected, c.Data().Data())
}

func Test_Sum1_big(t *testing.T) {
	aMat := make([]float32, 50000)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 2
	}
	a := Variable(mat.WithShape(500, 100)).SetData(aMat)
	c := Sum(a, 1)

	c.forward()

	expected := mat.Sum(a.mat, 1).Data()

	assert.Equal(t, expected, c.Data().Data())
}

func Test_Sum1_bigDifferentShape(t *testing.T) {
	aMat := make([]float32, 200000)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 2
	}
	a := Variable(mat.WithShape(2000, 100)).SetData(aMat)
	c := Sum(a, 1)

	c.forward()

	expected := mat.Sum(a.mat, 1).Data()

	assert.Equal(t, expected, c.Data().Data())
}

func Test_Sum0_small(t *testing.T) {
	a := Variable(mat.WithShape(3, 4)).SetData([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	c := Sum(a, 0)

	c.forward()

	expected := mat.Sum(a.mat, 0).Data()

	assert.Equal(t, expected, c.Data().Data())
}

func Test_Sum0_big(t *testing.T) {
	aMat := make([]float32, 50000)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 2
	}
	a := Variable(mat.WithShape(500, 100)).SetData(aMat)
	c := Sum(a, 0)

	c.forward()

	expected := mat.Sum(a.mat, 0).Data()

	assert.Equal(t, expected, c.Data().Data())
}

func Test_Sum0_bigDifferentShape(t *testing.T) {
	aMat := make([]float32, 200000)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 2
	}
	a := Variable(mat.WithShape(2000, 100)).SetData(aMat)
	c := Sum(a, 0)

	c.forward()

	expected := mat.Sum(a.mat, 0).Data()

	assert.Equal(t, expected, c.Data().Data())
}
