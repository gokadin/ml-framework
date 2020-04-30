package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_dot_forward_simple(t *testing.T) {
	a := OfShape(2, 2).SetData([]float32{1, 2, 3, 4})
	b := OfShape(2, 2).SetData([]float32{5, 6, 7, 8})
	c := Matmul(a, b)

	c.forward()

	expected := []float32{19, 22, 43, 50}
	assert.Equal(t, expected, c.ToFloat32())
}

func Test_dot_forward_square(t *testing.T) {
	size := 64
	aMat := make([]float32, size * size)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 1
	}
	bMat := make([]float32, size * size)
	for i := 0; i < len(bMat); i++ {
		bMat[i] = 1
	}
	a := OfShape(size, size).SetData(aMat)
	b := OfShape(size, size).SetData(bMat)
	c := Matmul(a, b)

	c.forward()

	expected := mat.MatMulParallel(mat.NewMat32f(mat.WithShape(size, size), aMat), mat.NewMat32f(mat.WithShape(size, size), bMat))
	assert.True(t, expected.Equals32f(c.ToMat32f()))
}

func Test_dot_forward_squareNotExponentialOfTwo(t *testing.T) {
	size := 50
	aMat := make([]float32, size * size)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 1
	}
	bMat := make([]float32, size * size)
	for i := 0; i < len(bMat); i++ {
		bMat[i] = 1
	}
	a := OfShape(size, size).SetData(aMat)
	b := OfShape(size, size).SetData(bMat)
	c := Matmul(a, b)

	c.forward()

	expected := mat.MatMulParallel(mat.NewMat32f(mat.WithShape(size, size), aMat), mat.NewMat32f(mat.WithShape(size, size), bMat))
	assert.True(t, expected.Equals32f(c.ToMat32f()))
}

func Test_dot_forward_simple2(t *testing.T) {
	size := 4
	size2 := 6
	aMat := make([]float32, size * size2)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 1
	}
	bMat := make([]float32, size2 * size)
	for i := 0; i < len(bMat); i++ {
		bMat[i] = 1
	}
	a := OfShape(size, size2).SetData(aMat)
	b := OfShape(size2, size).SetData(bMat)
	c := Matmul(a, b)

	c.forward()

	expected := mat.MatMulParallel(mat.NewMat32f(mat.WithShape(size, size2), aMat), mat.NewMat32f(mat.WithShape(size2, size), bMat))
	assert.True(t, expected.Equals32f(c.ToMat32f()))
}

func Test_dot_forward_differentSimple(t *testing.T) {
	a := OfShape(2, 2).SetData([]float32{1, 0, 0, 1})
	b := OfShape(2, 2).SetData([]float32{4, 1, 2, 2})
	c := Matmul(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(mat.WithShape(a.Shape().X, a.Shape().Y), []float32{4, 1, 2, 2}).Equals32f(c.ToMat32f()))
}

func Test_dot_forward_differentSizes(t *testing.T) {
	a := OfShape(2, 3).SetData([]float32{1, 0, 0, 1, 2, 3})
	b := OfShape(3, 2).SetData([]float32{4, 1, 2, 2, 0, 1})
	c := Matmul(a, b)

	c.forward()

	assert.True(t, mat.NewMat32f(mat.WithShape(2, 2), []float32{4, 1, 8, 8}).Equals32f(c.ToMat32f()))
}

func Test_dot_forward_allDifferentBigSizes(t *testing.T) {
	size := 1000
	size2 := 784
	size3 := 128
	aMat := make([]float32, size * size2)
	for i := 0; i < len(aMat); i++ {
		aMat[i] = 2.8
	}
	bMat := make([]float32, size2 * size3)
	for i := 0; i < len(bMat); i++ {
		bMat[i] = 3.5
	}
	a := OfShape(size, size2).SetData(aMat)
	b := OfShape(size2, size3).SetData(bMat)
	c := Matmul(a, b)

	c.forward()

	expected := mat.MatMulParallel(mat.NewMat32f(mat.WithShape(size, size2), aMat), mat.NewMat32f(mat.WithShape(size2, size3), bMat))
	assert.True(t, expected.Equals32f(c.ToMat32f()))
}

func Test_dot_backward(t *testing.T) {
	a := OfShape(2, 2).SetData([]float32{1, 0, 0, 1})
	a.isGradientEnabled = true
	b := OfShape(2, 2).SetData([]float32{4, 1, 2, 2})
	b.isGradientEnabled = true
	c := Matmul(a, b)
	c.SetGradient(mat.Ones32f(c.Shape().X * c.Shape().Y))
	c.forward()

	c.backward()

	expectedAGrad := mat.MatMul(c.GradientToMat32(), mat.Transpose(b.ToMat32f())).Data()
	assert.Equal(t, expectedAGrad, a.GradientToFloat32())
	expectedBGrad := mat.Transpose(mat.MatMul(mat.Transpose(c.GradientToMat32()), a.ToMat32f())).Data()
	assert.Equal(t, expectedBGrad, b.GradientToFloat32())
}

func Test_dot_backward_big(t *testing.T) {
	aMat := make([]float32, 1000 * 128)
	for i := 0; i < len(aMat); i++ {
		if i == 10 {
			aMat[i] = 0
		} else {
			aMat[i] = 2
		}
	}
	bMat := make([]float32, 128 * 10)
	for i := 0; i < len(bMat); i++ {
		bMat[i] = 3
	}
	a := OfShape(1000, 128).SetData(aMat)
	a.isGradientEnabled = true
	b := OfShape(128, 10).SetData(bMat)
	b.isGradientEnabled = true
	c := Matmul(a, b)
	c.SetGradient(mat.Ones32f(c.Shape().Size()))
	c.forward()

	c.backward()

	expectedAGrad := mat.MatMul(c.GradientToMat32(), mat.Transpose(b.ToMat32f())).Data()
	assert.Equal(t, expectedAGrad, a.GradientToFloat32())
	expectedBGrad := mat.Transpose(mat.MatMul(mat.Transpose(c.GradientToMat32()), a.ToMat32f())).Data()
	assert.Equal(t, expectedBGrad, b.GradientToFloat32())
}

