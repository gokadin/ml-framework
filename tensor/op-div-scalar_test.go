package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_divScalar_forward(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 3, 4})
	c := DivScalar(a, 2)

	c.forward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{0.5, 1, 1.5, 2}).Equals32f(c.mat))
}

func Test_divScalar_backward(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 3, 4})
	a.isGradientEnabled = true
	c := DivScalar(a, 2)
	c.grad = mat.NewMat32fOnes(c.mat.Shape())
	c.forward()

	c.backward()

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{0.5, 0.5, 0.5, 0.5}).Equals32f(a.grad))
}
