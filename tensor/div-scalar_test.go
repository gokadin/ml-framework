package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_divScalar_forward(t *testing.T) {
	a := Variable(2, 2).SetData([]float32{1, 2, 3, 4})
	c := DivScalar(a, 2)

	c.forward()

	assert.True(t, mat.NewMat32f(mat.WithShape(a.Shape().X, a.Shape().Y), []float32{0.5, 1, 1.5, 2}).Equals32f(c.ToMat32f()))
}

func Test_divScalar_backward(t *testing.T) {
	a := Variable(2, 2).SetData([]float32{1, 2, 3, 4})
	a.isGradientEnabled = true
	c := DivScalar(a, 2)
	c.SetGradient(mat.NewMat32fOnes(mat.WithShape(c.Shape().X, c.Shape().Y)).Data())
	c.forward()

	c.backward()

	assert.True(t, mat.NewMat32f(mat.WithShape(a.Shape().X, a.Shape().Y), []float32{0.5, 0.5, 0.5, 0.5}).Equals32f(a.GradientToMat32()))
}
