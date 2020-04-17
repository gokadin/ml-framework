package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_add_forward(t *testing.T) {
	a := Variable(1, 2).SetData([]float32{1, 2})
	b := Variable(1, 2).SetData([]float32{2, 3})
	c := Add(a, b)

	c.forward()

	assert.Equal(t, []float32{3, 5}, c.ToFloat32())
}

func Test_add_forward_multipleAssociations(t *testing.T) {
	a := Variable(2, 2).SetData([]float32{1, 2, 2, 3})
	b := Variable(2, 2).SetData([]float32{2, 3, 1, 2})
	c := Add(a, b)

	c.forward()

	assert.Equal(t, []float32{3, 5, 3, 5}, c.ToFloat32())
}

func Test_add_backward(t *testing.T) {
	a := Variable(2, 2).SetData([]float32{1, 2, 2, 3})
	a.isGradientEnabled = true
	b := Variable(2, 2).SetData([]float32{2, 3, 1, 2})
	b.isGradientEnabled = true
	c := Add(a, b)
	c.forward()
	c.SetGradient(mat.Ones32f(c.Size()))

	c.backward()

	assert.Equal(t, []float32{1, 1, 1, 1}, a.GradientToFloat32())
	assert.Equal(t, []float32{1, 1, 1, 1}, b.GradientToFloat32())
}

func Test_add_backward_isGradientsAreDisabled(t *testing.T) {
	a := Variable(2, 2).SetData([]float32{1, 2, 2, 3})
	b := Variable(2, 2).SetData([]float32{2, 3, 1, 2})
	c := Add(a, b)
	c.forward()
	c.SetGradient(mat.Ones32f(c.Size()))

	c.backward()

	assert.Nil(t, a.GradientToFloat32())
	assert.Nil(t, b.GradientToFloat32())
}
