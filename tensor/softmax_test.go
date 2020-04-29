package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_softmax_forward(t *testing.T) {
	a := Variable(2, 4).SetData([]float32{0.2, 0.4, 0.3, 0.8,
															   0.1, 2.4, 0.2, 0.9})
	c := Softmax(a)

	c.forward()

	assert.True(t, mat.NewMat32f(mat.WithShape(2, 1), []float32{1.0, 1.0}).Equals32f(mat.Sum(c.ToMat32f(), 1)))
}
