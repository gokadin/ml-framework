package tensor

//import (
//	"github.com/gokadin/ml-framework/mat"
//	"github.com/stretchr/testify/assert"
//	"testing"
//)
//
//func Test_pow_forward(t *testing.T) {
//	a := OfShape(2, 2).SetData([]float32{1, 2, 3, 4})
//	c := Pow(a, 3)
//
//	c.forward()
//
//	assert.True(t, mat.NewMat32f(mat.WithShape(a.Shape().X, a.Shape().Y), []float32{1, 8, 27, 64}).Equals32f(c.ToMat32f()))
//}
//
//func Test_pow_backward(t *testing.T) {
//	a := OfShape(2, 2).SetData([]float32{1, 2, 3, 4})
//	a.isGradientEnabled = true
//	c := Pow(a, 3)
//	c.SetGradient(mat.Ones32f(c.Shape().Size()))
//	c.forward()
//
//	c.backward()
//
//	assert.True(t, mat.NewMat32f(mat.WithShape(a.Shape().X, a.Shape().Y), []float32{3, 12, 27, 48}).Equals32f(a.GradientToMat32()))
//}
//
//func Test_pow_backward_forPowerOfTwo(t *testing.T) {
//	a := OfShape(2, 2).SetData([]float32{1, 2, 3, 4})
//	a.isGradientEnabled = true
//	c := Pow(a, 2)
//	c.SetGradient(mat.Ones32f(c.Shape().Size()))
//	c.forward()
//
//	c.backward()
//
//	assert.True(t, mat.NewMat32f(mat.WithShape(a.Shape().X, a.Shape().Y), []float32{2, 4, 6, 8}).Equals32f(a.GradientToMat32()))
//}
