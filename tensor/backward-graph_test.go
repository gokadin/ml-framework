package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_markGradientDependencies_withSingleNode(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))

	buildBackwardGraph([]*Tensor{a}, a)

	assert.True(t, !a.isGradientEnabled)
}

func Test_markGradientDependencies_withSingleNodeAndOneDependency(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	b := Pow(a, 2)

	buildBackwardGraph([]*Tensor{a}, b)

	assert.True(t, a.isGradientEnabled)
	assert.True(t, b.isGradientEnabled)
}

func Test_markGradientDependencies_withSingleNodeAndTwoDependenciesWithSingleDerivative(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	b := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	c := Add(a, b)

	buildBackwardGraph([]*Tensor{a}, c)

	assert.True(t, a.isGradientEnabled)
	assert.True(t, !b.isGradientEnabled)
	assert.True(t, c.isGradientEnabled)
}

func Test_markGradientDependencies_withSingleNodeAndTwoDependenciesWithBothDerivative(t *testing.T) {
	a := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	b := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	c := Add(a, b)

	buildBackwardGraph([]*Tensor{a, b}, c)

	assert.True(t, a.isGradientEnabled)
	assert.True(t, b.isGradientEnabled)
	assert.True(t, c.isGradientEnabled)
}

func Test_markGradientDependencies_withMultipleNodesAndSingleDerivative(t *testing.T) {
	x := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	y := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	a := Add(x, y)
	b := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	c := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	d := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	e := Add(a, b)
	f := Add(c, d)
	g := Add(e, f)
	h := Pow(g, 2)

	buildBackwardGraph([]*Tensor{a}, h)

	assert.True(t, !x.isGradientEnabled)
	assert.True(t, !y.isGradientEnabled)
	assert.True(t, a.isGradientEnabled)
	assert.True(t, !b.isGradientEnabled)
	assert.True(t, !c.isGradientEnabled)
	assert.True(t, !d.isGradientEnabled)
	assert.True(t, e.isGradientEnabled)
	assert.True(t, !f.isGradientEnabled)
	assert.True(t, g.isGradientEnabled)
	assert.True(t, h.isGradientEnabled)
}

func Test_markGradientDependencies_withMultipleNodesAndMultipleDerivatives(t *testing.T) {
	x := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	y := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	a := Add(x, y)
	b := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	c := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	d := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	e := Add(a, b)
	f := Add(c, d)
	g := Add(e, f)
	h := Pow(g, 2)

	buildBackwardGraph([]*Tensor{a, d}, h)

	assert.True(t, !x.isGradientEnabled)
	assert.True(t, !y.isGradientEnabled)
	assert.True(t, a.isGradientEnabled)
	assert.True(t, !b.isGradientEnabled)
	assert.True(t, !c.isGradientEnabled)
	assert.True(t, d.isGradientEnabled)
	assert.True(t, e.isGradientEnabled)
	assert.True(t, f.isGradientEnabled)
	assert.True(t, g.isGradientEnabled)
	assert.True(t, h.isGradientEnabled)
}

func Test_markGradientDependencies_withMultipleNodesAndMultipleDerivativesOnSamePath(t *testing.T) {
	x := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	y := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	a := Add(x, y)
	b := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	c := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	d := Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	e := Add(a, b)
	f := Add(c, d)
	g := Add(e, f)
	h := Pow(g, 2)

	buildBackwardGraph([]*Tensor{a, e}, h)

	assert.True(t, !x.isGradientEnabled)
	assert.True(t, !y.isGradientEnabled)
	assert.True(t, a.isGradientEnabled)
	assert.True(t, !b.isGradientEnabled)
	assert.True(t, !c.isGradientEnabled)
	assert.True(t, !d.isGradientEnabled)
	assert.True(t, e.isGradientEnabled)
	assert.True(t, !f.isGradientEnabled)
	assert.True(t, g.isGradientEnabled)
	assert.True(t, h.isGradientEnabled)
}
