package modules

import (
	"github.com/stretchr/testify/assert"
	"math"
	"ml-framework/mat"
	"ml-framework/tensor"
	"testing"
)

func Test_Criterion_meanSquared_oneAssociation(t *testing.T) {
	target := tensor.OfShape(1, 1).SetData([]float32{0.5})
	pred := tensor.OfShape(1, 1).SetData([]float32{1})
	c := NewCriterion(LossMeanSquared)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	expected := tensor.OfShape(1, 1).SetData([]float32{0.25})
	assert.Equal(t, expected.ToFloat32(), loss.ToFloat32())
}

func Test_Criterion_meanSquared_multipleOutputs(t *testing.T) {
	target := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{0.5, 0.5, 0.5}))
	pred := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{1, 1, 1}))
	c := NewCriterion(LossMeanSquared)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{0.25, 0.25, 0.25}))
	assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_meanSquared_multipleAssociations(t *testing.T) {
	target := tensor.Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{0.5, 0.5, 0.5, 0.5}))
	pred := tensor.Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 1, 1, 1}))
	c := NewCriterion(LossMeanSquared)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 2), []float32{0.25, 0.25}))
	assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_crossEntropy_onePositiveAssociation(t *testing.T) {
	target := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
	pred := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{0.5}))
	c := NewCriterion(LossSoftmaxCrossEntropy)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{float32(-math.Log(0.5))}))
	assert.True(t, expected.Data().Equals32f(loss.Data()))
}

func Test_Criterion_crossEntropy_multipleClassesOneAssociation(t *testing.T) {
	target := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{1, 0, 0}))
	pred := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{0.5, 0.4, 0.1}))
	c := NewCriterion(LossSoftmaxCrossEntropy)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{float32(-math.Log(0.5))}))
	assert.True(t, expected.Data().Equals32f(loss.Data()))
}

func Test_Criterion_crossEntropy_multipleClassesMultipleAssociations(t *testing.T) {
	target := tensor.Constant(mat.NewMat32f(mat.WithShape(3, 3), []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}))
	pred := tensor.Constant(mat.NewMat32f(mat.WithShape(3, 3), []float32{0.5, 0.4, 0.1, 0.5, 0.1, 0.4, 0.5, 0.3, 0.2}))
	c := NewCriterion(LossSoftmaxCrossEntropy)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{float32((-(math.Log(0.5)) + (-math.Log(0.1)) + (-math.Log(0.2))) / 3)}))
	assert.True(t, expected.Data().Equals32f(loss.Data()))
}

func Test_Criterion_crossEntropy_backward(t *testing.T) {
	target := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 4), []float32{0, 1, 0, 0}))
	pred := tensor.Softmax(tensor.Constant(mat.NewMat32f(mat.WithShape(1, 4), []float32{0.3, 0.5, 0.1, 0.1})))
	c := NewCriterion(LossSoftmaxCrossEntropy)
	graph := tensor.NewGraph()
	loss := c.Forward(pred, target)
	graph.Forward(loss)

	graph.Backward(loss, pred)

	expected := mat.Sub(pred.Data(), target.Data())
	assert.True(t, expected.Equals32f(pred.Gradient()))
}

func Test_Criterion_crossEntropy_backward_multiDimension(t *testing.T) {
	target := tensor.Constant(mat.NewMat32f(mat.WithShape(2, 4), []float32{0, 1, 0, 0, 0, 1, 0, 0}))
	pred := tensor.Constant(mat.NewMat32f(mat.WithShape(2, 4), []float32{0.2, 0.5, 0.1, 0.1, 0.2, 0.5, 0.1, 0.1}))
	c := NewCriterion(LossSoftmaxCrossEntropy)
	graph := tensor.NewGraph()
	loss := c.Forward(pred, target)
	graph.Forward(loss)

	graph.Backward(loss, pred)

	expected := mat.Sub(pred.Data(), target.Data())
	assert.True(t, expected.Equals32f(pred.Gradient()))
}
