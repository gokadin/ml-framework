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
	assert.True(t, expected.ToMat32f().Equals32f(loss.ToMat32f()))
}

func Test_Criterion_meanSquared_multipleOutputs(t *testing.T) {
	target := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 3), []float32{0.5, 0.5, 0.5}))
	pred := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 3), []float32{1, 1, 1}))
	c := NewCriterion(LossMeanSquared)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	expected := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 3), []float32{0.25, 0.25, 0.25}))
	assert.True(t, expected.ToMat32f().Equals32f(loss.ToMat32f()))
}

func Test_Criterion_meanSquared_multipleAssociations(t *testing.T) {
	target := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(2, 2), []float32{0.5, 0.5, 0.5, 0.5}))
	pred := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(2, 2), []float32{1, 1, 1, 1}))
	c := NewCriterion(LossMeanSquared)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	expected := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 2), []float32{0.25, 0.25}))
	assert.True(t, expected.ToMat32f().Equals32f(loss.ToMat32f()))
}

func Test_Criterion_crossEntropy_onePositiveAssociation(t *testing.T) {
	predMat := mat.FromSlice32f(mat.WithShape(1, 1), []float32{0.5})
	target := tensor.FromMat32(predMat)
	pred := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 1), []float32{0.5}))
	c := NewCriterion(LossSoftmaxCrossEntropy)

	assert.Panics(t, func() {
		c.Forward(pred, target)
	})
}

func Test_Criterion_crossEntropy_multipleClassesOneAssociation(t *testing.T) {
	predMat := mat.FromSlice32f(mat.WithShape(1, 3), []float32{0.5, 0.4, 0.1})
	target := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 3), []float32{1, 0, 0}))
	pred := tensor.FromMat32(predMat.Copy())
	c := NewCriterion(LossSoftmaxCrossEntropy)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	softmaxPredMat := mat.Softmax(predMat)
	expected := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 1), []float32{float32(-math.Log(float64(softmaxPredMat.Data()[0])))}))
	assert.True(t, expected.ToMat32f().Equals32f(loss.ToMat32f()))
}

func Test_Criterion_crossEntropy_multipleClassesMultipleAssociations(t *testing.T) {
	predMat := mat.FromSlice32f(mat.WithShape(3, 3), []float32{0.5, 0.4, 0.1, 0.5, 0.1, 0.4, 0.5, 0.3, 0.2})
	target := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(3, 3), []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}))
	pred := tensor.FromMat32(predMat)
	c := NewCriterion(LossSoftmaxCrossEntropy)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	softmaxPredMat := mat.Softmax(predMat)
	expected := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 1), []float32{
		float32((-(math.Log(float64(softmaxPredMat.Data()[0]))) + (-math.Log(float64(softmaxPredMat.Data()[2]))) + (-math.Log(float64(softmaxPredMat.Data()[8])))) / 3),
	}))
	assert.True(t, expected.ToMat32f().Equals32f(loss.ToMat32f()))
}

func Test_Criterion_crossEntropy_backward(t *testing.T) {
	target := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 4), []float32{0, 1, 0, 0}))
	pred := tensor.Softmax(tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 4), []float32{0.3, 0.5, 0.1, 0.1})))
	c := NewCriterion(LossSoftmaxCrossEntropy)
	graph := tensor.NewGraph()
	loss := c.Forward(pred, target)
	graph.Forward(loss)

	graph.Backward(loss, pred)

	expected := mat.Sub(pred.ToMat32f(), target.ToMat32f())
	assert.True(t, expected.Equals32f(pred.GradientToMat32()))
}

func Test_Criterion_crossEntropy_backward_multiDimension(t *testing.T) {
	target := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(2, 4), []float32{0, 1, 0, 0, 0, 1, 0, 0}))
	pred := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(2, 4), []float32{0.2, 0.5, 0.1, 0.1, 0.2, 0.5, 0.1, 0.1}))
	c := NewCriterion(LossSoftmaxCrossEntropy)
	graph := tensor.NewGraph()
	loss := c.Forward(pred, target)
	graph.Forward(loss)

	graph.Backward(loss, pred)

	expected := mat.Sub(pred.ToMat32f(), target.ToMat32f())
	assert.True(t, expected.Equals32f(pred.GradientToMat32()))
}

func Test_Criterion_binaryCrossEntropy_multipleClassesOneAssociation(t *testing.T) {
	predMat := mat.FromSlice32f(mat.WithShape(1, 4), []float32{0.8, 0.2, 0.6, 0.9})
	target := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(1, 4), []float32{1, 0, 0, 1}))
	pred := tensor.FromMat32(predMat.Copy())
	c := NewCriterion(LossBinaryCrossEntropy)
	graph := tensor.NewGraph()

	loss := c.Forward(pred, target)
	graph.Forward(loss)

	expected := []float32{0.367}
	assert.InDeltaSlice(t, expected, loss.ToMat32f().Data(), 0.0001)
}

func Test_Criterion_binaryCrossEntropy_backward_multiDimension(t *testing.T) {
	target := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(2, 4), []float32{0, 1, 0, 0, 0, 1, 0, 0}))
	pred := tensor.FromMat32(mat.FromSlice32f(mat.WithShape(2, 4), []float32{0.2, 0.5, 0.1, 0.1, 0.2, 0.5, 0.1, 0.1}))
	c := NewCriterion(LossBinaryCrossEntropy)
	graph := tensor.NewGraph()
	loss := c.Forward(pred, target)
	graph.Forward(loss)

	graph.Backward(loss, pred)

	expected := mat.Sub(pred.ToMat32f(), target.ToMat32f())
	assert.True(t, expected.Equals32f(pred.GradientToMat32()))
}
