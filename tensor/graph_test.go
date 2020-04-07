package tensor

import (
	"github.com/gokadin/ml-framework/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_graph_some(t *testing.T) {
	graph := NewGraph()
	a := Variable(mat.WithShape(1, 1)).SetData([]float32{2})
	b := Variable(mat.WithShape(1, 1)).SetData([]float32{5})
	c := Add(a, b)
	d := Variable(mat.WithShape(1, 1)).SetData([]float32{4})
	e := Add(c, d)

	graph.Forward(e)

	assert.True(t, mat.NewMat32f(e.Shape(), []float32{11}).Equals32f(e.mat))
}

func Test_nonsnese(t *testing.T) {
	a := Variable(mat.WithShape(1, 1)).SetData([]float32{2})
	b := Variable(mat.WithShape(1, 1)).SetData([]float32{5})
	c := Variable(mat.WithShape(1, 1)).SetData([]float32{6})
	d := Variable(mat.WithShape(1, 1)).SetData([]float32{4})
	e := Add(a, b)
	f := Add(c, d)
	g := Add(e, f)
	graph := NewGraph()

	graph.Forward(g)

	assert.True(t, mat.NewMat32f(g.Shape(), []float32{17}).Equals32f(g.mat))
}

func Test_nonsnese2(t *testing.T) {
	x := Variable(mat.WithShape(1, 1)).SetData([]float32{1})
	y := Variable(mat.WithShape(1, 1)).SetData([]float32{1})
	a := Add(x, y)
	b := Variable(mat.WithShape(1, 1)).SetData([]float32{5})
	c := Variable(mat.WithShape(1, 1)).SetData([]float32{6})
	d := Variable(mat.WithShape(1, 1)).SetData([]float32{4})
	e := Add(a, b)
	f := Sub(c, d)
	g := Add(e, f)
	graph := NewGraph()

	graph.Forward(g)

	assert.True(t, mat.NewMat32f(g.Shape(), []float32{9}).Equals32f(g.mat))
}

func Test_nonsnese3(t *testing.T) {
	x := Variable(mat.WithShape(1, 1)).SetData([]float32{1}).SetName("x")
	y := Variable(mat.WithShape(1, 1)).SetData([]float32{1}).SetName("y")
	a := Add(x, y).SetName("a")
	b := Variable(mat.WithShape(1, 1)).SetData([]float32{5}).SetName("b")
	c := Variable(mat.WithShape(1, 1)).SetData([]float32{6}).SetName("c")
	d := Variable(mat.WithShape(1, 1)).SetData([]float32{4}).SetName("d")
	e := Add(a, b).SetName("e")
	f := Sub(c, d).SetName("f")
	g := Add(e, f).SetName("g")
	graph := NewGraph()

	graph.Backward(g, y)

	assert.True(t, mat.NewMat32f(y.Shape(), []float32{1}).Equals32f(y.grad))
}

func Test_nonsnese4(t *testing.T) {
	x := Variable(mat.WithShape(1, 1)).SetData([]float32{1}).SetName("x")
	y := Variable(mat.WithShape(1, 1)).SetData([]float32{1}).SetName("y")
	a := Add(x, y).SetName("a")
	b := Variable(mat.WithShape(1, 1)).SetData([]float32{5}).SetName("b")
	c := Variable(mat.WithShape(1, 1)).SetData([]float32{6}).SetName("c")
	d := Variable(mat.WithShape(1, 1)).SetData([]float32{4}).SetName("d")
	e := Add(a, b).SetName("e") //7
	f := Sub(c, d).SetName("f") //2
	g := Add(e, f).SetName("g") //9
	h := Pow(g, 3)              //729
	graph := NewGraph()

	graph.Forward(h)
	graph.Backward(h, y)

	assert.True(t, mat.NewMat32f(h.Shape(), []float32{729}).Equals32f(h.mat))
	assert.True(t, mat.NewMat32f(y.Shape(), []float32{243}).Equals32f(y.grad))

	graph.Backward(h, x)

	assert.True(t, mat.NewMat32f(x.Shape(), []float32{243}).Equals32f(x.grad))
}

func Test_nonsnese5(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 2, 1}).SetName("a")
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{0, 3, 1, 1}).SetName("b")
	e := Matmul(a, b).SetName("e")
	graph := NewGraph()
	graph.Forward(e)

	graph.Backward(e, a)

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{3, 2, 3, 2}).Equals32f(a.grad))
}

func Test_nonsnese6(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 2, 1}).SetName("a")
	e := Sum(a, 0).SetName("e")
	graph := NewGraph()
	graph.Forward(e)

	graph.Forward(e)

	assert.True(t, mat.NewMat32f(mat.WithShape(1, 2), []float32{3, 3}).Equals32f(e.mat))
}

func Test_nonsnese7(t *testing.T) {
	a := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 2, 2, 1}).SetName("a")
	b := Variable(mat.WithShape(2, 2)).SetData([]float32{0, 3, 1, 1}).SetName("b")
	e := Matmul(a, b).SetName("e")
	graph := NewGraph()
	graph.Forward(e)

	graph.Backward(e, a)

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{3, 2, 3, 2}).Equals32f(a.grad))

	graph.Backward(e, a)

	assert.True(t, mat.NewMat32f(a.Shape(), []float32{3, 2, 3, 2}).Equals32f(a.grad))
}

func Test_Autograd_Gradient_DerivativeOfDotSubAndPow2(t *testing.T) {
	x := Variable(mat.WithShape(1, 2)).SetData([]float32{0, 1}).SetName("x")
	w := Variable(mat.WithShape(2, 2)).SetData([]float32{1, 1, 1, 1}).SetName("w")
	yHat := Variable(mat.WithShape(1, 2)).SetData([]float32{2, 2}).SetName("yHat")
	e := Pow(Sub(Matmul(x, w), yHat), 2)
	graph := NewGraph()
	graph.Forward(e)

	graph.Backward(e, w)

	assert.True(t, mat.NewMat32f(mat.WithShape(1, 2), []float32{1, 1}).Equals32f(e.grad))
}
