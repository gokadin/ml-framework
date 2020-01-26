package tensor2

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_graph_some(t *testing.T) {
	graph := NewGraph()
	a := Constant([][]float64{{2}})
	b := Constant([][]float64{{5}})
	c := Add(a, b)
	d := Constant([][]float64{{4}})
	e := Add(c, d)

	graph.Forward(e)

	assert.Equal(t, [][]float64{{11}}, e.mat)
}


func Test_nonsnese(t *testing.T) {
	a := Constant([][]float64{{2}})
	b := Constant([][]float64{{5}})
	c := Constant([][]float64{{6}})
	d := Constant([][]float64{{4}})
	e := Add(a, b)
	f := Add(c, d)
	g := Add(e, f)
	graph := NewGraph()

	graph.Forward(g)

	assert.Equal(t, [][]float64{{17}}, g.mat)
}

func Test_nonsnese2(t *testing.T) {
	x := Constant([][]float64{{1}})
	y := Constant([][]float64{{1}})
	a := Add(x, y)
	b := Constant([][]float64{{5}})
	c := Constant([][]float64{{6}})
	d := Constant([][]float64{{4}})
	e := Add(a, b)
	f := Sub(c, d)
	g := Add(e, f)
	graph := NewGraph()

	graph.Forward(g)

	assert.Equal(t, [][]float64{{9}}, g.mat)
}

func Test_nonsnese3(t *testing.T) {
	x := Constant([][]float64{{1}}).SetName("x")
	y := Constant([][]float64{{1}}).SetName("y")
	a := Add(x, y).SetName("a")
	b := Constant([][]float64{{5}}).SetName("b")
	c := Constant([][]float64{{6}}).SetName("c")
	d := Constant([][]float64{{4}}).SetName("d")
	e := Add(a, b).SetName("e")
	f := Sub(c, d).SetName("f")
	g := Add(e, f).SetName("g")
	graph := NewGraph()

	graph.Backward(y, g)

	assert.Equal(t, [][]float64{{1}}, y.grad)
}

func Test_nonsnese4(t *testing.T) {
	x := Constant([][]float64{{1}}).SetName("x")
	y := Constant([][]float64{{1}}).SetName("y")
	a := Add(x, y).SetName("a")
	b := Constant([][]float64{{5}}).SetName("b")
	c := Constant([][]float64{{6}}).SetName("c")
	d := Constant([][]float64{{4}}).SetName("d")
	e := Add(a, b).SetName("e")
	f := Sub(c, d).SetName("f")
	g := Add(e, f).SetName("g")
	h := Pow(g, 3)
	graph := NewGraph()

	graph.Backward(y, h)

	assert.Equal(t, [][]float64{{729}}, h.mat)
	assert.Equal(t, [][]float64{{243}}, y.grad)

	graph.Backward(x, h)

	assert.Equal(t, [][]float64{{243}}, x.grad)
}
