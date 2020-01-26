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

	graph.Backward(g) // stopped here

	assert.Equal(t, [][]float64{{9}}, g.mat)
}
