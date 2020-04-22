package tensor

//#cgo CFLAGS: -I.
//#cgo LDFLAGS: -L${SRCDIR} -Wl,-rpath,${SRCDIR}
//#include <graph.h>
import "C"

type Graph struct {
	forwardGraphs map[string]*forwardGraph
	backwardGraphs map[string]*backwardGraph
}

func NewGraph() *Graph {
	return &Graph{
		forwardGraphs: make(map[string]*forwardGraph),
		backwardGraphs: make(map[string]*backwardGraph),
	}
}

func (g *Graph) Forward(tensor *Tensor) {
	if _, ok := g.forwardGraphs[tensor.id]; !ok {
		g.forwardGraphs[tensor.id] = buildForwardGraph(tensor)
	}

	//C.testgraph(tensor.op.dependencies()[0]._tensor, tensor.op.dependencies()[1]._tensor, tensor._tensor)

	g.forwardGraphs[tensor.id].run()
}

func (g *Graph) Backward(of *Tensor, derivatives ...*Tensor) {
	id := backwardGraphId(of, derivatives)
	if _, ok := g.backwardGraphs[id]; !ok {
		g.backwardGraphs[id] = buildBackwardGraph(derivatives, of)
	}

	g.backwardGraphs[id].run()
}

func backwardGraphId(of *Tensor, derivatives []*Tensor) string {
	id := of.id
	for _, derivative := range derivatives {
		id += derivative.id
	}
	return id // truncate (md5)
}
