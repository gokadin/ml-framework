package tensor2

type Graph struct {
	forwardGraphs map[string]*forwardGraph
}

func NewGraph() *Graph {
	return &Graph{
		forwardGraphs: make(map[string]*forwardGraph),
	}
}

func (g *Graph) Forward(tensor *Tensor) {
	if _, ok := g.forwardGraphs[tensor.id]; !ok {
		g.forwardGraphs[tensor.id] = buildForwardGraph(tensor)
	}

	g.forwardGraphs[tensor.id].run()
}

func (g *Graph) Backward(tensor *Tensor) {

}