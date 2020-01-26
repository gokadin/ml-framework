package tensor2

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

	g.forwardGraphs[tensor.id].run()
}

func (g *Graph) Backward(derivative, of *Tensor) {
	if _, ok := g.backwardGraphs[derivative.id + of.id]; !ok {
		g.backwardGraphs[derivative.id + of.id] = buildBackwardGraph(derivative, of)
	}

	if _, ok := g.forwardGraphs[of.id]; !ok {
		g.Forward(of)
	}

	g.backwardGraphs[derivative.id + of.id].run()
}