package tensor2

type Graph struct {
	cache map[string]*executionGraph
}

func NewGraph() *Graph {
	return &Graph{
		cache: make(map[string]*executionGraph),
	}
}

func (g *Graph) Forward(tensor *Tensor) {
	if _, ok := g.cache[tensor.id]; !ok {
		g.cache[tensor.id] = newExecutionGraph(tensor)
	}

	g.cache[tensor.id].run()
}

func (g *Graph) Backward(tensor *Tensor) {

}