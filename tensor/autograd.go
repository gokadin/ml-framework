package tensor

import "log"

type Autograd struct {
	cache map[string][]*operation
}

func NewAutograd() *Autograd {
	return &Autograd{
		cache: make(map[string][]*operation),
	}
}

func (a *Autograd) Derivative(c, f *Tensor) [][]float64 {
	return computeDerivative(a.getDerivativeGraph(c, f))
}

func (a *Autograd) getDerivativeGraph(c, f *Tensor) []*operation {
	hash := calculateDerivativeHash(c, f)
	if _, ok := a.cache[hash]; ok {
		return a.cache[hash]
	}
	
    a.cache[hash] = createDerivativeGraph(c.operation, f.operation)
    return a.cache[hash]
}

func calculateDerivativeHash(c, f *Tensor) string {
	return c.id + f.id
}

func createDerivativeGraph(c, f *operation) []*operation {
    graph := make([]*operation, 0)
	graph = append(graph, f)

    current := f
	for !current.isLeaf() {
		current = findThatSpecialChild(c, current.children)
		current.isMarked = true
		graph = append(graph, current)
	}

    return graph
}

func findThatSpecialChild(c *operation, children []*operation) *operation {
    for _, child := range children {
    	if child.tensor.id == c.tensor.id {
    		return child
		}
        if leafIsInPath(c, child) {
        	return child
		}
	}
    
    log.Fatal("no child was special")
    return nil
}

func leafIsInPath(c, root *operation) bool {
    if root.tensor.id == c.tensor.id {
    	return true
	}

    for _, child := range root.children {
		if leafIsInPath(c, child) {
			return true
		}
	}

    return false
}

func computeDerivative(graph []*operation) [][]float64 {
	return graph[0].differentiate()
}

