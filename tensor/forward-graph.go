package tensor

type forwardGraph struct {
	startChannels []chan bool
	done chan bool
}

func buildForwardGraph(tensor *Tensor) *forwardGraph {
	fg := &forwardGraph{
		startChannels: make([]chan bool, 0),
		done: make(chan bool),
	}

	fg.build(tensor, fg.done)

	return fg
}

func (fg *forwardGraph) build(tensor *Tensor, lastChan chan bool) {
	c := make(chan bool)

	activeDependencyCount := 0
	for _, dependency := range tensor.op.dependencies() {
		if dependency.op != nil {
			activeDependencyCount++
			fg.build(dependency, c)
		}
	}

	if activeDependencyCount == 0 {
		activeDependencyCount = 1
		fg.startChannels = append(fg.startChannels, c)
	}

	go executeForwardOp(tensor, c, lastChan, activeDependencyCount)
}

func (fg *forwardGraph) run() {
	for _, startChannel := range fg.startChannels {
		startChannel <- true
	}

	<-fg.done
}

func executeForwardOp(tensor *Tensor, in, out chan bool, threshold int) {
	counter := 0
	for range in {
		counter++
		if counter != threshold {
			continue
		}

		counter = 0
		if !tensor.ready {
			tensor.reshapeMat(tensor.op.forwardShape().ToArray()...)
			tensor.forward()
		}
		out <- true
	}
}
