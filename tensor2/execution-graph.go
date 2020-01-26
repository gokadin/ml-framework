package tensor2

type executionGraph struct {
	startChannels []chan bool
	done chan bool
}

func newExecutionGraph(tensor *Tensor) *executionGraph {
	eg := &executionGraph{
		startChannels: make([]chan bool, 0),
		done: make(chan bool),
	}

	eg.build(tensor, eg.done)

	return eg
}

func (eg *executionGraph) build(tensor *Tensor, lastChan chan bool) {
	c := make(chan bool)

	activeDependencyCount := 0
	for _, dependency := range tensor.op.dependencies() {
		if dependency.op != nil {
			activeDependencyCount++
			eg.build(dependency, c)
		}
	}

	if activeDependencyCount == 0 {
		activeDependencyCount = 1
		eg.startChannels = append(eg.startChannels, c)
	}

	go executeOp(tensor, c, lastChan, activeDependencyCount)
}

func (eg *executionGraph) run() {
	for _, startChannel := range eg.startChannels {
		startChannel <- true
	}

	<-eg.done
}

func executeOp(tensor *Tensor, in, out chan bool, threshold int) {
	counter := 0
	for range in {
		counter++
		if counter != threshold {
			continue
		}

		counter = 0
		tensor.forward()
		out <- true
	}
}
