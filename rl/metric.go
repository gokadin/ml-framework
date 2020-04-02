package rl

import (
	"fmt"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"sync"
	"time"
)

type metricEvents struct {
	trainingStarted chan bool
	trainingFinished chan bool
	epochStarted chan bool
	epochFinished chan bool
	batchStarted chan int
	batchFinished chan bool
	forwardStarted chan bool
	forwardFinished chan bool
	backwardStarted chan bool
	backwardFinished chan bool
	gameActionTaken chan bool
	gameWon chan bool
	gameFinished chan bool
	statusUpdate chan bool
	loss chan float32
}

func makeMetricEvents() metricEvents {
	return metricEvents{
		trainingStarted: make(chan bool),
		trainingFinished: make(chan bool),
		epochStarted: make(chan bool),
		epochFinished: make(chan bool),
		batchStarted: make(chan int),
		batchFinished: make(chan bool),
		forwardStarted: make(chan bool),
		forwardFinished: make(chan bool),
		backwardStarted: make(chan bool),
		backwardFinished: make(chan bool),
		gameActionTaken: make(chan bool),
		gameWon: make(chan bool),
		gameFinished: make(chan bool),
		statusUpdate: make(chan bool),
		loss: make(chan float32),
	}
}

type metricTiming struct {
	timeMs int64
	timeAveMs int64
	iteration int
}

type metric struct {
	timings map[string]*metricTiming
	events metricEvents
	counters map[string]float64
	p *plot.Plot
	lossLine *plotter.Line
}

func newMetric() *metric {
	m := &metric{
		events: makeMetricEvents(),
		timings: make(map[string]*metricTiming),
		counters: make(map[string]float64),
	}
	m.timings["epoch"] = &metricTiming{}
	m.timings["batch"] = &metricTiming{}
	m.timings["forward"] = &metricTiming{}
	m.timings["backward"] = &metricTiming{}
	p, _ := plot.New()
	m.p = p
	m.p.X.Label.Text = "Epochs"
	m.p.Y.Label.Text = "Loss"
	line, _ := plotter.NewLine(plotter.XYs{})
	m.lossLine = line
	return m
}

func (m *metric) start() {
	go m.receiveEvents()
}

func (m *metric) receiveEvents() {
	for {
		select {
		case <- m.events.trainingStarted:
			fmt.Println("beginning training")
		case <- m.events.trainingFinished:
			fmt.Println("training finished")
		case <- m.events.epochStarted:
			m.timings["epoch"].timeMs = time.Now().UnixNano()
		case <- m.events.epochFinished:
			m.counters["epochs"]++
			//epochTimeMs := (time.Now().UnixNano() - m.timings["epoch"].timeMs) / int64(time.Millisecond)
			//batchTimeAveMs := m.timings["batch"].timeAveMs / (int64(m.timings["batch"].iteration) + 1)
			//m.timings["batch"].timeAveMs = 0
			//forwardTimeAveMs := m.timings["forward"].timeAveMs / (int64(m.timings["forward"].iteration) + 1)
			//m.timings["forward"].timeAveMs = 0
			//m.timings["forward"].iteration = 0
			//backwardTimeAveMs := m.timings["backward"].timeAveMs / (int64(m.timings["backward"].iteration) + 1)
			//m.timings["backward"].timeAveMs = 0
			//m.timings["backward"].iteration = 0
			////fmt.Printf("epoch %d finished in %dms with loss %f\n", m.timings["epoch"].iteration, epochTimeMs, -1.2)
			//fmt.Printf("ave batch: %dms ave forward: %dms ave backward: %dms\n", batchTimeAveMs, forwardTimeAveMs, backwardTimeAveMs)
		//case batch := <- m.events.batchStarted:
		//	m.timings["batch"].iteration = batch
		//	m.timings["batch"].timeMs = time.Now().UnixNano()
		//case <- m.events.batchFinished:
		//	batchTimeMs := (time.Now().UnixNano() - m.timings["batch"].timeMs) / int64(time.Millisecond)
		//	m.timings["batch"].timeAveMs += batchTimeMs
		//case <- m.events.forwardStarted:
		//	m.timings["forward"].iteration++
		//	m.timings["forward"].timeMs = time.Now().UnixNano()
		//case <- m.events.forwardFinished:
		//	forwardTimeMs := (time.Now().UnixNano() - m.timings["forward"].timeMs) / int64(time.Millisecond)
		//	m.timings["forward"].timeAveMs += forwardTimeMs
		//case <- m.events.backwardStarted:
		//	m.timings["backward"].iteration++
		//	m.timings["backward"].timeMs = time.Now().UnixNano()
		//case <- m.events.backwardFinished:
		//	backwardTimeMs := (time.Now().UnixNano() - m.timings["backward"].timeMs) / int64(time.Millisecond)
		//	m.timings["backward"].timeAveMs += backwardTimeMs
		case <- m.events.gameActionTaken:
			m.counters["gameActionsTotal"]++
		case <- m.events.gameWon:
			m.counters["gameWins"]++
		case <- m.events.gameFinished:
			m.counters["totalGames"]++
		case loss := <- m.events.loss:
			m.counters["lossTotal"] += float64(loss)
			m.lossLine.XYs = append(m.lossLine.XYs, plotter.XY{Y: float64(loss), X: m.counters["epochs"]})
		case <- m.events.statusUpdate:
			fmt.Println(fmt.Sprintf("epoch %d   loss %f   aveMoves %2.f   success %2.f%%",
				int(m.counters["epochs"]),
				m.counters["lossTotal"] / m.counters["gameActionsTotal"],
				m.counters["gameActionsTotal"] / m.counters["epochs"],
				m.counters["gameWins"] * 100 / m.counters["totalGames"]))
		}
	}
}

func (m *metric) finalize(wg *sync.WaitGroup) {
	defer wg.Done()

	m.p.Add(m.lossLine)
	if err := m.p.Save(10*vg.Inch, 10*vg.Inch, "loss-per-epoch-w4.png"); err != nil {
		panic(err)
	}
}
