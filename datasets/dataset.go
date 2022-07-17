package datasets

import (
	"log"
	"math/rand"
	"ml-framework/mat"
)

const TrainingSetX = "trainingSetX"
const TrainingSetY = "trainingSetY"
const ValidationSetX = "validationSetX"
const ValidationSetY = "validationSetY"

const defaultBatchSize = 1

type Dataset struct {
	name          string
	sets          map[string]*set
	batchSize     int
	batchCounter  int
	shouldShuffle bool
}

func NewDataset() *Dataset {
	return &Dataset{
		sets:      make(map[string]*set),
		batchSize: defaultBatchSize,
	}
}

func (d *Dataset) SetName(name string) {
	d.name = name
}

func (d *Dataset) AddData(name string, mat *mat.M32f) *set {
	d.sets[name] = &set{mat}
	return d.sets[name]
}

func (d *Dataset) Get(name string) *set {
	if set, ok := d.sets[name]; ok {
		return set
	}

	log.Fatalf("set %s does not exist", name)
	return nil
}

func (d *Dataset) Shape() mat.Shape {
	if set, ok := d.sets[TrainingSetX]; ok {
		return set.data.Shape()
	}

	log.Fatal("training set does not exist")
	return mat.Shape{}
}

func (d *Dataset) Shuffle() *Dataset {
	d.shouldShuffle = true
	return d
}

func (d *Dataset) DisableShuffle() *Dataset {
	d.shouldShuffle = false
	return d
}

func (d *Dataset) BatchSize() int {
	return d.batchSize
}

func (d *Dataset) NumBatches() int {
	return d.sets[TrainingSetX].data.Shape().D[0] / d.BatchSize()
}

func (d *Dataset) SetBatchSize(batchSize int) *Dataset {
	d.batchSize = batchSize
	return d
}

func (d *Dataset) HasNextBatch() bool {
	result := d.batchCounter < d.NumBatches()
	if !result {
		d.batchCounter = 0
	}
	return result
}

func (d *Dataset) ResetBatchCounter() {
	d.batchCounter = 0
}

func (d *Dataset) NextBatch() (*mat.M32f, *mat.M32f) {
	if d.batchCounter == 0 && d.shouldShuffle {
		d.shuffleData()
	}

	fromIndex := d.batchCounter * d.batchSize
	_ = fromIndex // remove
	toIndex := (d.batchCounter + 1) * d.batchSize
	if toIndex > d.sets[TrainingSetX].data.Shape().D[0] {
		toIndex = d.sets[TrainingSetX].data.Shape().D[0]
	}

	d.batchCounter++

	return d.sets[TrainingSetX].data.Slice(mat.Dim(), mat.Dim()), d.sets[TrainingSetY].data.Slice(mat.Dim(), mat.Dim()) // fix
	//return d.sets[TrainingSetX].data.Slice(fromIndex, toIndex), d.sets[TrainingSetY].data.Slice(fromIndex, toIndex)
}

func (d *Dataset) BatchCounter() int {
	return d.batchCounter
}

func (d *Dataset) shuffleData() {
	matX := d.sets[TrainingSetX].data.Data()
	matY := d.sets[TrainingSetY].data.Data()
	rand.Shuffle(d.sets[TrainingSetX].data.Shape().D[0]*d.sets[TrainingSetX].data.Shape().D[1], func(i, j int) {
		matX[i], matX[j] = matX[j], matX[i]
		matY[i], matY[j] = matY[i], matY[i]
	})
}
