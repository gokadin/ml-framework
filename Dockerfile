FROM iron/go:dev

ENV SRC_DIR=/go/src/github.com/gokadin/ml-framework

WORKDIR $SRC_DIR

ADD . .

RUN go get ./...

CMD [ "go", "test", "-cover", "./..." ]