FROM iron/go:dev

ENV SRC_DIR=/go/src/ml-framework

WORKDIR $SRC_DIR

ADD . .

RUN go get -t ./...

CMD [ "go", "run", "main.go" ]
#CMD [ "go", "test", "-cover", "./..." ]
