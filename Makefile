.PHONY: build test fmt lint clean help ci

help:
	@echo "Available targets:"
	@echo "  make build      - Build the project"
	@echo "  make test       - Run tests"
	@echo "  make fmt        - Format code (swift-format)"
	@echo "  make lint       - Run SwiftLint"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make ci         - Run full CI pipeline"

build:
	swift build -c release

test:
	swift test -v

fmt:
	swift-format -i -r .

fmt-check:
	@if swift-format -p -r . > /dev/null ; then echo "✓ Format check passed"; else exit 1; fi

lint:
	swiftlint

clean:
	swift package clean

ci: fmt-check lint test

.DEFAULT_GOAL := help
