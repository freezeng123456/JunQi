all:
	$(MAKE) -C legacy_engine
	$(MAKE) -C legacy_gui

clean:
	$(MAKE) -C legacy_engine clean
	$(MAKE) -C legacy_gui clean

legacy-engine:
	$(MAKE) -C legacy_engine

legacy-gui:
	$(MAKE) -C legacy_gui

test:
	python3 run_tests.py --profile core

test-rl:
	python3 run_tests.py --profile rl

lint-critical:
	python3 -m ruff check junqi_core junqi_rl scripts tests \
		--select E9,F63,F7,F82,F811

check: lint-critical test

run:
	./run_mac.sh

.PHONY: all clean legacy-engine legacy-gui test test-rl lint-critical check run
