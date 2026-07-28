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

legacy-sanitize:
	$(MAKE) -C legacy_engine clean sanitize
	python3 tools/test_legacy_protocol.py legacy_engine/bin/JunQiEngine
	$(MAKE) -C legacy_gui clean sanitize

test:
	python3 run_tests.py --profile core

test-rl:
	python3 run_tests.py --profile rl

lint-critical:
	python3 -m ruff check junqi_core junqi_rl scripts tests \
		--select E9,F63,F7,F82,F811

lint-core:
	python3 -m ruff check junqi_core

typecheck:
	python3 -m mypy junqi_core

check: lint-critical lint-core typecheck test

run:
	./run_mac.sh

.PHONY: all clean legacy-engine legacy-gui legacy-sanitize test test-rl lint-critical lint-core typecheck check run
