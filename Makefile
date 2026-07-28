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

.PHONY: all clean legacy-engine legacy-gui
