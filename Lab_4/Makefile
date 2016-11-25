# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

EXECUTABLES	= gold scode pcode

gold:
	+$(MAKE) -C serial_code_golden
scode:
	+$(MAKE) -C serial_code_optimized
pcode:
	+$(MAKE) -C parallel_code
clean:
	rm -f $(EXECUTABLES) $(OBJ)
