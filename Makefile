check:
	bash check.sh

fast:
	FAST=1 bash check.sh

eq1:
	bash reproduce_eq1.sh

eq2:
	bash reproduce_eq2.sh

.PHONY: check fast eq1 eq2
