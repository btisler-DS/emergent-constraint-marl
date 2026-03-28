.PHONY: test pilot clean

test:
	cd backend && python -m pytest tests/ -v

pilot:
	cd backend && python run_p6_pilot.py

clean:
	rm -f confirmatory_run_p6_pilot.log p6_pilot_summary.json
	rm -rf backend/__pycache__ backend/simulation/__pycache__
