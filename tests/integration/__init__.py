"""Integration tests: need downloaded data or a trained model.

Marked ``@pytest.mark.integration`` and excluded from the default run. Each
must also SKIP rather than fail when its inputs are absent, so a clean checkout
stays green.
"""
