import time


class DrowsyState:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = None
        self.triggered = False

    def confirm(self):
        """
        Confirms CRITICAL state only after persistence
        """
        if self.start_time is None:
            self.start_time = time.time()

        elif time.time() - self.start_time >= self.time_limit:
            if not self.triggered:
                self.triggered = True
                return True

        return False

    def reset(self):
        """
        Resets state when driver becomes alert
        """
        self.start_time = None
        self.triggered = False
