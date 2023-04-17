class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_value, initial_value=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_value = final_value
        self.initial_value = initial_value

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_value + fraction * (self.final_value - self.initial_value)