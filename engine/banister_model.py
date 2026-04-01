import math

class BanisterModel:
    """
    Implementation of the Banister Fitness-Fatigue (Training Impulse) model.
    Tracks chronic training load (fitness) and acute training load (fatigue)
    using exponential decay.
    """

    def __init__(self, initial_fitness: float = 0.0, initial_fatigue: float = 0.0):
        """
        Initialize the Banister model with baseline metrics.

        Args:
            initial_fitness (float): Starting chronic training load.
            initial_fatigue (float): Starting acute training load.
        """
        # Standard physiological time constants (in days)
        self.tau_fitness = 42.0
        self.tau_fatigue = 7.0
        
        self.fitness = initial_fitness
        self.fatigue = initial_fatigue

    def add_daily_load(self, training_load: float):
        """
        Update fitness and fatigue scores based on daily training load.

        Args:
            training_load (float): The quantifiable stress score of the workout. 
                                   Input 0.0 for a rest day.
        """
        # Calculate exponential decay factors based on respective time constants
        fitness_decay_factor = math.exp(-1 / self.tau_fitness)
        fatigue_decay_factor = math.exp(-1 / self.tau_fatigue)
        
        # Apply decay to existing scores and integrate new training load
        self.fitness = (self.fitness * fitness_decay_factor) + (training_load * (1 - fitness_decay_factor))
        self.fatigue = (self.fatigue * fatigue_decay_factor) + (training_load * (1 - fatigue_decay_factor))

    def get_readiness(self) -> float:
        """
        Calculate the current Training Stress Balance (Readiness).

        Returns:
            float: The difference between fitness and fatigue.
        """
        return self.fitness - self.fatigue


if __name__ == "__main__":
    # Module execution test
    athlete = BanisterModel(initial_fitness=20.0, initial_fatigue=10.0)
    print(f"Pre-workout State  -> Fitness: {athlete.fitness:.1f}, Fatigue: {athlete.fatigue:.1f}, Readiness: {athlete.get_readiness():.1f}")
    
    athlete.add_daily_load(150.0)
    print(f"Post-workout State -> Fitness: {athlete.fitness:.1f}, Fatigue: {athlete.fatigue:.1f}, Readiness: {athlete.get_readiness():.1f}")