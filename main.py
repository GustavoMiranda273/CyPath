"""
CyPath API Gateway
Handles routing, form submissions, and interacts with the Banister model.
"""

from flask import Flask, render_template, request
from engine.banister_model import BanisterModel

app = Flask(__name__)

# Route: Dashboard initialization and form handling
@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Dashboard route.
    Handles both initial page loads (GET) and new workout submissions (POST).
    """
    # Initialize the modeling engine with baseline parameters
    athlete = BanisterModel(initial_fitness=20.0, initial_fatigue=10.0)
    
    # Check if the user submitted a new workout via the HTML form
    if request.method == 'POST':
        try:
            # Extract the training load from the form input
            training_load = float(request.form.get('training_load', 0))
            athlete.add_daily_load(training_load)
        except ValueError:
            # Fallback in case of invalid input (e.g., empty or non-numeric)
            athlete.add_daily_load(0.0)
    else:
        # Default behavior for a normal page load (rest day / 0 load)
        athlete.add_daily_load(0.0)
    
    # Extract and format metrics for the frontend presentation layer
    current_fitness = round(athlete.fitness, 1)
    current_fatigue = round(athlete.fatigue, 1)
    current_readiness = round(athlete.get_readiness(), 1)
    
    return render_template('dashboard.html', 
                           fitness=current_fitness, 
                           fatigue=current_fatigue, 
                           readiness=current_readiness)

if __name__ == '__main__':
    # Execute the application in development mode
    app.run(debug=True)