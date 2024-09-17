from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the saved model
pipeline = joblib.load('classifier.pkl')

days = {'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed', 'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'}

# Helper function to parse input time
def parse_time_input(time_str):
    hour, minute = map(int, time_str.split(':'))
    return hour, minute

# Home route with form for input
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect inputs from the form
        day = days[request.form['weekday']]
        min_time = request.form['min_time']
        max_time = request.form['max_time']
        duration = int(request.form['duration'])

        # Parse input times
        min_hour, min_minute = parse_time_input(min_time)
        max_hour, max_minute = parse_time_input(max_time)

        # Convert times to datetime objects for easier manipulation
        start_time = datetime(2000, 1, 1, min_hour, min_minute)
        end_time = datetime(2000, 1, 1, max_hour, max_minute)

        # List to hold all potential times and predictions
        potential_times = []

        current_time = start_time
        while current_time <= end_time:
            # Create time entry
            current_entry = {
                'Weekday': day,
                'Total_minutes': current_time.hour * 60 + current_time.minute,
                'Hour': current_time.hour,
                'Minute': current_time.minute
            }

            # Make the prediction
            current_df = pd.DataFrame([current_entry])
            predicted_count = pipeline.predict(current_df)[0]
            current_entry['Pred_Count'] = predicted_count

            # Append to the list
            potential_times.append(current_entry)

            # Increment time by 1 minute
            current_time += timedelta(minutes=1)

        # Convert list of times to a DataFrame
        potential_times_df = pd.DataFrame(potential_times)

        # Generate intervals and compute the average predicted count for each interval
        intervals = {}
        for start_min in range(min_hour * 60 + min_minute, max_hour * 60 + max_minute - duration + 1):
            interval = potential_times_df[
                (potential_times_df['Total_minutes'] >= start_min) &
                (potential_times_df['Total_minutes'] < start_min + duration)
            ]
            if not interval.empty:
                avg_count = interval['Pred_Count'].mean()
                start_time_str = f"{interval['Hour'].iloc[0]:02}:{interval['Minute'].iloc[0]:02}"
                intervals[f"{start_min} to {start_min + duration}"] = [avg_count, start_time_str]

        # Sort intervals by the average predicted count and get the best time
        best_interval = min(intervals.items(), key=lambda item: item[1][0])

        # Display results on the same page
        return render_template('index.html', best_time=best_interval[1][1], avg_count=best_interval[1][0])

    # GET request, show the form
    return render_template('index.html')

# Run the app locally
if __name__ == '__main__':
    app.run(debug=True)
