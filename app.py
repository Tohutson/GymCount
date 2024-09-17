from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved model
pipeline = joblib.load('classifier.pkl')
days = {'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed', 'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'}

def convert_to_12hr_format(hour, minute):
    period = "AM"
    if hour >= 12:
        period = "PM"
        if hour > 12:
            hour -= 12
    if hour == 0:
        hour = 12
    return f"{hour:02d}:{minute:02d} {period}"

@app.route("/", methods=["GET", "POST"])
def index():
    best_time = None
    avg_count = None
    
    if request.method == "POST":
        # Collect form data
        weekday = request.form['weekday']
        min_hour = int(request.form['min_hour'])
        min_minute = int(request.form['min_minute'])
        min_period = request.form['min_period']

        max_hour = int(request.form['max_hour'])
        max_minute = int(request.form['max_minute'])
        max_period = request.form['max_period']

        duration_hours = int(request.form['duration_hours'])
        duration_minutes = int(request.form['duration_minutes'])

        # Convert duration to total minutes
        duration = duration_hours * 60 + duration_minutes

        # Convert start time to 24-hour format
        if min_period == 'PM' and min_hour != 12:
            min_hour += 12
        if min_period == 'AM' and min_hour == 12:
            min_hour = 0

        # Convert end time to 24-hour format
        if max_period == 'PM' and max_hour != 12:
            max_hour += 12
        if max_period == 'AM' and max_hour == 12:
            max_hour = 0

        # Calculate minute ranges
        start_min = min_hour * 60 + min_minute
        end_min = max_hour * 60 + max_minute

        # Create a DataFrame to store potential times and predictions
        potential_times = pd.DataFrame(columns=['Weekday', 'Total_minutes', 'Hour', 'Minute', 'Pred_Count'])
        minute = min_minute
        hour = min_hour
        current_min = start_min
        
        while current_min <= end_min:
            current_time = {'Weekday': days[weekday], 'Total_minutes': current_min, 'Hour': hour, 'Minute': minute}
            current_df = pd.DataFrame([current_time])
            predicted_count = pipeline.predict(current_df)[0]
            current_time['Pred_Count'] = predicted_count
            potential_times = potential_times._append(current_time, ignore_index=True)
            minute += 1
            if minute > 59:
                minute = 0
                hour += 1
            current_min += 1

        intervals = {}
        current_min = start_min
        while current_min + duration <= end_min:
            interval = potential_times[
                (potential_times['Total_minutes'] >= current_min) &
                (potential_times['Total_minutes'] <= current_min + duration)
            ]
            if not interval.empty:
                avg_count_val = interval['Pred_Count'].mean()
                start_hour = interval['Hour'].iloc[0]
                start_minute = interval['Minute'].iloc[0]
                time_str = convert_to_12hr_format(start_hour, start_minute)
                intervals[str(current_min) + " to " + str(current_min + duration)] = [avg_count_val, time_str]
            current_min += 1

        if intervals:
            intervals = {k: v for k, v in sorted(intervals.items(), key=lambda item: item[1])}
            best_time = list(intervals.values())[0][1]
            avg_count = round(list(intervals.values())[0][0], 2)

    return render_template('index.html', best_time=best_time, avg_count=avg_count)

if __name__ == "__main__":
    app.run(debug=True)
