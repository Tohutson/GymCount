# Gym Attendance Predictor

This is a Flask-based web application that predicts the best time to visit the gym based on the time range you provide. It uses a machine learning model (`classifier.pkl`) to predict the gym crowd size based on features such as the day of the week, start time, end time, and workout duration. The model was trained on data from the University of Pittsburgh found [here](https://www.studentaffairs.pitt.edu/campus-recreation/facilities-hours/live-facility-counts).

## Features

- Input your preferred workout day, earliest start time, latest end time, and workout duration.
- The app provides the best time to go to the gym based on historical data.
- Results are displayed in an easy-to-read 12-hour `AM/PM` format.

## Requirements

Before running the app, ensure you have the following dependencies installed:

- Python 3.x
- Flask
- pandas
- joblib

You can install the required Python libraries by running:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/gym-attendance-predictor.git
cd gym-attendance-predictor
```
2. Run train.py
```bash
python train.py
```
3. Run the Flask app
```bash
python app.py
```
4. Access the web page
Visit http://127.0.0.1:5000/

## Future Improvements
-Allow for different locations
-Retrain with more data

## License
This project is licensed under the MIT License - see the [License](https://www.mit.edu/~amini/LICENSE.md)

## Author: Trey Hutson (Tohutson)