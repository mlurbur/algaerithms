import argparse

# argparse argument validators
def validate_year(year):
    year_as_int = int(year)
    MIN_YEAR, MAX_YEAR = 2003, 2021
    if year_as_int < MIN_YEAR or year_as_int > MAX_YEAR:
        raise argparse.ArgumentTypeError(f"{year_as_int} is not in the range [{MIN_YEAR}, {MAX_YEAR}]")
    return year_as_int

def validate_start_day(start_day):
    start_day_as_int = int(start_day)
    MIN_START_DAY, MAX_START_DAY = 1, 365
    if start_day_as_int < MIN_START_DAY or start_day_as_int > MAX_START_DAY:
        raise argparse.ArgumentTypeError(f"{start_day} is not in the range [{MIN_START_DAY}, {MAX_START_DAY}]")
    return start_day_as_int

def validate_end_day(end_day):
    end_day_as_int = int(end_day)
    MIN_END_DAY, MAX_END_DAY = 1, 365
    if end_day_as_int < MIN_END_DAY or end_day_as_int > MAX_END_DAY:
        raise argparse.ArgumentTypeError(f"{end_day_as_int} is not in the range [{MIN_END_DAY}, {MAX_END_DAY}]")
    return end_day_as_int

def validate_time_window(time_window):
    time_window_as_int = int(time_window)
    MIN_TIME_WINDOW, MAX_TIME_WINDOW = 0, 5
    if time_window_as_int < MIN_TIME_WINDOW or time_window_as_int > MAX_TIME_WINDOW:
        raise argparse.ArgumentTypeError(f"{time_window_as_int} is not in the range [{MIN_TIME_WINDOW}, {MAX_TIME_WINDOW}]")
    return time_window_as_int

def validate_num_neighbors(num_neighbors):
    num_neighbors_as_int = int(num_neighbors)
    MIN_NUM_NEIGHBORS, MAX_NUM_NEIGHBORS = 0, 5
    if num_neighbors_as_int < MIN_NUM_NEIGHBORS or num_neighbors_as_int > MAX_NUM_NEIGHBORS:
        raise argparse.ArgumentTypeError(f"{num_neighbors_as_int} is not in the range [{MIN_NUM_NEIGHBORS}, {MAX_NUM_NEIGHBORS}]")
    return num_neighbors_as_int

def validate_data_types (data_types):
    data_types = data_types.split(",")
    VALID_DATA_TYPES = ["chlorophyll", "par", "sst", "ice", "depth", "meanlat", "meanlon"]
    for data_type in data_types:
        if data_type not in VALID_DATA_TYPES:
            raise argparse.ArgumentTypeError(f"{data_type} is not a valid data type")
    return data_types

def validate_model(model):
    VALID_MODELS = ["FFN", "RNN"]
    if model not in VALID_MODELS:
        raise argparse.ArgumentTypeError(f"{model} is not a valid model")
    return model

# argparse argument relation validators
def validate_day_range(start_day, end_day):
    if start_day > end_day:
        raise argparse.ArgumentTypeError(f"start day {start_day} is greater than end day {end_day}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=validate_model, help="the model to run, either FFN (feed-forward) or RNN (LSTM)")
    
    parser.add_argument("year", type=validate_year, help="the year whose data the model should take in")
    parser.add_argument("start_day", type=validate_start_day, help="julian day on which the data should begin")
    parser.add_argument("end_day", type=validate_end_day, help="julian day on which the data should end")
    parser.add_argument("time_window", type=validate_time_window, help="the number of previous days whose data should be included")
    parser.add_argument("num_neighbors", type=validate_num_neighbors, help="the width of the points surrounding the point of interest whose data should be included")
    parser.add_argument("data_types", type=validate_data_types, help="the types of data to include in the inputs, separated by commas (valid data types are chlorophyll, par, sst, ice, depth, meanlat, meanlon)")
    parser.add_argument("-p", "--preprocess", help="whether or not the data should be preprocessed first", action="store_true")
    parser.add_argument("-r", "--run", help="whether or not the model should be trained and tested", action="store_true")
    parser.add_argument("-v", "--verbose", help="print preprocessing steps verbosely", action="store_true")

    args = parser.parse_args()

    validate_day_range(args.start_day, args.end_day)

    return args
