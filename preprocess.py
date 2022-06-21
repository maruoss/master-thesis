# FEATURE ENGINEERING

# import pandas as pd

def feature_engineer(data):
    """
    Arguments:
    data: pandas.DataFrame that must have specific columns.

    """
    # Bid-Ask spread: (Ask - Bid) / Ask
    data["best_bid"] = (data["best_offer"] - data["best_bid"]) / (data["best_offer"])
    data = data.rename(columns={"best_bid": "ba_spread_option"}).drop(["best_offer"], axis=1)

    # Gamma: multiply by spotprice and divide by 100
    data["gamma"] = data["gamma"] * data["spotprice"] / 100 #following Bali et al. (2021)

    # Theta: scale by spotprice
    data["theta"] = data["theta"] / data["spotprice"] #following Bali et al. (2021)

    # Vega: scale by spotprice
    data["vega"] = data["vega"] / data["spotprice"] #following Bali et al. (2021)

    # Time to Maturity: cale by number of days in year: 365
    data["days_to_exp"] = data["days_to_exp"] / 365

    # Moneyness: Strike / Spot (K / S)
    data["strike_price"] = data["strike_price"] / data["spotprice"] # K / S
    data = data.rename(columns={"strike_price": "moneyness"})

    # Forward Price ratio: Forward / Spot
    data["forwardprice"] = data["forwardprice"] / data["spotprice"]

    # Drop redundant/ unimportant columns
    data = data.drop(["cfadj", "days_no_trading", "spotprice", "adj_spot"], axis=1)

    return data


# binary y label generator
def binary_categorize(y):
    """
    Input: continuous target variable 

    Output: 1 for positive returns, 
            0 for negative returns
    """
    if y > 0:
        return 1
    else:
        return 0


# multiclass y label generator
def multi_categorize(y):
    """
    Input: continuous target variable

    Output: multi class
    """
    if y > 0.05:
        return 1
    elif y < -0.05:
        return -1
    else:
        return 0

# create gridsearch timeseries splits
class CV_splitter:
    """ Generator for sklearn gridsearch cv
    Args:
    dates: pandas.Series of datetime,
    init_train_length: int,
    val_length: int
    """
    def __init__(self, dates, init_train_length=5, val_length=2):
        # find indeces where years change (will ignore last year end in dates)
        self.val_length = val_length
        self.eoy_idx =  np.where((dates.dt.year.diff() == 1))[0] - 1
        self.eoy_idx = np.append(self.eoy_idx, len(dates) - 1) #append end of year of last year in dates

        assert init_train_length + val_length <= len(self.eoy_idx) + 1, "defined train and val are larger "\
            "than number of years in dataset"
        assert init_train_length > 0, "init_train_length must be strictly greater than 0"

        # align
        self.train_start_idx = init_train_length - 1

        self.train_indeces = self.eoy_idx[self.train_start_idx:]
        self.val_indeces = self.eoy_idx[self.train_start_idx + val_length:]

    def generate(self):
        for i in range(len(self.eoy_idx) - (self.train_start_idx + self.val_length)):
            yield (list(range(self.train_indeces[i] + 1)), 
                   list(range(self.train_indeces[i]+1, self.val_indeces[i]+1)))

