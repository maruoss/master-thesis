import numpy as np


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


# Binary y label generator.
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


# Multiclass y label generator.
def multi_categorize(y: float, classes:int):
    """
    Creates categorical labels from continuous values.

        Args:
            y (float):      continuous target variable (option return)
            classes (int):  number of classes to create
        Returns:
            (int):          class assignment
        CAREFUL: classes have to be between [0, C) for F.crossentropyloss.
    """
    if classes == 3:
        # thresholds: +/- 5%
        if y > 0.05:
            return 2
        elif y < -0.05:
            return 0
        else:
            return 1
    elif classes == 5:
        # thresholds: +/- 2.5% and +/- 5%
        if y > 0.05:
            return 4
        elif (y > 0.025 and y <= 0.05):
            return 3
        elif (y >= -0.05 and y < -0.025):
            return 1
        elif (y < -0.05):
            return 0
        else:
            return 2
    else:
        raise ValueError("Only multi for 3 or 5 classes implemented right now.")


class YearEndIndeces:
    """Generator for indices where years change.

        Args:
            dates (pandas.Series):      series of datetimes,
            init_train_length (int):    initial train length,
            val_length (int):           validation length
    """
    def __init__(self, dates, init_train_length, val_length, test_length):
        # Find indeces where years change.
        self.val_length = val_length
        self.test_length = test_length
        # Get end of month indeces for slicing.
        # TECHNICALLY its start of month indeces, i.e. first row of January 31,
        # but because for slicing [:idx], idx is not included, we name it end of
        # year here.
        self.eoy_idx =  np.where((dates.dt.year.diff() == 1))[0]
        # Append last row as end of year of last year.
        self.eoy_idx = np.append(self.eoy_idx, len(dates))

        assert init_train_length + val_length + test_length <= len(self.eoy_idx), \
            ("defined train and val are larger than eoy_indeces generated")
        assert init_train_length > 0, "init_train_length must be strictly greater than 0"

        # The 4th idx in eoy_idx is the end of year 5. -> Subtract 1.
        self.train_start_idx = init_train_length - 1

        self.train_eoy = self.eoy_idx[self.train_start_idx:-(val_length+test_length)]
        self.val_eoy = self.eoy_idx[self.train_start_idx + val_length:-test_length]
        # For generate_idx():
        self.test_eoy = self.eoy_idx[self.train_start_idx + val_length + test_length:]

    # def generate(self):
    #     for i in range(len(self.eoy_idx) - (self.train_start_idx + self.val_length)):
    #         yield (list(range(self.train_eoy[i])),
    #                list(range(self.train_eoy[i], self.val_eoy[i])))

    def generate_idx(self):
        for i in range(len(self.eoy_idx) - (self.train_start_idx + self.val_length 
                        + self.test_length)):
            yield ({"train": self.train_eoy[i], 
                    "val": self.val_eoy[i], 
                    "test": self.test_eoy[i]}
                )
    